//! A read-only picture of the world for the renderer and UI, published by the sim thread.

use std::sync::Arc;

use crate::body::PartShape;
use crate::genome::Traits;
use crate::params::SimParams;
use crate::sim::{LifeStats, ObstacleShape, Perf, Sim, SpeciesInfo};
use crate::stats::{Sample, Stats};
use crate::terrain::Terrain;

#[derive(Clone, Debug)]
pub struct PartView {
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub shape: PartShape,
}

#[derive(Clone, Debug)]
pub struct CreatureView {
    pub id: u64,
    pub color: [f32; 3],
    pub parts: Vec<PartView>,
    pub energy: f32,
    pub health: f32,
    pub diet: f32,
    pub generation: u32,
    pub age: f32,
    pub neurons: u32,
    pub attacking: bool,
    pub mating: bool,
    pub drowning: bool,
    /// How submerged the torso is, 0..1.
    pub in_water: f32,
    /// What its life has achieved so far (see `Sim::score_parts`).
    pub score: f32,
    pub mass: f32,
    pub children: u32,
    pub kills: u32,
    /// Food and meat eaten.
    pub food: f32,
    /// Net distance travelled.
    pub distance: f32,
    /// Its voice genes (see `Traits::voice`), how loudly it's calling (0 silent) and at what pitch.
    pub voice: [f32; 4],
    pub call: f32,
    pub call_pitch: f32,
    /// Ground speed of the torso, m/s.
    pub speed: f32,
}

/// Something worth watching, for the camera director and the sound effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HighlightKind {
    /// `subject` was born to `other`.
    Birth,
    /// `subject` bit `other`.
    Attack,
    /// `subject` killed `other` (at `other`'s last position).
    Kill,
    Drowning,
    /// `subject` is the first of a new generation.
    Record,
    /// `subject` hit the water.
    Splash,
    /// Part of `subject` hit the ground; `strength` says how hard.
    Impact,
    /// `subject` is calling out; `strength` says how loudly.
    Call,
}

#[derive(Clone, Copy, Debug)]
pub struct Highlight {
    /// Increases by one per highlight within a world.
    pub seq: u64,
    pub kind: HighlightKind,
    pub subject: u64,
    pub other: Option<u64>,
    pub pos: [f32; 3],
    /// How hard, for impacts: about 1 for a creature landing from the sky.
    pub strength: f32,
}

#[derive(Clone, Debug)]
pub struct FoodView {
    pub id: u64,
    pub pos: [f32; 3],
    pub energy: f32,
}

#[derive(Clone, Debug)]
pub struct ObstacleView {
    pub id: u64,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub shape: ObstacleShape,
    pub dynamic: bool,
}

#[derive(Clone, Debug)]
pub struct CreatureDetail {
    pub id: u64,
    pub generation: u32,
    pub parents: Option<(u64, u64)>,
    pub rained: bool,
    pub age: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub health: f32,
    pub max_health: f32,
    pub oxygen: f32,
    pub hydration: f32,
    pub mate_drive: f32,
    /// Muscle power per kg, W/kg.
    pub exertion: f32,
    pub mass: f32,
    /// Torso plus spine segments.
    pub trunk: usize,
    /// Limb segments.
    pub limbs: usize,
    pub hinges: usize,
    pub swivels: usize,
    pub traits: Traits,
    pub life: LifeStats,
    pub inputs: Vec<f32>,
    pub hidden: Vec<f32>,
    pub outputs: Vec<f32>,
    /// Mean absolute size of the lifetime-learned weight offsets.
    pub learned: f32,
    /// Mean absolute size of the inherited output weights, to compare `learned` with.
    pub inherited: f32,
    /// Its score so far: eating, hunting, breeding, travelling.
    pub score: [f32; 4],
    pub brain_cost: f32,
    pub synapses: usize,
    pub neurons: usize,
    pub basal_cost: f32,
    /// A sketch of the brain's wiring, refreshed about once a second.
    pub wiring: Arc<crate::wiring::Wiring>,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub tick: u64,
    pub time: f64,
    pub creatures: Vec<CreatureView>,
    pub plants: Vec<FoodView>,
    pub meat: Vec<FoodView>,
    pub obstacles: Vec<ObstacleView>,
    pub terrain: Arc<Terrain>,
    pub params: SimParams,
    pub weather: crate::weather::Weather,
    pub stats: Stats,
    pub history: Arc<Vec<Sample>>,
    pub species: Vec<SpeciesInfo>,
    pub perf: Perf,
    pub brain_backend: String,
    /// The going energy price of brains (1 is the `brain_energy_cost` param as set).
    pub brain_price: f32,
    pub brain_memory_bytes: u64,
    pub selected: Option<CreatureDetail>,
    pub grabbing: bool,
    pub pool_size: usize,
    pub pool_best: f32,
    // Filled in by the runner.
    pub paused: bool,
    pub speed: f32,
    pub steps_per_sec: f32,
    /// CPU used by the whole program, in percent of one core.
    pub cpu_percent: f32,
    pub events: Vec<String>,
    /// The latest highlights, oldest first.
    pub highlights: Arc<Vec<Highlight>>,
    /// Changes whenever a different world is loaded or created (ids restart in each world).
    pub world_epoch: u64,
    /// Everything in the zoo (shared across worlds and sessions).
    pub zoo: Arc<Vec<crate::zoo::ZooMeta>>,
    pub zoo_dir: String,
    /// This world first, then any background worlds.
    pub worlds: Vec<crate::islands::WorldStatus>,
}

impl Sim {
    pub fn snapshot(&self, terrain: Arc<Terrain>, history: Arc<Vec<Sample>>) -> Snapshot {
        let bodies = &self.world.bodies;
        let pose = |h| {
            bodies
                .get(h)
                .map(|b: &rapier3d::prelude::RigidBody| {
                    (b.translation().to_array(), b.rotation().to_array())
                })
                .unwrap_or(([0.0; 3], [0.0, 0.0, 0.0, 1.0]))
        };
        let creatures = self
            .creatures
            .iter()
            .map(|c| CreatureView {
                id: c.id,
                color: c.genome.color(),
                parts: c
                    .parts
                    .iter()
                    .zip(&c.plan)
                    .map(|(p, spec)| {
                        let (pos, rot) = pose(p.body);
                        PartView {
                            pos,
                            rot,
                            shape: spec.shape,
                        }
                    })
                    .collect(),
                energy: (c.energy / c.max_energy).clamp(0.0, 1.0),
                health: (c.health / c.max_health).clamp(0.0, 1.0),
                diet: c.genome.traits.diet,
                generation: c.generation,
                age: c.age,
                neurons: c.genome.brain.h as u32,
                attacking: c.attack_flash > 0.0,
                mating: c.mate_flash > 0.0,
                drowning: c.drowning,
                in_water: c.in_water,
                score: self.life_score(c),
                mass: c.mass,
                children: c.life.children,
                kills: c.life.kills,
                food: c.life.food(),
                distance: c.life.distance,
                voice: c.genome.traits.voice,
                call: crate::life::call_loudness(c),
                call_pitch: c.output(crate::brain::output::CALL_PITCH),
                speed: bodies
                    .get(c.parts[0].body)
                    .map(|b| {
                        let v = b.linvel();
                        (v.x * v.x + v.z * v.z).sqrt()
                    })
                    .unwrap_or(0.0),
            })
            .collect();
        let food = |f: &crate::sim::Food| FoodView {
            id: f.id,
            pos: f.pos.to_array(),
            energy: f.energy,
        };
        let obstacles = self
            .obstacles
            .iter()
            .map(|o| {
                let (pos, rot) = pose(o.body);
                ObstacleView {
                    id: o.id,
                    pos,
                    rot,
                    shape: o.shape,
                    dynamic: o.dynamic,
                }
            })
            .collect();
        let selected = self.selected.and_then(|id| self.creature(id)).map(|c| {
            let p = &self.params;
            let t = &c.genome.traits;
            CreatureDetail {
                id: c.id,
                generation: c.generation,
                parents: c.parents,
                rained: c.rained,
                age: c.age,
                energy: c.energy,
                max_energy: c.max_energy,
                health: c.health,
                max_health: c.max_health,
                oxygen: c.oxygen,
                hydration: c.hydration,
                mate_drive: c.mate_drive,
                exertion: c.exertion,
                mass: c.mass,
                trunk: c.plan.iter().filter(|p| p.trunk).count(),
                limbs: c.plan.iter().filter(|p| !p.trunk).count(),
                hinges: c.joints.iter().filter(|j| j.kind == crate::body::JointKind::Hinge).count(),
                swivels: c.joints.iter().filter(|j| j.kind == crate::body::JointKind::Swivel).count(),
                traits: t.clone(),
                life: c.life.clone(),
                inputs: c.brain.inputs.clone(),
                hidden: c.brain.hidden.clone(),
                outputs: c.brain.outputs.clone(),
                learned: if c.brain.plastic.is_empty() {
                    0.0
                } else {
                    c.brain.plastic.iter().map(|v| v.abs()).sum::<f32>() / c.brain.plastic.len() as f32
                },
                score: self.score_parts(c),
                inherited: {
                    let w = &c.genome.brain.w_out.0;
                    w.iter().map(|v| v.to_f32().abs()).sum::<f32>() / w.len().max(1) as f32
                },
                brain_cost: c.brain_cost(p, self.brain_price),
                synapses: c.genome.brain.synapses(),
                neurons: c.genome.brain.h,
                basal_cost: p.basal_metabolism
                    * c.mass.powf(0.75)
                    * (1.0 + 0.5 * t.armor + 0.3 * t.breath),
                wiring: Arc::default(),
            }
        });
        Snapshot {
            tick: self.tick,
            time: self.time,
            creatures,
            plants: self.plants.iter().map(food).collect(),
            meat: self.meat.iter().map(food).collect(),
            obstacles,
            terrain,
            params: self.params.clone(),
            weather: self.weather,
            stats: self.stats.clone(),
            history,
            species: self.species.clone(),
            perf: self.perf,
            brain_backend: self.brain_backend.clone(),
            brain_price: self.brain_price,
            brain_memory_bytes: self.brains.as_ref().map(|g| g.memory_bytes()).unwrap_or(0),
            selected,
            grabbing: self.grab.is_some(),
            pool_size: self.pool.entries.len(),
            pool_best: self.pool.entries.first().map(|e| e.score).unwrap_or(0.0),
            paused: false,
            speed: 1.0,
            steps_per_sec: 0.0,
            cpu_percent: 0.0,
            events: Vec::new(),
            highlights: Arc::default(),
            worlds: Vec::new(),
            world_epoch: 0,
            zoo: Arc::new(Vec::new()),
            zoo_dir: String::new(),
        }
    }
}
