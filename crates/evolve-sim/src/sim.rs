//! The world: physics, terrain, plants, meat, obstacles and creatures, advanced in fixed steps.

use std::collections::HashMap;
use std::time::Instant;

use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use crate::body::{self, ContactFilter, JointRef, PartRef, PartSpec, Tag};
use crate::brain::{BrainState, output};
use crate::genome::Genome;
use crate::math::{Pose, Quat, Vec3};
use crate::params::SimParams;
use crate::rng::Rng;
use crate::snapshot::{Highlight, HighlightKind};
use crate::stats::{GenePool, Sample, Stats};
use crate::terrain::Terrain;

pub const DT: f32 = 1.0 / 60.0;
pub const TICKS_PER_SECOND: u64 = 60;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeathCause {
    Starved,
    Killed,
    OldAge,
    Drowned,
    Dehydrated,
    Removed,
    Glitch,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LifeStats {
    pub plant_eaten: f32,
    pub meat_eaten: f32,
    pub children: u32,
    pub kills: u32,
    pub attacks: u32,
    pub distance: f32,
    /// Energy handed to its children.
    #[serde(default)]
    pub invested: f32,
}

impl LifeStats {
    pub fn food(&self) -> f32 {
        self.plant_eaten + self.meat_eaten
    }
}

fn full() -> f32 {
    1.0
}

fn long_ago() -> f32 {
    1e9
}

/// A death within this many seconds of a bite counts as a kill, credited to the biter.
const RECENT_ATTACK: f32 = 5.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Creature {
    pub id: u64,
    pub genome: Genome,
    pub plan: Vec<PartSpec>,
    pub parts: Vec<PartRef>,
    pub joints: Vec<JointRef>,
    pub brain: BrainState,
    pub generation: u32,
    pub parents: Option<(u64, u64)>,
    pub mass: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub health: f32,
    pub max_health: f32,
    pub oxygen: f32,
    /// 1 fully watered, 0 dried out.
    #[serde(default = "full")]
    pub hydration: f32,
    /// The urge to mate, 0..1 (a sense, not a rule: mating still needs the conditions below).
    #[serde(default)]
    pub mate_drive: f32,
    pub age: f32,
    pub mate_cooldown: f32,
    pub attack_cooldown: f32,
    pub pain: f32,
    pub in_water: f32,
    pub drowning: bool,
    pub last_attacker: Option<u64>,
    /// Seconds since it was last bitten.
    #[serde(default = "long_ago")]
    pub attacked_ago: f32,
    pub attack_flash: f32,
    pub mate_flash: f32,
    pub life: LifeStats,
    pub last_pos: Vec3,
    /// Each part's vertical speed after the last step, to notice it hitting the ground.
    #[serde(skip)]
    pub fall: Vec<f32>,
    /// Seconds until a call in progress is announced again.
    #[serde(skip)]
    pub call_timer: f32,
    /// Muscle power per kg of body, smoothed over about a second, W/kg.
    #[serde(skip)]
    pub exertion: f32,
    /// Earth it's carrying, m³.
    #[serde(default)]
    pub soil: f32,
    pub rained: bool,
    pub dead: Option<DeathCause>,
    /// Which physics world its body is in (see `Sim::shards`).
    #[serde(default)]
    pub shard: usize,
}

impl Creature {
    pub fn torso_half(&self) -> Vec3 {
        match self.plan[0].shape {
            body::PartShape::Box { half } => Vec3::from(half),
            _ => Vec3::splat(0.2),
        }
    }

    pub fn radius(&self) -> f32 {
        self.torso_half().max_element()
    }

    /// What the creature does: the brain's output plus its exploration.
    pub fn output(&self, k: usize) -> f32 {
        (self.brain.outputs[k] + self.brain.noise.get(k).copied().unwrap_or(0.0)).clamp(-1.0, 1.0)
    }

    pub fn output_bias(&self) -> crate::brain::OutputBias {
        let t = &self.genome.traits;
        crate::brain::OutputBias {
            eat: t.appetite,
            attack: t.aggression,
            mate: t.libido,
        }
    }

    /// Energy per second this brain costs to run, at the going price of brain (see `gpu_target`).
    pub fn brain_cost(&self, p: &SimParams, price: f32) -> f32 {
        p.brain_energy_cost * price * self.genome.brain.synapses() as f32 / 1e6
    }

    pub fn plant_efficiency(&self) -> f32 {
        (1.0 - self.genome.traits.diet).max(0.05)
    }

    pub fn meat_efficiency(&self) -> f32 {
        self.genome.traits.diet.max(0.05)
    }

    /// Can be the second parent: mature, willing and fed, but with half the initiator's energy bar.
    pub fn willing_partner(&self, p: &SimParams) -> bool {
        self.ready_to_mate(p, p.mate_energy_fraction * 0.5)
    }

    pub fn ready_to_mate(&self, p: &SimParams, energy_fraction: f32) -> bool {
        self.dead.is_none()
            && self.age >= p.maturity_age
            && self.mate_cooldown <= 0.0
            && self.energy >= energy_fraction * self.max_energy
            && self.output(output::MATE) > 0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Food {
    pub id: u64,
    pub pos: Vec3,
    pub energy: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObstacleShape {
    Box { half: [f32; 3] },
    Ball { radius: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Obstacle {
    pub id: u64,
    pub body: RigidBodyHandle,
    pub shape: ObstacleShape,
    pub dynamic: bool,
    /// Which physics world it's in (see `Sim::shards`).
    #[serde(default)]
    pub shard: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Grab {
    /// A creature and the part held, or an obstacle (by id: bodies change handles when they
    /// move between physics worlds).
    pub creature: Option<u64>,
    pub part: usize,
    pub obstacle: Option<u64>,
    pub local: Vec3,
    pub target: Vec3,
}

/// One of the physics worlds the world is split into. Each holds the ground and walls and some
/// of the creatures and obstacles: those that might touch are kept in the same one, and the
/// worlds are stepped in parallel (rapier solves each world's articulations on one thread).
#[derive(Serialize, Deserialize)]
pub struct Shard {
    pub world: PhysicsWorld,
    pub terrain_collider: ColliderHandle,
    pub walls: Vec<ColliderHandle>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Perf {
    pub think_ms: f32,
    pub physics_ms: f32,
    pub other_ms: f32,
    /// GPU time per brain tick.
    pub gpu_ms: f32,
    /// Time the sim thread waited for the GPU per brain tick (0 when the GPU keeps up).
    pub gpu_wait_ms: f32,
    /// Fraction of wall time the GPU spends on brains.
    pub gpu_busy: f32,
    /// Regrouping and environment forces, before the physics step.
    pub forces_ms: f32,
    /// The slowest physics world's step, and all of them added up (their ratio to the physics
    /// time shows how well the worlds run in parallel).
    pub shard_slowest_ms: f32,
    pub shard_total_ms: f32,
}

#[derive(Serialize, Deserialize)]
pub struct Sim {
    pub params: SimParams,
    /// The physics worlds (see `Shard`); creatures and obstacles record which they're in.
    pub shards: Vec<Shard>,
    pub terrain: Terrain,
    pub creatures: Vec<Creature>,
    pub plants: Vec<Food>,
    pub meat: Vec<Food>,
    pub obstacles: Vec<Obstacle>,
    pub rng: Rng,
    pub tick: u64,
    pub time: f64,
    pub next_id: u64,
    pub stats: Stats,
    pub history: Vec<Sample>,
    pub pool: GenePool,
    pub selected: Option<u64>,
    pub grab: Option<Grab>,
    pub rain_timer: f32,
    pub plant_spawn_acc: f32,
    /// Multiplies the energy cost of brains; the runner steers it toward `gpu_target`.
    #[serde(default = "full")]
    pub brain_price: f32,
    /// How far below its target the GPU is running (negative when above), set by the runner at
    /// Max speed and 0 otherwise.
    #[serde(skip)]
    pub gpu_headroom: f32,
    /// Sim time the physics ground last took on the reshaped terrain.
    #[serde(skip)]
    pub settled_at: f64,
    /// A background world, whose physics threads run at a lower priority.
    #[serde(skip)]
    pub background: bool,
    /// Recent rain, 0 to 1, averaged over about half an hour: how soaked the land is.
    #[serde(default)]
    pub soak: f32,
    /// Simulated seconds of running water not yet worn into the land.
    #[serde(default)]
    pub erosion_due: f32,
    /// Storms, whirlwinds and the like called down by hand.
    #[serde(default)]
    pub acts: crate::acts::Acts,
    #[serde(skip)]
    pub index: HashMap<u64, usize>,
    #[serde(skip)]
    pub perf: Perf,
    #[serde(skip)]
    pub events: Vec<String>,
    /// Refreshed with each history sample.
    #[serde(skip)]
    pub species: Vec<SpeciesInfo>,
    /// Brains run here when a GPU is available; otherwise on the CPU.
    #[serde(skip)]
    pub brains: Option<crate::gpu::GpuBrains>,
    #[serde(skip)]
    pub brain_backend: String,
    #[serde(skip)]
    pub last_collect: Option<Instant>,
    /// The sky right now; a function of the clock, so not saved.
    #[serde(skip)]
    pub weather: crate::weather::Weather,
    /// Notable moments since the runner last collected them.
    #[serde(skip)]
    pub highlights: Vec<Highlight>,
    #[serde(skip)]
    pub highlight_seq: u64,
}

pub const MAX_HISTORY: usize = 4000;
/// Radius of a full-grown plant's base, as drawn.
pub const PLANT_FOOTPRINT: f32 = 0.35;

#[derive(Clone, Debug, Default)]
pub struct SpeciesInfo {
    /// The best-scoring member alive, used to jump to the species.
    pub representative: u64,
    pub best_score: f32,
    pub color: [f32; 3],
    pub count: u32,
    pub max_generation: u32,
    /// Children had by the members alive now.
    pub births: u32,
    pub mean_diet: f32,
    pub mean_limbs: f32,
    pub mean_neurons: f32,
    pub mean_mass: f32,
}

impl Sim {
    /// A background world: its physics threads run at a lower priority.
    pub fn new_background(params: SimParams) -> Self {
        let mut sim = Self::new_with(params, true);
        sim.background = true;
        sim
    }

    pub fn new(params: SimParams) -> Self {
        Self::new_with(params, false)
    }

    fn new_with(params: SimParams, background: bool) -> Self {
        let terrain = Terrain::generate(&params, 1);
        let rng = Rng::new(params.seed.wrapping_mul(0x9E37_79B9).wrapping_add(7));
        let mut sim = Sim {
            params,
            shards: Vec::new(),
            terrain,
            creatures: Vec::new(),
            plants: Vec::new(),
            meat: Vec::new(),
            obstacles: Vec::new(),
            rng,
            tick: 0,
            time: 0.0,
            next_id: 1,
            stats: Stats::default(),
            history: Vec::new(),
            pool: GenePool::default(),
            selected: None,
            grab: None,
            rain_timer: 0.0,
            plant_spawn_acc: 0.0,
            brain_price: 1.0,
            gpu_headroom: 0.0,
            settled_at: 0.0,
            background,
            soak: 0.0,
            erosion_due: 0.0,
            acts: Default::default(),
            index: HashMap::new(),
            perf: Perf::default(),
            events: Vec::new(),
            species: Vec::new(),
            brains: None,
            brain_backend: String::new(),
            last_collect: None,
            weather: Default::default(),
            highlights: Vec::new(),
            highlight_seq: 0,
        };
        sim.refresh_weather();
        sim.init_brains();
        sim.resize_shards();
        sim.sync_physics_params(None);
        // Start with the land half full of plants.
        let target = sim.params.plant_max_count / 2;
        let energy = sim.params.plant_max_energy * 0.6;
        let mut grid = sim.plant_grid();
        let mut tries = 0;
        while (sim.plants.len() as u32) < target && tries < target * 20 {
            tries += 1;
            sim.try_spawn_random_plant(energy, &mut grid);
        }
        sim
    }

    /// Rebuild state that isn't saved.
    pub fn after_load(&mut self) {
        // Brains saved before a change of layout are rewired; their running state starts over.
        for c in &mut self.creatures {
            if c.genome.upgrade() {
                c.brain = BrainState::new(&c.genome.brain);
            }
        }
        for e in &mut self.pool.entries {
            e.genome.upgrade();
        }
        self.rebuild_index();
        self.species = self.species_summary();
        self.perf = Perf::default();
        self.events.clear();
        self.refresh_weather();
        self.init_brains();
        // (Also gives worlds loaded from a save their threads.)
        self.resize_shards();
    }

    /// Opens the GPU for brains (unless `EVOLVE_CPU_BRAINS` is set or there is none) and uploads
    /// every living creature's brain.
    pub fn init_brains(&mut self) {
        self.brains = None;
        self.last_collect = None;
        self.brain_backend = "CPU".into();
        if std::env::var_os("EVOLVE_CPU_BRAINS").is_none() {
            match crate::gpu::GpuBrains::new() {
                Ok(g) => {
                    self.brain_backend = format!("GPU ({})", g.adapter_name);
                    self.brains = Some(g);
                }
                Err(e) => self.brain_backend = format!("CPU ({e})"),
            }
        }
        self.apply_gpu_budget();
        if let Some(g) = self.brains.as_mut() {
            let (lo, _) = self.params.neuron_range();
            for c in &mut self.creatures {
                if g.register(c.id, &c.genome.brain, Some(&c.brain)).is_ok() {
                    continue;
                }
                // A brain the GPU can't hold (a smaller GPU or budget than it was saved with)
                // would never think: trim it until it fits, losing what it had learned.
                let mut result = Err(String::new());
                while result.is_err() && c.genome.brain.h > lo as usize {
                    let nh = (c.genome.brain.h / 2).max(lo as usize);
                    c.genome.brain = c.genome.brain.resize(nh, &mut self.rng);
                    c.brain = BrainState::new(&c.genome.brain);
                    result = g.register(c.id, &c.genome.brain, None);
                }
                match result {
                    Ok(()) => self.events.push(format!("creature #{}'s brain was trimmed to fit the GPU", c.id)),
                    Err(e) => self.events.push(format!("creature #{} has no brain on the GPU: {e}", c.id)),
                }
            }
        }
    }

    /// Each world gets an equal share of the brain memory budget.
    fn apply_gpu_budget(&mut self) {
        let p = &self.params;
        if let Some(g) = self.brains.as_mut() {
            let share = p.gpu_memory_gb.max(0.1) as f64 * 1e9 / (1 + p.parallel_worlds) as f64;
            g.budget_bytes = share as u64;
        }
    }

    /// Pulls the live brain state (activations, learned weights) back from the GPU, so a save
    /// captures it.
    pub fn sync_brains(&mut self) {
        let Some(g) = self.brains.as_mut() else {
            return;
        };
        let mut states = g.read_states();
        for c in &mut self.creatures {
            if let Some(s) = states.remove(&c.id) {
                c.brain.hidden = s.hidden;
                c.brain.plastic = s.plastic;
                c.brain.traces = s.traces;
                c.brain.outputs = s.outputs;
            }
        }
    }

    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, c) in self.creatures.iter().enumerate() {
            self.index.insert(c.id, i);
        }
    }

    pub fn creature(&self, id: u64) -> Option<&Creature> {
        self.index.get(&id).and_then(|&i| self.creatures.get(i))
    }

    pub fn creature_mut(&mut self, id: u64) -> Option<&mut Creature> {
        let i = *self.index.get(&id)?;
        self.creatures.get_mut(i)
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        self.events.push(msg.into());
    }

    pub fn highlight(&mut self, kind: HighlightKind, subject: u64, other: Option<u64>, pos: Vec3) {
        self.highlight_with(kind, subject, other, pos, 0.0);
    }

    pub(crate) fn highlight_with(&mut self, kind: HighlightKind, subject: u64, other: Option<u64>, pos: Vec3, strength: f32) {
        self.highlight_seq += 1;
        self.highlights.push(Highlight {
            seq: self.highlight_seq,
            kind,
            subject,
            other,
            pos: pos.to_array(),
            strength,
        });
        // Nobody collects them in a headless run.
        if self.highlights.len() > 512 {
            self.highlights.drain(..256);
        }
    }

    /// A new physics world with the ground and walls in it.
    fn new_shard(&self) -> Shard {
        let mut world = PhysicsWorld::new();
        let terrain_collider = world.colliders.insert(
            self.terrain
                .collider()
                .friction(self.params.ground_friction)
                .user_data(Tag::Terrain.encode()),
        );
        let mut shard = Shard { world, terrain_collider, walls: Vec::new() };
        Self::build_walls(&self.terrain, &mut shard);
        shard
    }

    /// Matches the number of physics worlds to the `physics_shards` param, moving the creatures
    /// and obstacles of any world dropped into the remaining ones.
    pub(crate) fn resize_shards(&mut self) {
        let want = (self.params.physics_shards.max(1) as usize).min(64);
        while self.shards.len() < want {
            let shard = self.new_shard();
            self.shards.push(shard);
        }
        if self.shards.len() > want {
            for ci in 0..self.creatures.len() {
                if self.creatures[ci].shard >= want {
                    let to = ci % want;
                    self.move_creature(ci, to);
                }
            }
            for oi in 0..self.obstacles.len() {
                if self.obstacles[oi].shard >= want {
                    self.move_obstacle(oi, oi % want);
                }
            }
            self.shards.truncate(want);
        }
        // Each world's physics runs on a thread of its own: left to share the world pool, rapier
        // spreads every world's solver over all of it, and the worlds' workers wait on each other.
        for shard in &mut self.shards {
            if shard.world.physics_pipeline.thread_pool().is_none() {
                let pool = crate::islands::pool("evolve-physics", 1, self.background);
                shard.world.physics_pipeline.set_thread_pool(Some(std::sync::Arc::new(pool)));
            }
        }
        self.sync_physics_params(None);
    }

    fn build_walls(terrain: &Terrain, shard: &mut Shard) {
        for h in shard.walls.drain(..) {
            shard.world.remove_collider(h);
        }
        let half = terrain.half();
        let t = 1.0;
        let height = 60.0;
        let specs = [
            (Vec3::new(half + t, 0.0, 0.0), Vec3::new(t, height, half + 2.0 * t)),
            (Vec3::new(-half - t, 0.0, 0.0), Vec3::new(t, height, half + 2.0 * t)),
            (Vec3::new(0.0, 0.0, half + t), Vec3::new(half + 2.0 * t, height, t)),
            (Vec3::new(0.0, 0.0, -half - t), Vec3::new(half + 2.0 * t, height, t)),
        ];
        for (pos, he) in specs {
            let c = ColliderBuilder::cuboid(he.x, he.y, he.z)
                .translation(pos)
                .user_data(Tag::Wall.encode())
                .build();
            shard.walls.push(shard.world.colliders.insert(c));
        }
    }

    /// Push the physics-related params into rapier. `old` is the previous param set, if any.
    fn sync_physics_params(&mut self, old: Option<&SimParams>) {
        let p = &self.params;
        for shard in &mut self.shards {
            let world = &mut shard.world;
            world.gravity = Vec3::new(0.0, -p.gravity, 0.0);
            world.integration_parameters.dt = DT;
            world.integration_parameters.num_solver_iterations = p.solver_iterations.max(1) as usize;
            world.integration_parameters.num_internal_pgs_iterations = p.solver_passes.max(1) as usize;
            if let Some(c) = world.colliders.get_mut(shard.terrain_collider) {
                c.set_friction(p.ground_friction);
            }
        }
        let friction_changed = old.is_none_or(|o| o.body_friction != p.body_friction);
        let damping_changed = old.is_none_or(|o| o.air_drag != p.air_drag);
        if friction_changed || damping_changed {
            for c in &self.creatures {
                let world = &mut self.shards[c.shard].world;
                for part in &c.parts {
                    if friction_changed && let Some(col) = world.colliders.get_mut(part.collider) {
                        col.set_friction(p.body_friction);
                    }
                    if damping_changed && let Some(b) = world.bodies.get_mut(part.body) {
                        b.set_linear_damping(p.air_drag);
                    }
                }
            }
        }
    }

    pub fn set_params(&mut self, new: SimParams) {
        let old = std::mem::replace(&mut self.params, new);
        self.sync_physics_params(Some(&old));
        if old.physics_shards != self.params.physics_shards {
            self.resize_shards();
        }
        self.apply_gpu_budget();
    }

    /// The physics world a creature is in.
    pub fn world_of(&self, c: &Creature) -> &PhysicsWorld {
        &self.shards[c.shard].world
    }

    /// A creature's torso body.
    pub fn torso_of(&self, c: &Creature) -> Option<&RigidBody> {
        self.world_of(c).bodies.get(c.parts.first()?.body)
    }

    /// Rebuild the terrain from the world params, keeping everything that lives on it.
    pub fn regenerate_terrain(&mut self) {
        let version = self.terrain.version + 1;
        self.install_terrain(Terrain::generate(&self.params, version));
        let margin = 2.0;
        let half = self.terrain.half() - margin;
        // Re-seat creatures above the new ground.
        for ci in 0..self.creatures.len() {
            let Some(pos) = self.torso_of(&self.creatures[ci]).map(|b| b.translation()) else {
                continue;
            };
            // As at birth: clear of the walls and above the highest ground the body reaches.
            let extent = body::plan_extent(&self.creatures[ci].plan);
            let room = (self.terrain.half() - 1.0 - extent).max(0.0);
            let x = pos.x.clamp(-room, room);
            let z = pos.z.clamp(-room, room);
            let depth = body::plan_depth(&self.creatures[ci].plan);
            let y = self.ground_over(x, z, extent) + depth + 1.0;
            self.translate_creature(ci, Vec3::new(x, y, z) - pos);
        }
        let terrain = &self.terrain;
        self.plants.retain(|f| terrain.contains(f.pos.x, f.pos.z, 0.5));
        for f in self.plants.iter_mut().chain(self.meat.iter_mut()) {
            f.pos.x = f.pos.x.clamp(-half, half);
            f.pos.z = f.pos.z.clamp(-half, half);
            f.pos.y = terrain.height_at(f.pos.x, f.pos.z);
        }
        for i in 0..self.plants.len() {
            let p = self.plants[i].pos;
            self.plants[i].pos.y = self.ground_under(p.x, p.z, PLANT_FOOTPRINT);
        }
        let terrain = &self.terrain;
        for o in &self.obstacles {
            if let Some(b) = self.shards[o.shard].world.bodies.get_mut(o.body) {
                let mut t = b.translation();
                t.x = t.x.clamp(-half, half);
                t.z = t.z.clamp(-half, half);
                t.y = terrain.height_at(t.x, t.z) + obstacle_lift(o.shape);
                b.set_translation(t, true);
            }
        }
        self.log("Terrain regenerated");
    }

    /// The params births use: while the GPU has headroom, brains also tend to grow when they
    /// mutate, and to shrink when it's over its target, so evolution fills the GPU in a few dozen
    /// generations instead of drifting there over hours.
    pub fn breeding_params(&self) -> SimParams {
        let mut p = self.params.clone();
        p.brain_growth = (p.brain_growth + 0.5 * self.gpu_headroom).clamp(-0.5, 0.5);
        p
    }

    /// Swaps in a new ground surface and the walls around it.
    pub fn install_terrain(&mut self, terrain: Terrain) {
        self.terrain = terrain;
        // The same ground in every physics world (the heightfield shape is shared).
        let shape = self.terrain.shape();
        for shard in &mut self.shards {
            shard.world.remove_collider(shard.terrain_collider);
            shard.terrain_collider = shard.world.colliders.insert(
                ColliderBuilder::new(shape.clone())
                    .friction(self.params.ground_friction)
                    .user_data(Tag::Terrain.encode()),
            );
            Self::build_walls(&self.terrain, shard);
        }
    }

    fn translate_creature(&mut self, ci: usize, delta: Vec3) {
        let c = &self.creatures[ci];
        let world = &mut self.shards[c.shard].world;
        for part in &c.parts {
            if let Some(b) = world.bodies.get_mut(part.body) {
                let t = b.translation();
                b.set_translation(t + delta, true);
                b.set_linvel(Vec3::ZERO, true);
                b.set_angvel(Vec3::ZERO, true);
            }
        }
        // A multibody follows its root: re-pose the links from the moved torso, and stop them.
        if let Some(link) = c.parts.first().and_then(|p| world.multibody_joints.rigid_body_link(p.body)).copied()
            && let Some(mb) = world.multibody_joints.get_multibody_mut(link.multibody)
        {
            mb.forward_kinematics(&world.bodies, true);
            mb.update_rigid_bodies(&mut world.bodies, true);
            mb.generalized_velocity_mut().fill(0.0);
        }
        self.creatures[ci].last_pos += delta;
    }

    /// Two different physics worlds, mutably.
    fn two_worlds(&mut self, a: usize, b: usize) -> (&mut PhysicsWorld, &mut PhysicsWorld) {
        assert_ne!(a, b);
        if a < b {
            let (x, y) = self.shards.split_at_mut(b);
            (&mut x[a].world, &mut y[0].world)
        } else {
            let (x, y) = self.shards.split_at_mut(a);
            (&mut y[0].world, &mut x[b].world)
        }
    }

    /// Moves creature `ci` into physics world `to`, just as it is.
    pub(crate) fn move_creature(&mut self, ci: usize, to: usize) {
        let from = self.creatures[ci].shard;
        if from == to || to >= self.shards.len() {
            return;
        }
        let p = self.params.clone();
        let c = &self.creatures[ci];
        let (plan, parts, joints, id, mass) = (c.plan.clone(), c.parts.clone(), c.joints.clone(), c.id, c.mass);
        let (a, b) = self.two_worlds(from, to);
        if let Some((parts, joints)) = body::transplant(a, b, &plan, &parts, &joints, id, mass, &p) {
            let c = &mut self.creatures[ci];
            c.parts = parts;
            c.joints = joints;
            c.shard = to;
        }
    }

    /// Moves obstacle `oi` into physics world `to`, just as it is.
    pub(crate) fn move_obstacle(&mut self, oi: usize, to: usize) {
        let o = self.obstacles[oi].clone();
        if o.shard == to || to >= self.shards.len() {
            return;
        }
        let (a, b) = self.two_worlds(o.shard, to);
        let Some(old) = a.bodies.get(o.body) else {
            return;
        };
        let (pose, lin, ang) = (*old.position(), old.linvel(), old.angvel());
        a.remove_body(o.body);
        let (body, _) = b.insert(obstacle_body(o.id, o.dynamic, pose), obstacle_collider(o.id, o.shape));
        if let Some(rb) = b.bodies.get_mut(body) {
            rb.set_linvel(lin, true);
            rb.set_angvel(ang, true);
        }
        let o = &mut self.obstacles[oi];
        o.body = body;
        o.shard = to;
    }

    /// The physics world for something new at `at`: the one holding whatever is nearest within
    /// touching distance, or else the least busy.
    fn shard_for(&self, at: Vec3) -> usize {
        let near = self
            .creatures
            .iter()
            .filter_map(|c| Some((c.shard, self.torso_of(c)?.translation().distance(at))))
            .chain(self.obstacles.iter().filter_map(|o| {
                Some((o.shard, self.shards[o.shard].world.bodies.get(o.body)?.translation().distance(at)))
            }))
            .filter(|(_, d)| *d < REGROUP_REACH)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        match near {
            Some((s, _)) => s,
            None => {
                let mut load = vec![0usize; self.shards.len()];
                for c in &self.creatures {
                    load[c.shard] += c.parts.len();
                }
                (0..load.len()).min_by_key(|&s| load[s]).unwrap_or(0)
            }
        }
    }

    /// Regroups creatures and obstacles among the physics worlds: anything that might touch
    /// before the next regroup shares a world, and the worlds' loads are kept even by moving
    /// whole groups out of busy worlds (a few at a time, so nothing churns).
    pub(crate) fn regroup(&mut self) {
        let k = self.shards.len();
        if k <= 1 {
            return;
        }
        // Things to place: creatures, then obstacles, with the box around their parts as they
        // are now, how much they cost to simulate, and which world they're in.
        let mut spots: Vec<(Vec3, Vec3, usize, usize)> = Vec::with_capacity(self.creatures.len() + self.obstacles.len());
        for c in &self.creatures {
            let world = &self.shards[c.shard].world;
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for (q, spec) in c.parts.iter().zip(&c.plan) {
                if let Some(b) = world.bodies.get(q.body) {
                    let r = spec.shape.bounding_radius();
                    lo = lo.min(b.translation() - Vec3::splat(r));
                    hi = hi.max(b.translation() + Vec3::splat(r));
                }
            }
            if lo.x > hi.x {
                (lo, hi) = (c.last_pos, c.last_pos);
            }
            spots.push(((lo + hi) * 0.5, (hi - lo) * 0.5, c.parts.len(), c.shard));
        }
        for o in &self.obstacles {
            let at = self.shards[o.shard].world.bodies.get(o.body).map_or(Vec3::ZERO, |b| b.translation());
            spots.push((at, Vec3::splat(obstacle_radius(o.shape)), 1, o.shard));
        }
        let n = spots.len();
        // Groups: union-find over boxes that come within a margin of each other, found on a grid.
        let size = spots.iter().map(|s| s.1.x.max(s.1.z)).fold(0.0f32, f32::max);
        let cell = (2.0 * size + REGROUP_MARGIN).max(1.0);
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (i, s) in spots.iter().enumerate() {
            grid.entry(((s.0.x / cell).floor() as i32, (s.0.z / cell).floor() as i32)).or_default().push(i);
        }
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let near = |a: &(Vec3, Vec3, usize, usize), b: &(Vec3, Vec3, usize, usize)| {
            let d = (a.0 - b.0).abs() - a.1 - b.1;
            d.x < REGROUP_MARGIN && d.y < REGROUP_MARGIN && d.z < REGROUP_MARGIN
        };
        for (i, s) in spots.iter().enumerate() {
            let (gx, gz) = ((s.0.x / cell).floor() as i32, (s.0.z / cell).floor() as i32);
            for dx in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid.get(&(gx + dx, gz + dz)).map_or(&[][..], |v| v) {
                        if j > i && near(s, &spots[j]) {
                            let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                            parent[a] = b;
                        }
                    }
                }
            }
        }
        // Each group's home is the world most of it is already in.
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let r = root(&mut parent, i);
            members.entry(r).or_default().push(i);
        }
        let mut groups: Vec<(usize, usize, Vec<usize>)> = members
            .into_values()
            .map(|m| {
                let mut votes = vec![0usize; k];
                for &i in &m {
                    votes[spots[i].3.min(k - 1)] += spots[i].2;
                }
                let home = (0..k).max_by_key(|&s| (votes[s], usize::MAX - s)).unwrap_or(0);
                let weight = m.iter().map(|&i| spots[i].2).sum();
                (home, weight, m)
            })
            .collect();
        groups.sort_by_key(|g| (g.1, g.2[0]));
        // Even out the load: move the smallest groups out of worlds well above the average.
        let mut load = vec![0usize; k];
        for g in &groups {
            load[g.0] += g.1;
        }
        let fair = load.iter().sum::<usize>() / k;
        let mut budget = REGROUP_MOVES;
        for g in groups.iter_mut() {
            if budget < g.1 {
                break;
            }
            if load[g.0] * 4 > fair * 5 + 20 {
                let to = (0..k).min_by_key(|&s| load[s]).unwrap_or(0);
                if load[to] + g.1 < load[g.0] {
                    load[g.0] -= g.1;
                    load[to] += g.1;
                    g.0 = to;
                    budget -= g.1;
                }
            }
        }
        let creatures = self.creatures.len();
        for (home, _, m) in groups {
            for i in m {
                if spots[i].3 != home {
                    if i < creatures {
                        self.move_creature(i, home);
                    } else {
                        self.move_obstacle(i - creatures, home);
                    }
                }
            }
        }
    }

    // ── Spawning ───────────────────────────────────────────────────────

    pub fn random_land_point(&mut self, margin: f32) -> Vec3 {
        let half = self.terrain.half() - margin;
        let mut best = Vec3::ZERO;
        for i in 0..40 {
            let x = self.rng.range(-half, half);
            let z = self.rng.range(-half, half);
            let y = self.terrain.height_at(x, z);
            best = Vec3::new(x, y, z);
            // Keep clear of the shore too: a raindrop that bounces into the sea drowns.
            let wl = self.params.water_level + 0.5;
            if (y > wl && self.ground_under(x, z, 3.0) > wl) || i == 39 {
                break;
            }
        }
        best
    }

    /// Adds a creature with its torso centered over `at` (x, z), `lift` meters above the ground,
    /// holding `energy` (capped at what its body can store).
    pub fn spawn_creature(
        &mut self,
        genome: Genome,
        at: Vec3,
        lift: f32,
        energy: f32,
        generation: u32,
        parents: Option<(u64, u64)>,
    ) -> u64 {
        let id = self.next_id();
        let plan = body::body_plan(&genome);
        let mass = body::plan_mass(&plan, self.params.body_density);
        let half = self.terrain.half() - 1.0 - body::plan_extent(&plan);
        let x = at.x.clamp(-half, half);
        let z = at.z.clamp(-half, half);
        // The highest ground the body could reach, so no limb starts inside a hillside.
        let ground = self.ground_over(x, z, body::plan_extent(&plan));
        let y = (ground + body::plan_depth(&plan) + lift).max(at.y);
        let yaw = self.rng.range(0.0, std::f32::consts::TAU);
        let root = Pose::from_parts(Vec3::new(x, y, z), Quat::from_rotation_y(yaw));
        let shard = self.shard_for(Vec3::new(x, y, z));
        let (parts, joints) = body::spawn_body(&mut self.shards[shard].world, &plan, root, id, mass, &self.params);
        let max_energy = self.params.energy_per_mass * mass;
        let max_health = 100.0 * (mass / 60.0).sqrt();
        let mut genome = genome;
        // A brain from before the last change of brain size grows into the new era.
        let (lo, hi) = self.params.neuron_range();
        genome.brain.bring_up_to_era(lo as usize, hi as usize, &mut self.rng);
        if let Some(g) = self.brains.as_mut() {
            // A brain the GPU can't hold is trimmed until it fits.
            let (lo, _) = self.params.neuron_range();
            let mut result = g.register(id, &genome.brain, None);
            while result.is_err() && genome.brain.h > lo as usize {
                let nh = (genome.brain.h / 2).max(lo as usize);
                genome.brain = genome.brain.resize(nh, &mut self.rng);
                result = g.register(id, &genome.brain, None);
            }
            if let Err(e) = result {
                self.events.push(format!("creature #{id}: {e}"));
            }
        }
        let brain = BrainState::new(&genome.brain);
        let creature = Creature {
            id,
            genome,
            plan,
            parts,
            joints,
            brain,
            generation,
            parents,
            mass,
            energy: energy.clamp(0.0, max_energy),
            max_energy,
            health: max_health,
            max_health,
            oxygen: 1.0,
            hydration: 1.0,
            mate_drive: 0.0,
            age: 0.0,
            mate_cooldown: 0.0,
            attack_cooldown: 0.0,
            pain: 0.0,
            in_water: 0.0,
            drowning: false,
            last_attacker: None,
            attacked_ago: long_ago(),
            attack_flash: 0.0,
            mate_flash: 0.0,
            life: LifeStats::default(),
            last_pos: Vec3::new(x, y, z),
            fall: Vec::new(),
            call_timer: 0.0,
            exertion: 0.0,
            soil: 0.0,
            rained: parents.is_none(),
            dead: None,
            shard,
        };
        if generation > self.stats.max_generation && parents.is_some() {
            self.highlight(HighlightKind::Record, id, None, Vec3::new(x, y, z));
        }
        self.stats.max_generation = self.stats.max_generation.max(generation);
        self.index.insert(id, self.creatures.len());
        self.creatures.push(creature);
        id
    }

    /// A creature dropped from the sky: bred from the gene pool or entirely random.
    pub fn rain_creature(&mut self, at: Option<Vec3>) -> u64 {
        let from_pool = self.rng.chance(self.params.rain_from_pool);
        let bred = if from_pool {
            let p = self.breeding_params();
            self.pool.breed(&mut self.rng, &p)
        } else {
            None
        };
        let from_pool = bred.is_some();
        let (genome, generation) =
            bred.unwrap_or_else(|| (Genome::random(&mut self.rng, &self.params), 0));
        let near_kin = (from_pool && self.rng.chance(self.params.rain_near_kin))
            .then(|| self.near_kin_point(&genome))
            .flatten();
        let at = at.or(near_kin).unwrap_or_else(|| self.random_land_point(4.0));
        let lift = self.params.rain_height;
        let e = self.params.start_energy;
        self.stats.rained += 1;
        self.spawn_creature(genome, at, lift, e, generation, None)
    }

    /// How well a lifetime went, for the gene pool and the zoo. Children count most; food eaten
    /// and real distance covered are the stepping stones that let the gene pool improve before
    /// anyone manages to breed. Lifespan is left out on purpose: it rewards giant bodies that sit
    /// still and starve slowly.
    pub fn life_score(&self, c: &Creature) -> f32 {
        self.score_parts(c).iter().sum()
    }

    /// What a life achieved, in "plants' worth": eating (food and meat), hunting (kills),
    /// breeding (children, and the energy given them) and travelling (net progress). Simply
    /// lasting earns nothing, and neither do children given almost nothing.
    pub fn score_parts(&self, c: &Creature) -> [f32; 4] {
        let plant = self.params.plant_max_energy.max(1.0);
        [
            c.life.food() / plant,
            c.life.kills as f32 * 3.0,
            c.life.children as f32 * 0.5 + c.life.invested / plant,
            c.life.distance / 50.0,
        ]
    }

    /// Drops `count` mutated copies of a genome from the sky, together (so they can breed), at `at`
    /// or a random spot. They count as the next generation.
    pub fn drop_genome(&mut self, genome: &Genome, generation: u32, at: Option<Vec3>, count: u32) {
        let center = at.unwrap_or_else(|| self.random_land_point(6.0));
        let p = self.breeding_params();
        for k in 0..count.max(1) {
            // Each copy is a slightly mutated variant, like a child: close enough to interbreed.
            let mut g = genome.clone();
            g.mutate(&mut self.rng, &p);
            let spot = if k == 0 {
                center
            } else {
                let a = self.rng.range(0.0, std::f32::consts::TAU);
                let d = self.rng.range(1.5, 5.0);
                center + Vec3::new(a.cos() * d, 0.0, a.sin() * d)
            };
            let (lift, e) = (self.params.rain_height, self.params.start_energy);
            self.spawn_creature(g, spot, lift, e, generation + 1, None);
        }
    }

    /// A spot a few meters from a random living creature this genome could breed with.
    fn near_kin_point(&mut self, genome: &Genome) -> Option<Vec3> {
        let threshold = self.params.species_threshold;
        let kin: Vec<usize> = (0..self.creatures.len())
            .filter(|&i| self.creatures[i].genome.distance(genome) < threshold)
            .collect();
        if kin.is_empty() {
            return None;
        }
        let i = kin[self.rng.below(kin.len())];
        let base = self.torso_of(&self.creatures[i])?.translation();
        let a = self.rng.range(0.0, std::f32::consts::TAU);
        let d = self.rng.range(3.0, 8.0);
        Some(base + Vec3::new(a.cos() * d, 0.0, a.sin() * d))
    }

    /// Lowest ground within `r` of (x, z), so something with a flat base never hangs over a slope.
    pub fn ground_under(&self, x: f32, z: f32, r: f32) -> f32 {
        let t = &self.terrain;
        let mut lo = t.height_at(x, z);
        for k in 0..8 {
            let a = k as f32 * std::f32::consts::FRAC_PI_4;
            lo = lo.min(t.height_at(x + a.cos() * r, z + a.sin() * r));
        }
        lo
    }

    /// Highest ground within `r` of a point.
    pub fn ground_over(&self, x: f32, z: f32, r: f32) -> f32 {
        let t = &self.terrain;
        let mut hi = t.height_at(x, z);
        for ring in [0.5, 1.0] {
            for k in 0..8 {
                let a = k as f32 * std::f32::consts::FRAC_PI_4;
                hi = hi.max(t.height_at(x + a.cos() * r * ring, z + a.sin() * r * ring));
            }
        }
        hi
    }

    pub fn add_plant(&mut self, pos: Vec3, energy: f32) -> Option<u64> {
        if !self.terrain.contains(pos.x, pos.z, 0.3) {
            return None;
        }
        let id = self.next_id();
        let y = self.ground_under(pos.x, pos.z, PLANT_FOOTPRINT);
        self.plants.push(Food {
            id,
            pos: Vec3::new(pos.x, y, pos.z),
            energy,
        });
        Some(id)
    }

    pub fn add_meat(&mut self, pos: Vec3, energy: f32) -> u64 {
        let id = self.next_id();
        let half = self.terrain.half() - 0.5;
        let x = pos.x.clamp(-half, half);
        let z = pos.z.clamp(-half, half);
        let y = self.terrain.height_at(x, z).max(pos.y.min(self.terrain.height_at(x, z) + 0.2));
        self.meat.push(Food {
            id,
            pos: Vec3::new(x, y, z),
            energy,
        });
        id
    }

    pub fn add_obstacle(&mut self, pos: Vec3, shape: ObstacleShape, dynamic: bool, yaw: f32) -> u64 {
        let id = self.next_id();
        let y = self.terrain.height_at(pos.x, pos.z) + obstacle_lift(shape);
        let at = Vec3::new(pos.x, y, pos.z);
        let shard = self.shard_for(at);
        let rb = obstacle_body(id, dynamic, Pose::from_parts(at, Quat::from_rotation_y(yaw)));
        let (body, _) = self.shards[shard].world.insert(rb, obstacle_collider(id, shape));
        self.obstacles.push(Obstacle {
            id,
            body,
            shape,
            dynamic,
            shard,
        });
        id
    }

    pub fn remove_obstacle(&mut self, id: u64) {
        if let Some(i) = self.obstacles.iter().position(|o| o.id == id) {
            let o = self.obstacles.swap_remove(i);
            if self.grab.as_ref().is_some_and(|g| g.obstacle == Some(o.id)) {
                self.grab = None;
            }
            self.shards[o.shard].world.remove_body(o.body);
        }
    }

    /// The sky now: the clock's weather, and any storm called down on top of it.
    pub fn refresh_weather(&mut self) {
        self.weather = crate::weather::at(&self.params, self.time);
        let s = self.acts.storm_strength(self.time);
        if s > 0.0 {
            crate::weather::summoned(&mut self.weather, s);
        }
    }

    /// Marks the hand of god at work on a creature, for the miracle the app draws there.
    pub(crate) fn miracle(&mut self, id: u64, kind: HighlightKind) {
        let Some(c) = self.creature(id) else {
            return;
        };
        let size = c.radius();
        if let Some(at) = self.torso_of(c).map(|b| b.translation()) {
            self.highlight_with(kind, id, None, at, size);
        }
    }

    pub fn kill(&mut self, id: u64, cause: DeathCause) {
        if let Some(c) = self.creature_mut(id) {
            c.dead.get_or_insert(cause);
        }
    }

    // ── Stepping ───────────────────────────────────────────────────────

    pub fn step(&mut self) {
        // Water is moved on every couple of seconds (rain changes slowly).
        const HYDROLOGY_EVERY: u64 = 2;
        let brain_every = (TICKS_PER_SECOND as f32 / self.params.brain_hz.max(0.1))
            .round()
            .max(1.0) as u64;
        let brain_tick = self.tick % brain_every == 0;
        let bdt = brain_every as f32 * DT;

        let t0 = Instant::now();
        if self.tick % 6 == 0 {
            self.acts.expire(self.time);
            self.refresh_weather();
        }
        if brain_tick {
            self.think(bdt);
        }
        let t1 = Instant::now();
        if self.tick % REGROUP_EVERY == 0 {
            self.regroup();
        }
        self.move_whirlwinds(DT);
        self.environment_forces();
        let t_forces = Instant::now();
        // The physics worlds step in parallel: rapier solves a world's articulations on one
        // thread.
        let times: Vec<f32> = {
            use rayon::prelude::*;
            self.shards
                .par_iter_mut()
                .map(|s| {
                    let t = Instant::now();
                    s.world.step_with_events(&ContactFilter, &());
                    t.elapsed().as_secs_f32() * 1000.0
                })
                .collect()
        };
        let t2 = Instant::now();
        let k = 0.05;
        let slowest = times.iter().copied().fold(0.0, f32::max);
        let total: f32 = times.iter().sum();
        self.perf.shard_slowest_ms += (slowest - self.perf.shard_slowest_ms) * k;
        self.perf.shard_total_ms += (total - self.perf.shard_total_ms) * k;
        self.perf.forces_ms += ((t_forces - t1).as_secs_f32() * 1000.0 - self.perf.forces_ms) * k;
        self.after_physics();
        if brain_tick {
            self.interact(bdt);
            self.metabolism(bdt);
        }
        if self.tick % TICKS_PER_SECOND == 0 {
            self.ecology(1.0);
        }
        if self.tick % (TICKS_PER_SECOND * HYDROLOGY_EVERY) == 0 {
            self.hydrology(HYDROLOGY_EVERY as f32);
        }
        if self.tick % (TICKS_PER_SECOND * 2) == 0 {
            self.sample_history();
        }
        self.rain(DT);
        self.process_deaths();
        self.tick += 1;
        self.time += DT as f64;
        let t3 = Instant::now();

        // Smoothed per-step timings for the UI.
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1000.0;
        let k = 0.05;
        self.perf.think_ms += (ms(t0, t1) - self.perf.think_ms) * k;
        self.perf.physics_ms += (ms(t1, t2) - self.perf.physics_ms) * k;
        self.perf.other_ms += (ms(t2, t3) - self.perf.other_ms) * k;
    }

    /// Water buoyancy and drag, wind, plus the user's grab spring.
    fn environment_forces(&mut self) {
        let p = &self.params;
        let sea = p.water_level;
        let g = p.gravity;
        let wind = Vec3::new(self.weather.wind[0], 0.0, self.weather.wind[1]);
        // Air drag, 1/2 rho Cd v^2 per m^2 facing the wind (rho 1.2, Cd 1).
        let wind_push = 0.6 * p.wind_force * DT;
        let mut splashes = Vec::new();
        let kicks: HashMap<u64, Vec3> = self.acts.kicks.drain(..).collect();
        // Everything here is a force for this step only. (Forces, not impulses: a creature's
        // parts form a multibody, which keeps no velocity set on a part directly.)
        for o in &self.obstacles {
            if let Some(b) = self.shards[o.shard].world.bodies.get_mut(o.body) {
                b.reset_forces(false);
                b.reset_torques(false);
                if b.is_dynamic() {
                    for w in &self.acts.whirlwinds {
                        let pos = b.translation();
                        let air = w.wind_at(pos, self.terrain.height_at(pos.x, pos.z), self.time);
                        if air != Vec3::ZERO {
                            let pull = ((air - b.linvel()) / 0.6).clamp_length_max(40.0);
                            let m = b.mass();
                            b.add_force((pull + Vec3::Y * (air.y / 18.0 * g * 1.1)) * m, true);
                        }
                    }
                }
            }
        }
        for c in &mut self.creatures {
            let world = &mut self.shards[c.shard].world;
            for (k, part) in c.parts.iter().enumerate() {
                let Some(b) = world.bodies.get_mut(part.body) else {
                    continue;
                };
                b.reset_forces(false);
                b.reset_torques(false);
                // A throw: the same change of velocity for every part, over this one step.
                if let Some(&dv) = kicks.get(&c.id) {
                    let m = b.mass();
                    b.add_force(dv * m / DT, true);
                }
                // A whirlwind drags each part toward its air (over about half a second), and
                // holds up some of its weight in the core, so it lifts heavy bodies too.
                for w in &self.acts.whirlwinds {
                    let pos = b.translation();
                    let air = w.wind_at(pos, self.terrain.height_at(pos.x, pos.z), self.time);
                    if air != Vec3::ZERO {
                        let m = b.mass();
                        let pull = ((air - b.linvel()) / 0.6).clamp_length_max(40.0);
                        let carry = air.y / 18.0 * g * 1.1;
                        b.add_force((pull + Vec3::Y * carry) * m, true);
                    }
                }
                let shape = c.plan[k].shape;
                let pos = b.translation();
                let vr = shape.vertical_radius().max(0.02);
                // The sea, or a lake, pond or stream standing higher than it here.
                let wl = self.terrain.water_surface(pos.x, pos.z, sea);
                let frac = ((wl - (pos.y - vr)) / (2.0 * vr)).clamp(0.0, 1.0);
                if k == 0 {
                    if c.in_water < 0.3 && frac >= 0.3 && b.linvel().y < -1.5 {
                        splashes.push((c.id, pos));
                    }
                    c.in_water = frac;
                }
                if wind_push > 0.0 && frac < 1.0 {
                    let rel = wind - b.linvel();
                    let rel = Vec3::new(rel.x, 0.0, rel.z);
                    let a = shape.areas();
                    let area = (a[0] + a[1] + a[2]) / 3.0;
                    b.add_force(rel * rel.length() * (wind_push * area * (1.0 - frac)) / DT, false);
                }
                if frac <= 0.0 {
                    continue;
                }
                let mass = b.mass().max(1e-3);
                let buoy = p.water_density * shape.volume() * frac * g * DT;
                b.add_force(Vec3::Y * buoy / DT, false);
                // Drag at two points along the part's long axis, so a limb sweeping about its
                // joint feels more push at its fast-moving end.
                let pose = *b.position();
                let rot = pose.rotation;
                let areas = shape.areas();
                let reach = match shape {
                    body::PartShape::Capsule { half_height, .. } => Vec3::new(0.0, half_height * 0.5, 0.0),
                    body::PartShape::Box { half } => {
                        let k = if half[0] >= half[1].max(half[2]) { 0 } else if half[1] >= half[2] { 1 } else { 2 };
                        let mut r = Vec3::ZERO;
                        r[k] = half[k] * 0.5;
                        r
                    }
                };
                for side in [-1.0, 1.0] {
                    let at = pose.transform_point(reach * side);
                    let v_local = rot.inverse() * b.velocity_at_point(at);
                    let mut imp = Vec3::ZERO;
                    for axis in 0..3 {
                        let v = v_local[axis];
                        let coeff = p.water_drag + 0.5 * p.water_density * p.swim_drag * v.abs();
                        let damp = (coeff * areas[axis] * 0.5 * frac * DT / mass).min(0.45);
                        imp[axis] = -mass * damp * v;
                    }
                    b.add_force_at_point(rot * imp / DT, at, false);
                }
                let ang_damp = (p.water_drag / 400.0 * 3.0 * frac * DT).min(0.5);
                let w = b.angvel();
                b.add_torque(-w * (spin_inertia(b) * ang_damp / DT), false);
            }
        }
        for (id, at) in splashes {
            self.highlight(HighlightKind::Splash, id, None, at);
        }
        if let Some(grab) = &self.grab {
            let held = grab.creature.and_then(|id| self.index.get(&id)).map(|&i| &self.creatures[i]);
            let creature_mass = held.map(|c| c.mass);
            let body = match (held, grab.obstacle) {
                (Some(c), _) => c.parts.get(grab.part).or(c.parts.first()).map(|q| (c.shard, q.body)),
                (None, Some(id)) => self.obstacles.iter().find(|o| o.id == id).map(|o| (o.shard, o.body)),
                _ => None,
            };
            if let Some(b) = body.and_then(|(s, h)| self.shards[s].world.bodies.get_mut(h)) {
                let m = creature_mass.unwrap_or_else(|| b.mass());
                let anchor = b.position().transform_point(grab.local);
                let v = b.velocity_at_point(anchor);
                let k = 60.0;
                let c = 2.0 * k.sqrt();
                let mut force = ((grab.target - anchor) * k - v * c + Vec3::Y * g) * m;
                let max = 120.0 * m;
                if force.length() > max {
                    force = force.normalize() * max;
                }
                b.add_force_at_point(force, anchor, true);
            }
            // Calm the rest of a dangling creature so it doesn't spin like a flail.
            if let Some(&i) = grab.creature.and_then(|id| self.index.get(&id)) {
                let c = &self.creatures[i];
                for part in &c.parts {
                    if let Some(b) = self.shards[c.shard].world.bodies.get_mut(part.body) {
                        let w = b.angvel();
                        b.add_torque(-w * (spin_inertia(b) * 0.03 / DT), false);
                    }
                }
            }
        }
    }

    /// Motor energy accounting, speed limits, breath, and sanity checks.
    fn after_physics(&mut self) {
        let p = &self.params;
        let max_v = p.max_body_speed;
        let max_w = p.max_body_spin;
        let terrain = &self.terrain;
        let rain = self.weather.rain;
        let wl = p.water_level;
        let mut drowned = Vec::new();
        let mut impacts = Vec::new();
        for c in &mut self.creatures {
            if c.dead.is_some() {
                continue;
            }
            let world = &mut self.shards[c.shard].world;
            let mut work = 0.0;
            for j in &c.joints {
                let w = body::muscle_work(world, &c.parts, j, DT);
                work += match j.kind {
                    body::JointKind::Hinge => w,
                    body::JointKind::Swivel => w * p.swivel_energy_cost,
                };
            }
            c.energy -= work * p.motor_energy_cost;

            let mut bad = false;
            let mut slow = 1.0f32;
            c.fall.resize(c.parts.len(), 0.0);
            for (k, part) in c.parts.iter().enumerate() {
                let Some(b) = world.bodies.get_mut(part.body) else {
                    bad = true;
                    continue;
                };
                let v = b.linvel();
                // A part that was falling fast and has just been stopped, at the ground (not in
                // water), has hit it: the harder and heavier, the louder.
                let was = c.fall[k];
                c.fall[k] = v.y;
                if was < -2.0 && v.y > was * 0.3 {
                    let at = b.translation();
                    let reach = c.plan[k].shape.bounding_radius() + 0.25;
                    if at.y > terrain.water_surface(at.x, at.z, wl) && at.y - terrain.height_at(at.x, at.z) < reach {
                        let strength = (-was * b.mass().sqrt() / 15.0).min(1.5);
                        if strength > 0.08 {
                            impacts.push((c.id, at, strength));
                        }
                    }
                }
                let w = b.angvel();
                if !v.is_finite() || !w.is_finite() || !b.translation().is_finite() {
                    bad = true;
                    continue;
                }
                slow = slow.min(max_v / v.length().max(1e-6)).min(max_w / w.length().max(1e-6));
            }
            if bad {
                c.dead = Some(DeathCause::Glitch);
                continue;
            }
            // Over the speed or spin cap: the whole creature slows together (its joints stay
            // together; slowing one part alone would wrench them).
            if slow < 1.0 {
                match body::multibody_mut(world, &c.parts) {
                    Some(mb) => {
                        let mut v = mb.generalized_velocity_mut();
                        v *= slow;
                    }
                    None => {
                        if let Some(b) = world.bodies.get_mut(c.parts[0].body) {
                            let (v, w) = (b.linvel(), b.angvel());
                            b.set_linvel(v * slow, false);
                            b.set_angvel(w * slow, false);
                        }
                    }
                }
            }
            let torso = &world.bodies[c.parts[0].body];
            let pos = torso.translation();
            if pos.y < terrain.height_at(pos.x, pos.z) - 3.0 || !terrain.contains(pos.x, pos.z, -3.0) {
                // Tunneled through the ground or out of the world.
                c.dead = Some(DeathCause::Glitch);
                continue;
            }

            // Breathing: muscles burn oxygen in proportion to their power per kg of body; the
            // creature breathes it back in (at a rate set by its lungs) while its head is out of
            // the water. Underwater it holds its breath, and drowns when that runs out.
            let half = c.torso_half();
            let mouth = torso.position().transform_point(Vec3::new(half.x, 0.0, 0.0));
            let lungs = c.genome.traits.breath;
            let power = work / DT / c.mass.max(0.1);
            c.exertion += (power - c.exertion) * (DT / 1.0);
            let exertion = p.exertion_oxygen * power;
            let surface = terrain.water_surface(mouth.x, mouth.z, p.water_level);
            let head_under = mouth.y < surface - 0.05 || c.in_water >= 0.95;
            if head_under {
                let hold = p.breath_time * (0.3 + 1.7 * lungs);
                c.oxygen -= DT / hold.max(0.1) + exertion * DT;
            } else {
                c.oxygen += (p.breath_rate * (0.6 + 0.8 * lungs) - exertion) * DT;
            }
            c.oxygen = c.oxygen.min(1.0);
            let suffocating = head_under && c.oxygen <= 0.0;
            if !c.drowning && suffocating {
                drowned.push((c.id, pos));
            }
            c.drowning = suffocating;
            if c.oxygen <= 0.0 {
                c.oxygen = 0.0;
                if c.drowning {
                    c.health -= p.drowning_damage * DT;
                }
            }

            // Drinking: any water will do, sea, lake, stream or pond, as long as the mouth is in it.
            if mouth.y < surface + 0.1 {
                c.hydration += p.drink_rate * DT;
            }
            c.hydration += p.rain_drinking * rain * DT;
            if p.thirst_time > 0.0 {
                c.hydration -= DT / p.thirst_time;
            }
            c.hydration = c.hydration.clamp(0.0, 1.0);
            if c.hydration <= 0.0 {
                c.health -= p.dehydration_damage * DT;
            }
        }
        for (id, at) in drowned {
            self.highlight(HighlightKind::Drowning, id, None, at);
        }
        for (id, at, strength) in impacts {
            self.highlight_with(HighlightKind::Impact, id, None, at, strength);
        }
    }

    fn metabolism(&mut self, bdt: f32) {
        let p = &self.params;
        for c in &mut self.creatures {
            let t = &c.genome.traits;
            let basal = p.basal_metabolism * c.mass.powf(0.75) * (1.0 + 0.5 * t.armor + 0.3 * t.breath);
            let brain = c.brain_cost(p, self.brain_price);
            let calling = p.call_cost * crate::life::call_loudness(c);
            c.energy -= (basal + brain + calling) * bdt;
            if c.health < c.max_health && c.energy > 0.0 {
                let h = (p.health_regen * bdt).min(c.max_health - c.health);
                c.health += h;
                c.energy -= h * 0.5;
            }
            // The mating drive builds while it could breed (mature, past its cooldown, fed enough
            // to be a partner) and fades quickly while it couldn't.
            let ready = c.age >= p.maturity_age
                && c.mate_cooldown <= 0.0
                && c.energy >= p.mate_energy_fraction * 0.5 * c.max_energy;
            c.mate_drive = if ready {
                let rate = 2f32.powf(c.genome.traits.libido.clamp(-1.0, 1.0)) / p.mate_drive_time.max(0.1);
                (c.mate_drive + rate * bdt).min(1.0)
            } else {
                c.mate_drive * (1.0 - bdt).max(0.0)
            };
            c.age += bdt;
            c.mate_cooldown -= bdt;
            c.attack_cooldown -= bdt;
            c.attack_flash -= bdt;
            c.mate_flash -= bdt;
            // Pain fades by the same amount each second whatever the brain rate.
            c.pain *= 0.85f32.powf(bdt * 30.0);
            c.attacked_ago += bdt;
            if c.dead.is_none() {
                if c.energy <= 0.0 {
                    c.dead = Some(DeathCause::Starved);
                } else if c.health <= 0.0 {
                    c.dead = Some(if c.attacked_ago < RECENT_ATTACK {
                        DeathCause::Killed
                    } else if c.drowning || c.oxygen < 0.5 {
                        DeathCause::Drowned
                    } else if c.hydration <= 0.0 {
                        DeathCause::Dehydrated
                    } else {
                        DeathCause::Killed
                    });
                } else if c.age > p.lifespan {
                    c.dead = Some(DeathCause::OldAge);
                }
            }
        }
    }

    fn rain(&mut self, dt: f32) {
        self.rain_timer -= dt;
        if (self.creatures.len() as u32) < self.params.min_population && self.rain_timer <= 0.0 {
            self.rain_timer = self.params.rain_interval;
            self.rain_creature(None);
        }
    }

    /// Dead creatures leave meat where their body parts were.
    fn process_deaths(&mut self) {
        if !self.creatures.iter().any(|c| c.dead.is_some()) {
            return;
        }
        // Credit kills first, while every killer (even one that died this step too) is still in
        // the index.
        let mut kills = Vec::new();
        for c in self.creatures.iter().filter(|c| c.dead == Some(DeathCause::Killed)) {
            if let Some(k) = c.last_attacker.filter(|_| c.attacked_ago < RECENT_ATTACK) {
                let at = self.torso_of(c).map(|b| b.translation()).unwrap_or(c.last_pos);
                kills.push((k, c.id, at));
            }
        }
        for (killer, victim, at) in kills {
            if let Some(k) = self.creature_mut(killer) {
                k.life.kills += 1;
                self.highlight(HighlightKind::Kill, killer, Some(victim), at);
            }
        }
        let mut dead = Vec::new();
        let mut i = 0;
        while i < self.creatures.len() {
            if self.creatures[i].dead.is_some() {
                dead.push(self.creatures.swap_remove(i));
            } else {
                i += 1;
            }
        }
        self.rebuild_index();
        // Lives good enough for the gene pool go in with part of what they learned, read back
        // (in one round trip) before their brains are freed.
        let mut learned = HashMap::new();
        if self.params.learned_inheritance > 0.0 {
            let worthy: Vec<u64> = dead
                .iter()
                .filter(|c| !matches!(c.dead, Some(DeathCause::Removed | DeathCause::Glitch)))
                .filter(|c| self.pool.would_take(self.life_score(c)))
                .map(|c| c.id)
                .collect();
            learned = match self.brains.as_mut() {
                Some(g) => g.read_plastic(&worthy),
                None => dead
                    .iter()
                    .filter(|c| worthy.contains(&c.id))
                    .map(|c| (c.id, c.brain.plastic.clone()))
                    .collect(),
            };
        }
        for c in dead {
            let cause = c.dead.unwrap_or(DeathCause::Removed);
            match cause {
                DeathCause::Starved => self.stats.deaths_starved += 1,
                DeathCause::Killed => self.stats.deaths_killed += 1,
                DeathCause::OldAge => self.stats.deaths_old += 1,
                DeathCause::Drowned => self.stats.deaths_drowned += 1,
                DeathCause::Dehydrated => self.stats.deaths_thirst += 1,
                DeathCause::Removed | DeathCause::Glitch => self.stats.deaths_other += 1,
            }
            if !matches!(cause, DeathCause::Removed | DeathCause::Glitch) {
                let score = self.life_score(&c);
                match learned.get(&c.id) {
                    Some(p) => {
                        let mut g = c.genome.clone();
                        g.brain.absorb(p, self.params.learned_inheritance);
                        self.pool.consider(score, c.generation, &g);
                    }
                    None => self.pool.consider(score, c.generation, &c.genome),
                }
                self.stats.record_life(c.life.food(), c.life.distance, c.life.children as f32, c.age);
                // Meat where each body part lay.
                let leftover = c.energy.max(0.0) * 0.5;
                // Shares by volume of the body as it was built (the density param may have
                // changed since).
                let volume: f32 = c.plan.iter().map(|s| s.shape.volume()).sum::<f32>().max(1e-6);
                for (k, part) in c.parts.iter().enumerate() {
                    let Some(b) = self.shards[c.shard].world.bodies.get(part.body) else {
                        continue;
                    };
                    let share = c.plan[k].shape.volume() / volume;
                    let part_mass = c.mass * share;
                    let e = part_mass * self.params.meat_per_mass + leftover * share;
                    if e > 0.5 {
                        let pos = b.translation();
                        self.add_meat(pos, e);
                    }
                }
            }
            // Earth it was carrying drops where it died.
            if c.soil > 0.0 {
                let at = self
                    .torso_of(&c)
                    .map(|b| b.translation())
                    .filter(|p| p.is_finite())
                    .unwrap_or(c.last_pos);
                let (depth, height) = (self.params.dig_depth, self.params.pile_height);
                // Where the ground there is already piled as high as it goes, spill around it.
                let mut left = c.soil;
                for (dx, dz) in [(0.0, 0.0), (1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5)] {
                    if left <= 1e-6 || !at.is_finite() {
                        break;
                    }
                    left -= self.terrain.reshape(at.x + dx, at.z + dz, left, depth, height);
                }
            }
            if self.grab.as_ref().is_some_and(|g| g.creature == Some(c.id)) {
                self.grab = None;
            }
            if self.selected == Some(c.id) {
                self.selected = None;
            }
            body::remove_body(&mut self.shards[c.shard].world, &c.parts, &c.joints);
            if let Some(g) = self.brains.as_mut() {
                g.unregister(c.id);
            }
        }
        self.settle_terrain();
    }

    pub fn sample_history(&mut self) {
        self.species = self.species_summary();
        let n = self.creatures.len().max(1) as f32;
        let mean = |f: &dyn Fn(&Creature) -> f32| self.creatures.iter().map(f).sum::<f32>() / n;
        let sample = Sample {
            time: self.time as f32,
            population: self.creatures.len() as u32,
            plants: self.plants.len() as u32,
            meat: self.meat.len() as u32,
            mean_generation: mean(&|c| c.generation as f32),
            max_generation: self.creatures.iter().map(|c| c.generation).max().unwrap_or(0),
            mean_diet: mean(&|c| c.genome.traits.diet),
            mean_neurons: mean(&|c| c.genome.brain.h as f32),
            mean_limbs: mean(&|c| (c.parts.len() - 1) as f32),
            mean_mass: mean(&|c| c.mass),
            life_food: self.stats.life_food,
            life_distance: self.stats.life_distance,
            births: self.stats.births,
            species: self.species.len() as u32,
        };
        self.history.push(sample);
        if self.history.len() > MAX_HISTORY {
            // Keep the whole run by thinning older samples instead of dropping them.
            let mut k = 0;
            self.history.retain(|_| {
                k += 1;
                k % 2 == 0
            });
        }
    }

    /// A rough count of how many species exist.
    pub fn count_species(&self) -> u32 {
        self.species_summary().len() as u32
    }

    /// Greedy clustering by genetic distance (the same distance that decides who can interbreed),
    /// largest species first.
    pub fn species_summary(&self) -> Vec<SpeciesInfo> {
        let threshold = self.params.species_threshold.max(0.01);
        let mut reps: Vec<(&Genome, SpeciesInfo)> = Vec::new();
        for c in &self.creatures {
            let slot = reps
                .iter()
                .position(|(g, _)| g.distance(&c.genome) < threshold)
                .or_else(|| {
                    // Past 64 clusters, lump stragglers into the closest one.
                    (reps.len() >= 64).then(|| {
                        reps.iter()
                            .enumerate()
                            .min_by(|a, b| a.1.0.distance(&c.genome).total_cmp(&b.1.0.distance(&c.genome)))
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    })
                });
            let i = match slot {
                Some(i) => i,
                None => {
                    reps.push((
                        &c.genome,
                        SpeciesInfo {
                            representative: c.id,
                            color: c.genome.color(),
                            ..SpeciesInfo::default()
                        },
                    ));
                    reps.len() - 1
                }
            };
            let s = &mut reps[i].1;
            s.count += 1;
            s.mean_diet += c.genome.traits.diet;
            s.mean_limbs += (c.parts.len() - 1) as f32;
            s.mean_neurons += c.genome.brain.h as f32;
            s.mean_mass += c.mass;
            s.births += c.life.children;
            s.max_generation = s.max_generation.max(c.generation);
            // Its star: the member doing best so far.
            let score = self.life_score(c);
            if score >= s.best_score {
                s.best_score = score;
                s.representative = c.id;
            }
        }
        let mut out: Vec<SpeciesInfo> = reps
            .into_iter()
            .map(|(_, mut s)| {
                let n = s.count.max(1) as f32;
                s.mean_diet /= n;
                s.mean_limbs /= n;
                s.mean_neurons /= n;
                s.mean_mass /= n;
                s
            })
            .collect();
        out.sort_by(|a, b| b.count.cmp(&a.count));
        out
    }
}

pub fn obstacle_lift(shape: ObstacleShape) -> f32 {
    match shape {
        ObstacleShape::Box { half } => half[1] * 0.85,
        ObstacleShape::Ball { radius } => radius * 0.8,
    }
}

/// A body's typical moment of inertia (the mean about its principal axes), kg·m², for damping
/// its spin with a torque.
fn spin_inertia(b: &rapier3d::prelude::RigidBody) -> f32 {
    let i = b.mass_properties().local_mprops.principal_inertia();
    (i.x + i.y + i.z) / 3.0
}

/// How near the boxes around two things' parts must come for them to share a physics world, m:
/// more than they can close in on each other before the next regroup.
const REGROUP_MARGIN: f32 = 0.6;
/// Something new joins the world of anything within this, m.
const REGROUP_REACH: f32 = 6.0;
/// Physics ticks between regroups.
pub(crate) const REGROUP_EVERY: u64 = 6;
/// Most body parts moved between worlds per regroup to even out their loads.
const REGROUP_MOVES: usize = 60;

fn obstacle_body(id: u64, dynamic: bool, pose: Pose) -> RigidBodyBuilder {
    if dynamic { RigidBodyBuilder::dynamic() } else { RigidBodyBuilder::fixed() }
        .pose(pose)
        .user_data(Tag::Obstacle { id }.encode())
}

fn obstacle_collider(id: u64, shape: ObstacleShape) -> ColliderBuilder {
    match shape {
        ObstacleShape::Box { half } => ColliderBuilder::cuboid(half[0], half[1], half[2]),
        ObstacleShape::Ball { radius } => ColliderBuilder::ball(radius),
    }
    .density(1500.0)
    .friction(0.9)
    .user_data(Tag::Obstacle { id }.encode())
}

/// How far an obstacle reaches from its center, m.
fn obstacle_radius(shape: ObstacleShape) -> f32 {
    match shape {
        ObstacleShape::Box { half } => (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
        ObstacleShape::Ball { radius } => radius,
    }
}
