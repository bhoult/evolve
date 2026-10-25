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

/// Where a creature came from.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Origin {
    /// From a save made before origins were noted.
    #[default]
    Unknown,
    /// Born in the world to parents.
    Born,
    /// Fell from the sky, made at random.
    Random,
    /// Fell from the sky, a mutated copy of a genome in the gene pool.
    PoolClone,
    /// Fell from the sky, a cross of two genomes in the gene pool.
    PoolCross,
    /// Dropped in by hand or from the zoo.
    Dropped,
    /// Fell from the sky, an exact copy of a gene-pool genome, living again to check its score.
    Retest,
    /// Fell from the sky, a mutated copy of a creature in the zoo.
    Zoo,
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
    /// Where the rest of its energy went.
    #[serde(default)]
    pub spent: Spending,
    /// Meters closed on food it could sense while it had room to eat (see `life::approach_reward`).
    #[serde(default)]
    pub approached: f32,
    /// Food eaten (plants and meat) at least `score_food_away` meters from where it came into the
    /// world: what the life score counts, so landing beside a plant earns nothing.
    #[serde(default)]
    pub earned: f32,
    /// Kills made at least `score_food_away` meters from where it came into the world: the ones
    /// the life score counts, so biting whatever lands against its head earns nothing.
    #[serde(default)]
    pub kills_away: u32,
}

/// Energy a creature has used up, by what it went on.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Spending {
    /// Staying alive: the resting burn of its body.
    pub basal: f32,
    pub brain: f32,
    /// Work done by its joint motors.
    pub muscle: f32,
    /// Healing its wounds.
    pub healing: f32,
    pub digging: f32,
    /// Calling and attacking.
    pub other: f32,
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
    /// Where it came into the world: where it fell, or was born.
    #[serde(default)]
    pub home: Vec3,
    /// The fingerprint of the gene-pool entry it's an exact copy of, if it fell to retest it.
    #[serde(default)]
    pub retest_of: Option<u64>,
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
    /// Births in a row behind it, back to an ancestor that fell from the sky (0 if it fell).
    #[serde(default)]
    pub lineage: u32,
    #[serde(default)]
    pub origin: Origin,
    pub dead: Option<DeathCause>,
    /// Under the blessing of heaven (a power): when it next changes, sim seconds. It glows,
    /// can't die but by being removed, and every `heaven_mutation_interval` it's remade from a
    /// mutated copy of its genome where it stands.
    #[serde(default)]
    pub heaven: Option<f64>,
    /// Which physics world its body is in (see `Sim::shards`).
    #[serde(default)]
    pub shard: usize,
    /// What nearby totems stir in it, 0..1: love, war, fear (see `acts`).
    #[serde(skip)]
    pub urge: [f32; 3],
    /// Where a totem of war drives it, as an acceleration (m/s²): toward a fight.
    #[serde(skip)]
    pub urge_push: Vec3,
    /// What it holds in its mouth (see `carry`).
    #[serde(default)]
    pub held: Option<crate::carry::Held>,
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
        // A totem's urge overrides the creature's own leaning.
        crate::brain::OutputBias {
            eat: t.appetite,
            attack: t.aggression + 10.0 * self.urge[1],
            mate: t.libido + 10.0 * self.urge[0],
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
        // A totem of love drowns out caution about being fed enough, or about having eaten its
        // own way (though a child's body still has to be paid for).
        self.dead.is_none()
            && self.age >= p.maturity_age
            && self.mate_cooldown <= 0.0
            && self.life.food() >= p.breeding_food * (1.0 - self.urge[0])
            && self.energy >= energy_fraction * (1.0 - self.urge[0]) * self.max_energy
            && self.output(output::MATE) > 0.0
    }
}

/// Plants and meat both. A plant is grass, which stays small, or a tree seedling, which turns
/// into a woody tree (see `forest`) once it has grown full without being eaten.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Food {
    pub id: u64,
    pub pos: Vec3,
    pub energy: f32,
    /// Saves from before trees hold only grass.
    #[serde(default)]
    pub kind: PlantKind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlantKind {
    #[default]
    Grass,
    Seedling,
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
    /// The ground, in `Terrain::tiles` squared pieces, row by row.
    #[serde(default)]
    pub ground: Vec<ColliderHandle>,
    /// The ground in one piece, as saves made before it was split have it (moved to `ground` on
    /// loading).
    #[serde(default, skip_serializing)]
    pub terrain_collider: Option<ColliderHandle>,
    pub walls: Vec<ColliderHandle>,
    /// Tree trunks, by tree id: like the ground, every tree stands in every world.
    #[serde(default)]
    pub trunks: HashMap<u64, ColliderHandle>,
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
    /// Each phase of the step (named in `PHASE_NAMES`), ms per step, smoothed. Phases that run
    /// only now and then (brains, ecology, water) count as their cost spread over the steps
    /// between.
    pub phases: [f32; PHASES],
    /// This step's phases so far, before smoothing.
    pub this_step: [f32; PHASES],
    /// The runner's work for the screen, ms each time (smoothed): building a snapshot, and taking
    /// a fresh copy of the terrain when it has changed.
    pub publish_ms: f32,
    pub terrain_copy_ms: f32,
}

/// Runs `f` on every creature together with the physics world it's in, the worlds in parallel
/// (all of a creature's bodies are in its own world, so their work never overlaps), and returns
/// what `f` gave for each creature, in creature order. Each world takes its creatures in creature
/// order, so the outcome is the same however many threads there are.
pub(crate) fn in_shards<R: Send>(
    creatures: &mut [Creature],
    shards: &mut [Shard],
    f: impl Fn(&mut Creature, &mut PhysicsWorld) -> R + Sync,
) -> Vec<R> {
    use rayon::prelude::*;
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); shards.len()];
    for (i, c) in creatures.iter().enumerate() {
        groups[c.shard].push(i);
    }
    let all = Creatures(creatures.as_mut_ptr());
    let done: Vec<Vec<(usize, R)>> = shards
        .par_iter_mut()
        .zip(groups.par_iter())
        .map(|(shard, group)| {
            group
                .iter()
                .map(|&i| {
                    // SAFETY: every creature index is in exactly one world's group and each group
                    // is run by one task, so no two tasks reach the same creature; the slice
                    // outlives the tasks, which end before this function returns.
                    let c = unsafe { &mut *all.at(i) };
                    (i, f(c, &mut shard.world))
                })
                .collect()
        })
        .collect();
    let mut slots: Vec<Option<R>> = std::iter::repeat_with(|| None).take(creatures.len()).collect();
    for (i, r) in done.into_iter().flatten() {
        slots[i] = Some(r);
    }
    slots.into_iter().map(|r| r.expect("every creature is in a world")).collect()
}

/// The creatures, shared between the worlds' tasks in `in_shards`, each task taking only its own.
struct Creatures(*mut Creature);

impl Creatures {
    fn at(&self, i: usize) -> *mut Creature {
        // SAFETY: `i` is an index into the slice the pointer was taken from (see `in_shards`).
        unsafe { self.0.add(i) }
    }
}

// SAFETY: the tasks that share it never touch the same creature (see `in_shards`).
unsafe impl Send for Creatures {}
unsafe impl Sync for Creatures {}

/// The phases of a step, in order, as `Perf::phases` times them.
pub const PHASES: usize = 17;
/// Steps kept in `Sim::step_log`: half a minute at 60 a second.
pub const STEP_LOG: usize = 1800;

pub const PHASE_NAMES: [&str; PHASES] = [
    "weather and totems",
    "brain results",
    "calls and lookup grids",
    "senses",
    "brains",
    "muscles",
    "regrouping",
    "forces",
    "physics",
    "after physics",
    "interacting",
    "metabolism",
    "ecology",
    "water",
    "history",
    "rain",
    "deaths",
];

impl Perf {
    /// The phases of a step in a line, e.g. "senses 1.20, physics 12.4, ... ms".
    pub fn phases_line(&self) -> String {
        let parts: Vec<String> = PHASE_NAMES.iter().zip(&self.phases).map(|(n, ms)| format!("{n} {ms:.2}")).collect();
        format!("{} ms per step", parts.join(", "))
    }

    /// Adds the time since `since` to this step's phase `k`, and moves `since` on to now.
    pub(crate) fn lap(&mut self, k: usize, since: &mut Instant) {
        let now = Instant::now();
        self.this_step[k] += (now - *since).as_secs_f32() * 1000.0;
        *since = now;
    }
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
    /// Woody trees (see `forest`); seedlings are still plants.
    #[serde(default)]
    pub trees: Vec<crate::forest::Tree>,
    /// Fruit hanging in the trees and lying under them.
    #[serde(default)]
    pub fruit: Vec<crate::forest::Fruit>,
    /// Whether the world has had its first trees (false in saves from before trees).
    #[serde(default)]
    pub forest_planted: bool,
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
    /// Creatures owed to the rain so far (it falls when this reaches one), the death count net
    /// of births when last looked, and the deaths net of births per second lately.
    #[serde(default)]
    pub rain_due: f32,
    #[serde(skip, default = "not_seen")]
    pub rain_seen: f32,
    #[serde(default)]
    pub rain_net_deaths: f32,
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
    /// The last `STEP_LOG` steps' wall time and phase times (ms), for how even the steps are
    /// (`Sim::step_spread`).
    #[serde(skip)]
    pub step_log: std::collections::VecDeque<(f32, [f32; PHASES])>,
    /// Each physics world's time to step, ms, smoothed like `perf`.
    #[serde(skip)]
    pub shard_ms: Vec<f32>,
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
    /// Zoo creatures (and their generations) for creatures falling from the sky to be copied
    /// from (`rain_from_zoo`). Whoever holds the zoo refills it (see `zoo::refresh_rain_sample`).
    #[serde(skip)]
    pub zoo_sample: Vec<(Genome, u32)>,
    #[serde(skip)]
    pub zoo_sample_version: u64,
    /// Notable moments since the runner last collected them.
    #[serde(skip)]
    pub highlights: Vec<Highlight>,
    #[serde(skip)]
    pub highlight_seq: u64,
    /// Recent deaths and births, for reports (see `observe`).
    #[serde(skip)]
    pub census: crate::observe::Census,
    /// Creatures being followed to see how they move (see `observe::Watch`).
    #[serde(skip)]
    pub watches: Vec<crate::observe::Watch>,
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
        let mut terrain = Terrain::generate(&params, 1);
        terrain.set_collision_cell(params.ground_collision_cell);
        let rng = Rng::new(params.seed.wrapping_mul(0x9E37_79B9).wrapping_add(7));
        let mut sim = Sim {
            params,
            shards: Vec::new(),
            terrain,
            creatures: Vec::new(),
            plants: Vec::new(),
            meat: Vec::new(),
            trees: Vec::new(),
            fruit: Vec::new(),
            forest_planted: true,
            obstacles: Vec::new(),
            rng,
            tick: 0,
            time: 0.0,
            next_id: 1,
            stats: Stats::default(),
            history: Vec::new(),
            pool: GenePool::default(),
            zoo_sample: Vec::new(),
            zoo_sample_version: 0,
            selected: None,
            grab: None,
            rain_timer: 0.0,
            rain_due: 0.0,
            rain_seen: f32::NAN,
            rain_net_deaths: 0.0,
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
            step_log: Default::default(),
            shard_ms: Vec::new(),
            events: Vec::new(),
            species: Vec::new(),
            brains: None,
            brain_backend: String::new(),
            last_collect: None,
            weather: Default::default(),
            highlights: Vec::new(),
            highlight_seq: 0,
            census: Default::default(),
            watches: Vec::new(),
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
        sim.plant_forest();
        sim.scatter_stones();
        sim
    }

    /// Loose stones lying about the island in patches (the makings of anything creatures build).
    fn scatter_stones(&mut self) {
        let total = self.params.stones;
        let patches = total.div_ceil(10);
        let mut placed = 0;
        for _ in 0..patches {
            let center = self.random_land_point(8.0);
            for _ in 0..10 {
                if placed >= total {
                    return;
                }
                let a = self.rng.range(0.0, std::f32::consts::TAU);
                let r = self.rng.range(0.0, 4.0);
                let at = center + Vec3::new(a.cos() * r, 0.0, a.sin() * r);
                if self.terrain.height_at(at.x, at.z) <= self.params.water_level {
                    continue;
                }
                let shape = crate::carry::stone_shape(self.rng.range(0.0, 1.0));
                let yaw = self.rng.range(0.0, std::f32::consts::TAU);
                self.add_obstacle(at, shape, true, yaw);
                placed += 1;
            }
        }
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
        // The ground as this version splits and samples it (one piece in saves from before it
        // was split).
        self.terrain.set_collision_cell(self.params.ground_collision_cell);
        let n = self.terrain.tiles();
        if self.terrain.stride() > 1 || self.shards.iter().any(|s| s.ground.len() != n * n) {
            let tiles = self.ground_tiles();
            for shard in &mut self.shards {
                Self::place_ground(&tiles, self.params.ground_friction, shard);
            }
        }
        self.rebuild_index();
        self.species = self.species_summary();
        self.perf = Perf::default();
        self.events.clear();
        self.refresh_weather();
        self.init_brains();
        // (Also gives worlds loaded from a save their threads.)
        self.resize_shards();
        // A world saved before trees grows its forest now, rather than over hours from seedlings.
        if !self.forest_planted {
            self.plant_forest();
            self.forest_planted = true;
        }
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
            // Shared by population: a background world holds `background_population` of this
            // world's creatures, so it gets that much less of the memory (an even split left the
            // world on screen room for brains of only a few hundred neurons when it was full).
            let bg = p.background_population.clamp(0.05, 1.0) as f64;
            let weight = if self.background { bg } else { 1.0 };
            let share = p.gpu_memory_gb.max(0.1) as f64 * 1e9 * weight / (1.0 + p.parallel_worlds as f64 * bg);
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
        let mut shard = Shard { world: PhysicsWorld::new(), ground: Vec::new(), terrain_collider: None, walls: Vec::new(), trunks: HashMap::new() };
        Self::place_ground(&self.ground_tiles(), self.params.ground_friction, &mut shard);
        Self::build_walls(&self.terrain, &mut shard);
        shard
    }

    /// The pieces of the physics ground, row by row (see `Terrain::tile_shape`).
    fn ground_tiles(&self) -> Vec<(Vec3, SharedShape)> {
        let n = self.terrain.tiles();
        (0..n * n).map(|k| self.terrain.tile_shape(k / n, k % n)).collect()
    }

    /// Puts the ground into a physics world, in place of what was there.
    fn place_ground(tiles: &[(Vec3, SharedShape)], friction: f32, shard: &mut Shard) {
        for h in std::mem::take(&mut shard.ground).into_iter().chain(shard.terrain_collider.take()) {
            shard.world.remove_collider(h);
        }
        for (at, shape) in tiles {
            let c = ColliderBuilder::new(shape.clone()).translation(*at).friction(friction).user_data(Tag::Terrain.encode());
            shard.ground.push(shard.world.colliders.insert(c));
        }
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
        self.sync_trunks();
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
            for &h in &shard.ground {
                if let Some(c) = world.colliders.get_mut(h) {
                    c.set_friction(p.ground_friction);
                }
            }
        }
        let friction_changed = old.is_none_or(|o| o.body_friction != p.body_friction);
        let damping_changed = old.is_none_or(|o| o.air_drag != p.air_drag);
        let ccd_changed = old.is_none_or(|o| o.soft_ccd != p.soft_ccd);
        if friction_changed || damping_changed || ccd_changed {
            for c in &self.creatures {
                let world = &mut self.shards[c.shard].world;
                for part in &c.parts {
                    if friction_changed && let Some(col) = world.colliders.get_mut(part.collider) {
                        col.set_friction(p.body_friction);
                    }
                    if let Some(b) = world.bodies.get_mut(part.body) {
                        if damping_changed {
                            b.set_linear_damping(p.air_drag);
                        }
                        if ccd_changed {
                            b.set_soft_ccd_prediction(p.soft_ccd);
                        }
                    }
                }
            }
        }
    }

    pub fn set_params(&mut self, new: SimParams) {
        let old = std::mem::replace(&mut self.params, new);
        self.sync_physics_params(Some(&old));
        if old.ground_collision_cell != self.params.ground_collision_cell {
            let stride = self.terrain.stride();
            self.terrain.set_collision_cell(self.params.ground_collision_cell);
            if self.terrain.stride() != stride {
                let tiles = self.ground_tiles();
                for shard in &mut self.shards {
                    Self::place_ground(&tiles, self.params.ground_friction, shard);
                }
            }
        }
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
        self.reseat_trees(None);
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
        self.terrain.set_collision_cell(self.params.ground_collision_cell);
        // The same ground in every physics world (the heightfield shapes are shared).
        let tiles = self.ground_tiles();
        for shard in &mut self.shards {
            Self::place_ground(&tiles, self.params.ground_friction, shard);
            Self::build_walls(&self.terrain, shard);
        }
    }

    pub(crate) fn translate_creature(&mut self, ci: usize, delta: Vec3) {
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
        self.creatures[ci].home += delta;
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
                // (Obstacles too: a new world's stones are laid out before anything lives.)
                for o in &self.obstacles {
                    load[o.shard] += 1;
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
        self.place_brain(id, &mut genome);
        let brain = BrainState::new(&genome.brain);
        let lineage = parents
            .map(|(a, b)| {
                let line = |id| self.creature(id).map(|c| c.lineage).unwrap_or(0);
                line(a).max(line(b)) + 1
            })
            .unwrap_or(0);
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
            urge: [0.0; 3],
            urge_push: Vec3::ZERO,
            held: None,
            last_attacker: None,
            attacked_ago: long_ago(),
            attack_flash: 0.0,
            mate_flash: 0.0,
            life: LifeStats::default(),
            last_pos: Vec3::new(x, y, z),
            home: Vec3::new(x, y, z),
            retest_of: None,
            fall: Vec::new(),
            call_timer: 0.0,
            exertion: 0.0,
            soil: 0.0,
            rained: parents.is_none(),
            lineage,
            origin: if parents.is_some() { Origin::Born } else { Origin::Dropped },
            dead: None,
            heaven: None,
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

    /// Readies a creature's brain: one from before the last change of brain size grows into the
    /// new era, and on the GPU it gets its place (trimmed until it fits, if it doesn't).
    fn place_brain(&mut self, id: u64, genome: &mut Genome) {
        let (lo, hi) = self.params.neuron_range();
        genome.brain.bring_up_to_era(lo as usize, hi as usize, &mut self.rng);
        if let Some(g) = self.brains.as_mut() {
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
    }

    /// Rebuilds a living creature from a new genome where it stands, facing the same way: a new
    /// body and brain, the same creature otherwise (its id, age, energy and health in proportion,
    /// history). What it held is let go.
    pub fn remake_creature(&mut self, id: u64, genome: Genome) {
        let Some(&ci) = self.index.get(&id) else {
            return;
        };
        if self.creatures[ci].held.is_some() {
            self.let_go(ci, 0.0);
        }
        let c = &self.creatures[ci];
        let (shard, parts, joints) = (c.shard, c.parts.clone(), c.joints.clone());
        let (at, yaw) = match self.shards[shard].world.bodies.get(parts[0].body) {
            Some(b) => {
                let f = b.rotation() * Vec3::X;
                (b.translation(), (-f.z).atan2(f.x))
            }
            None => (c.last_pos, 0.0),
        };
        body::remove_body(&mut self.shards[shard].world, &parts, &joints);
        let plan = body::body_plan(&genome);
        let mass = body::plan_mass(&plan, self.params.body_density);
        let ground = self.ground_over(at.x, at.z, body::plan_extent(&plan));
        let root = Pose::from_parts(Vec3::new(at.x, (ground + body::plan_depth(&plan) + 0.3).max(at.y), at.z), Quat::from_rotation_y(yaw));
        let shard = self.shard_for(root.translation);
        let (parts, joints) = body::spawn_body(&mut self.shards[shard].world, &plan, root, id, mass, &self.params);
        let mut genome = genome;
        self.place_brain(id, &mut genome);
        let (max_energy, max_health) = (self.params.energy_per_mass * mass, 100.0 * (mass / 60.0).sqrt());
        let c = &mut self.creatures[ci];
        let health = c.health / c.max_health.max(1e-3);
        c.brain = BrainState::new(&genome.brain);
        c.genome = genome;
        (c.plan, c.parts, c.joints, c.mass, c.shard) = (plan, parts, joints, mass, shard);
        (c.max_energy, c.max_health) = (max_energy, max_health);
        c.energy = c.energy.min(max_energy);
        c.health = health * max_health;
    }

    /// Lays the blessing of heaven on a creature, or lifts it.
    pub fn bless_from_heaven(&mut self, id: u64, on: bool) {
        let next = self.time + self.params.heaven_mutation_interval as f64;
        let Some(c) = self.creature_mut(id) else {
            return;
        };
        c.heaven = on.then_some(next);
        if on {
            c.energy = c.max_energy;
            c.health = c.max_health;
            self.miracle(id, HighlightKind::Blessing);
            self.log(format!("#{id} has the blessing of heaven"));
        } else {
            self.log(format!("The blessing of heaven is lifted from #{id}"));
        }
    }

    /// The blessed of heaven whose hour has come are remade from a mutated copy of their genome.
    fn heavenly_changes(&mut self) {
        let due: Vec<u64> = self
            .creatures
            .iter()
            // (One marked dead this step is revived at its end, as the blessed are.)
            .filter(|c| !matches!(c.dead, Some(DeathCause::Removed | DeathCause::Glitch)) && c.heaven.is_some_and(|t| self.time >= t))
            .map(|c| c.id)
            .collect();
        for id in due {
            let Some(c) = self.creature(id) else { continue };
            let mut g = c.genome.clone();
            let p = self.breeding_params();
            g.mutate(&mut self.rng, &p);
            self.remake_creature(id, g);
            let next = self.time + self.params.heaven_mutation_interval as f64;
            if let Some(c) = self.creature_mut(id) {
                c.heaven = Some(next);
            }
            self.miracle(id, HighlightKind::Blessing);
            self.log(format!("#{id}, blessed of heaven, is changed"));
        }
    }

    /// A creature dropped from the sky: bred from the gene pool or entirely random.
    pub fn rain_creature(&mut self, at: Option<Vec3>) -> u64 {
        // Which of the three sources (`rain_from_world`, `rain_from_zoo`, `rain_from_random`).
        let p = &self.params;
        let (world, zoo) = (p.rain_from_world.max(0.0), p.rain_from_zoo.max(0.0));
        let total = world + zoo + p.rain_from_random.max(0.0);
        let roll = self.rng.f32() * total;
        let from_zoo = roll >= world && roll < world + zoo;
        let zoo_pick = if from_zoo && !self.zoo_sample.is_empty() {
            let (g, n) = &self.zoo_sample[self.rng.below(self.zoo_sample.len())];
            let mut g = g.clone();
            let p = self.breeding_params();
            g.mutate(&mut self.rng, &p);
            Some((g, n + 1))
        } else {
            None
        };
        if from_zoo && zoo_pick.is_none() {
            // No zoo here (a headless run, a test): the world stands in for it.
            self.stats.rain_without_zoo += 1;
        }
        let from_pool = zoo_pick.is_none() && (roll < world || from_zoo);
        let retest = if from_pool && self.rng.chance(self.params.pool_retest) {
            self.pool.retest(&mut self.rng)
        } else {
            None
        };
        let bred = if from_pool && retest.is_none() {
            let p = self.breeding_params();
            self.pool.breed(&mut self.rng, &p)
        } else {
            None
        };
        let from_pool = retest.is_some() || bred.is_some();
        let origin = match (&retest, &bred, &zoo_pick) {
            (_, _, Some(_)) => Origin::Zoo,
            (Some(_), _, _) => Origin::Retest,
            (_, Some((_, _, true)), _) => Origin::PoolCross,
            (_, Some(_), _) => Origin::PoolClone,
            _ => Origin::Random,
        };
        let retest_of = retest.as_ref().map(|r| r.2);
        let (genome, generation) = zoo_pick
            .or(retest.map(|(g, n, _)| (g, n)))
            .or(bred.map(|(g, n, _)| (g, n)))
            .unwrap_or_else(|| (Genome::random(&mut self.rng, &self.params), 0));
        let near_kin = (from_pool && self.rng.chance(self.params.rain_near_kin))
            .then(|| self.near_kin_point(&genome))
            .flatten();
        let at = at.or(near_kin).unwrap_or_else(|| self.random_land_point(4.0));
        let lift = self.params.rain_height;
        let e = self.params.start_energy;
        self.stats.rained += 1;
        let id = self.spawn_creature(genome, at, lift, e, generation, None);
        let energy = match self.creature_mut(id) {
            Some(c) => {
                c.origin = origin;
                c.retest_of = retest_of;
                c.energy
            }
            None => 0.0,
        };
        self.record_sky(energy);
        id
    }

    /// How well a lifetime went, for the gene pool and the zoo. Children count most; food eaten
    /// and real distance covered are the stepping stones that let the gene pool improve before
    /// anyone manages to breed. Lifespan is left out on purpose: it rewards giant bodies that sit
    /// still and starve slowly.
    pub fn life_score(&self, c: &Creature) -> f32 {
        self.score_parts(c).iter().sum()
    }

    /// What a life achieved, in "plants' worth": eating (food and meat, away from where it fell or
    /// was born: see `score_food_away`), hunting (kills, also away from there: `score_kill`),
    /// breeding (children, and the energy given them) and travelling (net progress, and above all
    /// closing in on food it has room to eat: `score_approach` m of that is worth a plant). Simply lasting earns
    /// nothing, and neither do children given almost nothing.
    ///
    /// Closing in on food is there because eating alone mostly measured luck: creatures from the
    /// sky that landed beside a plant scored, and ones that set off toward a plant 5 m away and
    /// starved on the way scored nothing, so walking to food never got a foothold. It wasn't
    /// enough: a lucky landing filled a 200 kg tank, worth more than any walk, so food only counts
    /// once a creature has gone to get it.
    pub fn score_parts(&self, c: &Creature) -> [f32; 4] {
        let plant = self.params.plant_max_energy.max(1.0);
        [
            c.life.earned / plant,
            c.life.kills_away as f32 * self.params.score_kill,
            c.life.children as f32 * 0.5 + c.life.invested / plant,
            c.life.distance / 50.0 + c.life.approached.max(0.0) / self.params.score_approach.max(0.1),
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
        self.add_plant_of(pos, energy, PlantKind::Grass)
    }

    pub fn add_plant_of(&mut self, pos: Vec3, energy: f32, kind: PlantKind) -> Option<u64> {
        if !self.terrain.contains(pos.x, pos.z, 0.3) {
            return None;
        }
        let id = self.next_id();
        let y = self.ground_under(pos.x, pos.z, PLANT_FOOTPRINT);
        self.plants.push(Food {
            id,
            pos: Vec3::new(pos.x, y, pos.z),
            energy,
            kind,
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
            kind: PlantKind::Grass,
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
        self.perf.this_step = [0.0; PHASES];
        let mut lap = t0;
        if self.tick % 6 == 0 {
            self.acts.expire(self.time);
            self.refresh_weather();
        }
        if brain_tick {
            self.totem_urges();
            self.perf.lap(0, &mut lap);
            self.think(bdt);
        } else {
            self.perf.lap(0, &mut lap);
        }
        let t1 = Instant::now();
        lap = t1;
        if self.tick % REGROUP_EVERY == 0 {
            self.regroup();
        }
        self.perf.lap(6, &mut lap);
        self.move_whirlwinds(DT);
        if self.tick % 30 == 0 {
            self.wash_away();
        }
        self.environment_forces();
        self.carry();
        let t_forces = Instant::now();
        self.perf.lap(7, &mut lap);
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
        self.perf.lap(8, &mut lap);
        let k = 0.05;
        self.shard_ms.resize(times.len(), 0.0);
        for (s, &t) in self.shard_ms.iter_mut().zip(&times) {
            *s += (t - *s) * k;
        }
        let slowest = times.iter().copied().fold(0.0, f32::max);
        let total: f32 = times.iter().sum();
        self.perf.shard_slowest_ms += (slowest - self.perf.shard_slowest_ms) * k;
        self.perf.shard_total_ms += (total - self.perf.shard_total_ms) * k;
        self.perf.forces_ms += ((t_forces - t1).as_secs_f32() * 1000.0 - self.perf.forces_ms) * k;
        self.after_physics();
        self.perf.lap(9, &mut lap);
        if brain_tick {
            self.interact(bdt);
            self.perf.lap(10, &mut lap);
            self.metabolism(bdt);
            self.perf.lap(11, &mut lap);
        }
        if self.tick % TICKS_PER_SECOND == 0 {
            self.heavenly_changes();
            self.ecology(1.0);
            self.perf.lap(12, &mut lap);
        }
        if self.tick % (TICKS_PER_SECOND * HYDROLOGY_EVERY) == 0 {
            self.hydrology(HYDROLOGY_EVERY as f32);
            self.perf.lap(13, &mut lap);
        }
        if self.tick % (TICKS_PER_SECOND * 2) == 0 {
            self.sample_history();
        }
        if !self.watches.is_empty() && self.tick % 6 == 0 {
            self.sample_watches();
        }
        self.perf.lap(14, &mut lap);
        self.rain(DT);
        self.perf.lap(15, &mut lap);
        self.process_deaths();
        self.tick += 1;
        self.time += DT as f64;
        let t3 = Instant::now();
        self.perf.lap(16, &mut lap);

        // Smoothed per-step timings for the UI.
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1000.0;
        let k = 0.05;
        self.perf.think_ms += (ms(t0, t1) - self.perf.think_ms) * k;
        self.perf.physics_ms += (ms(t1, t2) - self.perf.physics_ms) * k;
        self.perf.other_ms += (ms(t2, t3) - self.perf.other_ms) * k;
        let Perf { phases, this_step, .. } = &mut self.perf;
        for (x, v) in phases.iter_mut().zip(this_step.iter()) {
            *x += (v - *x) * k;
        }
        if self.step_log.len() >= STEP_LOG {
            self.step_log.pop_front();
        }
        self.step_log.push_back((ms(t0, Instant::now()), self.perf.this_step));
    }

    /// How even the last `STEP_LOG` steps were: their wall time at the median, 90th and 99th
    /// percentiles and worst, and the phases that make the slowest twentieth slow (their time
    /// there against the median step's).
    pub fn step_spread(&self) -> Option<String> {
        let n = self.step_log.len();
        if n < 60 {
            return None;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| self.step_log[a].0.total_cmp(&self.step_log[b].0));
        let at = |q: f32| self.step_log[order[((n - 1) as f32 * q) as usize]].0;
        let slow = &order[n - (n / 20).max(1)..];
        let typical = &order[n / 4..3 * n / 4];
        let mean = |set: &[usize], k: usize| set.iter().map(|&i| self.step_log[i].1[k]).sum::<f32>() / set.len() as f32;
        let mut extra: Vec<(f32, usize)> = (0..PHASES).map(|k| (mean(slow, k) - mean(typical, k), k)).collect();
        extra.sort_by(|a, b| b.0.total_cmp(&a.0));
        let what: Vec<String> = extra.iter().take(4).filter(|e| e.0 > 0.05).map(|(d, k)| format!("{} +{d:.1}", PHASE_NAMES[*k])).collect();
        Some(format!(
            "Step times over the last {} steps: median {:.1} ms, 90th percentile {:.1}, 99th {:.1}, worst {:.1}; the slowest twentieth spend more than a typical step on {}.",
            n,
            at(0.5),
            at(0.9),
            at(0.99),
            at(1.0),
            if what.is_empty() { "nothing in particular".to_string() } else { what.join(", ") }
        ))
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
        let mut kicks: HashMap<u64, Vec3> = HashMap::new();
        for (id, dv) in self.acts.kicks.drain(..) {
            *kicks.entry(id).or_default() += dv;
        }
        // Everything here is a force for this step only. (Forces, not impulses: a creature's
        // parts form a multibody, which keeps no velocity set on a part directly.)
        for o in &self.obstacles {
            if let Some(b) = self.shards[o.shard].world.bodies.get_mut(o.body) {
                b.reset_forces(false);
                b.reset_torques(false);
                if b.is_dynamic() {
                    for w in &self.acts.whirlwinds {
                        let pos = b.translation();
                        let air = w.wind_at(pos, || self.terrain.height_at(pos.x, pos.z), self.time);
                        if air != Vec3::ZERO {
                            let pull = ((air - b.linvel()) / 0.6).clamp_length_max(40.0);
                            let m = b.mass();
                            b.add_force((pull + Vec3::Y * (air.y / 18.0 * g * 1.1)) * m, true);
                        }
                    }
                }
            }
        }
        // Each creature's parts, its world's creatures together, the worlds in parallel.
        let (acts, terrain, time, kicks) = (&self.acts, &self.terrain, self.time, &kicks);
        let creature_splashes = in_shards(&mut self.creatures, &mut self.shards, |c, world| {
            let mut splash = None;
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
                // A totem of war drives it into a fight.
                if c.urge_push != Vec3::ZERO {
                    let m = b.mass();
                    b.add_force(c.urge_push * m, true);
                }
                // An earthquake shakes every part (the same for a whole creature).
                let shake = acts.shaking(b.translation().x, b.translation().z, time, (c.id % 97) as f32 * 0.37);
                if shake != Vec3::ZERO {
                    let m = b.mass();
                    b.add_force(shake * m, true);
                }
                // A whirlwind drags each part toward its air (over about half a second), and
                // holds up some of its weight in the core, so it lifts heavy bodies too.
                for w in &acts.whirlwinds {
                    let pos = b.translation();
                    let air = w.wind_at(pos, || terrain.height_at(pos.x, pos.z), time);
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
                // The sea, or a lake, pond or stream standing higher than it here, or a tsunami.
                let wave = acts.tsunami_surface(pos.x, pos.z, time, sea);
                let wl = terrain.water_surface(pos.x, pos.z, sea).max(wave.unwrap_or(f32::MIN));
                let frac = ((wl - (pos.y - vr)) / (2.0 * vr)).clamp(0.0, 1.0);
                if k == 0 {
                    if c.in_water < 0.3 && frac >= 0.3 && b.linvel().y < -1.5 {
                        splash = Some((c.id, pos));
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
                // A tsunami's water rushes on with it: drag is against the moving water, so it
                // carries what it covers.
                let flow = match (&acts.tsunami, wave) {
                    (Some(t), Some(_)) => Vec3::new(t.dir[0], 0.0, t.dir[1]) * crate::acts::TSUNAMI_SPEED,
                    _ => Vec3::ZERO,
                };
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
                    let v_local = rot.inverse() * (b.velocity_at_point(at) - flow);
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
            splash
        });
        splashes.extend(creature_splashes.into_iter().flatten());
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
        let (acts, time) = (&self.acts, self.time);
        // Each creature on its own world's thread (see `in_shards`): whether it began to drown,
        // and the parts that hit the ground hard.
        let results = in_shards(&mut self.creatures, &mut self.shards, |c, world| {
            let mut drowned = None;
            let mut impacts = Vec::new();
            if c.dead.is_some() {
                return (drowned, impacts);
            }
            let mut work = 0.0;
            for j in &c.joints {
                let w = body::muscle_work(world, &c.parts, j, DT);
                work += match j.kind {
                    body::JointKind::Hinge => w,
                    body::JointKind::Swivel => w * p.swivel_energy_cost,
                };
            }
            c.energy -= work * p.motor_energy_cost;
            c.life.spent.muscle += work * p.motor_energy_cost;

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
                return (drowned, impacts);
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
                return (drowned, impacts);
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
            let wave = acts.tsunami_surface(mouth.x, mouth.z, time, p.water_level);
            let surface = terrain.water_surface(mouth.x, mouth.z, p.water_level).max(wave.unwrap_or(f32::MIN));
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
                drowned = Some((c.id, pos));
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
            (drowned, impacts)
        });
        for (drowned, impacts) in results {
            if let Some((id, at)) = drowned {
                self.highlight(HighlightKind::Drowning, id, None, at);
            }
            for (id, at, strength) in impacts {
                self.highlight_with(HighlightKind::Impact, id, None, at, strength);
            }
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
            c.life.spent.basal += basal * bdt;
            c.life.spent.brain += brain * bdt;
            c.life.spent.other += calling * bdt;
            if c.health < c.max_health && c.energy > 0.0 {
                let h = (p.health_regen * bdt).min(c.max_health - c.health);
                c.health += h;
                c.energy -= h * 0.5;
                c.life.spent.healing += h * 0.5;
            }
            // The mating drive builds while it could breed (mature, past its cooldown, has eaten its
            // own way, fed enough to be a partner) and fades quickly while it couldn't.
            let ready = c.age >= p.maturity_age
                && c.mate_cooldown <= 0.0
                && c.life.food() >= p.breeding_food * (1.0 - c.urge[0])
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

    /// Creatures fall at the rate that replaces the dead (net of births, averaged over the last
    /// 10 s or so) plus a share of the gap up to `max_population` (`rain_fill_time`), all of
    /// it below `min_population` and easing off smoothly from there to nothing at
    /// `max_population`; below the minimum, the gap up to it besides. So the population climbs to the minimum however fast creatures die
    /// (with a fixed rate it settled wherever deaths matched the rain: about 200 against a
    /// minimum of 790), then settles in the band, where the rain just replaces the dead.
    fn rain(&mut self, dt: f32) {
        let p = &self.params;
        // Deaths net of births, per second, averaged (from the second look on: a loaded world's
        // counts go back to its start).
        let net = self.stats.deaths() as f32 - self.stats.births as f32;
        let seen = std::mem::replace(&mut self.rain_seen, net);
        if !seen.is_nan() {
            let k = (dt / 10.0).min(1.0);
            self.rain_net_deaths += ((net - seen) / dt - self.rain_net_deaths) * k;
        }
        if p.min_population == 0 {
            self.rain_due = 0.0;
            return;
        }
        let n = self.creatures.len() as f32;
        let (lo, hi) = (p.min_population as f32, (p.max_population as f32).max(p.min_population as f32 + 1.0));
        let t = ((n - lo) / (hi - lo)).clamp(0.0, 1.0);
        let ease = 1.0 - t * t * (3.0 - 2.0 * t);
        let fill = p.rain_fill_time.max(1.0);
        let rate = (self.rain_net_deaths.max(0.0) + (hi - n).max(0.0) / fill) * ease + (lo - n).max(0.0) / fill;
        self.rain_due = (self.rain_due + rate * dt).min(8.0);
        self.rain_timer -= dt;
        if self.rain_due >= 1.0 && self.rain_timer <= 0.0 && n < p.max_population as f32 {
            self.rain_due -= 1.0;
            self.rain_timer = p.rain_interval;
            self.rain_creature(None);
        }
    }

    /// Dead creatures leave meat where their body parts were.
    fn process_deaths(&mut self) {
        if !self.creatures.iter().any(|c| c.dead.is_some()) {
            return;
        }
        // The blessed of heaven don't die, of hunger, wounds, drowning, thirst or age: they're
        // made whole enough to go on (being removed, or the blessing lifted first, still ends
        // them; so does falling out of the world).
        for c in self.creatures.iter_mut().filter(|c| c.heaven.is_some()) {
            if matches!(c.dead, Some(cause) if !matches!(cause, DeathCause::Removed | DeathCause::Glitch)) {
                c.dead = None;
                c.energy = c.energy.max(0.5 * c.max_energy);
                c.health = c.health.max(0.5 * c.max_health);
                c.hydration = c.hydration.max(0.5);
                c.oxygen = c.oxygen.max(0.5);
            }
        }
        if !self.creatures.iter().any(|c| c.dead.is_some()) {
            return;
        }
        // What the dead held falls where they were.
        for ci in 0..self.creatures.len() {
            if self.creatures[ci].dead.is_some() && self.creatures[ci].held.is_some() {
                self.drop_held(ci);
            }
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
        let away = self.params.score_food_away;
        for (killer, victim, at) in kills {
            if let Some(k) = self.creature_mut(killer) {
                k.life.kills += 1;
                if Vec3::new(at.x - k.home.x, 0.0, at.z - k.home.z).length() >= away {
                    k.life.kills_away += 1;
                }
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
                let at = self.torso_of(&c).map(|b| b.translation()).filter(|p| p.is_finite()).unwrap_or(c.last_pos);
                self.record_death(&c, at);
                let score = self.life_score(&c);
                let retested = c.retest_of.is_some_and(|fp| self.pool.retested(fp, score));
                if !retested && self.pool.would_take(score) {
                    let parts = self.score_parts(&c);
                    self.stats.record_admitted(parts);
                }
                match learned.get(&c.id) {
                    _ if retested => {}
                    Some(p) => {
                        let mut g = c.genome.clone();
                        g.brain.absorb(p, self.params.learned_inheritance);
                        self.pool.consider(score, c.generation, &g);
                    }
                    None => self.pool.consider(score, c.generation, &c.genome),
                }
                self.stats.record_life(c.life.food(), c.life.distance, c.life.children as f32, c.age);
                // Meat where each body part lay, plus half the energy it still held. A body born
                // in the world was paid for by its parents (`life::birth_cost`), so it leaves that,
                // `meat_per_mass` a kg. One that fell from the sky cost nothing, so it leaves meat
                // (`sky_meat_per_mass` a kg) only when something killed it: hunting pays, but
                // starving, drowning and old age leave nothing. Otherwise the population floor
                // rains free carcasses, and big carrion-eaters that never move win.
                let killed = cause == DeathCause::Killed;
                let body_meat = match (c.rained, killed) {
                    (false, _) => self.params.meat_per_mass,
                    (true, true) => self.params.sky_meat_per_mass,
                    (true, false) => 0.0,
                };
                let leftover = if !c.rained || killed { c.energy.max(0.0) * 0.5 } else { 0.0 };
                // Shares by volume of the body as it was built (the density param may have
                // changed since).
                let volume: f32 = c.plan.iter().map(|s| s.shape.volume()).sum::<f32>().max(1e-6);
                for (k, part) in c.parts.iter().enumerate() {
                    let Some(b) = self.shards[c.shard].world.bodies.get(part.body) else {
                        continue;
                    };
                    let share = c.plan[k].shape.volume() / volume;
                    let part_mass = c.mass * share;
                    let e = part_mass * body_meat + leftover * share;
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
pub(crate) fn obstacle_radius(shape: ObstacleShape) -> f32 {
    match shape {
        ObstacleShape::Box { half } => (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
        ObstacleShape::Ball { radius } => radius,
    }
}

fn not_seen() -> f32 {
    f32::NAN
}
