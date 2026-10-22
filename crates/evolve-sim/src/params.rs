//! Every tunable number in the simulation. The UI builds its sliders from `SimParams::visit`,
//! so adding a param here is all it takes to make it adjustable.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// Takes effect immediately.
    Live,
    /// Stored now, applied when the world is regenerated.
    NewWorld,
}

pub struct ParamInfo {
    pub group: &'static str,
    pub name: &'static str,
    pub doc: &'static str,
    pub min: f64,
    pub max: f64,
    pub kind: ParamKind,
}

pub enum ParamRef<'a> {
    F32(&'a mut f32),
    U32(&'a mut u32),
    U64(&'a mut u64),
    Bool(&'a mut bool),
}

pub trait AsParamRef {
    fn as_param_ref(&mut self) -> ParamRef<'_>;
}
impl AsParamRef for f32 {
    fn as_param_ref(&mut self) -> ParamRef<'_> {
        ParamRef::F32(self)
    }
}
impl AsParamRef for u32 {
    fn as_param_ref(&mut self) -> ParamRef<'_> {
        ParamRef::U32(self)
    }
}
impl AsParamRef for u64 {
    fn as_param_ref(&mut self) -> ParamRef<'_> {
        ParamRef::U64(self)
    }
}
impl AsParamRef for bool {
    fn as_param_ref(&mut self) -> ParamRef<'_> {
        ParamRef::Bool(self)
    }
}

macro_rules! params {
    ($( $group:literal { $( $(#[doc = $doc:literal])* $name:ident : $ty:ty = $default:expr, [$min:expr, $max:expr], $kind:ident; )* } )*) => {
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        #[serde(default)]
        pub struct SimParams {
            $($( $(#[doc = $doc])* pub $name: $ty, )*)*
        }

        impl Default for SimParams {
            fn default() -> Self {
                Self { $($( $name: $default, )*)* }
            }
        }

        impl SimParams {
            pub fn visit(&mut self, f: &mut dyn FnMut(&ParamInfo, ParamRef<'_>)) {
                $($(
                    f(
                        &ParamInfo {
                            group: $group,
                            name: stringify!($name),
                            doc: concat!($($doc),*),
                            min: $min as f64,
                            max: $max as f64,
                            kind: ParamKind::$kind,
                        },
                        self.$name.as_param_ref(),
                    );
                )*)*
            }
        }
    };
}

params! {
    "World" {
        /// Random seed for terrain and fertility.
        seed: u64 = 1, [0, 1_000_000], NewWorld;
        /// Width and depth of the square world, ocean included, in meters.
        world_size: f32 = 220.0, [60, 800], NewWorld;
        /// Height samples along each side of the terrain grid.
        terrain_resolution: u32 = 177, [33, 513], NewWorld;
        /// Radius of the island as a fraction of half the world's width. Above about 1.3 the land
        /// reaches the edges and there is no ocean.
        island_size: f32 = 0.8, [0.2, 1.6], NewWorld;
        /// How ragged the coastline is: 0 a round island, 0.5 deep bays and headlands.
        coast_roughness: f32 = 0.25, [0, 0.6], NewWorld;
        /// Height of the tallest hills, in meters.
        terrain_height: f32 = 22.0, [0, 80], NewWorld;
        /// Typical width of hills and valleys, in meters.
        terrain_feature_size: f32 = 70.0, [10, 300], NewWorld;
        /// How much small-scale bumpiness is layered on the hills (0.2 smooth, 0.8 rugged).
        terrain_roughness: f32 = 0.5, [0.2, 0.8], NewWorld;
        /// Height of the water surface. Raise it to flood the world.
        water_level: f32 = 5.0, [-5, 60], Live;
    }
    "Weather" {
        /// Let the sun rise and set. Off: always late morning.
        day_night: bool = true, [0, 1], Live;
        /// Simulated seconds from one sunrise to the next (nights take about a third of it).
        day_length: f32 = 900.0, [30, 36000], Live;
        /// Clouds, rain and storms come and go. Off: fair weather forever.
        weather: bool = true, [0, 1], Live;
        /// Typical simulated seconds between changes in the weather.
        weather_period: f32 = 600.0, [30, 36000], Live;
        /// How often it rains and storms: 0 mostly fair, 1 mostly wet.
        storminess: f32 = 0.3, [0, 1], Live;
        /// Extra plant growth while it rains: 0.5 means half again as fast in a downpour.
        rain_plant_growth: f32 = 0.5, [0, 5], Live;
        /// Strength of the wind pushing on creatures (1 is real air drag; storms blow at up to
        /// about 20 m/s).
        wind_force: f32 = 1.0, [0, 20], Live;
    }
    "Physics" {
        /// Gravitational acceleration, m/s^2.
        gravity: f32 = 9.81, [0, 30], Live;
        /// Constraint solver iterations per step. More is stiffer joints and slower physics.
        solver_iterations: u32 = 4, [1, 16], Live;
        /// Constraint passes within each solver iteration. One pass lets lively multi-jointed
        /// bodies (snakes thrashing their tails) bend joints a radian or more past their limits;
        /// two holds them, for about 12% more physics time.
        solver_passes: u32 = 2, [1, 8], Live;
        /// Friction of the ground.
        ground_friction: f32 = 1.0, [0, 3], Live;
        /// Friction of creature bodies.
        body_friction: f32 = 0.9, [0, 3], Live;
        /// Density of newly born creature bodies, kg/m^3. Below the water density they float.
        body_density: f32 = 850.0, [100, 3000], Live;
        /// Density of water, kg/m^3 (drives buoyancy).
        water_density: f32 = 1000.0, [0, 3000], Live;
        /// Linear (viscous) drag of water on moving body parts, N per m^2 per m/s.
        water_drag: f32 = 150.0, [0, 3000], Live;
        /// Pressure drag of water (the drag coefficient): grows with the square of speed, so a
        /// fast stroke pushes much harder than a slow recovery. This is what lets paddles, fins
        /// and wriggling bodies swim.
        swim_drag: f32 = 1.0, [0, 5], Live;
        /// Velocity damping in air.
        air_drag: f32 = 0.05, [0, 2], Live;
        /// Hard cap on body part speed, m/s, to keep the solver from exploding.
        max_body_speed: f32 = 30.0, [5, 200], Live;
    }
    "Motors" {
        /// Spring stiffness of the joint motors chasing the brain's target angles.
        motor_stiffness: f32 = 200.0, [0, 2000], Live;
        /// Damping of the joint motors.
        motor_damping: f32 = 20.0, [0, 500], Live;
        /// Maximum motor torque per kg of creature mass (scaled by each limb's strength gene).
        motor_torque_per_kg: f32 = 3.0, [0, 30], Live;
        /// Energy per unit of work for two-axis swivel joints (tentacles), relative to hinges.
        swivel_energy_cost: f32 = 2.0, [1, 10], Live;
    }
    "Population" {
        /// When fewer creatures than this are alive, new ones rain from the sky.
        min_population: u32 = 50, [0, 1000], Live;
        /// No births beyond this many creatures.
        max_population: u32 = 300, [1, 3000], Live;
        /// Seconds between raindrops while the population is under the minimum.
        rain_interval: f32 = 0.25, [0, 10], Live;
        /// Height above the ground at which raining creatures appear.
        rain_height: f32 = 10.0, [1, 60], Live;
        /// Fraction of raining creatures bred from the gene pool of past successes (the rest are random).
        rain_from_pool: f32 = 0.7, [0, 1], Live;
        /// Fraction of those pool-bred raindrops that land near a living relative they could breed
        /// with, instead of at a random spot. More births, but in tests walking failed to evolve
        /// in 2 of 3 worlds (breeding without moving gets rewarded), so it's off by default.
        rain_near_kin: f32 = 0.0, [0, 1], Live;
        /// Lifespan in seconds before dying of old age.
        lifespan: f32 = 500.0, [20, 10000], Live;
        /// Age in seconds before a creature can breed.
        maturity_age: f32 = 15.0, [0, 600], Live;
    }
    "Brain" {
        /// Hidden neurons in a brain made from scratch, and the fewest a brain can have.
        brain_min_neurons: u32 = 32, [1, 1024], Live;
        /// Most hidden neurons a brain can evolve. A 2048-neuron brain has 4.3 million connections
        /// (8.6 MB of GPU memory); thousands of creatures that size need a big GPU.
        brain_max_neurons: u32 = 2048, [1, 8192], Live;
        /// Energy per second per million connections. Connections grow with the square of the
        /// neuron count, so big brains get expensive.
        brain_energy_cost: f32 = 0.1, [0, 5], Live;
        /// Chance per birth that the brain grows or shrinks (by about 20%, at least one neuron).
        brain_resize_rate: f32 = 0.5, [0, 1], Live;
        /// Average size step of those changes, in log units: 0.1 means brains tend to grow ~10%
        /// per resize; negative values make them shrink. Energy costs push back.
        brain_growth: f32 = 0.15, [-0.5, 0.5], Live;
        /// Brain updates per second.
        brain_hz: f32 = 30.0, [1, 60], Live;
        /// Scales lifetime learning (creatures trying small variations on their movements and
        /// keeping what pays off); 0 turns it off and leaves only evolution.
        learning_rate: f32 = 1.0, [0, 10], Live;
        /// Scales every creature's exploration (small random variations on its movements, which
        /// learning needs); 0 makes them repeat themselves exactly.
        exploration: f32 = 1.0, [0, 4], Live;
        /// Seconds for what a creature learned to fade halfway back to its inherited wiring.
        learned_memory: f32 = 300.0, [5, 100000], Live;
        /// Fraction of what each parent learned that is written into its children's genes.
        /// 0 is purely Darwinian; 1 passes on everything.
        learned_inheritance: f32 = 0.5, [0, 1], Live;
        /// Distance at which food and other creatures can be sensed, m.
        sense_radius: f32 = 25.0, [1, 100], Live;
        /// Length of the three obstacle-sensing rays, m.
        ray_length: f32 = 8.0, [0.5, 40], Live;
    }
    "Energy" {
        /// Energy a creature can store per kg of body mass.
        energy_per_mass: f32 = 1.0, [0.1, 10], Live;
        /// Resting energy burn per second, times mass^0.75.
        basal_metabolism: f32 = 0.012, [0, 0.5], Live;
        /// Energy per joule of mechanical work done by joint motors.
        motor_energy_cost: f32 = 0.0005, [0, 0.05], Live;
        /// Energy that creatures dropped from the sky start with, capped at what their body can
        /// store. A fixed amount rather than a fraction, so a big body is no free fuel tank.
        start_energy: f32 = 80.0, [1, 2000], Live;
        /// Health regenerated per second (costs energy).
        health_regen: f32 = 1.0, [0, 20], Live;
    }
    "Plants" {
        /// Most plants the world can hold.
        plant_max_count: u32 = 1600, [0, 20000], Live;
        /// Energy a plant gains per second on fully fertile ground.
        plant_growth: f32 = 0.4, [0, 10], Live;
        /// Energy in a fully grown plant.
        plant_max_energy: f32 = 25.0, [1, 500], Live;
        /// Chance per second that a grown plant drops a seed nearby.
        plant_spread_rate: f32 = 0.02, [0, 1], Live;
        /// How far seeds land from their parent plant, m.
        plant_spread_radius: f32 = 6.0, [0.5, 50], Live;
        /// New plants sprouting at random places per second.
        plant_spawn_rate: f32 = 2.0, [0, 100], Live;
        /// Minimum spacing between plants, m.
        plant_min_spacing: f32 = 1.0, [0.1, 10], Live;
    }
    "Eating" {
        /// Energy eaten per second while biting (scaled by body size).
        bite_rate: f32 = 10.0, [0, 200], Live;
        /// How far beyond its mouth a creature can reach food, m.
        eat_reach: f32 = 0.6, [0, 5], Live;
        /// Meat energy left per kg of a dead creature.
        meat_per_mass: f32 = 0.8, [0, 10], Live;
        /// Fraction of meat energy lost per second to rot.
        meat_decay: f32 = 0.01, [0, 0.5], Live;
    }
    "Drinking" {
        /// Simulated seconds for a full creature to run dry without drinking (0: never thirsty).
        thirst_time: f32 = 360.0, [0, 5000], Live;
        /// Fraction of a full drink taken per second with the mouth in water.
        drink_rate: f32 = 0.5, [0, 10], Live;
        /// Health lost per second once dried out.
        dehydration_damage: f32 = 3.0, [0, 100], Live;
        /// Fraction of a full drink per second from catching rain, in a downpour.
        rain_drinking: f32 = 0.03, [0, 1], Live;
    }
    "Voices" {
        /// How far a full-throated call carries, m (quieter calls carry less).
        call_range: f32 = 40.0, [1, 300], Live;
        /// Energy per second of calling at full volume.
        call_cost: f32 = 0.08, [0, 5], Live;
    }
    "Breeding" {
        /// Fraction of full energy a creature needs before it can breed.
        mate_energy_fraction: f32 = 0.5, [0, 1], Live;
        /// Least energy a child can start with; parents who can't give that much can't breed.
        min_child_energy: f32 = 5.0, [0, 100], Live;
        /// Fraction of each parent's energy handed to the child.
        child_energy_share: f32 = 0.3, [0.01, 0.9], Live;
        /// How close two creatures must be to mate, m (between bodies).
        mate_range: f32 = 2.5, [0.1, 20], Live;
        /// Seconds between matings.
        mate_cooldown: f32 = 15.0, [0, 300], Live;
        /// Seconds of being ready to breed for the mating drive to build from nothing to full
        /// (half that with the highest libido gene, twice that with the lowest).
        mate_drive_time: f32 = 20.0, [0.5, 600], Live;
        /// Largest genetic distance at which two creatures can still interbreed (speciation).
        species_threshold: f32 = 0.35, [0, 2], Live;
        /// Let creatures breed alone (clone plus mutation) when they have plenty of energy.
        asexual: bool = false, [0, 1], Live;
        /// Fraction of full energy needed to breed alone.
        asexual_energy_fraction: f32 = 0.9, [0, 1], Live;
    }
    "Mutation" {
        /// Chance that each gene changes in a child.
        mutation_rate: f32 = 0.05, [0, 1], Live;
        /// Size of each change, as a fraction of the gene's range.
        mutation_strength: f32 = 0.15, [0, 2], Live;
        /// Chance of structural changes: adding or removing limbs and neurons.
        structural_mutation: f32 = 0.03, [0, 1], Live;
    }
    "Hardware" {
        /// Extra worlds evolving in the background on spare CPU cores (and the GPU). Their best
        /// genomes migrate into this world's gene pool, so evolution runs several times faster.
        parallel_worlds: u32 = default_background_worlds(), [0, 16], Live;
        /// Simulated seconds between migrations of the best genomes between worlds.
        migration_interval: f32 = 60.0, [5, 1200], Live;
        /// How busy the brains should keep the GPU at Max speed (0..1), by the GPU's own timer.
        /// System monitors read higher: 0.5 shows as roughly 70% in nvidia-smi. Below the target
        /// brains get cheaper and tend to grow; above it, dearer and tend to shrink. 0 turns the
        /// steering off.
        gpu_target: f32 = 0.5, [0, 1], Live;
        /// GPU memory for brains, shared by all worlds, in GB. Brains that don't fit are trimmed.
        gpu_memory_gb: f32 = 8.0, [0.5, 64], Live;
    }
    "Zoo" {
        /// Simulated seconds between each world putting its best creature in the zoo.
        zoo_interval: f32 = 300.0, [10, 7200], Live;
        /// Share of a new world's starting population that comes from the zoo (pairs of randomly
        /// chosen creatures, so they can breed); the rest are random.
        zoo_seed_fraction: f32 = 0.7, [0, 1], Live;
        /// Most creatures the zoo keeps (pinned ones don't count); the weakest go first.
        zoo_max_entries: u32 = 500, [10, 5000], Live;
    }
    "Combat" {
        /// Health removed by one bite of an attack.
        attack_damage: f32 = 15.0, [0, 200], Live;
        /// Energy an attack costs the attacker.
        attack_cost: f32 = 2.0, [0, 50], Live;
        /// Seconds between attacks.
        attack_cooldown: f32 = 0.6, [0.05, 10], Live;
        /// Seconds a creature can hold its breath (scaled by its breath gene).
        breath_time: f32 = 20.0, [1, 600], Live;
        /// Fraction of a full breath taken in per second while the head is out of the water
        /// (scaled by the lung gene). Hard work can use oxygen faster than this, and a creature
        /// short of breath is weak until it recovers.
        breath_rate: f32 = 0.25, [0.01, 5], Live;
        /// Oxygen used per second for each watt of muscle power per kg of body.
        exertion_oxygen: f32 = 0.08, [0, 2], Live;
        /// Health lost per second while out of breath underwater.
        drowning_damage: f32 = 5.0, [0, 100], Live;
    }
}

/// About one background world per 4 cores. Each world runs in its own thread pool (8 threads for
/// the world you watch, 4 for each background world); measured on 24 cores, 6 background worlds
/// gave 5.7x the single world's steps without slowing the watched world down.
fn default_background_worlds() -> u32 {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u32;
    (cores / 4).min(12)
}

impl SimParams {
    pub fn neuron_range(&self) -> (u32, u32) {
        let lo = self.brain_min_neurons.max(1);
        (lo, self.brain_max_neurons.max(lo))
    }

    /// True if anything that needs a world regeneration differs.
    pub fn world_shape_differs(&self, other: &SimParams) -> bool {
        self.seed != other.seed
            || self.world_size != other.world_size
            || self.terrain_resolution != other.terrain_resolution
            || self.terrain_height != other.terrain_height
            || self.terrain_feature_size != other.terrain_feature_size
            || self.terrain_roughness != other.terrain_roughness
            || self.island_size != other.island_size
            || self.coast_roughness != other.coast_roughness
    }
}
