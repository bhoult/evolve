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
        /// Loose stones lying about a new world in patches, for creatures to carry (and build with).
        stones: u32 = 150, [0, 2000], NewWorld;
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
        /// Solver substeps per physics step. A creature's joints are solved exactly (it's a
        /// multibody), so one is enough; more makes contacts a little firmer and physics slower.
        solver_iterations: u32 = 1, [1, 16], Live;
        /// Constraint passes within each substep.
        solver_passes: u32 = 8, [1, 16], Live;
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
        /// Hard cap on body part spin, rad/s.
        max_body_spin: f32 = 40.0, [5, 1000], Live;
        /// Physics worlds the world is split into and stepped in parallel. Creatures that might
        /// touch share one, and rapier solves each world on one thread, so more use more cores.
        physics_shards: u32 = 12, [1, 32], Live;
    }
    "Motors" {
        /// How far a joint can be pushed off the angle its brain asks for before its motor
        /// pushes back with full strength, radians. The motors are servos: they hold the angle
        /// against whatever load (a limb on the ground lifting the body) up to their strength.
        motor_reach: f32 = 0.2, [0.02, 2], Live;
        /// Damping of the joint motors, as a multiple of critical damping for the limb they swing
        /// (reckoned from the limb alone; chains of joints need more than 1 to stay steady: at 1
        /// some bodies with several swivels in a row shook themselves apart).
        motor_damping_ratio: f32 = 2.0, [0, 5], Live;
        /// Maximum motor torque per kg of creature mass (scaled by each limb's strength gene).
        motor_torque_per_kg: f32 = 6.0, [0, 30], Live;
        /// Fastest a joint can swing, rad/s, for a limb segment half a meter long; longer
        /// segments swing slower and shorter ones faster, like pendulums. Keeps limbs from
        /// flicking between extremes faster than a body could.
        max_joint_speed: f32 = 8.0, [0.5, 60], Live;
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
        /// Food and kills only count toward the life score (which picks the gene pool and the
        /// zoo) when eaten or made at least this many meters from where the creature fell or was
        /// born. Counting all food rewarded landing beside a plant: big bodies that never walked
        /// filled their tanks where they fell and won, so the pool bred for sitting still. 0
        /// counts everything.
        score_food_away: f32 = 4.0, [0, 50], Live;
        /// A kill's worth in the life score, in plants, when made at least `score_food_away` meters
        /// from where the killer fell or was born. Counting every kill rewarded heavy bodies that
        /// sat still biting whatever landed against them.
        score_kill: f32 = 3.0, [0, 50], Live;
        /// Meters closed on food (while it has room to eat) worth as much in the life score as
        /// eating a plant.
        score_approach: f32 = 10.0, [0.1, 1000], Live;
        /// Fraction of the creatures bred from the gene pool that are instead exact copies of a
        /// pool genome judged on fewer than 3 lives, whose scores are averaged into its own. One
        /// life mostly measures luck (where it fell); this keeps the pool from filling with the
        /// luckiest.
        pool_retest: f32 = 0.0, [0, 1], Live;
        /// Lifespan in seconds before dying of old age.
        lifespan: f32 = 500.0, [20, 10000], Live;
        /// Age in seconds before a creature can breed.
        maturity_age: f32 = 15.0, [0, 600], Live;
    }
    "Brain" {
        /// Hidden neurons in a brain made from scratch, and the fewest a brain can have.
        brain_min_neurons: u32 = 64, [1, 1024], Live;
        /// Most hidden neurons a brain can evolve. A 4096-neuron brain has 17 million connections
        /// (34 MB of GPU memory); thousands of creatures that size need a big GPU.
        brain_max_neurons: u32 = 4096, [1, 8192], Live;
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
        /// How strongly a brain made from scratch is wired to the senses that point at food, prey
        /// and mates, times the usual input strength. Random brains are dominated by their
        /// rhythm, so where food lies barely changes what they do, leaving no steering for
        /// evolution to pick from.
        random_sense_gain: f32 = 1.0, [0, 20], Live;
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
        motor_energy_cost: f32 = 0.000125, [0, 0.05], Live;
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
        /// Energy in each kg of body: what parents pay to grow a child's body, and what its
        /// carcass leaves as meat (plus half the energy it still held). Bodies that fell from
        /// the sky cost nothing, so they leave only the half. (At 0.8 a child cost its parents
        /// so much of a full-size body that what was left for it to live on was a crumb.)
        meat_per_mass: f32 = 0.5, [0, 10], Live;
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
    "Earthworks" {
        /// Earth a 30 kg creature moves per second digging or piling flat out, m³ (scaled by
        /// the square root of body mass).
        dig_rate: f32 = 0.05, [0, 2], Live;
        /// Earth a creature can carry per kg of body, m³.
        soil_per_kg: f32 = 0.01, [0, 0.2], Live;
        /// Energy per m³ of earth dug or piled.
        dig_cost: f32 = 10.0, [0, 500], Live;
        /// How deep below the original ground creatures can dig, and how high above it they can
        /// pile earth, m.
        dig_depth: f32 = 1.5, [0, 20], Live;
        pile_height: f32 = 1.5, [0, 20], Live;
        /// Steepest slope that dug, piled or water-cut ground holds, degrees: steeper, the earth
        /// slides down until it holds (90: walls stand forever).
        slump_angle: f32 = 35.0, [10, 90], Live;
        /// Minutes for dug and piled ground to weather halfway back to the land as it was made.
        earth_recovery: f32 = 120.0, [1, 100000], Live;
    }
    "Land and water" {
        /// Rain runs downhill into streams and collects in hollows, and running water slowly
        /// cuts the land into channels.
        hydrology: bool = true, [0, 1], Live;
        /// How much of the rain runs off downhill (bigger: bigger streams).
        runoff: f32 = 1.0, [0, 5], Live;
        /// Rain at full strength, mm per hour of water gathered into hollows from everything that
        /// drains into them (several times real rain, so lakes fill within a session).
        rain_rate: f32 = 40.0, [0, 1000], Live;
        /// Water lakes and ponds lose to evaporation and seepage, mm per hour.
        evaporation: f32 = 8.0, [0, 500], Live;
        /// How fast running water cuts its bed, m per hour for a stream draining 100 m² of hillside
        /// at a 1-in-10 slope in steady rain (a big valley's stream cuts several times faster).
        erosion_rate: f32 = 0.08, [0, 20], Live;
        /// The deepest water can wear the land below where the world made it, m.
        erosion_max_depth: f32 = 2.5, [0, 20], Live;
        /// How quickly feet wear the ground bare (1: a few dozen passes of a 50 kg creature).
        trampling: f32 = 1.0, [0, 20], Live;
        /// Minutes for trampled ground to grow halfway back.
        wear_recovery: f32 = 60.0, [1, 2000], Live;
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
        /// Least share of its own energy store a child can start with (whichever is more, this or
        /// min_child_energy). Without it most children were born with a few seconds' worth and
        /// starved at once.
        child_start_fraction: f32 = 0.25, [0, 1], Live;
        /// Food a creature must have eaten in its life before it can breed, so births come from
        /// foraging rather than from the energy it fell from the sky with.
        breeding_food: f32 = 50.0, [0, 1000], Live;
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
        /// Background worlds' population, as a share of this world's minimum and maximum: big
        /// background worlds take the cores this world's physics needs.
        background_population: f32 = 0.25, [0.05, 1], Live;
        /// Simulated seconds between migrations of the best genomes between worlds.
        migration_interval: f32 = 60.0, [5, 1200], Live;
        /// How busy the brains should keep the GPU at Max speed (0..1), by the GPU's own timer.
        /// System monitors read higher: 0.5 showed as roughly 70% in nvidia-smi. Below the target
        /// brains get cheaper and tend to grow; above it, dearer and tend to shrink. 0 turns the
        /// steering off.
        gpu_target: f32 = 0.65, [0, 1], Live;
        /// GPU memory for brains, shared by all worlds in proportion to their populations, in GB.
        /// Brains that don't fit are trimmed.
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
        exertion_oxygen: f32 = 0.004, [0, 2], Live;
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
    /// Sets a parameter by name from text, e.g. ("min_population", "200").
    pub fn set_by_name(&mut self, name: &str, value: &str) -> Result<(), String> {
        let mut result = Err(format!("no parameter named {name}"));
        self.visit(&mut |info, r| {
            if info.name != name {
                return;
            }
            let bad = |_| format!("{name} can't be {value}");
            result = match r {
                ParamRef::F32(x) => value.parse().map(|v| *x = v).map_err(|e: std::num::ParseFloatError| bad(e.to_string())),
                ParamRef::U32(x) => value.parse().map(|v| *x = v).map_err(|e: std::num::ParseIntError| bad(e.to_string())),
                ParamRef::U64(x) => value.parse().map(|v| *x = v).map_err(|e: std::num::ParseIntError| bad(e.to_string())),
                ParamRef::Bool(x) => value.parse().map(|v| *x = v).map_err(|e: std::str::ParseBoolError| bad(e.to_string())),
            };
        });
        result
    }

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
