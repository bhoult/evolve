//! What creatures do: sense, think, move, eat, fight and breed. Plus the plants they live on.

use std::collections::HashMap;

use rapier3d::prelude::*;
use rayon::prelude::*;

use crate::body::{self, Tag};
use std::time::Instant;

use crate::brain::{self, Learning, N_IN, N_MEMORY, input, output};
use crate::gpu::BatchItem;
use crate::genome::Genome;
use crate::grid::Grid;
use crate::params::SimParams;
use crate::math::{Quat, Vec3, heading_frame};
use crate::sim::{Creature, Food, Sim, TICKS_PER_SECOND};

pub(crate) const GRID_CELL: f32 = 5.0;
/// Seconds between position samples for the distance-travelled tally.
const DISTANCE_WINDOW: u64 = 5;

struct SenseCtx<'a> {
    sim: &'a Sim,
    torso: Vec<Vec3>,
    plants: Grid,
    meat: Grid,
    /// Loose stones and boulders (and their positions and masses), for the stone sense.
    stones: Grid,
    stone_at: Vec<(Vec3, f32)>,
    creatures: Grid,
    /// How far any creature's body reaches from its torso, m.
    creature_reach: f32,
    /// Every obstacle, and how far the biggest reaches from its center, m.
    obstacles: Grid,
    obstacle_reach: f32,
    calls: Vec<Call>,
}

/// A creature calling out this brain tick.
struct Call {
    index: usize,
    pos: Vec3,
    loud: f32,
    pitch: f32,
    voice: [f32; 4],
}

/// How loudly a creature is calling, 0..1 (silent while its call output is below 0.2).
pub(crate) fn call_loudness(c: &Creature) -> f32 {
    ((c.output(output::CALL) - 0.2) / 0.8).max(0.0)
}

pub(crate) fn food_grid(size: f32, food: &[Food]) -> Grid {
    let mut g = Grid::new(size, GRID_CELL);
    g.rebuild(food.iter().map(|f| f.pos));
    g
}

impl Sim {
    pub(crate) fn torso_positions(&self) -> Vec<Vec3> {
        self.creatures
            .iter()
            .map(|c| self.torso_of(c).map(|b| b.translation()).unwrap_or(c.last_pos))
            .collect()
    }

    /// Sense, run every brain, and drive the joint motors. With a GPU, this tick's brains are
    /// submitted now and their outputs are applied at the next tick (a one-tick reaction delay),
    /// so the GPU thinks while the CPU runs the physics steps in between.
    pub(crate) fn think(&mut self, bdt: f32) {
        if let Some(gpu) = self.brains.as_mut() {
            let wait = Instant::now();
            let collected = gpu.collect();
            let waited = wait.elapsed().as_secs_f32() * 1000.0;
            if let Some(col) = collected {
                let now = Instant::now();
                let perf = &mut self.perf;
                perf.gpu_wait_ms += (waited - perf.gpu_wait_ms) * 0.05;
                if let Some(ms) = col.gpu_ms {
                    perf.gpu_ms += (ms - perf.gpu_ms) * 0.05;
                    if let Some(prev) = self.last_collect {
                        let wall = (now - prev).as_secs_f32() * 1000.0;
                        let busy = (ms / wall.max(1e-3)).min(1.0);
                        perf.gpu_busy += (busy - perf.gpu_busy) * 0.05;
                    }
                }
                self.last_collect = Some(now);
                for (id, out) in col.ids.iter().zip(&col.outputs) {
                    if let Some(&i) = self.index.get(id) {
                        let c = &mut self.creatures[i];
                        c.brain.outputs.copy_from_slice(out);
                        c.brain
                            .memory
                            .copy_from_slice(&out[output::MEMORY..output::MEMORY + N_MEMORY]);
                    }
                }
                if let Some(sel) = col.selected
                    && let Some(&i) = self.index.get(&sel.id)
                {
                    let b = &mut self.creatures[i].brain;
                    b.hidden = sel.hidden;
                    b.plastic = sel.plastic;
                }
            }
            // New exploration on the outputs just arrived; it rides along with this tick's batch
            // so the GPU pairs it with the hidden activity that produced them.
            let (tick, scale) = (self.tick, self.params.exploration);
            self.creatures.par_iter_mut().for_each(|c| explore(c, tick, bdt, scale));
        }

        let torso = self.torso_positions();
        // Who is calling, from what they decided last tick. A call is announced (for the sound
        // and the camera) when it starts and twice a second while it lasts.
        let mut calls = Vec::new();
        let mut announce = Vec::new();
        for (i, c) in self.creatures.iter_mut().enumerate() {
            let loud = call_loudness(c);
            if loud <= 0.0 {
                c.call_timer = 0.0;
                continue;
            }
            calls.push(Call {
                index: i,
                pos: torso[i],
                loud,
                pitch: c.output(output::CALL_PITCH),
                voice: c.genome.traits.voice,
            });
            c.call_timer -= bdt;
            if c.call_timer <= 0.0 {
                c.call_timer = 0.5;
                announce.push((c.id, torso[i], loud));
            }
        }
        for (id, at, loud) in announce {
            self.highlight_with(crate::snapshot::HighlightKind::Call, id, None, at, loud);
        }
        let inputs: Vec<[f32; N_IN]> = {
            let size = self.terrain.size;
            let mut creatures = Grid::new(size, GRID_CELL);
            creatures.rebuild(torso.iter().copied());
            let creature_reach = self
                .creatures
                .iter()
                .zip(&torso)
                .map(|(c, &at)| {
                    let world = &self.shards[c.shard].world;
                    c.parts
                        .iter()
                        .zip(&c.plan)
                        .filter_map(|(q, spec)| Some(world.bodies.get(q.body)?.translation().distance(at) + spec.shape.bounding_radius()))
                        .fold(0.0f32, f32::max)
                })
                .fold(0.0f32, f32::max);
            let taken: std::collections::HashSet<u64> = self
                .creatures
                .iter()
                .filter_map(|c| match c.held {
                    Some(crate::carry::Held::Obstacle { id, .. }) => Some(id),
                    _ => None,
                })
                .collect();
            let obstacle_at: Vec<Vec3> = self
                .obstacles
                .iter()
                .map(|o| self.shards[o.shard].world.bodies.get(o.body).map_or(Vec3::ZERO, |b| b.translation()))
                .collect();
            let mut obstacles = Grid::new(size, GRID_CELL);
            obstacles.rebuild(obstacle_at.iter().copied());
            let obstacle_reach = self.obstacles.iter().map(|o| crate::sim::obstacle_radius(o.shape)).fold(0.0f32, f32::max);
            let stone_at: Vec<(Vec3, f32)> = self
                .obstacles
                .iter()
                .filter(|o| o.dynamic && !taken.contains(&o.id))
                .filter_map(|o| self.shards[o.shard].world.bodies.get(o.body).map(|b| (b.translation(), b.mass())))
                .collect();
            let mut stones = Grid::new(size, GRID_CELL);
            stones.rebuild(stone_at.iter().map(|s| s.0));
            let cx = SenseCtx {
                sim: self,
                stones,
                stone_at,
                plants: food_grid(size, &self.plants),
                meat: food_grid(size, &self.meat),
                creatures,
                creature_reach,
                obstacles,
                obstacle_reach,
                torso,
                calls,
            };
            self.creatures
                .par_iter()
                .enumerate()
                .map(|(i, c)| sense(&cx, i, c))
                .collect()
        };

        let rate = self.params.learning_rate * brain::LEARNING_RATE;
        let plastic_decay = brain::per_tick_decay(self.params.learned_memory, bdt);
        let exploration = self.params.exploration;
        let pipelined = self.brains.is_some();
        let learning = |c: &mut Creature, x: &[f32; N_IN]| {
            approach_reward(c, x);
            // Variations are measured against the creature's own exploration, so a creature
            // that explores gently learns as readily as a bold one.
            let sigma = exploration_sigma(c, exploration).max(0.03);
            Learning {
                lr: brain::learning_signal(&mut c.brain, rate * c.genome.traits.plasticity / sigma, bdt),
                trace_decay: brain::per_tick_decay(c.genome.traits.trace_time, bdt),
                plastic_decay,
                pipelined,
            }
        };
        match self.brains.as_mut() {
            Some(gpu) => {
                let batch: Vec<BatchItem> = self
                    .creatures
                    .iter_mut()
                    .zip(&inputs)
                    .filter(|(c, _)| gpu.contains(c.id))
                    .map(|(c, x)| {
                        let learn = learning(c, x);
                        c.brain.reward = 0.0;
                        c.brain.inputs.copy_from_slice(x);
                        BatchItem {
                            id: c.id,
                            inputs: *x,
                            noise: std::array::from_fn(|k| c.brain.noise.get(k).copied().unwrap_or(0.0)),
                            learn,
                            bias: c.output_bias(),
                        }
                    })
                    .collect();
                gpu.submit(&batch, self.selected);
            }
            None => {
                let (tick, scale) = (self.tick, self.params.exploration);
                self.creatures
                    .par_iter_mut()
                    .zip(inputs.par_iter())
                    .for_each(|(c, x)| {
                        let learn = learning(c, x);
                        let bias = c.output_bias();
                        brain::step_cpu(&c.genome.brain, bias, &mut c.brain, x, learn);
                        let out = &c.brain.outputs;
                        let mem: [f32; N_MEMORY] = std::array::from_fn(|k| out[output::MEMORY + k]);
                        c.brain.memory = mem;
                        explore(c, tick, bdt, scale);
                    });
            }
        }

        let p = &self.params;
        for c in &self.creatures {
            for j in &c.joints {
                let (mid, half) = j.span();
                // Out of breath, muscles weaken (to a quarter at worst).
                let breath = (c.oxygen / 0.3).clamp(0.25, 1.0);
                let force = p.motor_torque_per_kg * j.strength * c.mass * breath;
                let (stiffness, damping) = body::servo(p, force, j.inertia);
                // The angle the joint aims for moves at most this far per brain tick, so a limb
                // swings rather than snapping between whatever the brain asks from tick to tick.
                let len = c.plan.get(j.child).map_or(0.5, |s| s.shape.length()).max(0.05);
                let step = p.max_joint_speed * (0.5 / len).sqrt().clamp(0.5, 1.5) * bdt;
                for &(axis, d) in j.axes() {
                    if let Some(data) = body::axis_joint_mut(&mut self.shards[c.shard].world, j, d) {
                        let want = mid + c.output(output::JOINTS + j.slot + d) * half;
                        let was = data.motors[axis as usize].target_pos;
                        let target = was + (want - was).clamp(-step, step);
                        data.set_motor_model(axis, MotorModel::ForceBased);
                        data.set_motor_position(axis, target, stiffness, damping);
                        data.set_motor_max_force(axis, force);
                    }
                }
            }
        }
    }

    pub(crate) fn interact(&mut self, bdt: f32) {
        self.grip();
        self.trample(bdt);
        self.eat(bdt);
        self.fight();
        self.breed();
        self.earthworks(bdt);
    }

    /// Creatures digging earth out at their mouth (carrying it, up to a load set by their size)
    /// or piling what they carry. Moving earth costs energy. The ground can't be dug or piled
    /// more than `dig_depth` / `pile_height` from where the world made it.
    fn earthworks(&mut self, bdt: f32) {
        let p = &self.params;
        let (dig_rate, soil_per_kg, dig_cost, dig_depth, pile_height) =
            (p.dig_rate, p.soil_per_kg, p.dig_cost, p.dig_depth, p.pile_height);
        if dig_rate <= 0.0 {
            return;
        }
        for ci in 0..self.creatures.len() {
            let c = &self.creatures[ci];
            let (dig, pile) = (c.output(output::DIG), c.output(output::PILE));
            let want = ((dig - 0.2) / 0.8).max(0.0) - ((pile - 0.2) / 0.8).max(0.0);
            if want == 0.0 || c.dead.is_some() {
                continue;
            }
            let Some(torso) = self.shards[c.shard].world.bodies.get(c.parts[0].body) else {
                continue;
            };
            let half = c.torso_half();
            let mouth = torso.position().transform_point(Vec3::new(half.x + 0.1, 0.0, 0.0));
            // It has to reach the ground.
            if mouth.y - self.terrain.height_at(mouth.x, mouth.z) > 0.4 + half.y {
                continue;
            }
            let capacity = soil_per_kg * c.mass;
            let rate = dig_rate * (c.mass / 30.0).sqrt() * bdt;
            let volume = if want > 0.0 {
                -(rate * want).min(capacity - c.soil).max(0.0)
            } else {
                (rate * -want).min(c.soil)
            };
            if volume == 0.0 {
                continue;
            }
            let moved = self.terrain.reshape(mouth.x, mouth.z, volume, dig_depth, pile_height);
            let c = &mut self.creatures[ci];
            c.soil = (c.soil - moved).clamp(0.0, capacity);
            c.energy -= dig_cost * moved.abs();
        }
        self.settle_terrain();
    }

    /// After earth moved: the physics ground takes the new shape, plants and meat on changed
    /// ground settle onto it, and loose obstacles resting there wake up to fall or roll.
    pub(crate) fn settle_terrain(&mut self) {
        // Swapping in a new ground shape costs a full heightfield rebuild and drops the cached
        // contacts of everything touching the ground, so it's done twice a simulated second at
        // most; earth moves only centimeters in that time.
        if self.terrain.dirty.is_none() || self.time - self.settled_at < 0.5 {
            return;
        }
        self.settled_at = self.time;
        let Some((i0, i1, j0, j1)) = self.terrain.take_dirty() else {
            return;
        };
        // The same ground in every physics world.
        let shape = self.terrain.shape();
        for shard in &mut self.shards {
            if let Some(col) = shard.world.colliders.get_mut(shard.terrain_collider) {
                col.set_shape(shape.clone());
            }
        }
        let (n, size) = ((self.terrain.res - 1) as f32, self.terrain.size);
        let at = |k: usize| (k as f32 / n - 0.5) * size;
        let (x0, x1, z0, z1) = (at(j0) - 1.0, at(j1) + 1.0, at(i0) - 1.0, at(i1) + 1.0);
        let inside = |pos: Vec3| pos.x >= x0 && pos.x <= x1 && pos.z >= z0 && pos.z <= z1;
        for k in 0..self.plants.len() {
            let pos = self.plants[k].pos;
            if inside(pos) {
                self.plants[k].pos.y = self.ground_under(pos.x, pos.z, crate::sim::PLANT_FOOTPRINT);
            }
        }
        for f in &mut self.meat {
            if inside(f.pos) {
                f.pos.y = self.terrain.height_at(f.pos.x, f.pos.z);
            }
        }
        // Ground can rise by meters at once (land raised, or thrown up by a quake or a meteor):
        // a stone it buried is set back on top, and so is a creature, by as much as its most
        // buried part needs.
        let terrain = &self.terrain;
        for o in self.obstacles.iter().filter(|o| o.dynamic) {
            if let Some(b) = self.shards[o.shard].world.bodies.get_mut(o.body)
                && inside(b.translation())
            {
                let t = b.translation();
                let ground = terrain.height_at(t.x, t.z);
                if t.y < ground {
                    b.set_translation(Vec3::new(t.x, ground + crate::sim::obstacle_lift(o.shape), t.z), true);
                    b.set_linvel(Vec3::ZERO, true);
                }
                b.wake_up(true);
            }
        }
        for ci in 0..self.creatures.len() {
            let c = &self.creatures[ci];
            let world = &self.shards[c.shard].world;
            let buried = c
                .parts
                .iter()
                .zip(&c.plan)
                .filter_map(|(q, spec)| {
                    let t = world.bodies.get(q.body)?.translation();
                    let ground = self.terrain.height_at(t.x, t.z);
                    (inside(t) && t.y < ground).then(|| ground + spec.shape.bounding_radius() - t.y)
                })
                .fold(0.0f32, f32::max);
            if buried > 0.0 {
                self.translate_creature(ci, Vec3::Y * buried);
            }
        }
    }

    fn eat(&mut self, bdt: f32) {
        let size = self.terrain.size;
        let pg = food_grid(size, &self.plants);
        let mg = food_grid(size, &self.meat);
        let reach_extra = self.params.eat_reach;
        let bite_rate = self.params.bite_rate;
        for c in &mut self.creatures {
            if c.dead.is_some() || c.output(output::EAT) <= 0.0 || c.energy >= c.max_energy {
                continue;
            }
            let Some(torso) = self.shards[c.shard].world.bodies.get(c.parts[0].body) else {
                continue;
            };
            let half = c.torso_half();
            let mouth = torso.position().transform_point(Vec3::new(half.x, 0.0, 0.0));
            let reach = reach_extra + 0.35 + half.y;
            let near = |grid: &Grid, food: &[Food]| {
                grid.nearest(
                    mouth,
                    reach,
                    |k| food[k as usize].pos,
                    |k| {
                        let f = &food[k as usize];
                        f.energy > 0.0 && (f.pos.y - mouth.y).abs() < reach + 0.6
                    },
                )
            };
            let plant = near(&pg, &self.plants);
            let meat = near(&mg, &self.meat);
            let (pe, me) = (c.plant_efficiency(), c.meat_efficiency());
            let (food, eff, is_meat) = match (plant, meat) {
                (Some((k, _)), Some((m, _))) => {
                    if me >= pe {
                        (&mut self.meat[m as usize], me, true)
                    } else {
                        (&mut self.plants[k as usize], pe, false)
                    }
                }
                (Some((k, _)), None) => (&mut self.plants[k as usize], pe, false),
                (None, Some((m, _))) => (&mut self.meat[m as usize], me, true),
                (None, None) => continue,
            };
            let bite = bite_rate * bdt * c.genome.traits.size;
            let room = c.max_energy - c.energy;
            let amount = bite.min(food.energy).min(room / eff);
            food.energy -= amount;
            let gain = amount * eff;
            c.energy += gain;
            if is_meat {
                c.life.meat_eaten += gain;
            } else {
                c.life.plant_eaten += gain;
            }

        }
        self.plants.retain(|f| f.energy > 0.05);
        self.meat.retain(|f| f.energy > 0.05);
    }

    /// A creature whose torso (its head) touches another and wants to attack, bites it.
    fn fight(&mut self) {
        let mut contacts = Vec::new();
        // Creatures that touch share a physics world (see `regroup`), so each world's own
        // contacts are all of them.
        for shard in &self.shards {
            let world = &shard.world;
            for pair in world.narrow_phase.contact_pairs() {
                if !pair.has_any_active_contact() {
                    continue;
                }
                let (Some(a), Some(b)) = (world.colliders.get(pair.collider1), world.colliders.get(pair.collider2)) else {
                    continue;
                };
                if let (Tag::Creature { id: ia, part: pa }, Tag::Creature { id: ib, part: pb }) =
                    (Tag::decode(a.user_data), Tag::decode(b.user_data))
                    && ia != ib
                {
                    contacts.push((ia, pa, ib, pb));
                }
            }
        }
        for (ia, pa, ib, pb) in contacts {
            if pa == 0 {
                self.try_attack(ia, ib);
            }
            if pb == 0 {
                self.try_attack(ib, ia);
            }
        }
    }

    fn try_attack(&mut self, attacker: u64, victim: u64) {
        let (Some(&ai), Some(&vi)) = (self.index.get(&attacker), self.index.get(&victim)) else {
            return;
        };
        let p = &self.params;
        let a = &self.creatures[ai];
        let v = &self.creatures[vi];
        if a.dead.is_some()
            || v.dead.is_some()
            || a.output(output::ATTACK) <= 0.0
            || a.attack_cooldown > 0.0
        {
            return;
        }
        let ratio = (a.mass / v.mass.max(1e-3)).sqrt().min(2.0);
        let damage = p.attack_damage * (0.5 + 0.5 * ratio) * (1.0 - 0.6 * v.genome.traits.armor);
        let (cost, cooldown) = (p.attack_cost, p.attack_cooldown);
        let v = &mut self.creatures[vi];
        v.health -= damage;
        v.pain = 1.0;
        v.last_attacker = Some(attacker);
        v.attacked_ago = 0.0;
        v.brain.reward -= damage / v.max_health * 2.0;
        let a = &mut self.creatures[ai];
        a.energy -= cost;
        a.attack_cooldown = cooldown;
        a.attack_flash = 0.3;
        a.life.attacks += 1;
        let at = self.torso_of(&self.creatures[vi]).map(|b| b.translation()).unwrap_or(Vec3::ZERO);
        self.highlight(crate::snapshot::HighlightKind::Attack, attacker, Some(victim), at);
    }

    fn breed(&mut self) {
        let n = self.creatures.len();
        if n == 0 || n as u32 >= self.params.max_population {
            return;
        }
        let p = self.params.clone();
        let torso = self.torso_positions();
        let mut grid = Grid::new(self.terrain.size, GRID_CELL);
        grid.rebuild(torso.iter().copied());
        let mut busy = vec![false; n];
        let mut pairs = Vec::new();
        for i in 0..n {
            let ci = &self.creatures[i];
            if busy[i] || !ci.ready_to_mate(&p, p.mate_energy_fraction.min(p.asexual_energy_fraction)) {
                continue;
            }
            let sexual_ready = ci.ready_to_mate(&p, p.mate_energy_fraction);
            let ri = ci.radius();
            let mut found = None;
            if sexual_ready {
                grid.for_each_near(torso[i], p.mate_range + ri + 1.0, |k| {
                    let k = k as usize;
                    if found.is_some() || k == i || busy[k] {
                        return;
                    }
                    let o = &self.creatures[k];
                    let gap = (torso[k] - torso[i]).length() - ri - o.radius();
                    // A child's body comes out like its parents', give or take a mutation.
                    if gap <= p.mate_range
                        && o.willing_partner(&p)
                        && ci.genome.distance(&o.genome) < p.species_threshold
                        && birth_cost(&p, &[ci.energy, o.energy], 0.5 * (ci.mass + o.mass)).is_some()
                    {
                        found = Some(k);
                    }
                });
            }
            if let Some(k) = found {
                busy[i] = true;
                busy[k] = true;
                pairs.push((i, k));
            } else if p.asexual
                && ci.ready_to_mate(&p, p.asexual_energy_fraction)
                && birth_cost(&p, &[ci.energy], ci.mass).is_some()
            {
                busy[i] = true;
                pairs.push((i, i));
            }
        }
        if pairs.is_empty() {
            return;
        }
        // What the parents learned, all fetched at once.
        let learned = if p.learned_inheritance > 0.0 {
            let parents: Vec<usize> = pairs.iter().flat_map(|&(i, k)| [i, k]).collect();
            self.learned(&parents)
        } else {
            HashMap::new()
        };
        for (i, k) in pairs {
            if self.creatures.len() as u32 >= p.max_population {
                break;
            }
            self.reproduce(i, k, torso[i], &learned);
        }
    }

    /// What these creatures have learned so far (output-weight offsets, N_OUT x h each), read
    /// back from the GPU in one round trip when brains live there.
    pub(crate) fn learned(&mut self, idx: &[usize]) -> HashMap<u64, Vec<f32>> {
        let ids: Vec<u64> = idx.iter().map(|&i| self.creatures[i].id).collect();
        match self.brains.as_mut() {
            Some(g) => g.read_plastic(&ids),
            None => idx
                .iter()
                .map(|&i| (self.creatures[i].id, self.creatures[i].brain.plastic.clone()))
                .collect(),
        }
    }

    /// A parent's genome with part of what it learned folded in.
    fn heritage(&self, i: usize, learned: &HashMap<u64, Vec<f32>>) -> Genome {
        let c = &self.creatures[i];
        let mut g = c.genome.clone();
        if let Some(p) = learned.get(&c.id) {
            g.brain.absorb(p, self.params.learned_inheritance);
        }
        g
    }

    /// Breed parents `i` and `k` (the same index for asexual reproduction).
    fn reproduce(&mut self, i: usize, k: usize, near: Vec3, learned: &HashMap<u64, Vec<f32>>) {
        let p = self.breeding_params();
        let mut genome = if i == k {
            self.heritage(i, learned)
        } else {
            let (a, b) = (self.heritage(i, learned), self.heritage(k, learned));
            Genome::crossover(&a, &b, &mut self.rng)
        };
        genome.mutate(&mut self.rng, &p);
        let parents: &[usize] = if i == k { &[i] } else { &[i, k] };
        // Priced by the body it actually got (a mutation can make it a little bigger).
        let mass = body::plan_mass(&body::body_plan(&genome), p.body_density);
        let held: Vec<f32> = parents.iter().map(|&j| self.creatures[j].energy).collect();
        let Some((body, gifts)) = birth_cost(&p, &held, mass) else {
            // The child came out too big to pay for: they wait a moment before trying again
            // (each try costs a crossover and a read of what they learned).
            for &j in parents {
                let c = &mut self.creatures[j];
                c.mate_cooldown = c.mate_cooldown.max(1.0);
            }
            return;
        };
        let mut energy = 0.0;
        for (&j, gift) in parents.iter().zip(gifts) {
            let c = &mut self.creatures[j];
            c.energy -= body + gift;
            energy += gift;
            c.life.invested += gift;
            c.mate_cooldown = p.mate_cooldown;
            c.mate_drive = 0.0;
            c.mate_flash = 0.6;
            c.life.children += 1;
        }
        let generation = self.creatures[i].generation.max(self.creatures[k].generation) + 1;
        let ids = Some((self.creatures[i].id, self.creatures[k].id));
        let a = self.rng.range(0.0, std::f32::consts::TAU);
        let offset = Vec3::new(a.cos(), 0.0, a.sin()) * (self.creatures[i].radius() + 0.8);
        let child = self.spawn_creature(genome, near + offset, 0.3, energy, generation, ids);
        let parent = self.creatures[i].id;
        self.highlight(crate::snapshot::HighlightKind::Birth, child, Some(parent), near + offset);
        self.stats.births += 1;
    }

    // ── Plants and meat ────────────────────────────────────────────────

    pub(crate) fn plant_grid(&self) -> Grid {
        let mut g = Grid::new(self.terrain.size, GRID_CELL.max(self.params.plant_min_spacing));
        g.rebuild(self.plants.iter().map(|f| f.pos));
        g
    }

    /// Plants a seed if the spot is fertile, in bounds and not crowded.
    pub(crate) fn try_plant_at(&mut self, pos: Vec3, energy: f32, grid: &mut Grid) -> bool {
        if self.plants.len() as u32 >= self.params.plant_max_count
            || !self.terrain.contains(pos.x, pos.z, 1.0)
        {
            return false;
        }
        let s = self.terrain.plant_suitability(
            pos.x,
            pos.z,
            self.params.water_level,
            self.params.terrain_height,
        );
        if s <= 0.0 || !self.rng.chance(s.sqrt()) {
            return false;
        }
        let sp = self.params.plant_min_spacing;
        let mut crowded = false;
        grid.for_each_near(pos, sp, |k| {
            let q = self.plants[k as usize].pos;
            if (q.x - pos.x).powi(2) + (q.z - pos.z).powi(2) < sp * sp {
                crowded = true;
            }
        });
        if crowded {
            return false;
        }
        if self.add_plant(pos, energy).is_some() {
            grid.insert(pos, (self.plants.len() - 1) as u32);
            true
        } else {
            false
        }
    }

    pub(crate) fn try_spawn_random_plant(&mut self, energy: f32, grid: &mut Grid) -> bool {
        let half = self.terrain.half() - 1.0;
        // Most of the world is sea, so look for land a few times before giving up.
        for _ in 0..6 {
            let x = self.rng.range(-half, half);
            let z = self.rng.range(-half, half);
            if self.terrain.height_at(x, z) > self.params.water_level {
                return self.try_plant_at(Vec3::new(x, 0.0, z), energy, grid);
            }
        }
        false
    }

    /// Once per simulated second: plants grow and seed, meat rots, distances are tallied.
    /// Rain runs off, collects and wears the land (see `hydro`).
    pub(crate) fn hydrology(&mut self, dt: f32) {
        // Running water is worn into the land in batches, so the physics ground (a full
        // heightfield rebuild) changes every half minute rather than every update.
        const EROSION_EVERY: f32 = 30.0;
        if !self.params.hydrology {
            if self.terrain.water.iter().chain(&self.terrain.pooled).any(|&w| w > 0.0) {
                self.terrain.water.fill(0.0);
                self.terrain.pooled.fill(0.0);
                self.terrain.version += 1;
            }
            return;
        }
        let rain = self.weather.rain;
        self.soak += (rain - self.soak) * (1.0 - (-dt / 1800.0).exp());
        self.erosion_due += dt;
        let erode = (self.erosion_due >= EROSION_EVERY).then(|| std::mem::take(&mut self.erosion_due));
        crate::hydro::update(&mut self.terrain, &self.params, rain, self.soak, dt, erode);
    }

    /// Feet wear the ground they move over: heavier and faster, more.
    fn trample(&mut self, bdt: f32) {
        let scale = self.params.trampling;
        if scale <= 0.0 {
            return;
        }
        self.terrain.ensure_layers();
        let wl = self.params.water_level;
        let area = self.terrain.cell() * self.terrain.cell();
        for c in &self.creatures {
            for (k, part) in c.parts.iter().enumerate() {
                let Some(b) = self.shards[c.shard].world.bodies.get(part.body) else {
                    continue;
                };
                let pos = b.translation();
                let ground = self.terrain.height_at(pos.x, pos.z);
                let reach = c.plan[k].shape.vertical_radius() + 0.15;
                if ground <= wl || pos.y - ground > reach {
                    continue;
                }
                let v = b.linvel();
                let speed = (v.x * v.x + v.z * v.z).sqrt().min(5.0);
                let at = self.terrain.nearest_point(pos.x, pos.z);
                self.terrain.wear[at] += b.mass() * speed * bdt / area * scale;
            }
        }
    }

    pub(crate) fn ecology(&mut self, dt: f32) {
        let p = self.params.clone();
        let rain = 1.0 + p.rain_plant_growth * self.weather.rain;
        for f in &mut self.plants {
            let s = self
                .terrain
                .plant_suitability(f.pos.x, f.pos.z, p.water_level, p.terrain_height);
            if s <= 0.0 {
                // Flooded or on bare rock: wither.
                f.energy -= p.plant_max_energy * 0.1 * dt;
            } else {
                f.energy = (f.energy + p.plant_growth * rain * (0.2 + 0.8 * s) * dt).min(p.plant_max_energy);
            }
        }
        self.plants.retain(|f| f.energy > 0.0);

        let mut grid = self.plant_grid();
        let mut seeds = Vec::new();
        for f in &self.plants {
            if f.energy > 0.5 * p.plant_max_energy && self.rng.chance(p.plant_spread_rate * dt) {
                let a = self.rng.range(0.0, std::f32::consts::TAU);
                let d = self
                    .rng
                    .range(p.plant_min_spacing, p.plant_spread_radius.max(p.plant_min_spacing));
                seeds.push(f.pos + Vec3::new(a.cos() * d, 0.0, a.sin() * d));
            }
        }
        for s in seeds {
            self.try_plant_at(s, 1.0, &mut grid);
        }
        self.plant_spawn_acc += p.plant_spawn_rate * dt;
        while self.plant_spawn_acc >= 1.0 {
            self.plant_spawn_acc -= 1.0;
            self.try_spawn_random_plant(1.0, &mut grid);
        }

        for m in &mut self.meat {
            m.energy -= m.energy * p.meat_decay * dt + 0.02 * dt;
        }
        self.meat.retain(|m| m.energy > 0.2);

        // Distance counts net progress over 5 s windows, so flailing in place doesn't score.
        if self.tick % (TICKS_PER_SECOND * DISTANCE_WINDOW) == 0 {
            let torso = self.torso_positions();
            for (c, pos) in self.creatures.iter_mut().zip(torso) {
                let d = Vec3::new(pos.x - c.last_pos.x, 0.0, pos.z - c.last_pos.z).length();
                if d < 20.0 * DISTANCE_WINDOW as f32 {
                    c.life.distance += d;
                }
                c.last_pos = pos;
            }
        }
        self.pool.age(0.998);
    }
}

/// What a child costs its parents (one, or two who split it): first its body, at the meat it
/// will be worth when it dies, so no round of breeding and dying makes energy from nothing; then
/// a share of what each has left, handed over as the child's energy. Returns each parent's part
/// of the body and its gift, or None if they can't pay for the body and still give the child a
/// real start in life (specks that could only spare a crumb once took over by breeding where they
/// landed).
fn birth_cost(p: &SimParams, energies: &[f32], child_mass: f32) -> Option<(f32, [f32; 2])> {
    // Going it alone costs more.
    let share = if energies.len() == 1 { (p.child_energy_share * 1.5).min(0.9) } else { p.child_energy_share };
    let body = p.meat_per_mass * child_mass / energies.len() as f32;
    let mut gifts = [0.0; 2];
    for (g, &e) in gifts.iter_mut().zip(energies) {
        if e < body {
            return None;
        }
        *g = (e - body) * share;
    }
    (gifts.iter().sum::<f32>() >= p.min_child_energy).then_some((body, gifts))
}

/// Exploration: each joint axis the body has wanders a little around what the brain asks for,
/// smoothly over about a quarter second, by the creature's `explore` gene. Other outputs get none
/// (a random bite or mating attempt costs too much to be worth trying).
fn explore(c: &mut Creature, tick: u64, bdt: f32, scale: f32) {
    let sigma = exploration_sigma(c, scale);
    let b = &mut c.brain;
    if b.noise.len() != brain::N_OUT {
        b.noise = vec![0.0; brain::N_OUT];
    }
    let keep = (-bdt / 0.25).exp();
    let fresh = (1.0 - keep * keep).sqrt() * sigma;
    let mut seed = (c.id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ tick.wrapping_mul(0xD1B5_4A32_D192_ED03)) | 1;
    let mut gauss = || {
        // Sum of four uniforms, scaled to unit variance: plenty normal enough for this.
        let mut s = 0.0;
        for _ in 0..4 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            s += (seed >> 40) as f32 / (1u64 << 24) as f32;
        }
        (s - 2.0) * 3f32.sqrt()
    };
    for j in &c.joints {
        for d in 0..j.kind.axes() {
            let n = &mut b.noise[output::JOINTS + j.slot + d];
            *n = *n * keep + fresh * gauss();
        }
    }
}

/// Spread of a creature's exploration noise on each joint output.
fn exploration_sigma(c: &Creature, scale: f32) -> f32 {
    0.3 * c.genome.traits.explore.clamp(0.0, 1.0) * scale
}

/// How strongly closing in on what it needs is felt.
const SEEK_REWARD: f32 = 10.0;

/// What movement is learned from: the feeling of closing in on food when hungry, on water when
/// thirsty and on a mate when the mating drive is up (and of falling back when moving away),
/// scaled by the `seek` gene. Pain (from bites, in `try_attack`) is the other teacher.
///
/// Eating, drinking and mating themselves aren't rewarded here: they happen standing still, so
/// crediting the leg movements just before them only scrambles a good gait. The approach feelings
/// are what point the creature at them.
///
/// Each closeness is compared with last tick's; a jump means the target changed (eaten, or a new
/// one came in range), not that the creature moved, so it doesn't count.
fn approach_reward(c: &mut Creature, x: &[f32; N_IN]) {
    use input::*;
    let diet = c.genome.traits.diet;
    let close = [
        x[PLANT + 2] * (1.0 - diet) + x[MEAT + 2] * diet,
        x[WATER + 2],
        x[MATE + 2],
    ];
    let need = [1.0 - x[ENERGY], 1.0 - x[HYDRATION], x[MATE_DRIVE]];
    let last = &mut c.brain.closeness;
    let mut felt = 0.0;
    for k in 0..3 {
        let change = close[k] - last[k];
        if change.abs() < 0.05 {
            felt += need[k] * change;
        }
        last[k] = close[k];
    }
    c.brain.reward += SEEK_REWARD * c.genome.traits.seek * felt;
}

const WATER_SENSE_RANGE: f32 = 24.0;

/// Offset to the nearest open water within sensing range, by sampling rings around a point,
/// then narrowing in on the shoreline so the distance changes smoothly as a creature walks
/// (the reward for closing in on water needs small steps, not ring-sized jumps).
fn nearest_water(sim: &Sim, pos: Vec3) -> Option<Vec3> {
    let wl = sim.params.water_level;
    let wet = |d: Vec3| {
        let (x, z) = (pos.x + d.x, pos.z + d.z);
        sim.terrain.height_at(x, z) < wl || sim.terrain.water_at(x, z) > 0.04
    };
    if wet(Vec3::ZERO) {
        return Some(Vec3::ZERO);
    }
    let mut inner = 0.0f32;
    for r in [1.5f32, 3.0, 6.0, 12.0, WATER_SENSE_RANGE] {
        let mut best: Option<(f32, Vec3)> = None;
        for k in 0..12 {
            let a = k as f32 * std::f32::consts::TAU / 12.0;
            let d = Vec3::new(a.cos() * r, 0.0, a.sin() * r);
            let (x, z) = (pos.x + d.x, pos.z + d.z);
            let pool = sim.terrain.water_at(x, z);
            let depth = (wl - sim.terrain.height_at(x, z)).max(if pool > 0.04 { pool } else { 0.0 });
            if depth > 0.0 && best.is_none_or(|(b, _)| depth > b) {
                best = Some((depth, d));
            }
        }
        if let Some((_, d)) = best {
            let dir = d / r;
            let (mut dry, mut hit) = (inner, r);
            for _ in 0..5 {
                let mid = 0.5 * (dry + hit);
                if wet(dir * mid) {
                    hit = mid;
                } else {
                    dry = mid;
                }
            }
            return Some(dir * hit);
        }
        inner = r;
    }
    None
}

fn sense(cx: &SenseCtx, i: usize, c: &Creature) -> [f32; N_IN] {
    use input::*;
    let sim = cx.sim;
    let p = &sim.params;
    let mut x = [0.0f32; N_IN];
    let Some(torso) = sim.torso_of(c) else {
        return x;
    };
    let pos = torso.translation();
    let rot = *torso.rotation();
    let (fwd, side) = heading_frame(rot);

    x[BIAS] = 1.0;
    let phase = std::f32::consts::TAU * c.genome.traits.cpg_freq * c.age;
    x[CPG_SIN] = phase.sin();
    x[CPG_COS] = phase.cos();
    x[ENERGY] = (c.energy / c.max_energy).clamp(0.0, 1.0);
    x[HEALTH] = (c.health / c.max_health).clamp(0.0, 1.0);
    x[OXYGEN] = c.oxygen;
    x[AGE] = (c.age / p.lifespan.max(1.0)).min(1.0);
    x[UPRIGHT] = (rot * Vec3::Y).y;
    x[PITCH] = (rot * Vec3::X).y;
    let v = torso.linvel();
    x[SPEED_FWD] = (v.dot(fwd) / 3.0).tanh();
    x[SPEED_SIDE] = (v.dot(side) / 3.0).tanh();
    x[YAW_RATE] = (torso.angvel().y / 3.0).tanh();

    let r = p.sense_radius;
    let put_dir = |x: &mut [f32; N_IN], at: usize, target: Vec3| {
        let rel = target - pos;
        let d = (rel.x * rel.x + rel.z * rel.z).sqrt().max(1e-3);
        x[at] = rel.dot(fwd) / d;
        x[at + 1] = rel.dot(side) / d;
        x[at + 2] = (1.0 - d / r).clamp(0.0, 1.0);
    };
    if let Some((k, _)) = cx.plants.nearest(pos, r, |k| sim.plants[k as usize].pos, |_| true) {
        put_dir(&mut x, PLANT, sim.plants[k as usize].pos);
    }
    if let Some((k, _)) = cx.meat.nearest(pos, r, |k| sim.meat[k as usize].pos, |_| true) {
        put_dir(&mut x, MEAT, sim.meat[k as usize].pos);
    }
    // The nearest stone (or boulder) light enough to lift.
    let can_lift = crate::carry::LIFT * c.mass;
    if let Some((k, _)) = cx.stones.nearest(pos, r, |k| cx.stone_at[k as usize].0, |k| cx.stone_at[k as usize].1 <= can_lift) {
        put_dir(&mut x, STONE, cx.stone_at[k as usize].0);
    }
    x[HOLDING] = if c.held.is_some() { 1.0 } else { 0.0 };
    let mut mate: Option<(usize, f32)> = None;
    cx.creatures.for_each_near(pos, r, |k| {
        let k = k as usize;
        if k == i {
            return;
        }
        let t = cx.torso[k];
        let d2 = (t.x - pos.x).powi(2) + (t.z - pos.z).powi(2);
        if d2 > r * r || mate.is_some_and(|(_, b)| d2 >= b) {
            return;
        }
        let o = &sim.creatures[k];
        if o.age >= p.maturity_age && c.genome.distance(&o.genome) < p.species_threshold {
            mate = Some((k, d2));
        }
    });
    if let Some((k, _)) = mate {
        put_dir(&mut x, MATE, cx.torso[k]);
    }
    if let Some((k, _)) = cx
        .creatures
        .nearest(pos, r, |k| cx.torso[k as usize], |k| k as usize != i)
    {
        let k = k as usize;
        let o = &sim.creatures[k];
        put_dir(&mut x, OTHER, cx.torso[k]);
        let kin = 1.0 - c.genome.distance(&o.genome) / p.species_threshold.max(0.01);
        x[OTHER_KIN] = kin.clamp(-1.0, 1.0);
        x[OTHER_SIZE] = (o.mass / c.mass.max(1e-3)).ln().tanh();
        x[OTHER_AGGRO] = o.output(output::ATTACK).max(0.0);
    }

    // Eyes see into every physics world with anything in their range (worlds hold things from
    // all over); the ground and walls, the same in each, are looked for in the first only.
    let mut worlds = 1u64;
    cx.creatures.for_each_near(pos, p.ray_length + cx.creature_reach, |k| worlds |= 1 << sim.creatures[k as usize].shard);
    cx.obstacles.for_each_near(pos, p.ray_length + cx.obstacle_reach, |k| worlds |= 1 << sim.obstacles[k as usize].shard);
    let own = c.id;
    let origin = pos + Vec3::Y * 0.05;
    let mut nearest = [f32::MAX; 3];
    for (s, shard) in sim.shards.iter().enumerate() {
        if worlds & (1 << s) == 0 {
            continue;
        }
        let not_self = move |_h: ColliderHandle, col: &Collider| match Tag::decode(col.user_data) {
            Tag::Terrain | Tag::Wall => s == 0,
            t => t.creature_id() != Some(own),
        };
        let qp = shard.world.query_pipeline_with_filter(QueryFilter::default().predicate(&not_self));
        for (n, ang) in [0.6f32, 0.0, -0.6].into_iter().enumerate() {
            let dir = Quat::from_rotation_y(ang) * fwd;
            if let Some((_, toi)) = qp.cast_ray(&Ray::new(origin, dir), p.ray_length, true) {
                nearest[n] = nearest[n].min(toi);
            }
        }
    }
    for (n, toi) in nearest.into_iter().enumerate() {
        if toi < f32::MAX {
            x[RAYS + n] = 1.0 - toi / p.ray_length.max(1e-3);
        }
    }

    x[PAIN] = c.pain.min(1.0);
    x[IN_WATER] = c.in_water;
    x[HYDRATION] = c.hydration;
    x[MATE_DRIVE] = c.mate_drive;
    x[SOIL] = (c.soil / (p.soil_per_kg * c.mass).max(1e-6)).min(1.0);
    // Hearing: the call that sounds loudest from here (louder calls carry further).
    let range = p.call_range.max(1.0);
    let heard = cx
        .calls
        .iter()
        .filter(|k| k.index != i)
        .map(|k| (k, k.loud * (1.0 - k.pos.distance(pos) / (range * k.loud.sqrt().max(0.1))).max(0.0)))
        .filter(|(_, level)| *level > 0.0)
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((k, level)) = heard {
        let rel = k.pos - pos;
        let d = (rel.x * rel.x + rel.z * rel.z).sqrt().max(1e-3);
        x[HEAR] = rel.dot(fwd) / d;
        x[HEAR + 1] = rel.dot(side) / d;
        x[HEAR + 2] = level;
        x[HEAR_PITCH] = k.pitch;
        let own = c.genome.traits.voice;
        let apart: f32 = own.iter().zip(&k.voice).map(|(a, b)| (a - b).abs()).sum::<f32>() / 4.0;
        x[HEAR_LIKE] = 1.0 - apart;
    }
    if let Some(d) = nearest_water(sim, pos) {
        let flat = Vec3::new(d.x, 0.0, d.z);
        let dist = flat.length();
        if dist > 1e-3 {
            x[WATER] = flat.dot(fwd) / dist;
            x[WATER + 1] = flat.dot(side) / dist;
        }
        x[WATER + 2] = 1.0 - (dist / WATER_SENSE_RANGE).min(1.0);
    }
    x[MEMORY..MEMORY + N_MEMORY].copy_from_slice(&c.brain.memory);
    for j in &c.joints {
        if let Some((angles, _)) = body::joint_state(sim.world_of(c), &c.parts, j) {
            let (mid, half) = j.span();
            for &(_, d) in j.axes() {
                x[JOINTS + j.slot + d] = ((angles[d] - mid) / half).clamp(-1.5, 1.5);
            }
        }
    }
    x
}
