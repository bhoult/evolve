//! Carrying: a creature can pick up what's at its mouth (a stone, a boulder, a plant, meat),
//! carry it (its weight drags on the body), set it down where it stands, or throw it. Whatever
//! creatures build out of stones, plants and food (walls, piles, caches, gardens) is made this
//! way, and whether to build is up to evolution: the grip and throw actions start switched off.

use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use crate::brain::output;
use crate::math::Vec3;
use crate::sim::{ObstacleShape, Sim};

/// What a creature holds in its mouth.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Held {
    /// A loose stone or boulder (by id), and its mass, kg.
    Obstacle { id: u64, mass: f32 },
    /// A plant (its energy): set down, it takes root again.
    Plant(f32),
    /// A piece of meat (its energy).
    Meat(f32),
}

/// The heaviest thing a creature can lift, as a fraction of its own mass.
pub const LIFT: f32 = 0.8;
/// How far beyond its mouth it can reach to pick something up, m (besides `eat_reach`).
const GRASP: f32 = 0.25;
/// Fastest it can throw, m/s.
const TOSS: f32 = 8.0;

/// A loose stone (a small block): its half extents for a size `s` (0..1).
pub fn stone_shape(s: f32) -> ObstacleShape {
    let a = 0.08 + 0.1 * s.clamp(0.0, 1.0);
    ObstacleShape::Box { half: [a, a * 0.6, a * 0.8] }
}

impl Sim {
    /// Where a creature's mouth is, which way it faces, and how fast its torso moves.
    fn mouth(&self, ci: usize) -> Option<(Vec3, Vec3, Vec3)> {
        let c = &self.creatures[ci];
        let torso = self.torso_of(c)?;
        let half = c.torso_half();
        let mouth = torso.position().transform_point(Vec3::new(half.x, 0.0, 0.0));
        let fwd = *torso.rotation() * Vec3::X;
        Some((mouth, fwd, torso.linvel()))
    }

    /// Each brain tick: a creature with its grip on and nothing in its mouth picks up what's
    /// there; with its grip off it lets go, and throwing tosses what it holds.
    pub(crate) fn grip(&mut self) {
        let mut reaching = Vec::new();
        for ci in 0..self.creatures.len() {
            let c = &self.creatures[ci];
            if c.dead.is_some() {
                continue;
            }
            let (grip, throw) = (c.output(output::GRIP), c.output(output::THROW));
            match c.held {
                Some(_) if throw > 0.5 => self.let_go(ci, TOSS * (throw - 0.5) / 0.5),
                Some(_) if grip < 0.0 => self.let_go(ci, 0.0),
                None if grip > 0.2 => reaching.push(ci),
                _ => {}
            }
        }
        if !reaching.is_empty() {
            self.pick_up(&reaching);
        }
    }

    /// Each of `reaching` picks up the nearest thing within reach of its mouth that it can lift
    /// (and nobody else took first), found through grids built once for all of them.
    fn pick_up(&mut self, reaching: &[usize]) {
        use crate::grid::Grid;
        use crate::life::{GRID_CELL, food_grid};
        let size = self.terrain.size;
        let reach = self.params.eat_reach + GRASP;
        let held: std::collections::HashSet<u64> = self
            .creatures
            .iter()
            .filter_map(|o| match o.held {
                Some(Held::Obstacle { id, .. }) => Some(id),
                _ => None,
            })
            .collect();
        // Loose stones and boulders: (index, center, mass, how far they reach).
        let loose: Vec<(usize, Vec3, f32, f32)> = self
            .obstacles
            .iter()
            .enumerate()
            .filter(|(_, o)| o.dynamic && !held.contains(&o.id))
            .filter_map(|(oi, o)| {
                let b = self.shards[o.shard].world.bodies.get(o.body)?;
                Some((oi, b.translation(), b.mass(), 0.7 * crate::sim::obstacle_radius(o.shape)))
            })
            .collect();
        let mut stones = Grid::new(size, GRID_CELL);
        stones.rebuild(loose.iter().map(|s| s.1));
        let (plants, meat) = (food_grid(size, &self.plants), food_grid(size, &self.meat));
        // What's been taken this tick, so two creatures can't pick up the same thing.
        let (mut took_stone, mut took_plant, mut took_meat) = (Vec::new(), Vec::new(), Vec::new());
        for &ci in reaching {
            let Some((mouth, _, _)) = self.mouth(ci) else {
                continue;
            };
            let can_lift = LIFT * self.creatures[ci].mass;
            // (Distances to a stone's surface; to food, how far it is along the ground.)
            let stone = stones.nearest(
                mouth,
                reach + 1.5,
                |k| loose[k as usize].1,
                |k| {
                    let (_, at, mass, r) = loose[k as usize];
                    mass <= can_lift && !took_stone.contains(&k) && at.distance(mouth) - r < reach
                },
            );
            let on_ground = |p: Vec3| (p.y - mouth.y).abs() <= 1.2;
            let plant = plants.nearest(mouth, reach + 0.3, |k| self.plants[k as usize].pos, |k| {
                !took_plant.contains(&k) && on_ground(self.plants[k as usize].pos)
            });
            let food = meat.nearest(mouth, reach + 0.3, |k| self.meat[k as usize].pos, |k| {
                !took_meat.contains(&k) && on_ground(self.meat[k as usize].pos)
            });
            let d = |o: Option<(u32, f32)>| o.map_or(f32::MAX, |(_, d)| d);
            let held = if d(stone) <= d(plant).min(d(food)) && let Some((k, _)) = stone {
                took_stone.push(k);
                let (oi, _, mass, _) = loose[k as usize];
                let o = &self.obstacles[oi];
                hold_body(&mut self.shards[o.shard].world, o.body);
                Held::Obstacle { id: o.id, mass }
            } else if d(plant) <= d(food) && let Some((k, _)) = plant {
                took_plant.push(k);
                Held::Plant(self.plants[k as usize].energy)
            } else if let Some((k, _)) = food {
                took_meat.push(k);
                Held::Meat(self.meat[k as usize].energy)
            } else {
                continue;
            };
            self.creatures[ci].held = Some(held);
        }
        // Food taken out of the world, last first so the indices stay good.
        took_plant.sort_unstable();
        for k in took_plant.into_iter().rev() {
            self.plants.swap_remove(k as usize);
        }
        took_meat.sort_unstable();
        for k in took_meat.into_iter().rev() {
            self.meat.swap_remove(k as usize);
        }
    }

    /// Lets go of what it holds, tossing it forward at `toss` m/s (0: setting it down).
    pub(crate) fn let_go(&mut self, ci: usize, toss: f32) {
        let Some(held) = self.creatures[ci].held.take() else {
            return;
        };
        // (A body that blew up, its motion gone to NaN, lets go where it was last seen.)
        let (mouth, fwd, vel) = self
            .mouth(ci)
            .filter(|(m, f, v)| m.is_finite() && f.is_finite() && v.is_finite())
            .unwrap_or((self.creatures[ci].last_pos, Vec3::X, Vec3::ZERO));
        match held {
            Held::Obstacle { id, .. } => {
                if let Some(o) = self.obstacles.iter().find(|o| o.id == id) {
                    let world = &mut self.shards[o.shard].world;
                    free_body(world, o.body);
                    if let Some(b) = world.bodies.get_mut(o.body) {
                        b.set_linvel(vel + fwd * toss + Vec3::Y * (0.3 * toss), true);
                    }
                }
            }
            // Food lands where it's set down (or, thrown, a little way off); a plant takes root
            // again if it lands on land.
            Held::Plant(energy) | Held::Meat(energy) => {
                let at = mouth + fwd * (toss * 0.4);
                if !self.terrain.contains(at.x, at.z, 0.5) {
                    return;
                }
                match held {
                    Held::Plant(_) => {
                        if self.terrain.height_at(at.x, at.z) > self.params.water_level {
                            self.add_plant(at, energy);
                        }
                    }
                    _ => {
                        self.add_meat(at, energy);
                    }
                }
            }
        }
    }

    /// Each physics step: what's held rides at the mouth, and its weight drags on the torso.
    pub(crate) fn carry(&mut self) {
        let g = self.params.gravity;
        for ci in 0..self.creatures.len() {
            let Some(Held::Obstacle { id, mass }) = self.creatures[ci].held else {
                continue;
            };
            let Some((mouth, fwd, _)) = self.mouth(ci) else {
                continue;
            };
            let Some(oi) = self.obstacles.iter().position(|o| o.id == id) else {
                // Taken away (by hand).
                self.creatures[ci].held = None;
                continue;
            };
            let (shard, body, shape) = (self.obstacles[oi].shard, self.obstacles[oi].body, self.obstacles[oi].shape);
            let rot = self.torso_of(&self.creatures[ci]).map(|t| *t.rotation()).unwrap_or_default();
            // (Moving between physics worlds rebuilds a body as it was made, so this is kept up.)
            hold_body(&mut self.shards[shard].world, body);
            if let Some(b) = self.shards[shard].world.bodies.get_mut(body) {
                let at = mouth + fwd * (crate::sim::obstacle_radius(shape) * 0.8 + 0.05);
                b.set_next_kinematic_position(Pose::from_parts(at, rot));
            }
            let c = &self.creatures[ci];
            if let Some(t) = self.shards[c.shard].world.bodies.get_mut(c.parts[0].body) {
                t.add_force(-Vec3::Y * (mass * g), true);
            }
        }
    }

    /// What a creature holds is dropped where it is (it died, or was taken away).
    pub(crate) fn drop_held(&mut self, ci: usize) {
        self.let_go(ci, 0.0);
    }
}

/// Makes a body ride along where it's put, passing through things while it's carried.
fn hold_body(world: &mut PhysicsWorld, body: RigidBodyHandle) {
    let Some(b) = world.bodies.get_mut(body) else {
        return;
    };
    // Already held (its colliders are made sensors along with it, and a body rebuilt in
    // another world comes back dynamic).
    if b.body_type() == RigidBodyType::KinematicPositionBased {
        return;
    }
    b.set_body_type(RigidBodyType::KinematicPositionBased, true);
    let colliders: Vec<ColliderHandle> = b.colliders().to_vec();
    for h in colliders {
        if let Some(col) = world.colliders.get_mut(h) {
            col.set_sensor(true);
        }
    }
}

/// Lets a carried body fall and collide again.
fn free_body(world: &mut PhysicsWorld, body: RigidBodyHandle) {
    let Some(b) = world.bodies.get_mut(body) else {
        return;
    };
    b.set_body_type(RigidBodyType::Dynamic, true);
    let colliders: Vec<ColliderHandle> = b.colliders().to_vec();
    for h in colliders {
        if let Some(col) = world.colliders.get_mut(h) {
            col.set_sensor(false);
        }
    }
}
