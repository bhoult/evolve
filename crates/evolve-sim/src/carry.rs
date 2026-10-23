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
        for ci in 0..self.creatures.len() {
            let c = &self.creatures[ci];
            if c.dead.is_some() {
                continue;
            }
            let (grip, throw) = (c.output(output::GRIP), c.output(output::THROW));
            match c.held {
                Some(_) if throw > 0.5 => self.let_go(ci, TOSS * (throw - 0.5) / 0.5),
                Some(_) if grip < 0.0 => self.let_go(ci, 0.0),
                None if grip > 0.2 => self.pick_up(ci),
                _ => {}
            }
        }
    }

    /// Picks up the nearest thing within reach of the mouth that it can lift.
    fn pick_up(&mut self, ci: usize) {
        let Some((mouth, _, _)) = self.mouth(ci) else {
            return;
        };
        let c = &self.creatures[ci];
        let reach = self.params.eat_reach + GRASP;
        let can_lift = LIFT * c.mass;
        let taken: Vec<u64> = self
            .creatures
            .iter()
            .filter_map(|o| match o.held {
                Some(Held::Obstacle { id, .. }) => Some(id),
                _ => None,
            })
            .collect();
        enum Pick {
            Obstacle(usize, f32),
            Plant(usize),
            Meat(usize),
        }
        let mut best: Option<(f32, Pick)> = None;
        let mut consider = |d: f32, pick: Pick| {
            if d < reach && best.as_ref().is_none_or(|(b, _)| d < *b) {
                best = Some((d, pick));
            }
        };
        for (oi, o) in self.obstacles.iter().enumerate() {
            if !o.dynamic || taken.contains(&o.id) {
                continue;
            }
            let Some(b) = self.shards[o.shard].world.bodies.get(o.body) else {
                continue;
            };
            let mass = b.mass();
            if mass > can_lift {
                continue;
            }
            let d = (b.translation().distance(mouth) - 0.7 * crate::sim::obstacle_radius(o.shape)).max(0.0);
            consider(d, Pick::Obstacle(oi, mass));
        }
        let ground_d = |p: Vec3| {
            let flat = Vec3::new(p.x - mouth.x, 0.0, p.z - mouth.z).length();
            if (p.y - mouth.y).abs() > 1.2 { f32::MAX } else { (flat - 0.3).max(0.0) }
        };
        for (k, f) in self.plants.iter().enumerate() {
            consider(ground_d(f.pos), Pick::Plant(k));
        }
        for (k, f) in self.meat.iter().enumerate() {
            consider(ground_d(f.pos), Pick::Meat(k));
        }
        let Some((_, pick)) = best else {
            return;
        };
        let held = match pick {
            Pick::Obstacle(oi, mass) => {
                let o = &self.obstacles[oi];
                hold_body(&mut self.shards[o.shard].world, o.body);
                Held::Obstacle { id: o.id, mass }
            }
            Pick::Plant(k) => Held::Plant(self.plants.swap_remove(k).energy),
            Pick::Meat(k) => Held::Meat(self.meat.swap_remove(k).energy),
        };
        self.creatures[ci].held = Some(held);
    }

    /// Lets go of what it holds, tossing it forward at `toss` m/s (0: setting it down).
    pub(crate) fn let_go(&mut self, ci: usize, toss: f32) {
        let Some(held) = self.creatures[ci].held.take() else {
            return;
        };
        let (mouth, fwd, vel) = self.mouth(ci).unwrap_or((self.creatures[ci].last_pos, Vec3::X, Vec3::ZERO));
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
    if b.body_type() != RigidBodyType::KinematicPositionBased {
        b.set_body_type(RigidBodyType::KinematicPositionBased, true);
    }
    let colliders: Vec<ColliderHandle> = b.colliders().to_vec();
    for h in colliders {
        if let Some(col) = world.colliders.get_mut(h)
            && !col.is_sensor()
        {
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
