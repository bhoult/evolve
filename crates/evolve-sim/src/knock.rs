//! Creatures knocking things about: fallen fruit and meat pushed along by a moving limb (or any
//! part of a body), sliding over the ground until friction stops them, and fruit shaken down
//! from a tree whose trunk is struck hard enough.

use std::collections::HashMap;

use crate::grid::Grid;
use crate::life::{GRID_CELL, food_grid, fruit_grid};
use crate::math::{Vec2, Vec3};
use crate::sim::Sim;

/// Slower than this, m/s, a body part pushes nothing.
const KNOCK_SPEED: f32 = 0.8;
/// How close above the ground a part must be to push what lies there, m (beyond its own size).
const KNOCK_HEIGHT: f32 = 0.3;
/// Fastest a knocked piece of food goes, m/s.
const KNOCKED_TOP: f32 = 5.0;
/// How quickly a sliding piece of food slows, 1/s.
const FRICTION: f32 = 3.0;
/// Slower than this, m/s, a sliding piece comes to rest.
const AT_REST: f32 = 0.05;
/// The steepest ground (sine of its slope, about 25 degrees) a piece of food rests on.
const HOLDS: f32 = 0.42;
/// Seconds before a tree that's been shaken gives up more fruit to another blow.
const SHAKE_EVERY: f64 = 1.0;

/// A body part on the move: where it is, its horizontal velocity, and how far it reaches.
#[derive(Clone, Copy, Debug)]
pub struct Swing {
    pub at: Vec3,
    pub vel: Vec2,
    pub reach: f32,
}

impl Sim {
    /// Every moving body part knocks the fallen fruit and meat it sweeps through and shakes the
    /// trees it strikes (see `knock_with`); then whatever was knocked slides on. Once a brain tick.
    pub(crate) fn knock(&mut self, bdt: f32) {
        if self.params.knock_strength > 0.0 || self.params.shake_speed > 0.0 {
            let mut swings = Vec::new();
            for c in self.creatures.iter().filter(|c| c.dead.is_none()) {
                let world = &self.shards[c.shard].world;
                for (q, spec) in c.parts.iter().zip(&c.plan) {
                    let Some(b) = world.bodies.get(q.body) else { continue };
                    let v = b.linvel();
                    let vel = Vec2::new(v.x, v.z);
                    if vel.length() >= KNOCK_SPEED {
                        swings.push(Swing { at: b.translation(), vel, reach: spec.shape.bounding_radius() });
                    }
                }
            }
            self.knock_with(&swings);
        }
        self.slide_food(bdt);
    }

    /// Body parts swinging as given knock what they touch: a piece of fallen fruit or meat
    /// within reach (and near the ground) takes on `knock_strength` of the part's horizontal
    /// velocity; a standing tree's trunk struck faster than `shake_speed` drops some of its
    /// hanging fruit, more the harder the blow. (Public so tests can swing parts directly.)
    pub fn knock_with(&mut self, swings: &[Swing]) {
        if swings.is_empty() {
            return;
        }
        let p = self.params.clone();
        let size = self.terrain.size;
        if p.knock_strength > 0.0 {
            let fruit = fruit_grid(size, &self.fruit);
            let meat = food_grid(size, &self.meat);
            for s in swings {
                let ground = self.terrain.height_at(s.at.x, s.at.z);
                if s.at.y - s.reach > ground + KNOCK_HEIGHT {
                    continue;
                }
                let push = (s.vel * p.knock_strength).clamp_length_max(KNOCKED_TOP);
                let near = s.reach + 0.15;
                let mut hit = Vec::new();
                fruit.for_each_near(s.at, near, |k| {
                    let f = &self.fruit[k as usize];
                    if !f.hanging() && Vec2::new(f.pos.x - s.at.x, f.pos.z - s.at.z).length() <= near {
                        hit.push(f.id);
                    }
                });
                meat.for_each_near(s.at, near, |k| {
                    let m = &self.meat[k as usize];
                    if Vec2::new(m.pos.x - s.at.x, m.pos.z - s.at.z).length() <= near {
                        hit.push(m.id);
                    }
                });
                for id in hit {
                    let v = self.sliding.entry(id).or_insert(Vec2::ZERO);
                    // A blow sets it going the part's way (not slower than it already slides).
                    if push.length() > v.length() {
                        *v = push;
                    }
                }
            }
        }
        if p.shake_speed > 0.0 {
            let mut trees = Grid::new(size, GRID_CELL);
            trees.rebuild(self.trees.iter().map(|t| t.pos));
            let mut shaken: Vec<(u64, f32)> = Vec::new();
            for s in swings {
                let speed = s.vel.length();
                if speed < p.shake_speed {
                    continue;
                }
                trees.for_each_near(s.at, s.reach + 1.0, |k| {
                    let t = &self.trees[k as usize];
                    let d = Vec2::new(t.pos.x - s.at.x, t.pos.z - s.at.z).length();
                    let struck = t.solid() && d <= t.trunk_radius() + s.reach + 0.1 && s.at.y < t.pos.y + t.height;
                    if struck && !shaken.iter().any(|x| x.0 == t.id) {
                        shaken.push((t.id, speed));
                    }
                });
            }
            for (tree, speed) in shaken {
                if self.shaken.get(&tree).is_some_and(|&at| self.time - at < SHAKE_EVERY) {
                    continue;
                }
                self.shaken.insert(tree, self.time);
                // From a tenth of the fruit at the threshold to most of it at three times it.
                let chance = (0.1 + 0.4 * (speed / p.shake_speed - 1.0)).clamp(0.1, 0.8);
                for i in 0..self.fruit.len() {
                    if self.fruit[i].tree == Some(tree) && self.rng.chance(chance) {
                        let f = &mut self.fruit[i];
                        f.tree = None;
                        f.age = 0.0;
                        f.pos.y = self.terrain.height_at(f.pos.x, f.pos.z);
                    }
                }
            }
            self.shaken.retain(|_, at| self.time - *at < SHAKE_EVERY);
        }
    }

    /// Knocked fruit and meat slide on: down the slope a little, slowing with friction, until
    /// they rest (or fall into the sea, where they stop).
    pub fn slide_food(&mut self, dt: f32) {
        if self.sliding.is_empty() {
            return;
        }
        let gravity = self.params.gravity;
        let half = self.terrain.half() - 1.0;
        let keep = (-FRICTION * dt).exp();
        let mut moving: HashMap<u64, Vec2> = std::mem::take(&mut self.sliding);
        let terrain = &self.terrain;
        // Moves one piece: None if it isn't sliding, else whether it slides on.
        let mut slide = |id: u64, pos: &mut Vec3| -> Option<bool> {
            let v = moving.get_mut(&id)?;
            // Down a slope too steep for it to rest on (gentler ones hold it, as friction would).
            let n = terrain.normal_at(pos.x, pos.z);
            let downhill = Vec2::new(n.x, n.z);
            let steep = (downhill.length() - HOLDS).max(0.0);
            *v = (*v + downhill.normalize_or_zero() * steep * gravity * dt) * keep;
            let to = Vec2::new(pos.x, pos.z) + *v * dt;
            pos.x = to.x.clamp(-half, half);
            pos.z = to.y.clamp(-half, half);
            pos.y = terrain.height_at(pos.x, pos.z);
            Some(v.length() >= AT_REST && terrain.water_at(pos.x, pos.z) < 0.3)
        };
        let mut still = Vec::new();
        for f in self.fruit.iter_mut().filter(|f| !f.hanging()) {
            if slide(f.id, &mut f.pos) == Some(false) {
                still.push(f.id);
            }
        }
        for m in self.meat.iter_mut() {
            if slide(m.id, &mut m.pos) == Some(false) {
                still.push(m.id);
            }
        }
        for id in still {
            moving.remove(&id);
        }
        // Whatever's gone (eaten, rotted) stops too.
        let alive: std::collections::HashSet<u64> = self.fruit.iter().map(|f| f.id).chain(self.meat.iter().map(|m| m.id)).collect();
        moving.retain(|id, _| alive.contains(id));
        self.sliding = moving;
    }
}
