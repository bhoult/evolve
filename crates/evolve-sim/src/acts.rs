//! Acts of god: storms, meteors, whirlwinds and manna, called down by whoever is at the controls.
//! What lasts (a storm, a whirlwind) is kept here as a function of the world's clock, so a saved
//! world resumes with the same sky.

use serde::{Deserialize, Serialize};

use crate::math::Vec3;
use crate::sim::{DeathCause, Sim};
use crate::snapshot::HighlightKind;

/// Seconds a summoned storm takes to build, and to die away.
pub const STORM_RISE: f64 = 30.0;
pub const STORM_FALL: f64 = 45.0;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Acts {
    /// A summoned storm: the sim time it was called and when it ends.
    #[serde(default)]
    pub storm: Option<(f64, f64)>,
    #[serde(default)]
    pub whirlwinds: Vec<Whirlwind>,
    /// An earthquake shaking the ground: its center (x, z) and the sim times it began and ends.
    #[serde(default)]
    pub quake: Option<([f32; 2], f64, f64)>,
    /// Throws waiting for the next step: a change of velocity for every part of a creature
    /// (a multibody keeps no velocity set on a part directly, so it goes in as a force).
    #[serde(skip)]
    pub kicks: Vec<(u64, Vec3)>,
}

/// A whirlwind: a funnel of spinning, rising air that wanders the island for a while.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Whirlwind {
    pub x: f32,
    pub z: f32,
    /// Which way it's wandering, radians.
    pub heading: f32,
    pub start: f64,
    pub end: f64,
    pub seed: u32,
}

/// The whirlwind's core (where it lifts), how far its pull reaches, how fast it wanders (m/s),
/// and how long it lasts (s).
pub const WHIRL_CORE: f32 = 3.0;
pub const WHIRL_REACH: f32 = 14.0;
const WHIRL_TRAVEL: f32 = 3.0;
pub const WHIRL_LIFE: f64 = 25.0;

impl Whirlwind {
    /// 0..1: it spins up over a few seconds and dies away over the last few.
    pub fn strength(&self, time: f64) -> f32 {
        ((time - self.start) / 3.0).min((self.end - time) / 4.0).clamp(0.0, 1.0) as f32
    }

    /// The air's velocity at `p` (m/s), `ground` the ground's height below it: round the core,
    /// in toward it, and up through it (the lift fading out between 6 and 18 m up, so what it
    /// catches whirls round at a few meters and is flung out, not carried off into the sky).
    pub fn wind_at(&self, p: Vec3, ground: f32, time: f64) -> Vec3 {
        let s = self.strength(time);
        let off = Vec3::new(p.x - self.x, 0.0, p.z - self.z);
        let r = off.length();
        if s <= 0.0 || r > WHIRL_REACH {
            return Vec3::ZERO;
        }
        let out = if r > 1e-3 { off / r } else { Vec3::X };
        let round = Vec3::new(-out.z, 0.0, out.x);
        let spin = 22.0 * s * if r < WHIRL_CORE { r / WHIRL_CORE } else { WHIRL_CORE / r };
        // Fades out toward the edge of its reach so nothing jolts as it passes.
        let edge = ((WHIRL_REACH - r) / 4.0).clamp(0.0, 1.0);
        let height = ((18.0 - (p.y - ground)) / 12.0).clamp(0.0, 1.0);
        let up = 18.0 * s * (-(r / (1.5 * WHIRL_CORE)).powi(2)).exp() * height;
        (round * spin - out * (0.35 * spin)) * edge + Vec3::Y * up
    }
}

/// How far an earthquake shakes the ground, m, and how long, s.
pub const QUAKE_REACH: f32 = 70.0;
const QUAKE_TIME: f64 = 6.0;
/// A chunk of land fallen from the sky: its radius and height, m.
pub const LANDFALL_RADIUS: f32 = 9.0;
const LANDFALL_HEIGHT: f32 = 4.0;

/// A meteor's crater radius, and how far its blast reaches, m.
pub const CRATER: f32 = 4.0;
pub const BLAST: f32 = 16.0;

impl Sim {
    /// Raises the land around `at` by up to `amount` m (lowers it if negative), less toward
    /// `radius`.
    pub fn terraform(&mut self, at: Vec3, radius: f32, amount: f32) {
        let r = radius.max(0.5);
        self.terrain.lift(at.x, at.z, r, true, |dx, dz| {
            let f = (1.0 - (dx * dx + dz * dz) / (r * r)).max(0.0);
            amount * f * f
        });
        self.settle_terrain();
    }

    /// Makes an eternal spring at `at`, or stops the one there.
    pub fn toggle_spring(&mut self, at: Vec3) {
        let t = &self.terrain;
        let (n, size) = (t.res, t.size);
        let pos = |k: u32| {
            let (i, j) = (k as usize / n, k as usize % n);
            ((j as f32 / (n - 1) as f32 - 0.5) * size, (i as f32 / (n - 1) as f32 - 0.5) * size)
        };
        let near = t.springs.iter().position(|&k| {
            let (x, z) = pos(k);
            (x - at.x).hypot(z - at.z) < 4.0
        });
        match near {
            Some(i) => {
                self.terrain.springs.swap_remove(i);
                self.log("A spring runs dry");
            }
            None if t.contains(at.x, at.z, 1.0) => {
                let k = t.nearest_point(at.x, at.z) as u32;
                self.terrain.springs.push(k);
                self.log("A spring wells up");
            }
            None => {}
        }
    }

    /// An earthquake splits the land along `path` (a jagged line of (x, z) points): a crack
    /// opens along it, the ground on one side heaves up, and everything near shakes for a while.
    pub fn earthquake(&mut self, path: &[[f32; 2]]) {
        if path.len() < 2 {
            return;
        }
        let pts: Vec<(f32, f32)> = path.iter().map(|p| (p[0], p[1])).collect();
        // Distance to the line, and which side of it.
        let near = |x: f32, z: f32| {
            let mut best = (f32::MAX, 0.0f32);
            for w in pts.windows(2) {
                let ((ax, az), (bx, bz)) = (w[0], w[1]);
                let (ex, ez) = (bx - ax, bz - az);
                let len2 = (ex * ex + ez * ez).max(1e-6);
                let t = (((x - ax) * ex + (z - az) * ez) / len2).clamp(0.0, 1.0);
                let (px, pz) = (ax + ex * t, az + ez * t);
                let d = (x - px).hypot(z - pz);
                if d < best.0 {
                    best = (d, (ex * (z - az) - ez * (x - ax)).signum());
                }
            }
            best
        };
        let (mut x0, mut z0, mut x1, mut z1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, z) in &pts {
            (x0, z0, x1, z1) = (x0.min(x), z0.min(z), x1.max(x), z1.max(z));
        }
        let (cx, cz) = ((x0 + x1) * 0.5, (z0 + z1) * 0.5);
        let reach = 0.5 * (x1 - x0).hypot(z1 - z0) + 8.0;
        // One side heaves up (the land remade), and the crack is torn open (raw earth).
        self.terrain.lift(cx, cz, reach, true, |dx, dz| {
            let (d, side) = near(cx + dx, cz + dz);
            if side > 0.0 && d < 8.0 { 0.9 * (1.0 - d / 8.0) } else { 0.0 }
        });
        self.terrain.lift(cx, cz, reach, false, |dx, dz| {
            let (d, _) = near(cx + dx, cz + dz);
            if d < 3.2 { -3.0 * (1.0 - d / 3.2).powf(1.5) } else { 0.0 }
        });
        self.settled_at = f64::NEG_INFINITY;
        self.settle_terrain();
        self.acts.quake = Some(([cx, cz], self.time, self.time + QUAKE_TIME));
        self.log("An earthquake splits the land");
    }

    /// A chunk of land falls from the sky onto `at`: a mound rises there (out of the sea, it
    /// makes new land), crushing what it lands on and throwing what's near.
    pub fn landfall(&mut self, at: Vec3) {
        if !self.terrain.contains(at.x, at.z, 2.0) {
            return;
        }
        let r = LANDFALL_RADIUS;
        // Out at sea, enough to stand clear of the water.
        let base = self.terrain.height_at(at.x, at.z);
        let lift = LANDFALL_HEIGHT.max(self.params.water_level + 2.5 - base);
        self.terrain.lift(at.x, at.z, r, true, |dx, dz| {
            let f = (1.0 - (dx * dx + dz * dz) / (r * r)).max(0.0);
            lift * f.powf(1.5)
        });
        self.settled_at = f64::NEG_INFINITY;
        self.settle_terrain();
        let center = Vec3::new(at.x, self.terrain.height_at(at.x, at.z), at.z);
        let mut dead = Vec::new();
        let mut kicks = Vec::new();
        for c in &mut self.creatures {
            let Some(pos) = self.shards[c.shard].world.bodies.get(c.parts[0].body).map(|b| b.translation()) else {
                continue;
            };
            let d = (pos.x - at.x).hypot(pos.z - at.z);
            if d < r * 0.6 {
                dead.push(c.id);
            } else if d < r * 2.0 {
                let f = 1.0 - d / (r * 2.0);
                c.health -= 60.0 * f * f;
                c.pain = 1.0;
                c.attacked_ago = 0.0;
                let away = Vec3::new(pos.x - at.x, 0.0, pos.z - at.z).normalize_or(Vec3::X);
                kicks.push((c.id, away * (10.0 * f) + Vec3::Y * (6.0 * f)));
            }
        }
        for id in dead {
            self.kill(id, DeathCause::Killed);
        }
        self.acts.kicks.extend(kicks);
        self.highlight_with(HighlightKind::Landfall, 0, None, center, r);
        self.log("Land falls from the sky");
    }

    /// A whirlwind touches down at `at` and wanders off.
    pub fn whirlwind(&mut self, at: Vec3) {
        if !self.terrain.contains(at.x, at.z, 2.0) {
            return;
        }
        let heading = self.rng.range(0.0, std::f32::consts::TAU);
        let seed = self.rng.next_u64() as u32;
        self.acts.whirlwinds.push(Whirlwind { x: at.x, z: at.z, heading, start: self.time, end: self.time + WHIRL_LIFE, seed });
        self.log("A whirlwind touches down");
    }

    /// Moves the whirlwinds on: each wanders, turning back from the sea and the world's edge,
    /// and tears up the plants and meat in its path.
    pub(crate) fn move_whirlwinds(&mut self, dt: f32) {
        if self.acts.whirlwinds.is_empty() {
            return;
        }
        let (time, wl) = (self.time, self.params.water_level);
        for w in &mut self.acts.whirlwinds {
            let wobble = ((time as f32) * 0.7 + w.seed as f32 * 1e-3).sin() * 0.8 + ((time as f32) * 0.23 + w.seed as f32).sin() * 0.5;
            w.heading += wobble * dt;
            let ahead = (w.x + w.heading.cos() * 8.0, w.z + w.heading.sin() * 8.0);
            if !self.terrain.contains(ahead.0, ahead.1, 10.0) || self.terrain.height_at(ahead.0, ahead.1) < wl + 0.3 {
                // Turn toward the middle of the island.
                let home = (-w.z).atan2(-w.x);
                let turn = (home - w.heading + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                w.heading += turn.clamp(-2.0 * dt, 2.0 * dt);
            }
            // It sits while it spins up, then wanders faster as it strengthens.
            let pace = WHIRL_TRAVEL * w.strength(time).powi(2);
            w.x += w.heading.cos() * pace * dt;
            w.z += w.heading.sin() * pace * dt;
        }
        let tears = |pos: Vec3, ws: &[Whirlwind]| {
            ws.iter().any(|w| w.strength(time) > 0.25 && (pos.x - w.x).hypot(pos.z - w.z) < WHIRL_CORE + 0.5)
        };
        let ws = &self.acts.whirlwinds;
        self.plants.retain(|f| !tears(f.pos, ws));
        self.meat.retain(|f| !tears(f.pos, ws));
    }

    /// A meteor strikes at `at`: it blasts out a crater (the earth thrown up into its rim),
    /// kills what it lands on, throws and hurts what's near, and scorches the plants.
    pub fn meteor(&mut self, at: Vec3) {
        let (x0, z0) = (at.x, at.z);
        if !self.terrain.contains(x0, z0, 1.0) {
            return;
        }
        // The crater: a bowl dug out, the earth piled into a ring around it, a grid point at a
        // time (earth moved between grid points would be smeared over four of them).
        let cell = self.terrain.cell();
        let (dig, pile) = (self.params.dig_depth, self.params.pile_height);
        let (size, cells) = (self.terrain.size, (self.terrain.res - 1) as f32);
        let grid = |k: f32| (k / cells - 0.5) * size;
        let (gx, gz) = (((x0 / size + 0.5) * cells).round(), ((z0 / size + 0.5) * cells).round());
        let n = (CRATER * 1.7 / cell).ceil() as i32;
        let mut dug = 0.0;
        let mut rim = Vec::new();
        for i in -n..=n {
            for j in -n..=n {
                let (x, z) = (grid(gx + j as f32), grid(gz + i as f32));
                let r = (x - x0).hypot(z - z0);
                if r < CRATER {
                    let depth = 1.8 * (1.0 - (r / CRATER).powi(2));
                    dug -= self.terrain.reshape(x, z, -depth * cell * cell, dig, pile);
                } else if r < CRATER * 1.7 {
                    rim.push((x, z, (std::f32::consts::PI * (r - CRATER) / (0.7 * CRATER)).sin()));
                }
            }
        }
        let weight: f32 = rim.iter().map(|r| r.2).sum::<f32>().max(1e-6);
        for (x, z, w) in rim {
            self.terrain.reshape(x, z, dug * w / weight, dig, pile);
        }
        self.settled_at = f64::NEG_INFINITY;
        self.settle_terrain();
        let ground = self.terrain.height_at(x0, z0);
        let center = Vec3::new(x0, ground, z0);

        // The blast: what it lands on dies; what's near is hurt and thrown, up and away.
        let mut dead = Vec::new();
        let mut kicks = Vec::new();
        for c in &mut self.creatures {
            let Some(pos) = self.shards[c.shard].world.bodies.get(c.parts[0].body).map(|b| b.translation()) else {
                continue;
            };
            let d = pos.distance(center);
            if d < CRATER * 0.75 + c.radius() {
                dead.push(c.id);
            } else if d < BLAST {
                let f = 1.0 - d / BLAST;
                c.health -= 120.0 * f * f;
                c.pain = 1.0;
                c.attacked_ago = 0.0;
                let away = Vec3::new(pos.x - x0, 0.0, pos.z - z0).normalize_or(Vec3::X);
                kicks.push((c.id, away * (16.0 * f) + Vec3::Y * (9.0 * f)));
            }
        }
        for id in dead {
            self.kill(id, DeathCause::Killed);
        }
        self.acts.kicks.extend(kicks);
        for o in &self.obstacles {
            if let Some(b) = self.shards[o.shard].world.bodies.get_mut(o.body)
                && b.is_dynamic()
            {
                let d = b.translation().distance(center);
                if d < BLAST {
                    let f = 1.0 - d / BLAST;
                    let away = Vec3::new(b.translation().x - x0, 0.0, b.translation().z - z0).normalize_or(Vec3::X);
                    let mass = b.mass();
                    b.apply_impulse((away * (16.0 * f) + Vec3::Y * (9.0 * f)) * mass, true);
                }
            }
        }
        // Plants near it burn, and meat at the heart of it is gone.
        let scorch = CRATER * 1.8;
        self.plants.retain(|f| f.pos.distance(center) > scorch);
        self.meat.retain(|f| f.pos.distance(center) > CRATER);
        self.highlight_with(HighlightKind::Meteor, 0, None, center, BLAST);
        self.log("A meteor strikes");
    }
}

impl Acts {
    /// How strong the summoned storm is at `time`, 0..1: building, raging, dying away.
    pub fn storm_strength(&self, time: f64) -> f32 {
        match self.storm {
            Some((start, end)) => ((time - start) / STORM_RISE).min((end - time) / STORM_FALL).clamp(0.0, 1.0) as f32,
            None => 0.0,
        }
    }

    /// Seconds left of the summoned storm (0 without one).
    pub fn storm_left(&self, time: f64) -> f32 {
        self.storm.map_or(0.0, |(_, end)| (end - time).max(0.0) as f32)
    }

    /// Calls a storm for `seconds` (more than its build and fade), or prolongs the one raging.
    pub fn summon_storm(&mut self, time: f64, seconds: f64) {
        let seconds = seconds.max(STORM_RISE + STORM_FALL);
        self.storm = Some(match self.storm {
            Some((start, end)) if end > time => (start, end.max(time + seconds)),
            _ => (time, time + seconds),
        });
    }

    /// Calms the summoned storm: it starts dying away now, from however strong it is.
    pub fn calm_storm(&mut self, time: f64) {
        let s = self.storm_strength(time) as f64;
        if let Some((start, end)) = self.storm.as_mut() {
            *end = end.min(time + STORM_FALL * s);
            // Still building: let it fade from where it got to, not jump to full first.
            *start = start.min(*end - STORM_RISE - STORM_FALL);
        }
    }

    /// Forgets what has run its course.
    pub fn expire(&mut self, time: f64) {
        if self.storm.is_some_and(|(_, end)| time >= end) {
            self.storm = None;
        }
        self.whirlwinds.retain(|w| time < w.end);
        if self.quake.is_some_and(|(_, _, end)| time >= end) {
            self.quake = None;
        }
    }

    /// How hard the ground shakes at (x, z), as an acceleration (m/s²) for something at `phase`.
    pub fn shaking(&self, x: f32, z: f32, time: f64, phase: f32) -> Vec3 {
        let Some(([cx, cz], start, end)) = self.quake else {
            return Vec3::ZERO;
        };
        let s = ((time - start) / 0.5).min((end - time) / 2.0).clamp(0.0, 1.0) as f32;
        let near = (1.0 - (x - cx).hypot(z - cz) / QUAKE_REACH).clamp(0.0, 1.0);
        let t = time as f32 + phase;
        Vec3::new((t * 13.0).sin(), 0.35 * (t * 17.0).sin(), (t * 11.0 + 1.3).cos()) * (8.0 * s * near)
    }
}
