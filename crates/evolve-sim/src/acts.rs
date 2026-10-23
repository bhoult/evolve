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
    #[serde(default)]
    pub totems: Vec<Totem>,
    #[serde(default)]
    pub tsunami: Option<Tsunami>,
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

    /// The air's velocity at `p` (m/s), `ground` giving the ground's height below it (asked only
    /// within its reach): round the core,
    /// in toward it, and up through it (the lift fading out between 6 and 18 m up, so what it
    /// catches whirls round at a few meters and is flung out, not carried off into the sky).
    pub fn wind_at(&self, p: Vec3, ground: impl FnOnce() -> f32, time: f64) -> Vec3 {
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
        let height = ((18.0 - (p.y - ground())) / 12.0).clamp(0.0, 1.0);
        let up = 18.0 * s * (-(r / (1.5 * WHIRL_CORE)).powi(2)).exp() * height;
        (round * spin - out * (0.35 * spin)) * edge + Vec3::Y * up
    }
}

/// What a totem stirs in the creatures near it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TotemKind {
    /// Uncontrollable breeding.
    Love,
    /// Uncontrollable fighting.
    War,
    /// Running away.
    Fear,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Totem {
    pub kind: TotemKind,
    pub x: f32,
    pub z: f32,
}

/// How far a totem's sway reaches, m (full over the inner half, fading to nothing at the edge).
pub const TOTEM_REACH: f32 = 16.0;

/// A tsunami: a great wave rolling in from the open sea across the island.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tsunami {
    /// Which way it travels (a unit vector in x, z).
    pub dir: [f32; 2],
    /// Where its crest started, along `dir` (m), and when.
    pub from: f32,
    pub start: f64,
}

/// The tsunami's height at sea (m), speed (m/s) and how long it runs (s).
pub const TSUNAMI_HEIGHT: f32 = 7.0;
pub const TSUNAMI_SPEED: f32 = 12.0;
pub const TSUNAMI_LIFE: f64 = 24.0;

impl Tsunami {
    /// Where its crest is, along `dir`.
    pub fn front(&self, time: f64) -> f32 {
        self.from + TSUNAMI_SPEED * (time - self.start) as f32
    }

    /// Its height now: rising as it builds, spent by the end.
    pub fn height(&self, time: f64) -> f32 {
        let u = ((time - self.start) / TSUNAMI_LIFE) as f32;
        TSUNAMI_HEIGHT * (u / 0.1).clamp(0.0, 1.0) * (1.0 - u * u).max(0.0)
    }
}

/// How far behind its crest a tsunami's flood reaches, m.
pub const TSUNAMI_BACK: f32 = 80.0;

/// The wave's shape, 0..1, `q` m ahead of (positive) or behind (negative) its crest: a steep face
/// 6 m long, then the flood behind it, falling slowly back over 80 m (water keeps pouring in for
/// several seconds after the crest has passed).
pub fn tsunami_profile(q: f32) -> f32 {
    if !(-TSUNAMI_BACK..=6.0).contains(&q) {
        0.0
    } else if q >= 0.0 {
        (1.0 - q / 6.0).powf(0.6)
    } else {
        (0.35 + 0.65 * (q / 8.0).exp()) * ((q + TSUNAMI_BACK) / 20.0).min(1.0)
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
    /// A tsunami rolls in from the open sea beyond `at`, toward the island's middle (or along
    /// `toward` if `at` is near the middle).
    pub fn tsunami(&mut self, at: Vec3, toward: Vec3) {
        let inward = Vec3::new(-at.x, 0.0, -at.z);
        let dir = if inward.length() > 10.0 { inward.normalize() } else { Vec3::new(toward.x, 0.0, toward.z).normalize_or(Vec3::X) };
        // It starts just beyond the farthest corner of the world behind it.
        let from = -self.terrain.half() * std::f32::consts::SQRT_2 - 8.0;
        self.acts.tsunami = Some(Tsunami { dir: [dir.x, dir.z], from, start: self.time });
        self.log("A tsunami is coming");
    }

    /// The tsunami sweeps away the plants and meat it covers deeply.
    pub(crate) fn wash_away(&mut self) {
        if self.acts.tsunami.is_none() {
            return;
        }
        let (time, sea) = (self.time, self.params.water_level);
        let (acts, terrain) = (&self.acts, &self.terrain);
        let deep = |p: Vec3| acts.tsunami_surface(p.x, p.z, time, sea).is_some_and(|s| s - terrain.height_at(p.x, p.z) > 0.8);
        self.plants.retain(|f| !deep(f.pos));
        self.meat.retain(|f| !deep(f.pos));
    }

    /// Food falls from heaven around `at`: a ring of full-grown plants.
    pub fn manna(&mut self, at: Vec3) {
        let full = self.params.plant_max_energy;
        for _ in 0..16 {
            let a = self.rng.range(0.0, std::f32::consts::TAU);
            let r = 6.0 * self.rng.range(0.0, 1.0).sqrt();
            self.add_plant(at + Vec3::new(a.cos() * r, 0.0, a.sin() * r), full);
        }
        self.highlight_with(HighlightKind::Manna, 0, None, at, 6.0);
        self.log("Manna falls from heaven");
    }

    /// Sets up a totem at `at`, or takes down the one there.
    pub fn toggle_totem(&mut self, at: Vec3, kind: TotemKind) {
        match self.acts.totems.iter().position(|t| (t.x - at.x).hypot(t.z - at.z) < 3.0) {
            Some(i) => {
                self.acts.totems.swap_remove(i);
                self.log("A totem is taken down");
            }
            None if self.terrain.contains(at.x, at.z, 1.0) => {
                self.acts.totems.push(Totem { kind, x: at.x, z: at.z });
                self.log(match kind {
                    TotemKind::Love => "A totem of love is raised",
                    TotemKind::War => "A totem of war is raised",
                    TotemKind::Fear => "A totem of fear is raised",
                });
            }
            None => {}
        }
    }

    /// What the totems stir in each creature, each brain tick. Love fills the mating drive,
    /// cuts the wait between matings and turns the urge to mate on (breeding still has to pay
    /// for each child); war turns the urge to bite on and draws it toward the nearest other
    /// creature; fear hurts, feels bad, and drives it away from the totem.
    pub(crate) fn totem_urges(&mut self) {
        if self.acts.totems.is_empty() {
            for c in &mut self.creatures {
                c.urge = [0.0; 3];
                c.urge_push = Vec3::ZERO;
            }
            return;
        }
        let torso = self.torso_positions();
        let totems = &self.acts.totems;
        // Where everyone is, for those a totem of war sends at the nearest.
        let everyone = totems.iter().any(|t| t.kind == TotemKind::War).then(|| {
            let mut grid = crate::grid::Grid::new(self.terrain.size, crate::life::GRID_CELL);
            grid.rebuild(torso.iter().copied());
            grid
        });
        for (i, c) in self.creatures.iter_mut().enumerate() {
            let p = torso[i];
            let mut urge = [0.0f32; 3];
            let mut away = Vec3::ZERO;
            for t in totems {
                let off = Vec3::new(p.x - t.x, 0.0, p.z - t.z);
                let f = ((TOTEM_REACH - off.length()) / (0.5 * TOTEM_REACH)).clamp(0.0, 1.0);
                let k = t.kind as usize;
                urge[k] = urge[k].max(f);
                if t.kind == TotemKind::Fear {
                    away += off.normalize_or(Vec3::X) * f;
                }
            }
            // Strong enough to drive even a body that can't move itself off over flat ground
            // (more than its friction holds it with).
            let mut push = away.normalize_or_zero() * (12.0 * urge[2]);
            if urge[1] > 0.0 {
                // The nearest other creature within reach, to go at.
                if let Some((j, _)) = everyone.as_ref().and_then(|g| g.nearest(p, 12.0, |j| torso[j as usize], |j| j as usize != i)) {
                    let q = torso[j as usize];
                    push += Vec3::new(q.x - p.x, 0.0, q.z - p.z).normalize_or_zero() * (10.0 * urge[1]);
                }
            }
            if urge[0] > 0.0 {
                c.mate_drive = c.mate_drive.max(urge[0]);
                // (Cut, not gone: a pair still rests a moment between children.)
                c.mate_cooldown = c.mate_cooldown.min(1.5 + 1.5 * (1.0 - urge[0]));
            }
            if urge[2] > 0.0 {
                c.pain = c.pain.max(urge[2]);
                c.brain.reward -= 0.05 * urge[2];
            }
            c.urge = urge;
            c.urge_push = push;
        }
    }

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
                c.last_attacker = None;
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
            // (Timed from its start, which keeps the precision an f32 loses on long runs.)
            let (t, a, b) = ((time - w.start) as f32, (w.seed % 6283) as f32 * 1e-3, (w.seed / 6283 % 6283) as f32 * 1e-3);
            let wobble = (t * 0.7 + a).sin() * 0.8 + (t * 0.23 + b).sin() * 0.5;
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
                c.last_attacker = None;
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
        if self.tsunami.as_ref().is_some_and(|t| time >= t.start + TSUNAMI_LIFE) {
            self.tsunami = None;
        }
    }

    /// The tsunami's water surface at (x, z), if it's there now: higher than the sea in and
    /// behind its crest, and nothing ahead of it.
    pub fn tsunami_surface(&self, x: f32, z: f32, time: f64, sea: f32) -> Option<f32> {
        let t = self.tsunami.as_ref()?;
        let q = x * t.dir[0] + z * t.dir[1] - t.front(time);
        let h = t.height(time) * tsunami_profile(q);
        (h > 0.01).then_some(sea + h)
    }

    /// How hard the ground shakes at (x, z), as an acceleration (m/s²) for something at `phase`.
    pub fn shaking(&self, x: f32, z: f32, time: f64, phase: f32) -> Vec3 {
        let Some(([cx, cz], start, end)) = self.quake else {
            return Vec3::ZERO;
        };
        let s = ((time - start) / 0.5).min((end - time) / 2.0).clamp(0.0, 1.0) as f32;
        let near = (1.0 - (x - cx).hypot(z - cz) / QUAKE_REACH).clamp(0.0, 1.0);
        let t = (time - start) as f32 + phase;
        Vec3::new((t * 13.0).sin(), 0.35 * (t * 17.0).sin(), (t * 11.0 + 1.3).cos()) * (8.0 * s * near)
    }
}
