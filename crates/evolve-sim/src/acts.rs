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
    /// Throws waiting for the next step: a change of velocity for every part of a creature
    /// (a multibody keeps no velocity set on a part directly, so it goes in as a force).
    #[serde(skip)]
    pub kicks: Vec<(u64, Vec3)>,
}

/// A meteor's crater radius, and how far its blast reaches, m.
pub const CRATER: f32 = 4.0;
pub const BLAST: f32 = 16.0;

impl Sim {
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
    }
}
