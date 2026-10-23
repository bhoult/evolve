//! Acts of god: storms, meteors, whirlwinds and manna, called down by whoever is at the controls.
//! What lasts (a storm, a whirlwind) is kept here as a function of the world's clock, so a saved
//! world resumes with the same sky.

use serde::{Deserialize, Serialize};

/// Seconds a summoned storm takes to build, and to die away.
pub const STORM_RISE: f64 = 30.0;
pub const STORM_FALL: f64 = 45.0;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Acts {
    /// A summoned storm: the sim time it was called and when it ends.
    #[serde(default)]
    pub storm: Option<(f64, f64)>,
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
