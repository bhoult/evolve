//! Day and night, clouds, rain and storms. Pure functions of the world's clock and seed, so
//! nothing extra goes into saves and a loaded world sees the same sky it had.

use serde::{Deserialize, Serialize};

use crate::noise::Perlin;
use crate::params::SimParams;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Weather {
    /// Solar time as a fraction of the day: 0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset.
    pub time_of_day: f32,
    /// Unit vector pointing at the sun (below the horizon at night).
    pub sun_dir: [f32; 3],
    /// 0 at night, 1 in full daylight, smooth through dawn and dusk.
    pub daylight: f32,
    /// Fraction of the sky covered by cloud, 0..1.
    pub cloud: f32,
    /// Rain intensity, 0..1.
    pub rain: f32,
    /// Storm intensity (gales, lightning), 0..1.
    pub storm: f32,
    /// Wind velocity along x and z, m/s.
    pub wind: [f32; 2],
}

/// How far the sun's daily path tilts away from straight overhead, radians.
const LATITUDE: f32 = 0.5;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Solar time for a point in the cycle. The sun moves faster below the horizon, so nights take
/// about a third of the cycle instead of half: there's more to see by day.
fn solar_time(cycle: f32) -> f32 {
    const NIGHT_SPEEDUP: f32 = 0.5;
    let u = cycle.rem_euclid(1.0);
    (u + NIGHT_SPEEDUP * (std::f32::consts::TAU * u).sin() / std::f32::consts::TAU).rem_euclid(1.0)
}

pub fn sun_direction(time_of_day: f32) -> [f32; 3] {
    let h = std::f32::consts::TAU * (time_of_day - 0.25);
    // Rises in the east (+x), sets in the west, passes south of overhead at noon.
    [h.cos(), h.sin() * LATITUDE.cos(), h.sin() * LATITUDE.sin()]
}

pub fn daylight(sun_y: f32) -> f32 {
    smoothstep(-0.12, 0.22, sun_y)
}

/// The sky over a world at a moment of simulated time.
pub fn at(p: &SimParams, time: f64) -> Weather {
    // Worlds start mid-morning.
    let cycle = if p.day_night {
        (time / p.day_length.max(1.0) as f64 + 0.32).fract() as f32
    } else {
        0.36
    };
    let time_of_day = solar_time(cycle);
    let sun_dir = sun_direction(time_of_day);
    let daylight = daylight(sun_dir[1]);

    let noise = Perlin::new(p.seed ^ 0x5EA7_4E12);
    // Weather drifts on the scale of `weather_period`; the offset keeps the lattice away from 0.
    let t = (time / p.weather_period.max(10.0) as f64) as f32 + 3.7;
    let slow = noise.fbm(t, 11.3, 3, 0.5);
    let gust = noise.noise(t * 23.0, 41.7);
    let heading = noise.fbm(t * 0.35, 77.1, 2, 0.5) * 4.0;
    let (cloud, rain, storm) = if p.weather {
        // `storminess` 0..1 shifts the whole distribution: 0 is mostly fair, 1 mostly wet.
        // `slow` has a spread of about 0.15.
        let cloud = (0.42 + 1.85 * slow + (p.storminess - 0.3) * 0.6).clamp(0.0, 1.0);
        (cloud, smoothstep(0.58, 0.78, cloud), smoothstep(0.78, 0.93, cloud))
    } else {
        (0.18, 0.0, 0.0)
    };
    let speed = (1.5 + 4.0 * cloud + 14.0 * storm) * (1.0 + gust * (0.4 + 0.6 * storm));
    Weather {
        time_of_day,
        sun_dir,
        daylight,
        cloud,
        rain,
        storm,
        wind: [heading.cos() * speed, heading.sin() * speed],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_are_longer_than_nights_and_weather_varies() {
        let p = SimParams::default();
        let n = 20_000;
        let mut day = 0;
        let (mut rainy, mut stormy, mut clear) = (0, 0, 0);
        for i in 0..n {
            let t = i as f64 * 30.0;
            let w = at(&p, t);
            if w.sun_dir[1] > 0.0 {
                day += 1;
            }
            if w.rain > 0.5 {
                rainy += 1;
            }
            if w.storm > 0.5 {
                stormy += 1;
            }
            if w.cloud < 0.3 {
                clear += 1;
            }
        }
        let f = |k: i32| k as f32 / n as f32;
        eprintln!("day {:.2} rainy {:.2} stormy {:.2} clear {:.2}", f(day), f(rainy), f(stormy), f(clear));
        assert!((0.6..0.72).contains(&f(day)));
        assert!((0.08..0.4).contains(&f(rainy)));
        assert!((0.02..0.2).contains(&f(stormy)));
        assert!(f(clear) > 0.2);
    }
}

