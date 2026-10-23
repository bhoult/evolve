//! Rain running off into streams and filling lakes, wearing channels, and feet wearing paths.

use evolve_sim::hydro;
use evolve_sim::terrain::Terrain;
use evolve_sim::{Command, Sim, SimParams};

fn quiet_params() -> SimParams {
    SimParams {
        min_population: 0,
        terrain_resolution: 65,
        weather: false,
        ..SimParams::default()
    }
}

/// Mean height of the eight neighbours of grid point k.
fn around(t: &Terrain, k: usize) -> f32 {
    let n = t.res;
    let (i, j) = (k / n, k % n);
    let mut sum = 0.0;
    for (di, dj) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
        sum += t.heights[(i as isize + di) as usize * n + (j as isize + dj) as usize];
    }
    sum / 8.0
}

#[test]
fn rain_drains_to_the_sea_through_every_point() {
    let p = SimParams::default();
    let t = Terrain::generate(&p, 1);
    let d = hydro::drainage(&t, p.water_level);
    assert_eq!(d.order.len(), t.heights.len(), "every point is reached from the sea");
    let mut place = vec![0usize; t.heights.len()];
    for (at, &k) in d.order.iter().enumerate() {
        place[k as usize] = at;
    }
    for (k, &r) in d.receiver.iter().enumerate() {
        if r != u32::MAX {
            assert!(place[r as usize] < place[k], "water is passed to a point nearer the sea");
            assert!(d.filled[k] >= d.filled[r as usize], "and never uphill");
        }
    }
}

#[test]
fn steady_rain_makes_streams_in_valleys_and_cuts_them_deeper() {
    let mut p = SimParams::default();
    p.erosion_rate = 2.0;
    let mut t = Terrain::generate(&p, 1);
    let start = t.heights.clone();
    for _ in 0..240 {
        hydro::update(&mut t, &p, 1.0, 1.0, 30.0, Some(30.0));
    }
    let land: Vec<usize> = (0..t.heights.len()).filter(|&k| start[k] > p.water_level + 0.5).collect();
    let streams: Vec<usize> = land.iter().copied().filter(|&k| t.water[k] > 0.1).collect();
    assert!(!streams.is_empty(), "two hours of rain should make streams");
    assert!(streams.len() < land.len() / 4, "streams, not a flood: {} of {}", streams.len(), land.len());
    // Streams run in the low ground, below what's around them.
    let inner: Vec<usize> = streams
        .iter()
        .copied()
        .filter(|&k| (1..t.res - 1).contains(&(k / t.res)) && (1..t.res - 1).contains(&(k % t.res)))
        .collect();
    let below = inner.iter().filter(|&&k| t.heights[k] < around(&t, k)).count();
    assert!(below * 10 > inner.len() * 7, "{below} of {} stream points lie below their banks", inner.len());

    let deepest = t.eroded.iter().copied().fold(0.0, f32::max);
    assert!(deepest > 0.2, "running water should have cut its bed: {deepest} m");
    assert!(deepest <= p.erosion_max_depth + 1e-4, "but no deeper than the limit: {deepest} m");
    for k in 0..t.heights.len() {
        assert!(t.heights[k].is_finite() && t.water[k].is_finite() && t.moisture[k].is_finite());
        assert!((start[k] - t.heights[k] - t.eroded[k]).abs() < 1e-3, "erosion is all accounted for");
        if start[k] > p.water_level - 0.3 {
            assert!(t.heights[k] >= p.water_level - 0.3 - 1e-4, "land isn't cut below the sea floor");
        }
        assert_eq!(t.disturbance(k), 0.0, "a stream bed isn't counted as earthworks");
    }
}

/// Running water (not lakes) on the land, summed.
fn streams(t: &Terrain) -> f32 {
    (0..t.water.len()).filter(|&k| t.pooled[k] <= 0.02).map(|k| t.water[k]).sum()
}

#[test]
fn streams_shrink_after_the_rain_and_the_land_dries() {
    let p = SimParams::default();
    let mut t = Terrain::generate(&p, 1);
    for _ in 0..60 {
        hydro::update(&mut t, &p, 1.0, 1.0, 30.0, None);
    }
    let raining = streams(&t);
    let mut soak = 1.0f32;
    for _ in 0..240 {
        soak *= (-30.0f32 / 1800.0).exp();
        hydro::update(&mut t, &p, 0.0, soak, 30.0, None);
    }
    let after = streams(&t);
    assert!(after < raining * 0.2, "two dry hours later: {after} vs {raining} while raining");
    let dry_land = |k: usize| t.heights[k] > p.water_level && t.pooled[k] <= 0.0;
    let land = (0..t.heights.len()).filter(|&k| dry_land(k)).count();
    let soaked = (0..t.heights.len()).filter(|&k| dry_land(k) && t.moisture[k] > 0.5).count();
    assert!(soaked < land / 4, "the land dries out: {soaked} of {land} still wet");
}

/// Digs a round hollow `radius` m across, `depth` m below the lowest point of its rim (so it
/// holds that much, even on a slope), somewhere its rim stands well above the sea. Returns its
/// center and grid points.
fn hollow(t: &mut Terrain, p: &SimParams, radius: f32, depth: f32) -> (f32, f32, Vec<usize>) {
    let n = t.res;
    let size = t.size;
    let at = move |k: usize| (((k % n) as f32 / (n - 1) as f32 - 0.5) * size, ((k / n) as f32 / (n - 1) as f32 - 0.5) * size);
    let ring = 2.0 * t.cell() / radius;
    let rim_of = |t: &Terrain, x: f32, z: f32| {
        (0..t.heights.len())
            .filter(|&k| {
                let (gx, gz) = at(k);
                (1.0..1.0 + ring).contains(&((gx - x).hypot(gz - z) / radius))
            })
            .map(|k| t.heights[k])
            .fold(f32::MAX, f32::min)
    };
    let (x, z, rim) = (0..t.heights.len())
        .step_by(7)
        .map(|k| {
            let (x, z) = at(k);
            (x, z, rim_of(t, x, z))
        })
        .filter(|&(x, z, rim)| t.contains(x, z, radius + 10.0) && rim > p.water_level + depth + 1.0 && rim < 1e6)
        // The gentlest: the least ground above the rim inside it.
        .min_by(|a, b| (t.height_at(a.0, a.1) - a.2).total_cmp(&(t.height_at(b.0, b.1) - b.2)))
        .expect("somewhere to dig");
    let mut cells = Vec::new();
    for k in 0..t.heights.len() {
        let (gx, gz) = at(k);
        let r = (gx - x).hypot(gz - z) / radius;
        if r < 1.0 {
            t.heights[k] = t.heights[k].min(rim - depth * (1.0 - r * r));
            cells.push(k);
        }
    }
    t.mark_dirty(0, n - 1, 0, n - 1);
    (x, z, cells)
}

fn lake_volume(t: &Terrain) -> f64 {
    let area = (t.cell() * t.cell()) as f64;
    t.pooled.iter().map(|&w| w as f64 * area).sum()
}

#[test]
fn a_hollow_fills_in_the_rain_holds_its_water_then_slowly_dries() {
    let p = SimParams::default();
    let mut t = Terrain::generate(&p, 1);
    let (_, _, cells) = hollow(&mut t, &p, 8.0, 2.0);
    let held = |t: &Terrain| -> f32 { cells.iter().map(|&k| t.pooled[k]).fold(0.0, f32::max) };
    assert_eq!(held(&t), 0.0);
    // An hour of steady rain.
    let mut soak = 0.0f32;
    for _ in 0..120 {
        soak += (1.0 - soak) * (1.0 - (-30.0f32 / 1800.0).exp());
        hydro::update(&mut t, &p, 1.0, soak, 30.0, None);
    }
    let full = held(&t);
    assert!(full > 0.2, "an hour of rain should leave a pond in the hollow: {full} m deep");
    let level = cells.iter().filter(|&&k| t.pooled[k] > 0.0).map(|&k| t.heights[k] + t.pooled[k]).collect::<Vec<_>>();
    let (lo, hi) = level.iter().fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
    assert!(hi - lo < 1e-3, "the pond lies level: {lo}..{hi}");
    // A short dry spell: it's still there.
    for _ in 0..60 {
        soak *= (-30.0f32 / 1800.0).exp();
        hydro::update(&mut t, &p, 0.0, soak, 30.0, None);
    }
    let after_short = held(&t);
    assert!(after_short > full * 0.8, "half an hour later the pond is still there: {full} -> {after_short} m");
    // A long one: it shrinks.
    for _ in 0..1440 {
        soak *= (-30.0f32 / 1800.0).exp();
        hydro::update(&mut t, &p, 0.0, soak, 30.0, None);
    }
    let after_long = held(&t);
    assert!(after_long < after_short - 0.05, "a dry half day later it has shrunk: {after_short} -> {after_long} m");
    // And the hollow's water shows on the ground for drinking and the renderer.
    let k = cells.iter().copied().max_by(|&a, &b| t.pooled[a].total_cmp(&t.pooled[b])).unwrap();
    assert!(t.water[k] >= t.pooled[k]);
}

#[test]
fn lake_water_is_only_gained_by_rain_and_lost_to_evaporation_or_the_sea() {
    let p = SimParams::default();
    let mut t = Terrain::generate(&p, 1);
    let (_, _, cells) = hollow(&mut t, &p, 10.0, 2.5);
    let (mut rained, mut evaporated, mut to_sea) = (0.0f64, 0.0f64, 0.0f64);
    let start = lake_volume(&t);
    for step in 0..400 {
        let rain = if step < 200 { 0.8 } else { 0.0 };
        // Creatures pile earth into the pond halfway through: the water it displaces must go
        // somewhere.
        if step == 250 {
            for &k in cells.iter().take(cells.len() / 2) {
                t.heights[k] += 1.0;
            }
        }
        let b = hydro::update(&mut t, &p, rain, 0.5, 30.0, None);
        rained += b.rained as f64;
        evaporated += b.evaporated as f64;
        to_sea += b.to_sea as f64;
    }
    let end = lake_volume(&t);
    assert!(rained > 0.0 && evaporated > 0.0 && end > 0.0);
    let expected = start + rained - evaporated - to_sea;
    assert!(
        (end - expected).abs() < 1e-3 * rained.max(1.0),
        "lakes hold {end} m³, but rain {rained} - evaporation {evaporated} - spilled to sea {to_sea} gives {expected}"
    );
    for k in 0..t.pooled.len() {
        assert!(t.pooled[k].is_finite() && t.pooled[k] >= 0.0);
    }
}

#[test]
fn creatures_float_in_a_lake_and_can_drown_in_it() {
    for sink in [false, true] {
        let mut p = quiet_params();
        if sink {
            p.water_density = 0.0;
            p.breath_time = 1.0;
        }
        let mut sim = Sim::new(p.clone());
        let (x, z, cells) = hollow(&mut sim.terrain, &p, 9.0, 4.0);
        let d = hydro::drainage(&sim.terrain, p.water_level);
        for &k in &cells {
            sim.terrain.pooled[k] = (d.filled[k] - sim.terrain.heights[k]).max(0.0);
            sim.terrain.water[k] = sim.terrain.pooled[k];
        }
        let surface = sim.terrain.height_at(x, z) + sim.terrain.water_at(x, z);
        assert!(sim.terrain.water_at(x, z) > 2.0, "a deep lake to swim in");
        // Let the physics ground take on the bowl first.
        for _ in 0..40 {
            sim.step();
        }
        let id = sim.spawn_creature(
            evolve_sim::genome::Genome::random(&mut evolve_sim::rng::Rng::new(3), &p),
            evolve_sim::math::Vec3::new(x, surface, z),
            0.0,
            f32::MAX,
            0,
            None,
        );
        for _ in 0..240 {
            sim.step();
        }
        let c = sim.creature(id).expect("alive");
        let y = sim.world_of(c).bodies.get(c.parts[0].body).unwrap().translation().y;
        if sink {
            assert!(y < surface - 1.0, "without buoyancy it sinks: {y} vs surface {surface}");
            assert!(c.drowning && c.oxygen == 0.0, "and drowns in the lake");
        } else {
            // (Whether it keeps its head up is up to how it moves: a body about as dense as
            // water floats low, and can drown at the surface too.)
            assert!((y - surface).abs() < 1.0 && c.in_water > 0.0, "it floats at the surface: {y} vs {surface}");
        }
    }
}

#[test]
fn worlds_saved_before_lakes_and_trampling_still_load() {
    let p = SimParams::default();
    let t = Terrain::generate(&p, 1);
    let mut json = serde_json::to_value(&t).unwrap();
    for layer in ["ground", "eroded", "wear", "water", "pooled", "moisture"] {
        json.as_object_mut().unwrap().remove(layer);
    }
    let mut old: Terrain = serde_json::from_value(json).unwrap();
    assert!(old.pooled.is_empty() && old.water.is_empty());
    assert_eq!(old.water_surface(0.0, 0.0, p.water_level), p.water_level);
    assert_eq!(old.disturbance(0), 0.0);
    hydro::update(&mut old, &p, 1.0, 1.0, 30.0, Some(30.0));
    assert_eq!(old.pooled.len(), old.heights.len());
    assert_eq!(old.ground.len(), old.heights.len());
}

#[test]
fn feet_wear_paths_where_creatures_walk_and_plants_avoid_them() {
    let mut sim = Sim::new(quiet_params());
    let at = sim.random_land_point(20.0);
    sim.drop_genome(&evolve_sim::genome::Genome::random(&mut evolve_sim::rng::Rng::new(4), &sim.params), 0, Some(at), 6);
    for _ in 0..1200 {
        sim.step();
    }
    let t = &sim.terrain;
    let worn: f32 = t.wear.iter().sum();
    assert!(worn > 0.0, "creatures moving about should wear the ground");
    let k = (0..t.wear.len()).max_by(|&a, &b| t.wear[a].total_cmp(&t.wear[b])).unwrap();
    let n = t.res;
    let (x, z) = (((k % n) as f32 / (n - 1) as f32 - 0.5) * t.size, ((k / n) as f32 / (n - 1) as f32 - 0.5) * t.size);
    assert!((x - at.x).hypot(z - at.z) < 25.0, "the most worn spot is where they were");

    let mut t = sim.terrain.clone();
    let before = t.plant_suitability(x, z, sim.params.water_level, sim.params.terrain_height);
    t.wear[k] = 1e6;
    let after = t.plant_suitability(x, z, sim.params.water_level, sim.params.terrain_height);
    assert!(after < before * 0.5 || before == 0.0, "plants struggle on a beaten path: {before} -> {after}");
}

#[test]
fn creatures_drink_from_a_stream() {
    let mut p = quiet_params();
    p.thirst_time = 10.0;
    p.hydrology = false;
    let mut sim = Sim::new(p.clone());
    let at = sim.random_land_point(20.0);
    let id = sim.spawn_creature(evolve_sim::genome::Genome::random(&mut evolve_sim::rng::Rng::new(6), &p), at, 0.3, f32::MAX, 0, None);
    // Drain the sea so the only water is what's put on the ground.
    p.water_level = -100.0;
    sim.apply(Command::SetParams(Box::new(p.clone())));
    // (With hydrology off, the water put down is cleared at the next update, every 2 s: the
    // last was at tick 360.)
    for _ in 0..361 {
        sim.step();
    }
    let dry = sim.creature(id).unwrap().hydration;
    sim.terrain.water.fill(0.5);
    for _ in 0..100 {
        sim.step();
    }
    let wet = sim.creature(id).unwrap().hydration;
    assert!(wet > dry + 0.3, "standing in a stream it should drink: {dry} -> {wet}");
}

#[test]
fn a_spring_creeps_downhill_fills_the_hollow_in_its_way_then_runs_on() {
    let mut p = quiet_params();
    p.world_size = 80.0;
    p.erosion_rate = 2.0;
    let mut t = Terrain::generate(&p, 1);
    let (n, size, wl) = (t.res, t.size, p.water_level);
    // A valley falling along x, with a round hollow 6 m across in its middle.
    let at = |v: usize| (v as f32 / (n - 1) as f32 - 0.5) * size;
    for i in 0..n {
        for j in 0..n {
            let (x, z) = (at(j), at(i));
            let r = x.hypot(z) / 6.0;
            let bowl = if r < 1.0 { 1.5 * (1.0 - r * r) } else { 0.0 };
            t.heights[i * n + j] = wl + 4.0 - 0.08 * x + 0.15 * z.abs() - bowl;
        }
    }
    t.ground = t.heights.clone();
    t.springs.push(t.nearest_point(-20.0, 0.0) as u32);
    let center = t.nearest_point(0.0, 0.0);
    let brim = hydro::drainage(&t, wl).filled[center];
    let below = t.nearest_point(16.0, 0.0);

    // When the water first gets to points down the valley, and when the hollow is full.
    let marks = [-18.0, -12.0, 8.0, 16.0, 28.0];
    let mut wet_at = [f32::MAX; 5];
    let mut full_at = f32::MAX;
    let dt = 2.0;
    for step in 1..=300 {
        let now = step as f32 * dt;
        hydro::update(&mut t, &p, 0.0, 0.0, dt, Some(dt));
        for (m, &x) in marks.iter().enumerate() {
            if wet_at[m] == f32::MAX && t.water[t.nearest_point(x, 0.0)] > 0.0 {
                wet_at[m] = now;
            }
        }
        if full_at == f32::MAX && t.heights[center] + t.pooled[center] >= brim - 0.01 {
            full_at = now;
        }
        if now < full_at {
            assert_eq!(t.water[below], 0.0, "nothing runs on below the hollow before it's full (at {now} s)");
            assert_eq!(t.eroded[below], 0.0, "nor wears the ground there");
        }
    }
    let [a, b, c, d, e] = wet_at;
    println!("wet at {wet_at:?} s, hollow full at {full_at} s");
    assert!(a < b && b < f32::MAX, "it creeps down the slope: {a} then {b}");
    assert!((8.0..=16.0).contains(&(b - a)), "at about half a meter a second: 6 m in {} s", b - a);
    assert!(full_at > b + 20.0 && full_at < f32::MAX, "the hollow fills, gradually: full at {full_at} s");
    assert!(c >= full_at && c < full_at + 10.0, "then it spills over the rim: {c} s");
    assert!(c < d && d < e && e < f32::MAX, "and runs on down the valley: {c}, {d}, {e}");
    assert!(t.eroded[below] > 0.0, "wearing a channel where it runs");
}
