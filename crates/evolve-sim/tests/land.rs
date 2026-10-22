//! Rain running off into streams, wearing channels, and feet wearing paths.

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

#[test]
fn streams_shrink_after_the_rain_and_ponds_dry_up() {
    let p = SimParams::default();
    let mut t = Terrain::generate(&p, 1);
    for _ in 0..60 {
        hydro::update(&mut t, &p, 1.0, 1.0, 30.0, None);
    }
    let raining: f32 = t.water.iter().sum();
    let mut soak = 1.0f32;
    for _ in 0..240 {
        soak *= (-30.0f32 / 1800.0).exp();
        hydro::update(&mut t, &p, 0.0, soak, 30.0, None);
    }
    let after: f32 = t.water.iter().sum();
    assert!(after < raining * 0.2, "two dry hours later: {after} vs {raining} while raining");
    let land = (0..t.heights.len()).filter(|&k| t.heights[k] > p.water_level).count();
    let soaked = (0..t.heights.len()).filter(|&k| t.heights[k] > p.water_level && t.moisture[k] > 0.5).count();
    assert!(soaked < land / 4, "the land dries out: {soaked} of {land} still wet");
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
