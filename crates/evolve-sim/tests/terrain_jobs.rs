//! How long each job over the whole terrain takes on a saved world, one run each:
//! `EVOLVE_DIAG_SAVE=world.evo cargo test --release --test terrain_jobs -- --ignored --nocapture`

use std::time::Instant;

use evolve_sim::hydro;
use evolve_sim::rng::Rng;

#[test]
#[ignore]
fn terrain_jobs() {
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        return;
    };
    let mut sim = evolve_sim::save::load(path.as_ref()).unwrap();
    let p = sim.params.clone();
    let t = &mut sim.terrain;
    println!("terrain {} a side over {} m ({:.2} m apart), {} pieces of ground", t.res, t.size, t.cell(), t.tiles() * t.tiles());
    let ms = |s: Instant| s.elapsed().as_secs_f64() * 1000.0;
    // The world steps on between updates, while a big terrain's coarse water is worked out on
    // another thread: a pause stands in for it, so what's timed is what the world waits for.
    let between = std::time::Duration::from_millis(400);
    for k in 0..4 {
        std::thread::sleep(between);
        let s = Instant::now();
        hydro::update(t, &p, 1.0, 1.0, 2.0, None);
        println!("water, run {k}: {:.1} ms", ms(s));
    }
    std::thread::sleep(between);
    let s = Instant::now();
    hydro::update(t, &p, 1.0, 1.0, 2.0, Some(30.0));
    println!("water with erosion: {:.1} ms", ms(s));
    std::thread::sleep(between);
    let s = Instant::now();
    hydro::update(t, &p, 1.0, 1.0, 2.0, None);
    println!("water after erosion (spreading what it cut): {:.1} ms", ms(s));
    t.take_dirty();
    // Dig 300 holes about the island, as creatures do.
    let mut rng = Rng::new(3);
    let half = t.half() * 0.6;
    for _ in 0..300 {
        let (x, z) = (rng.range(-half, half), rng.range(-half, half));
        t.lift(x, z, 0.6, false, |dx, dz| -0.4 * (1.0 - (dx * dx + dz * dz).sqrt() / 0.6).max(0.0));
    }
    let patches = t.dirty.len();
    let s = Instant::now();
    t.slump(p.slump_angle, p.dig_depth, p.pile_height, 24);
    println!("slump over {patches} dug patches: {:.1} ms", ms(s));
    t.take_dirty();
    let s = Instant::now();
    t.heal(30.0, p.earth_recovery * 60.0, 0.0);
    println!("heal: {:.1} ms, leaving {} patches", ms(s), t.dirty.len());
    let s = Instant::now();
    t.slump(p.slump_angle, p.dig_depth, p.pile_height, 24);
    println!("slump after heal: {:.1} ms", ms(s));
    t.take_dirty();
    let s = Instant::now();
    let tiles = t.tiles();
    for k in 0..tiles.min(16) {
        let _ = t.tile_shape(k, k);
    }
    println!("one piece of physics ground: {:.2} ms", ms(s) / tiles.min(16) as f64);
    let spots: Vec<(f32, f32)> = sim.plants.iter().map(|f| (f.pos.x, f.pos.z)).collect();
    let t = &sim.terrain;
    let s = Instant::now();
    let near: f32 = spots.iter().map(|&(x, z)| t.near_water(x, z, 5.0)).sum();
    println!("near_water for {} plants: {:.2} ms (mean {:.2})", spots.len(), ms(s), near / spots.len().max(1) as f32);
}
