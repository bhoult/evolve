use evolve_sim::body::{self, PartShape};
use evolve_sim::genome::Genome;
use evolve_sim::math::Vec3;
use evolve_sim::rng::Rng;
use evolve_sim::sim::DeathCause;
use evolve_sim::{Sim, SimParams, save};

fn quiet_params() -> SimParams {
    SimParams {
        min_population: 0,
        terrain_resolution: 65,
        ..SimParams::default()
    }
}

#[test]
fn random_body_plans_are_sane() {
    let p = SimParams::default();
    let mut rng = Rng::new(42);
    for _ in 0..2000 {
        let g = Genome::random(&mut rng, &p);
        let plan = body::body_plan(&g);
        assert!(!plan.is_empty() && plan.len() <= evolve_sim::genome::MAX_PARTS);
        let mut slots = std::collections::HashSet::new();
        for (k, part) in plan.iter().enumerate() {
            assert!(part.rest.translation.is_finite() && part.rest.rotation.is_finite());
            if k == 0 {
                assert!(part.parent.is_none() && part.joint.is_none());
                continue;
            }
            let parent = part.parent.expect("limbs have parents");
            assert!(parent < k);
            let j = part.joint.as_ref().expect("limbs have joints");
            assert!(j.frame1.translation.is_finite() && j.frame2.rotation.is_finite());
            assert!(slots.insert(j.slot), "each joint drives its own brain output");
            // At rest both joint frames must coincide in the torso frame (angle zero).
            let a = plan[parent].rest * j.frame1;
            let b = part.rest * j.frame2;
            assert!((a.translation - b.translation).length() < 1e-3);
            assert!(a.rotation.dot(b.rotation).abs() > 0.999);
            if let PartShape::Capsule { half_height, radius } = part.shape {
                assert!(half_height > 0.0 && radius > 0.0);
            }
        }
        let n = g.brain.h as u32;
        assert!(n >= p.brain_min_neurons && n <= p.brain_max_neurons);
    }
}

#[test]
fn neuron_count_stays_within_bounds_under_mutation() {
    let mut p = SimParams::default();
    p.structural_mutation = 1.0;
    p.brain_min_neurons = 3;
    p.brain_max_neurons = 7;
    let mut rng = Rng::new(7);
    let mut g = Genome::random(&mut rng, &p);
    for _ in 0..500 {
        let other = Genome::random(&mut rng, &p);
        g = Genome::crossover(&g, &other, &mut rng);
        g.mutate(&mut rng, &p);
        let b = &g.brain;
        assert!((3..=7).contains(&b.h), "{} neurons", b.h);
        assert_eq!(b.w_in.0.len(), b.h * evolve_sim::brain::N_IN);
        assert_eq!(b.w_rec.0.len(), b.h * b.h);
        assert_eq!(b.w_out.0.len(), b.h * evolve_sim::brain::N_OUT);
        assert_eq!(b.bias.len(), b.h);
    }
}

/// Random bodies with motors are where these sims explode. Drop many and run them for a while.
#[test]
fn random_creatures_stay_physically_stable() {
    let mut sim = Sim::new(quiet_params());
    let mut rng = Rng::new(99);
    for _ in 0..60 {
        let g = Genome::random(&mut rng, &sim.params);
        let at = sim.random_land_point(10.0);
        sim.spawn_creature(g, at, 0.5, f32::MAX, 0, None);
    }
    let spawned = sim.creatures.len();
    let mut max_speed = 0.0f32;
    for _ in 0..(60 * 60) {
        sim.step();
        for c in &sim.creatures {
            for p in &c.parts {
                let b = &sim.world.bodies[p.body];
                assert!(b.translation().is_finite());
                max_speed = max_speed.max(b.linvel().length());
            }
        }
    }
    let glitched = sim.stats.deaths_other;
    println!(
        "alive {}/{spawned}, glitched {glitched}, max part speed {max_speed:.1} m/s, think {:.2} ms, physics {:.2} ms",
        sim.creatures.len(),
        sim.perf.think_ms,
        sim.perf.physics_ms
    );
    assert!(glitched <= 1, "{glitched} creatures blew up or fell through the ground");
    assert!(max_speed <= sim.params.max_body_speed + 1.0);
}

#[test]
fn creatures_die_and_leave_meat() {
    let mut sim = Sim::new(quiet_params());
    let mut rng = Rng::new(3);
    let g = Genome::random(&mut rng, &sim.params);
    let at = sim.random_land_point(10.0);
    let id = sim.spawn_creature(g, at, 0.5, f32::MAX, 0, None);
    sim.step();
    sim.kill(id, DeathCause::Killed);
    let meat_before = sim.meat.len();
    sim.step();
    assert!(sim.creature(id).is_none());
    assert!(sim.meat.len() > meat_before);
    assert_eq!(sim.stats.deaths_killed, 1);
}

#[test]
fn population_rains_back_to_minimum() {
    let mut sim = Sim::new(SimParams {
        min_population: 15,
        rain_interval: 0.1,
        terrain_resolution: 65,
        ..SimParams::default()
    });
    for _ in 0..(60 * 3) {
        sim.step();
    }
    assert!(sim.creatures.len() >= 15, "only {} creatures", sim.creatures.len());
}

#[test]
fn save_and_load_resume_exactly() {
    let mut sim = Sim::new(SimParams {
        min_population: 25,
        rain_interval: 0.05,
        terrain_resolution: 65,
        ..SimParams::default()
    });
    for _ in 0..240 {
        sim.step();
    }
    let dir = std::env::temp_dir().join(format!("evolve-test-{}", std::process::id()));
    let path = dir.join("roundtrip.evo");
    save::save(&mut sim, &path).expect("save");
    let loaded = save::load(&path).expect("load");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(loaded.creatures.len(), sim.creatures.len());
    assert_eq!(loaded.plants.len(), sim.plants.len());
    assert_eq!(loaded.tick, sim.tick);
    for (a, b) in sim.creatures.iter().zip(&loaded.creatures) {
        assert_eq!(a.id, b.id);
        for (pa, pb) in a.parts.iter().zip(&b.parts) {
            let ta = sim.world.bodies[pa.body].translation();
            let tb = loaded.world.bodies[pb.body].translation();
            assert_eq!(ta, tb, "part poses survive the round trip exactly");
        }
    }
    // And the loaded world keeps running.
    let mut loaded = loaded;
    for _ in 0..120 {
        loaded.step();
    }
    let moved = loaded.creatures.iter().any(|c| {
        let t = loaded.world.bodies[c.parts[0].body].translation();
        (t - c.last_pos).length() > 1e-4 || t != Vec3::ZERO
    });
    assert!(moved);
}

#[test]
fn user_commands_work() {
    use evolve_sim::Command;
    use evolve_sim::sim::ObstacleShape;
    let mut sim = Sim::new(quiet_params());
    let mut rng = Rng::new(21);
    let g = Genome::random(&mut rng, &sim.params);
    let at = sim.random_land_point(20.0);
    let id = sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None);
    for _ in 0..120 {
        sim.step();
    }
    // Grab the torso and lift it three meters.
    let torso = sim.creature(id).unwrap().parts[0].body;
    let start = sim.world.bodies[torso].translation();
    let target = start + Vec3::new(2.0, 3.0, 0.0);
    sim.apply(Command::GrabCreature { creature: id, part: 0, local: [0.0; 3], target: target.to_array() });
    for _ in 0..180 {
        sim.step();
    }
    let held = sim.world.bodies[torso].translation();
    assert!((held - target).length() < 0.6, "grabbed creature should hang near the target: {held:?} vs {target:?}");
    sim.apply(Command::Release);
    assert!(sim.grab.is_none());

    let plants = sim.plants.len();
    sim.apply(Command::SpawnPlants { center: at.to_array(), radius: 3.0, count: 5 });
    assert_eq!(sim.plants.len(), plants + 5);
    let pid = sim.plants.last().unwrap().id;
    sim.apply(Command::RemovePlant(pid));
    assert_eq!(sim.plants.len(), plants + 4);

    sim.apply(Command::SpawnObstacle { pos: at.to_array(), shape: ObstacleShape::Ball { radius: 1.0 }, dynamic: true, yaw: 0.0 });
    sim.apply(Command::SpawnObstacle { pos: (at + Vec3::X * 5.0).to_array(), shape: ObstacleShape::Box { half: [1.0, 0.5, 1.0] }, dynamic: false, yaw: 0.3 });
    assert_eq!(sim.obstacles.len(), 2);
    let boulder = sim.obstacles[0].id;
    sim.apply(Command::GrabObstacle { obstacle: boulder, local: [0.0; 3], target: (at + Vec3::Y * 4.0).to_array() });
    assert!(sim.grab.is_some(), "boulders can be grabbed");
    for _ in 0..60 {
        sim.step();
    }
    sim.apply(Command::Release);
    sim.apply(Command::RemoveObstacle(boulder));
    assert_eq!(sim.obstacles.len(), 1);

    sim.creature_mut(id).unwrap().energy = 1.0;
    sim.apply(Command::FeedCreature(id));
    let c = sim.creature(id).unwrap();
    assert_eq!(c.energy, c.max_energy);

    sim.apply(Command::SpawnCreature { pos: None, from: Some(id), count: 2 });
    assert_eq!(sim.creatures.len(), 3);
    sim.apply(Command::RemoveCreature(id));
    sim.step();
    assert!(sim.creature(id).is_none());
    assert_eq!(sim.stats.deaths_other, 1);

    // Raising the water drowns plants that end up underwater.
    let mut p = sim.params.clone();
    p.water_level = 40.0;
    sim.apply(Command::SetParams(Box::new(p)));
    for _ in 0..(60 * 12) {
        sim.step();
    }
    assert!(sim.plants.len() < plants / 4, "flooding kills plants: {} left", sim.plants.len());
}

#[test]
fn regenerate_terrain_keeps_life_above_ground() {
    let mut sim = Sim::new(SimParams { min_population: 10, rain_interval: 0.0, terrain_resolution: 65, ..SimParams::default() });
    for _ in 0..120 {
        sim.step();
    }
    let mut p = sim.params.clone();
    p.seed = 77;
    p.terrain_height = 35.0;
    sim.set_params(p);
    sim.regenerate_terrain();
    for _ in 0..120 {
        sim.step();
    }
    assert_eq!(sim.stats.deaths_other, 0, "nobody should be stuck under the new ground");
    for c in &sim.creatures {
        let t = sim.world.bodies[c.parts[0].body].translation();
        assert!(t.y > sim.terrain.height_at(t.x, t.z) - 1.0);
    }
}

#[test]
fn brain_resize_keeps_surviving_wiring() {
    use evolve_sim::brain::{BrainGenes, N_IN, N_OUT};
    let mut rng = Rng::new(5);
    let b = BrainGenes::random(&mut rng, 10);
    let grown = b.resize(25, &mut rng);
    assert_eq!(grown.h, 25);
    assert_eq!(grown.w_rec.0.len(), 25 * 25);
    // The first 10 neurons keep their wiring among themselves.
    for i in 0..10 {
        assert_eq!(&grown.w_in.0[i * N_IN..(i + 1) * N_IN], &b.w_in.0[i * N_IN..(i + 1) * N_IN]);
        for j in 0..10 {
            assert_eq!(grown.w_rec.0[i * 25 + j], b.w_rec.0[i * 10 + j]);
        }
        for k in 0..N_OUT {
            assert_eq!(grown.w_out.0[k * 25 + i], b.w_out.0[k * 10 + i]);
        }
    }
    let shrunk = grown.resize(4, &mut rng);
    assert_eq!(shrunk.h, 4);
    assert_eq!(shrunk.w_out.0.len(), 4 * N_OUT);
    let child = BrainGenes::crossover(&b, &grown, &mut rng);
    assert!(child.h == 10 || child.h == 25);
    assert_eq!(child.w_rec.0.len(), child.h * child.h);
}

/// The GPU kernels must compute what the CPU reference computes, learning included.
#[test]
fn gpu_brains_match_cpu() {
    use evolve_sim::brain::{self, BrainGenes, BrainState, Learning, N_IN, N_OUT, OutputBias};
    use evolve_sim::gpu::{BatchItem, GpuBrains};
    let mut gpu = match GpuBrains::new() {
        Ok(g) => g,
        Err(e) => {
            println!("no GPU, skipping: {e}");
            return;
        }
    };
    let mut rng = Rng::new(17);
    let sizes = [1usize, 2, 5, 8, 63, 64, 333, 1024];
    let genes: Vec<BrainGenes> = sizes.iter().map(|&h| BrainGenes::random(&mut rng, h)).collect();
    let mut states: Vec<BrainState> = genes.iter().map(BrainState::new).collect();
    for (i, g) in genes.iter().enumerate() {
        gpu.register(100 + i as u64, g, None).expect("register");
    }
    let bias = OutputBias { eat: 0.3, attack: -0.2, mate: 0.1 };
    let mut worst = 0.0f32;
    // Several ticks of learning with fresh exploration noise each tick, so a mismatch anywhere
    // in the traces or learned weights shows up, not just in the forward pass.
    for tick in 0..8 {
        let mut batch = Vec::new();
        for (i, g) in genes.iter().enumerate() {
            let x: [f32; N_IN] = std::array::from_fn(|_| rng.signed());
            let learn = Learning {
                lr: if tick % 3 == 2 { 0.0 } else { 0.05 * rng.signed() },
                trace_decay: 0.9,
                plastic_decay: 0.995,
                pipelined: true,
            };
            let noise: [f32; N_OUT] = std::array::from_fn(|_| 0.2 * rng.signed());
            states[i].noise = noise.to_vec();
            brain::step_cpu(g, bias, &mut states[i], &x, learn);
            batch.push(BatchItem { id: 100 + i as u64, inputs: x, noise, learn, bias });
        }
        gpu.submit(&batch, Some(100 + 6));
        let got = gpu.collect().expect("results");
        assert_eq!(got.ids.len(), genes.len());
        for (i, out) in got.outputs.iter().enumerate() {
            for (a, b) in out.iter().zip(&states[i].outputs) {
                worst = worst.max((a - b).abs());
            }
        }
        let sel = got.selected.expect("selected hidden state");
        assert_eq!(sel.id, 106);
        for (a, b) in sel.hidden.iter().zip(&states[6].hidden) {
            worst = worst.max((a - b).abs());
        }
        for (a, b) in sel.plastic.iter().zip(&states[6].plastic) {
            worst = worst.max((a - b).abs());
        }
    }
    let saved = gpu.read_states();
    for (i, st) in states.iter().enumerate() {
        let s = &saved[&(100 + i as u64)];
        for (a, b) in s.plastic.iter().zip(&st.plastic).chain(s.traces.iter().zip(&st.traces)) {
            worst = worst.max((a - b).abs());
        }
        assert!(st.plastic.iter().any(|p| p.abs() > 1e-4), "the test should have learned something");
        for (a, b) in s.hidden.iter().zip(&st.hidden) {
            worst = worst.max((a - b).abs());
        }
    }
    println!("largest GPU/CPU difference: {worst:e}");
    assert!(worst < 2e-3, "GPU and CPU brains disagree by {worst}");
}

#[test]
fn zoo_stores_dedupes_prunes_and_drops() {
    use evolve_sim::zoo::Zoo;
    let dir = std::env::temp_dir().join(format!("evolve-zoo-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = SimParams::default();
    let mut rng = Rng::new(9);
    let genomes: Vec<Genome> = (0..6).map(|_| Genome::random(&mut rng, &p)).collect();
    {
        let mut zoo = Zoo::open(dir.clone());
        let first = zoo.add(&genomes[0], 5.0, 3, "test", false, 4).unwrap();
        assert!(first.is_some());
        assert!(zoo.add(&genomes[0], 9.0, 3, "test", false, 4).unwrap().is_none(), "same genome stored once");
        zoo.add(&genomes[1], 1.0, 1, "test", true, 4).unwrap(); // pinned, weakest
        for (k, g) in genomes.iter().enumerate().skip(2) {
            zoo.add(g, 2.0 + k as f32, 1, "test", false, 4).unwrap();
        }
        // 4 unpinned allowed plus the pinned one; the weakest unpinned (score 4.0) was pruned.
        assert_eq!(zoo.entries.len(), 5);
        assert!(zoo.entries.iter().any(|e| e.pinned && e.score == 1.0));
        assert!(!zoo.entries.iter().any(|e| e.score == 4.0));
    }
    let zoo = Zoo::open(dir.clone());
    assert_eq!(zoo.entries.len(), 5, "the zoo survives reopening");
    let id = zoo.entries.iter().find(|e| e.score == 5.0).unwrap().id;
    let g = zoo.genome(id).unwrap();
    assert_eq!(g.fingerprint(), genomes[0].fingerprint());

    let mut sim = Sim::new(quiet_params());
    sim.drop_genome(&g, 3, None, 5);
    assert_eq!(sim.creatures.len(), 5);
    // Each copy is a slightly mutated variant of the zoo creature, still able to interbreed.
    assert!(sim.creatures.iter().all(|c| c.generation == 4));
    assert!(sim.creatures.iter().any(|c| c.genome.fingerprint() != g.fingerprint()));
    let t = sim.params.species_threshold;
    assert!(sim.creatures.iter().all(|c| c.genome.distance(&g) < t * 0.5));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Pruned creatures go to the trash, a second process can only read the zoo, and a damaged
/// index is set aside and rebuilt from the genome files.
#[test]
fn zoo_survives_damage_and_a_second_process() {
    use evolve_sim::zoo::Zoo;
    let dir = std::env::temp_dir().join(format!("evolve-zoo-damage-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = SimParams::default();
    let mut rng = Rng::new(10);
    let genomes: Vec<Genome> = (0..4).map(|_| Genome::random(&mut rng, &p)).collect();
    let mut zoo = Zoo::open(dir.clone());
    assert!(!zoo.read_only());
    for (k, g) in genomes.iter().take(3).enumerate() {
        zoo.add(g, k as f32, 1, "test", false, 2).unwrap();
    }
    assert_eq!(zoo.entries.len(), 2);
    let trashed: Vec<_> = std::fs::read_dir(dir.join("trash")).unwrap().flatten().collect();
    assert_eq!(trashed.len(), 1, "the pruned creature is in the trash, not deleted");

    let mut second = Zoo::open(dir.clone());
    assert!(second.read_only(), "the zoo is locked by the first");
    assert_eq!(second.entries.len(), 2, "but can still be read");
    assert!(second.add(&genomes[3], 9.0, 1, "test", false, 10).is_err());
    drop(second);
    drop(zoo);

    std::fs::write(dir.join("index.json"), b"{ not json").unwrap();
    let mut zoo = Zoo::open(dir.clone());
    assert_eq!(zoo.entries.len(), 2, "rebuilt from the genome files");
    let set_aside = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .any(|f| f.file_name().to_string_lossy().starts_with("index.json.bad-"));
    assert!(set_aside, "the damaged index is kept");
    let id = zoo.add(&genomes[3], 9.0, 1, "test", false, 10).unwrap().unwrap();
    assert!(id > 3, "new ids don't reuse ones in the zoo or the trash");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The runner wiring: drop zoo creatures into this and a background world, and add one by hand.
#[test]
fn runner_drops_zoo_creatures_into_worlds() {
    use evolve_sim::zoo::Zoo;
    use evolve_sim::{Control, Runner, ZooTarget};
    let dir = std::env::temp_dir().join(format!("evolve-runner-zoo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: only this test reads EVOLVE_ZOO_DIR.
    unsafe { std::env::set_var("EVOLVE_ZOO_DIR", &dir) };
    let p = quiet_params();
    let mut rng = Rng::new(4);
    let g = Genome::random(&mut rng, &p);
    let id = Zoo::open(dir.clone()).add(&g, 3.0, 7, "test", false, 10).unwrap().unwrap();

    let runner = Runner::start(Sim::new(SimParams { parallel_worlds: 1, ..p }));
    runner.send(Control::Pause(true));
    runner.send(Control::ZooDrop { id, target: ZooTarget::All, pos: None, count: 3 });
    let wait = |cond: &dyn Fn(&evolve_sim::Snapshot) -> bool| {
        let start = std::time::Instant::now();
        while start.elapsed().as_secs() < 30 {
            let s = runner.snapshot();
            if cond(&s) {
                return s;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        panic!("timed out");
    };
    let snap = wait(&|s| s.creatures.len() == 3 && s.worlds.len() == 2 && s.worlds[1].population == 3);
    assert!(snap.zoo.iter().any(|e| e.id == id));
    let cid = snap.creatures[0].id;
    // Dropped copies are mutated variants, so keeping one adds a new (pinned) entry.
    runner.send(Control::ZooAdd(cid));
    wait(&|s| s.zoo.len() == 2 && s.zoo.iter().any(|e| e.pinned && e.id != id));
    drop(runner);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn new_worlds_start_with_zoo_creatures_and_some_random_ones() {
    use evolve_sim::zoo::{Zoo, seed_world};
    let dir = std::env::temp_dir().join(format!("evolve-zoo-seed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let zoo = Zoo::shared(dir.clone());
    let p = SimParams { min_population: 20, terrain_resolution: 65, ..SimParams::default() };
    let mut rng = Rng::new(31);
    for _ in 0..5 {
        let g = Genome::random(&mut rng, &p);
        zoo.lock().unwrap().add(&g, 1.0, 3, "test", false, 100).unwrap();
    }
    let mut sim = Sim::new(p.clone());
    seed_world(&zoo, &mut sim);
    assert_eq!(sim.creatures.len(), 14, "70% of the minimum population from the zoo");
    assert!(!sim.pool.entries.is_empty(), "their genomes seed the gene pool too");
    // The rest rain down as the world runs.
    for _ in 0..60 * 10 {
        sim.step();
    }
    assert!(sim.creatures.len() >= 18);
    let _ = std::fs::remove_dir_all(&dir);
}
