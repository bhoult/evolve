use evolve_sim::body::{self, PartShape};
use evolve_sim::genome::Genome;
use evolve_sim::math::Vec3;
use evolve_sim::rng::Rng;
use evolve_sim::sim::DeathCause;
use evolve_sim::{Sim, SimParams, save};

fn quiet_params() -> SimParams {
    SimParams {
        min_population: 0,
        stones: 0,
        tree_max_count: 0,
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
                let b = &sim.world_of(c).bodies[p.body];
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

/// A body that fell from the sky cost nothing, so it leaves no meat of its own: only half the
/// energy it still held. (A born body leaves what its parents paid for it; see
/// `a_child_killed_at_birth_leaves_no_more_meat_than_it_cost`.) Otherwise the population floor
/// rains free carcasses, and big carrion-eaters that never move win.
#[test]
fn a_killed_body_from_the_sky_leaves_sky_meat_and_half_its_energy() {
    let mut sim = Sim::new(quiet_params());
    let mut rng = Rng::new(3);
    let g = Genome::random(&mut rng, &sim.params);
    let at = sim.random_land_point(10.0);
    let id = sim.spawn_creature(g, at, 0.5, 20.0, 0, None);
    sim.step();
    let c = sim.creature(id).unwrap();
    assert!(c.rained);
    let (held, mass) = (c.energy, c.mass);
    sim.kill(id, DeathCause::Killed);
    let meat = |sim: &Sim| sim.meat.iter().map(|m| m.energy).sum::<f32>();
    let before = meat(&sim);
    sim.step();
    let left = meat(&sim) - before;
    let want = mass * sim.params.sky_meat_per_mass + held * 0.5;
    assert!(left <= want + 1e-3 && left > 0.9 * want, "left {left} of meat, want {want} ({mass} kg, holding {held})");
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
    // Saved in separate sections, and put back in the world after saving.
    assert_eq!(loaded.shards.len(), sim.shards.len());
    assert!(sim.terrain.res > 2 && loaded.terrain.heights == sim.terrain.heights);
    assert!(loaded.terrain.water == sim.terrain.water && loaded.terrain.fertility == sim.terrain.fertility);
    assert_eq!(loaded.pool.entries.len(), sim.pool.entries.len());
    for (a, b) in sim.creatures.iter().zip(&loaded.creatures) {
        assert_eq!(a.id, b.id);
        for (pa, pb) in a.parts.iter().zip(&b.parts) {
            let ta = sim.world_of(a).bodies[pa.body].translation();
            let tb = loaded.world_of(b).bodies[pb.body].translation();
            assert_eq!(ta, tb, "part poses survive the round trip exactly");
        }
    }
    // And the loaded world keeps running.
    let mut loaded = loaded;
    for _ in 0..120 {
        loaded.step();
    }
    let moved = loaded.creatures.iter().any(|c| {
        let t = loaded.world_of(c).bodies[c.parts[0].body].translation();
        (t - c.last_pos).length() > 1e-4 || t != Vec3::ZERO
    });
    assert!(moved);
}

#[test]
fn saves_from_before_sections_still_load() {
    use std::io::Write;
    let mut sim = Sim::new(SimParams {
        min_population: 10,
        terrain_resolution: 65,
        ..SimParams::default()
    });
    for _ in 0..60 {
        sim.step();
    }
    // EVOLVE06: the whole world as one gzipped stream of MessagePack.
    let path = std::env::temp_dir().join(format!("evolve-old-{}.evo", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"EVOLVE06").unwrap();
    let mut gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
    gz.write_all(&rmp_serde::to_vec_named(&sim).unwrap()).unwrap();
    gz.finish().unwrap();
    let loaded = save::load(&path).expect("load");
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.tick, sim.tick);
    assert_eq!(loaded.creatures.len(), sim.creatures.len());
    assert!(loaded.terrain.heights == sim.terrain.heights);
}

#[test]
fn terrain_layers_saved_as_numbers_still_load() {
    use evolve_sim::terrain::Layer;
    let samples = vec![0.5f32, -2.25, 1e6, 0.0];
    // How saves before EVOLVE07 held a layer: a list of numbers.
    let old: Layer = rmp_serde::from_slice(&rmp_serde::to_vec_named(&samples).unwrap()).unwrap();
    assert_eq!(*old, samples);
    let layer = Layer::from(samples.clone());
    let now: Layer = rmp_serde::from_slice(&rmp_serde::to_vec_named(&layer).unwrap()).unwrap();
    assert_eq!(*now, samples);
    let json: Layer = serde_json::from_str(&serde_json::to_string(&layer).unwrap()).unwrap();
    assert_eq!(*json, samples, "readable formats keep a list");
}

#[test]
fn experiment_arms_share_an_island_and_take_their_own_settings() {
    use evolve_sim::islands::trial_params;
    let main = SimParams::default();
    let set = vec![("score_child".to_string(), "2.5".to_string())];
    let control = trial_params(&main, 0, &[], 4).unwrap();
    let arm = trial_params(&main, 0, &set, 4).unwrap();
    let other = trial_params(&main, 1, &[], 4).unwrap();
    assert_eq!(control.seed, arm.seed, "the arms of a replicate share an island");
    assert_ne!(control.seed, other.seed, "each replicate has its own");
    assert_eq!(control.score_child, main.score_child);
    assert_eq!(arm.score_child, 2.5);
    assert_eq!(arm.parallel_worlds, 4, "for their shares of GPU brain memory");
    assert!(arm.max_population < main.max_population, "background-sized");
    assert!(trial_params(&main, 0, &[("no_such_setting".into(), "1".into())], 4).is_err());
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
    let torso = |sim: &Sim| sim.torso_of(sim.creature(id).unwrap()).unwrap().translation();
    let start = torso(&sim);
    let target = start + Vec3::new(2.0, 3.0, 0.0);
    sim.apply(Command::GrabCreature { creature: id, part: 0, local: [0.0; 3], target: target.to_array() });
    for _ in 0..180 {
        sim.step();
    }
    let held = torso(&sim);
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
    // (Counted from here: now and then a random body flies apart on its own, unrelated.)
    let lost = sim.stats.deaths_other;
    sim.regenerate_terrain();
    for _ in 0..120 {
        sim.step();
    }
    assert_eq!(sim.stats.deaths_other, lost, "nobody should be stuck under the new ground");
    for c in &sim.creatures {
        let t = sim.world_of(c).bodies[c.parts[0].body].translation();
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
    let mut genomes: Vec<Genome> = (0..6).map(|_| Genome::random(&mut rng, &p)).collect();
    // The fourth is the third's kind (the same body, a brain apart), so it's the pair to thin.
    genomes[3] = genomes[2].clone();
    genomes[3].brain.bias[0] += 1.0;
    {
        let mut zoo = Zoo::open(dir.clone());
        let first = zoo.add(&genomes[0], 5.0, 3, "test", false, 4).unwrap();
        assert!(first.is_some());
        assert!(zoo.add(&genomes[0], 9.0, 3, "test", false, 4).unwrap().is_none(), "same genome stored once");
        zoo.add(&genomes[1], 1.0, 1, "test", true, 4).unwrap(); // pinned, weakest
        for (k, g) in genomes.iter().enumerate().skip(2) {
            zoo.add(g, 2.0 + k as f32, 1, "test", false, 4).unwrap();
        }
        // 4 unpinned allowed plus the pinned one; the weaker of the two most alike (score 4.0)
        // was pruned.
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

/// The zoo can be cut down to the best of its most different kinds (the pinned stay), and single
/// creatures deleted; both go to the trash, and the zoo stays that way when reopened.
#[test]
fn the_zoo_can_be_cut_down_to_its_best_kinds_and_creatures_deleted() {
    use evolve_sim::zoo::Zoo;
    let dir = std::env::temp_dir().join(format!("evolve-zoo-keep-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = SimParams::default();
    let mut rng = Rng::new(11);
    let mut zoo = Zoo::open(dir.clone());
    // One kind in three variants (the same body, brains apart) holds the three best scores; two
    // other kinds score less, and a pinned one least of all.
    let kind = Genome::random(&mut rng, &p);
    for score in [7.0, 8.0, 9.0] {
        let mut g = kind.clone();
        g.brain.bias[0] += score;
        zoo.add(&g, score, 1, "test", false, 100).unwrap();
    }
    for score in [2.0, 1.0] {
        zoo.add(&Genome::random(&mut rng, &p), score, 1, "test", false, 100).unwrap();
    }
    zoo.add(&Genome::random(&mut rng, &p), 0.0, 1, "test", true, 100).unwrap();
    assert_eq!(zoo.keep_best(3).unwrap(), 2, "five unpinned, three kept");
    let mut scores: Vec<f32> = zoo.entries.iter().map(|e| e.score).collect();
    scores.sort_by(f32::total_cmp);
    assert_eq!(scores, [0.0, 1.0, 2.0, 9.0], "the best of each kind, not the three best, and the pinned one");
    assert_eq!(std::fs::read_dir(dir.join("trash")).unwrap().count(), 2, "the rest are in the trash");
    let best = zoo.entries.iter().find(|e| e.score == 9.0).unwrap().id;
    zoo.remove(best).unwrap();
    assert!(zoo.get(best).is_none());
    drop(zoo);
    let zoo = Zoo::open(dir.clone());
    assert_eq!(zoo.entries.len(), 3, "and it stays that way");
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

/// The world's physics is split into several worlds stepped in parallel: creatures that might
/// touch are always in the same one (fights come from contacts), and the load spreads over them.
#[test]
fn creatures_that_might_touch_share_a_physics_world() {
    let mut p = quiet_params();
    p.physics_shards = 6;
    let mut sim = Sim::new(p.clone());
    let mut rng = Rng::new(21);
    for _ in 0..60 {
        let g = Genome::random(&mut rng, &p);
        let at = sim.random_land_point(10.0);
        sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None);
    }
    // A huddle of four.
    let at = sim.random_land_point(20.0);
    for k in 0..4 {
        let g = Genome::random(&mut rng, &p);
        sim.spawn_creature(g, at + Vec3::new(k as f32 * 0.8, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    }
    // Each creature's parts as spheres around them.
    let spheres = |sim: &Sim, c: &evolve_sim::sim::Creature| -> Vec<(Vec3, f32)> {
        c.parts
            .iter()
            .zip(&c.plan)
            .map(|(q, spec)| (sim.world_of(c).bodies[q.body].translation(), spec.shape.bounding_radius()))
            .collect()
    };
    let mut touching = 0;
    for _ in 0..240 {
        sim.step();
        let alive: Vec<_> = sim.creatures.iter().collect();
        for (i, a) in alive.iter().enumerate() {
            let sa = spheres(&sim, a);
            for b in &alive[i + 1..] {
                let sb = spheres(&sim, b);
                let touch = sa.iter().any(|(pa, ra)| sb.iter().any(|(pb, rb)| pa.distance(*pb) < ra + rb));
                if touch {
                    touching += 1;
                    assert_eq!(a.shard, b.shard, "creatures whose parts touch share a physics world");
                }
            }
        }
    }
    assert!(touching > 0, "the huddle should have touched");
    let mut used = vec![0; 6];
    for c in &sim.creatures {
        used[c.shard] += 1;
    }
    eprintln!("creatures per physics world: {used:?}");
    assert!(used.iter().all(|&n| n > 0), "every world gets some: {used:?}");
    let (lo, hi) = (*used.iter().min().unwrap(), *used.iter().max().unwrap());
    assert!(hi <= lo * 3 + 4, "and the load is fairly even: {used:?}");
}

#[test]
fn a_summoned_storm_builds_rages_survives_a_save_and_dies_away() {
    use evolve_sim::Command;
    let mut p = quiet_params();
    p.weather = false;
    let mut sim = Sim::new(p);
    let run = |sim: &mut Sim, seconds: f32| {
        for _ in 0..(seconds * evolve_sim::sim::TICKS_PER_SECOND as f32) as u32 {
            sim.step();
        }
    };
    assert_eq!(sim.weather.rain, 0.0, "fair weather to begin with");
    sim.apply(Command::SummonStorm { seconds: 200.0 });
    run(&mut sim, 40.0);
    assert!(sim.weather.storm > 0.9 && sim.weather.rain > 0.99, "a full storm: {:?}", sim.weather);
    let wind = sim.weather.wind[0].hypot(sim.weather.wind[1]);
    assert!(wind > 17.0, "and a gale: {wind} m/s");
    // A saved world keeps its storm.
    let path = std::env::temp_dir().join(format!("evolve-storm-{}.evo", std::process::id()));
    save::save(&mut sim, &path).unwrap();
    let mut back = save::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(back.weather.storm > 0.9, "the storm is still raging after loading: {:?}", back.weather);
    back.apply(Command::CalmStorm);
    run(&mut back, 50.0);
    assert_eq!(back.weather.rain, 0.0, "calmed, it dies away: {:?}", back.weather);
    assert!(back.acts.storm.is_none(), "and is forgotten");
}

#[test]
fn a_meteor_digs_a_crater_kills_what_it_hits_and_throws_what_is_near() {
    use evolve_sim::Command;
    use evolve_sim::acts::{BLAST, CRATER};
    let mut p = quiet_params();
    p.weather = false;
    p.wind_force = 0.0;
    p.terrain_resolution = SimParams::default().terrain_resolution;
    let mut sim = Sim::new(p);
    let at = sim.random_land_point(30.0);
    let before = sim.terrain.heights.clone();
    let h0 = sim.terrain.height_at(at.x, at.z);
    // A legged body near it, and one right where it lands.
    let mut rng = Rng::new(5);
    let legged = loop {
        let g = Genome::random(&mut rng, &SimParams::default());
        if g.limbs.iter().filter(|l| l.present).count() >= 2 {
            break g;
        }
    };
    let near = sim.spawn_creature(legged.clone(), at + Vec3::new(8.0, 0.0, 0.0), 0.2, f32::MAX, 0, None);
    let hit = sim.spawn_creature(legged, at, 0.2, f32::MAX, 0, None);
    for _ in 0..120 {
        sim.step();
    }
    let pos = |sim: &Sim, id: u64| sim.torso_of(sim.creature(id).unwrap()).unwrap().translation();
    let health = sim.creature(near).unwrap().health;
    let v0 = sim.torso_of(sim.creature(near).unwrap()).unwrap().linvel();
    sim.apply(Command::Meteor { at: at.to_array() });
    // (A body part's velocity is refreshed from its multibody's at the start of the next step.)
    sim.step();
    sim.step();
    // The crater: the middle is lower, the rim higher, and no earth was made or lost.
    let h1 = sim.terrain.height_at(at.x, at.z);
    assert!(h1 < h0 - 1.0, "the middle should be dug out: {h0} -> {h1}");
    let rim = (0..16)
        .map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 16.0;
            let (x, z) = (at.x + a.cos() * CRATER * 1.35, at.z + a.sin() * CRATER * 1.35);
            sim.terrain.height_at(x, z) - before_height(&before, &sim, x, z)
        })
        .sum::<f32>()
        / 16.0;
    assert!(rim > 0.1, "the rim should be piled up: {rim}");
    let change: f32 = sim.terrain.heights.iter().zip(&before).map(|(a, b)| a - b).sum();
    let cell = sim.terrain.cell();
    assert!((change * cell * cell).abs() < 0.5, "earth is conserved: {} m³", change * cell * cell);
    // What it landed on is dead; what was near was thrown up and away, and hurt.
    assert!(sim.creature(hit).is_none(), "the creature it hit should be dead");
    let c = sim.creature(near).unwrap();
    assert!(c.health < health, "the near one is hurt");
    let v = sim.torso_of(c).unwrap().linvel() - v0;
    let f = 1.0 - (pos(&sim, near) - at).length().min(BLAST) / BLAST;
    eprintln!("thrown at {v:?} (expected about {:.1} out, {:.1} up)", 16.0 * f, 9.0 * f);
    assert!(v.x > 3.0 && v.y > 1.5, "thrown up and away: {v:?}");
    // A second strike on the same spot can't dig below the limit.
    sim.apply(Command::Meteor { at: at.to_array() });
    sim.apply(Command::Meteor { at: at.to_array() });
    let h2 = sim.terrain.height_at(at.x, at.z);
    let original = before_height(&before, &sim, at.x, at.z);
    assert!(h2 >= original - sim.params.dig_depth - 0.05, "dug no deeper than the limit: {h2} vs {original}");
}

/// The ground height at (x, z) in a copy of the heights from before.
fn before_height(before: &[f32], sim: &Sim, x: f32, z: f32) -> f32 {
    let mut t = sim.terrain.clone();
    t.heights = before.to_vec().into();
    t.height_at(x, z)
}

#[test]
fn a_whirlwind_lifts_creatures_tears_up_plants_and_dies_away() {
    use evolve_sim::Command;
    use evolve_sim::acts::WHIRL_LIFE;
    let mut p = quiet_params();
    p.weather = false;
    p.wind_force = 0.0;
    p.plant_spawn_rate = 0.0;
    let mut sim = Sim::new(p);
    let at = sim.random_land_point(40.0);
    let mut rng = Rng::new(9);
    let g = Genome::random(&mut rng, &SimParams::default());
    let id = sim.spawn_creature(g, at, 0.2, f32::MAX, 0, None);
    for k in 0..6 {
        let a = k as f32;
        sim.add_plant(at + Vec3::new(a.cos() * 1.5, 0.0, a.sin() * 1.5), 10.0);
    }
    for _ in 0..60 {
        sim.step();
    }
    let ground_gap = |sim: &Sim| {
        let t = sim.torso_of(sim.creature(id).unwrap()).unwrap().translation();
        t.y - sim.terrain.height_at(t.x, t.z)
    };
    let rest = ground_gap(&sim);
    sim.apply(Command::Whirlwind { at: at.to_array() });
    let mut highest: f32 = 0.0;
    for _ in 0..(8 * evolve_sim::sim::TICKS_PER_SECOND) {
        sim.step();
        if sim.creature(id).is_some() {
            highest = highest.max(ground_gap(&sim) - rest);
        }
    }
    eprintln!("lifted {highest:.1} m; {} plants left near it", sim.plants.iter().filter(|f| f.pos.distance(at) < 2.5).count());
    assert!(highest > 2.0, "the whirlwind should lift it off the ground: {highest:.2} m");
    assert!(highest < 25.0, "but not carry it off into the sky: {highest:.2} m");
    assert!(sim.plants.iter().all(|f| f.pos.distance(at) > 2.5), "the plants where it touched down are torn up");
    assert_eq!(sim.acts.whirlwinds.len(), 1);
    let w = &sim.acts.whirlwinds[0];
    assert!(sim.terrain.contains(w.x, w.z, 5.0), "it stays in the world");
    for _ in 0..((WHIRL_LIFE as u64 - 7) * evolve_sim::sim::TICKS_PER_SECOND) {
        sim.step();
    }
    assert!(sim.acts.whirlwinds.is_empty(), "and dies away");
}

#[test]
fn the_land_can_be_raised_lowered_split_and_added_to_and_springs_run() {
    use evolve_sim::Command;
    let mut p = quiet_params();
    p.weather = false;
    p.terrain_resolution = SimParams::default().terrain_resolution;
    let mut sim = Sim::new(p);
    let at = sim.random_land_point(30.0);
    let h = |sim: &Sim, x: f32, z: f32| sim.terrain.height_at(x, z);
    // Raised, then lowered back.
    let h0 = h(&sim, at.x, at.z);
    sim.apply(Command::Terraform { at: at.to_array(), radius: 6.0, amount: 2.0 });
    let h1 = h(&sim, at.x, at.z);
    assert!((h1 - h0 - 2.0).abs() < 0.4, "raised by about 2 m: {h0} -> {h1}");
    sim.apply(Command::Terraform { at: at.to_array(), radius: 6.0, amount: -2.0 });
    assert!((h(&sim, at.x, at.z) - h0).abs() < 0.1, "and lowered back");

    // A spring wells up (running off, or filling the hollow it's in: raised and lowered, the
    // ground isn't quite as it was), and stops when told.
    sim.apply(Command::Spring { at: at.to_array() });
    assert_eq!(sim.terrain.springs.len(), 1);
    for _ in 0..(10 * evolve_sim::sim::TICKS_PER_SECOND) {
        sim.step();
    }
    let wet = sim.terrain.water[sim.terrain.springs[0] as usize];
    assert!(wet > 0.02, "water wells up at the spring: {wet} m");
    sim.apply(Command::Spring { at: (at + Vec3::new(1.0, 0.0, 0.0)).to_array() });
    assert!(sim.terrain.springs.is_empty(), "clicking it again stops it");

    // An earthquake splits the land along its line and shakes, then calms.
    let line = [[at.x - 15.0, at.z], [at.x - 5.0, at.z + 1.0], [at.x + 5.0, at.z - 1.0], [at.x + 15.0, at.z]];
    let before = h(&sim, at.x, at.z);
    sim.apply(Command::Earthquake { path: line.to_vec() });
    assert!(h(&sim, at.x, at.z) < before - 1.0, "a crack opens: {before} -> {}", h(&sim, at.x, at.z));
    assert!(sim.acts.quake.is_some());
    for _ in 0..(7 * evolve_sim::sim::TICKS_PER_SECOND) {
        sim.step();
    }
    assert!(sim.acts.quake.is_none(), "the shaking stops");

    // Land fallen into the sea stands out of it.
    let wl = sim.params.water_level;
    let sea = (0..400)
        .map(|k| {
            let a = k as f32 * 0.37;
            let r = 20.0 + (k as f32 * 0.21) % (sim.terrain.half() - 30.0);
            Vec3::new(a.cos() * r, 0.0, a.sin() * r)
        })
        .find(|p| sim.terrain.height_at(p.x, p.z) < wl - 1.0 && sim.terrain.contains(p.x, p.z, 15.0))
        .expect("some sea");
    sim.apply(Command::Landfall { at: sea.to_array() });
    assert!(h(&sim, sea.x, sea.z) > wl + 1.0, "new land out of the sea: {}", h(&sim, sea.x, sea.z));
}

#[test]
fn a_tsunami_carries_creatures_inland_washes_plants_away_and_passes() {
    use evolve_sim::Command;
    use evolve_sim::acts::TSUNAMI_LIFE;
    let mut p = quiet_params();
    p.weather = false;
    p.wind_force = 0.0;
    p.plant_spawn_rate = 0.0;
    let mut sim = Sim::new(p);
    let wl = sim.params.water_level;
    // Low ground near the shore, well inside the world.
    let shore = (0..2000)
        .map(|k| {
            let a = k as f32 * 0.0731;
            let r = 15.0 + (k as f32 * 0.37) % (sim.terrain.half() - 25.0);
            Vec3::new(a.cos() * r, 0.0, a.sin() * r)
        })
        .find(|q| {
            let h = sim.terrain.height_at(q.x, q.z);
            h > wl + 0.3 && h < wl + 2.0 && sim.terrain.contains(q.x, q.z, 20.0)
        })
        .expect("a low shore");
    let id = sim.spawn_creature(Genome::random(&mut Rng::new(3), &SimParams::default()), shore, 0.2, f32::MAX, 0, None);
    sim.add_plant(shore + Vec3::new(1.5, 0.0, 0.0), 10.0);
    for _ in 0..60 {
        sim.step();
    }
    let start = sim.torso_of(sim.creature(id).unwrap()).unwrap().translation();
    sim.apply(Command::Tsunami { at: shore.to_array(), toward: [1.0, 0.0, 0.0] });
    let dir = sim.acts.tsunami.as_ref().map(|t| Vec3::new(t.dir[0], 0.0, t.dir[1])).unwrap();
    let (mut farthest, mut swept): (f32, f32) = (0.0, 0.0);
    for _ in 0..((TSUNAMI_LIFE as u64 + 1) * evolve_sim::sim::TICKS_PER_SECOND) {
        sim.step();
        if let Some(c) = sim.creature(id) {
            let now = sim.torso_of(c).unwrap().translation();
            farthest = farthest.max((now - start).dot(dir));
            swept = swept.max(sim.torso_of(c).unwrap().linvel().dot(dir));
        }
    }
    eprintln!("swept along at up to {swept:.1} m/s, carried {farthest:.1} m inland");
    // (How far depends on the shore: the flood runs up only as high as it reaches.)
    assert!(swept > 5.0, "the wave should sweep it along: {swept:.1} m/s");
    assert!(farthest > 3.0, "and carry it inland: {farthest:.1} m");
    assert!(sim.plants.iter().all(|f| f.pos.distance(shore) > 3.0), "the plant by the shore is washed away");
    assert!(sim.acts.tsunami.is_none(), "and the wave is spent");
}

/// Worlds offer the zoo their best of each kind: a creature much like one already there takes
/// its place only if it scored better, a different one joins as a new kind, and over the cap the
/// weaker of the two most alike goes.
#[test]
fn the_zoo_keeps_the_best_of_each_kind() {
    use evolve_sim::zoo::Zoo;
    let dir = std::env::temp_dir().join(format!("evolve-zoo-kinds-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = SimParams::default();
    let kind = p.zoo_kind_distance;
    let mut rng = Rng::new(31);
    let a = Genome::random(&mut rng, &p);
    let mut a2 = a.clone();
    a2.torso.half[0] *= 1.02;
    assert!(a.distance(&a2) < kind);
    let b = Genome::random(&mut rng, &p);
    assert!(a.distance(&b) > kind, "two random genomes are different kinds: {}", a.distance(&b));
    // Like b, but just past the line: a kind of its own, though the nearest one to b.
    let mut c = b.clone();
    c.traits.diet = if b.traits.diet < 0.5 { 1.0 } else { 0.0 };
    c.traits.armor = if b.traits.armor < 0.5 { 1.0 } else { 0.0 };
    c.traits.aggression = -b.traits.aggression.signum();
    let bc = b.distance(&c);
    assert!(bc > kind && bc < a.distance(&b) && bc < a.distance(&c), "c is its own kind, nearest b: {bc}");

    let mut zoo = Zoo::open(dir.clone());
    assert!(zoo.offer(&a, 5.0, 1, "test", 10, kind).unwrap().is_some());
    assert!(zoo.offer(&a2, 4.0, 1, "test", 10, kind).unwrap().is_none(), "a weaker one of the same kind is turned away");
    assert!(zoo.offer(&a2, 6.0, 1, "test", 10, kind).unwrap().is_some(), "a better one of the same kind gets in");
    assert_eq!(zoo.entries.len(), 1, "in place of the one it's like");
    assert_eq!(zoo.entries[0].score, 6.0);
    assert!(zoo.offer(&b, 1.0, 1, "test", 10, kind).unwrap().is_some(), "a different kind joins, however it scored");
    assert_eq!(zoo.entries.len(), 2);
    // Over a cap of two, the weaker of the two most alike (b and c) goes, not the weakest kind.
    assert!(zoo.offer(&c, 3.0, 1, "test", 2, kind).unwrap().is_some());
    let scores: Vec<f32> = zoo.entries.iter().map(|e| e.score).collect();
    assert_eq!(scores, vec![6.0, 3.0], "b went: {scores:?}");
    drop(zoo);
    let zoo = Zoo::open(dir.clone());
    assert!(zoo.entries.iter().all(|e| e.shape.is_some()), "shapes are kept in the index");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Creatures fall from the zoo in proportion to `rain_from_zoo`, as mutated copies of the
/// zoo creatures a world was handed; without any, the gene pool stands in and says so.
#[test]
fn creatures_fall_from_the_zoo_when_asked() {
    use evolve_sim::sim::Origin;
    let p = SimParams { rain_from_world: 0.0, rain_from_zoo: 1.0, rain_from_random: 0.0, ..quiet_params() };
    let mut sim = Sim::new(p);
    let g = Genome::random(&mut Rng::new(8), &sim.params);
    sim.zoo_sample = vec![(g.clone(), 3)];
    let ids: Vec<u64> = (0..8).map(|_| sim.rain_creature(None)).collect();
    for id in ids {
        let c = sim.creature(id).unwrap();
        assert_eq!(c.origin, Origin::Zoo);
        assert_eq!(c.generation, 4);
        assert!(c.genome.distance(&g) < sim.params.species_threshold, "a copy, mutated a little");
    }
    assert_eq!(sim.stats.rain_without_zoo, 0);
    sim.zoo_sample.clear();
    let id = sim.rain_creature(None);
    assert_ne!(sim.creature(id).unwrap().origin, Origin::Zoo);
    assert_eq!(sim.stats.rain_without_zoo, 1, "counted, for the report");
    assert!(sim.report(60.0).contains("came from the gene pool instead"));
}

/// A world takes its sample of zoo creatures to rain from out of the zoo, and again when the
/// zoo changes.
#[test]
fn worlds_take_a_sample_of_the_zoo_to_rain_from() {
    use evolve_sim::zoo::{Zoo, refresh_rain_sample};
    let dir = std::env::temp_dir().join(format!("evolve-zoo-sample-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let zoo = Zoo::shared(dir.clone());
    let p = SimParams::default();
    let mut rng = Rng::new(5);
    for k in 0..3 {
        zoo.lock().unwrap().offer(&Genome::random(&mut rng, &p), k as f32, 1, "test", 100, p.zoo_kind_distance).unwrap();
    }
    let mut sim = Sim::new(SimParams { rain_from_zoo: 0.5, ..quiet_params() });
    refresh_rain_sample(&zoo, &mut sim);
    assert_eq!(sim.zoo_sample.len(), 3);
    zoo.lock().unwrap().offer(&Genome::random(&mut rng, &p), 9.0, 1, "test", 100, p.zoo_kind_distance).unwrap();
    refresh_rain_sample(&zoo, &mut sim);
    assert_eq!(sim.zoo_sample.len(), 4, "the zoo changed, so the sample is taken again");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The rain brings the population up to the minimum however fast creatures die, eases off past
/// it, and never takes it past the maximum.
#[test]
fn rain_fills_to_the_minimum_and_eases_off_before_the_maximum() {
    let mut sim = Sim::new(SimParams {
        min_population: 30,
        max_population: 45,
        // Creatures that starve within seconds: the rain has to keep replacing them.
        start_energy: 15.0,
        terrain_resolution: 65,
        weather: false,
        stones: 0,
        ..SimParams::default()
    });
    let mut most = 0;
    for s in 0..60 {
        for _ in 0..60 {
            sim.step();
            most = most.max(sim.creatures.len());
        }
        if s == 40 {
            assert!(sim.creatures.len() >= 30, "at 40 s only {} alive", sim.creatures.len());
        }
    }
    assert!(sim.creatures.len() >= 30, "{} alive at the end", sim.creatures.len());
    assert!(most <= 45, "it rained past the maximum: {most}");
    assert!(sim.stats.deaths() > 30, "they did die: {}", sim.stats.deaths());
}

/// The step spreads creatures' work over threads (senses, muscles, forces, after physics), but
/// the world comes out the same however many threads there are: two worlds stepped side by side,
/// one on a single thread and one on eight, match to the bit.
#[test]
fn the_world_is_the_same_on_one_thread_or_many() {
    let params = SimParams { min_population: 24, max_population: 40, rain_fill_time: 2.0, ..quiet_params() };
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let mut sim = Sim::new(params.clone());
            for _ in 0..600 {
                sim.step();
            }
            sim.creatures
                .iter()
                .map(|c| {
                    let t = sim.torso_of(c).unwrap().translation();
                    (c.id, c.energy.to_bits(), c.health.to_bits(), [t.x.to_bits(), t.y.to_bits(), t.z.to_bits()])
                })
                .collect::<Vec<_>>()
        })
    };
    let (one, many) = (run(1), run(8));
    assert!(one.len() >= 20, "a world full of creatures: {}", one.len());
    assert_eq!(one, many);
}

/// A creature with the blessing of heaven can't die of hunger, changes when its hour comes (a
/// new genome, the same creature), and dies like any other once the blessing is lifted.
#[test]
fn the_blessing_of_heaven_keeps_a_creature_alive_and_changes_it() {
    use evolve_sim::Command;
    let p = SimParams { heaven_mutation_interval: 10.0, mutation_rate: 1.0, ..quiet_params() };
    let mut sim = Sim::new(p);
    let g = Genome::random(&mut Rng::new(12), &sim.params);
    let at = sim.random_land_point(20.0);
    let id = sim.spawn_creature(g, at, 0.3, 50.0, 0, None);
    let before = sim.creature(id).unwrap().genome.fingerprint();
    sim.apply(Command::Heaven { id, on: true });
    for _ in 0..(15 * 60) {
        sim.creature_mut(id).unwrap().energy = 0.0;
        sim.step();
    }
    let c = sim.creature(id).expect("it lives on without food");
    assert!(c.heaven.is_some());
    assert_ne!(c.genome.fingerprint(), before, "its hour came and it changed");
    assert_eq!(c.parts.len(), c.plan.len(), "with a body to match its new genome");
    sim.apply(Command::Heaven { id, on: false });
    sim.creature_mut(id).unwrap().energy = 0.0;
    for _ in 0..60 {
        sim.step();
    }
    assert!(sim.creature(id).is_none(), "without the blessing it starves");
}

/// What a background world hands on is marked as its, crosses with lines from here halve the
/// mark, and the report then tells the two apart.
#[test]
fn ancestry_from_background_worlds_is_traced() {
    use evolve_sim::islands::{Board, migrate};
    use evolve_sim::stats::GenePool;
    let p = quiet_params();
    let mut rng = Rng::new(12);
    let mut there = Genome::random(&mut rng, &p);
    let here = Genome::random(&mut rng, &p);
    assert_eq!(here.foreign, 0.0, "made from scratch here");

    let board: Board = std::sync::Arc::new(std::sync::Mutex::new(GenePool::default()));
    let mut background = GenePool::default();
    background.consider(5.0, 3, &there);
    migrate(&mut background, &board, 0.0, true);
    let mut world0 = GenePool::default();
    migrate(&mut world0, &board, 60.0, false);
    assert_eq!(world0.entries.len(), 1);
    assert_eq!(world0.entries[0].genome.foreign, 1.0, "came in from a background world");

    there.foreign = 1.0;
    assert_eq!(Genome::crossover(&there, &here, &mut rng).foreign, 0.5);

    let mut sim = Sim::new(quiet_params());
    assert!(!sim.report(60.0).contains("Ancestry from background worlds"), "nothing to say yet");
    assert!(!sim.summary_line(60.0).contains("background ancestry"));
    sim.pool = world0;
    assert!(sim.report(60.0).contains("Ancestry from background worlds"));
    assert!(sim.summary_line(60.0).contains("background ancestry: pool 100% (1 of 1 mostly)"));
}
