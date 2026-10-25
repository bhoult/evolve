//! Diagnostics for tuning, not correctness: `cargo test --release --test diag -- --ignored --nocapture`.

use evolve_sim::body;
use evolve_sim::genome::Genome;
use evolve_sim::math::Vec3;
use evolve_sim::rng::Rng;
use evolve_sim::{Sim, SimParams};

#[test]
#[ignore]
fn locomotion_of_random_creatures() {
    let mut sim = Sim::new(SimParams {
        min_population: 0,
        terrain_height: 0.0,
        water_level: -5.0,
        terrain_resolution: 65,
        ..SimParams::default()
    });
    let mut rng = Rng::new(5);
    let mut starts = Vec::new();
    for i in 0..40 {
        let g = Genome::random(&mut rng, &sim.params);
        let at = Vec3::new((i % 8) as f32 * 12.0 - 42.0, 0.0, (i / 8) as f32 * 12.0 - 24.0);
        let id = sim.spawn_creature(g, at, 0.2, f32::MAX, 0, None);
        starts.push((id, at));
    }
    let m = evolve_sim::brain::N_MOTORS;
    let mut ranges = vec![(f32::MAX, f32::MIN); 40 * m];
    let e0: Vec<f32> = sim.creatures.iter().map(|c| c.energy).collect();
    for _ in 0..(60 * 30) {
        sim.step();
        for (ci, c) in sim.creatures.iter().enumerate() {
            for j in &c.joints {
                if let Some((a, _)) = body::joint_state(sim.world_of(c), &c.parts, j) {
                    let r = &mut ranges[ci * m + j.slot];
                    r.0 = r.0.min(a[0]);
                    r.1 = r.1.max(a[0]);
                }
            }
        }
    }
    let mut disp = Vec::new();
    for (ci, c) in sim.creatures.iter().enumerate() {
        let t = sim.world_of(c).bodies[c.parts[0].body].translation();
        let start = starts.iter().find(|s| s.0 == c.id).unwrap().1;
        let d = Vec3::new(t.x - start.x, 0.0, t.z - start.z).length();
        let amp: f32 = c
            .joints
            .iter()
            .map(|j| {
                let r = ranges[ci * m + j.slot];
                (r.1 - r.0) / (j.range[1] - j.range[0])
            })
            .sum::<f32>()
            / c.joints.len().max(1) as f32;
        let torso_mass = c.plan[0].shape.volume() * sim.params.body_density;
        let spent = e0[ci] - c.energy;
        let basal = sim.params.basal_metabolism * c.mass.powf(0.75) * 30.0;
        println!(
            "mass {:6.1} torso {:5.1} limbs {:2} neurons {:2} disp {:6.2} m  joint swing {:4.2} of range  energy spent {:6.1} (basal ~{:5.1}) upright {:5.2}",
            c.mass,
            torso_mass,
            c.parts.len() - 1,
            c.genome.brain.h,
            d,
            amp,
            spent,
            basal,
            (sim.world_of(c).bodies[c.parts[0].body].rotation() * Vec3::Y).y
        );
        disp.push(d);
    }
    disp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "displacement in 30 s: median {:.2} m, best {:.2} m, top5 {:?}",
        disp[disp.len() / 2],
        disp.last().unwrap(),
        &disp[disp.len() - 5..]
    );
}

#[test]
#[ignore]
fn genome_distances() {
    let p = SimParams::default();
    let mut rng = Rng::new(11);
    let gs: Vec<Genome> = (0..200).map(|_| Genome::random(&mut rng, &p)).collect();
    let mut rand_d: Vec<f32> = (0..199).map(|i| gs[i].distance(&gs[i + 1])).collect();
    rand_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut kid_d = Vec::new();
    for i in 0..199 {
        let mut c = Genome::crossover(&gs[i], &gs[i], &mut rng);
        c.mutate(&mut rng, &p);
        kid_d.push(c.distance(&gs[i]));
    }
    kid_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |v: &Vec<f32>| (v[v.len() / 10], v[v.len() / 2], v[v.len() * 9 / 10]);
    println!("random pairs p10/p50/p90 {:?}", q(&rand_d));
    println!("parent-child p10/p50/p90 {:?}", q(&kid_d));
}

/// Where does the eat/mate pipeline break? `EVOLVE_DIAG_SAVE=world.evo cargo test --release
/// --test diag funnel -- --ignored --nocapture`
#[test]
#[ignore]
fn funnel() {
    use evolve_sim::brain::output;
    use evolve_sim::grid::Grid;
    use std::collections::HashMap;
    let path = std::env::var("EVOLVE_DIAG_SAVE").expect("set EVOLVE_DIAG_SAVE to a save file");
    let secs: usize = std::env::var("EVOLVE_DIAG_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(240);
    let mut sim = evolve_sim::save::load(path.as_ref()).unwrap();
    let p = sim.params.clone();
    #[derive(Default)]
    struct C {
        samples: f64,
        plant_in_reach: f64,
        reach_and_eat: f64,
        eat_on: f64,
        mature: f64,
        energy_ok: f64,
        cool_ok: f64,
        mate_on: f64,
        ready: f64,
        ready_partner_near: f64,
        ready_partner_compat: f64,
        nearest_other: Vec<f32>,
        energy_frac: Vec<f32>,
        intake: f64,
        basal: f64,
        brain: f64,
        motor_resid: f64,
        speed: Vec<f32>,
    }
    let mut c = C::default();
    let mut prev: HashMap<u64, (f32, f32, u32)> = HashMap::new();
    let births0 = sim.stats.births;
    let rained0 = sim.stats.rained;
    for step in 0..(secs * 60) {
        sim.step();
        if step % 3 != 0 {
            continue;
        }
        let dt = 3.0 / 60.0;
        let mut pg = Grid::new(sim.terrain.size, 5.0);
        pg.rebuild(sim.plants.iter().map(|f| f.pos));
        let torso: Vec<Vec3> = sim.creatures.iter().map(|c| sim.world_of(c).bodies[c.parts[0].body].translation()).collect();
        let ready: Vec<bool> = sim.creatures.iter().map(|c| c.ready_to_mate(&p, p.mate_energy_fraction)).collect();
        let willing: Vec<bool> = sim.creatures.iter().map(|c| c.willing_partner(&p)).collect();
        let mut next = HashMap::new();
        for (i, cr) in sim.creatures.iter().enumerate() {
            let b = &sim.world_of(cr).bodies[cr.parts[0].body];
            let half = cr.torso_half();
            let mouth = b.position().transform_point(Vec3::new(half.x, 0.0, 0.0));
            let reach = p.eat_reach + 0.35 + half.y;
            let near = pg
                .nearest(mouth, reach, |k| sim.plants[k as usize].pos, |k| (sim.plants[k as usize].pos.y - mouth.y).abs() < reach + 0.6)
                .is_some();
            let eat = cr.output(output::EAT) > 0.0;
            c.samples += 1.0;
            c.plant_in_reach += near as u8 as f64;
            c.reach_and_eat += (near && eat) as u8 as f64;
            c.eat_on += eat as u8 as f64;
            c.mature += (cr.age >= p.maturity_age) as u8 as f64;
            c.energy_ok += (cr.energy >= p.mate_energy_fraction * cr.max_energy) as u8 as f64;
            c.cool_ok += (cr.mate_cooldown <= 0.0) as u8 as f64;
            c.mate_on += (cr.output(output::MATE) > 0.0) as u8 as f64;
            c.energy_frac.push(cr.energy / cr.max_energy);
            c.speed.push(Vec3::new(b.linvel().x, 0.0, b.linvel().z).length());
            let mut best = f32::MAX;
            for (k, t) in torso.iter().enumerate() {
                if k != i {
                    best = best.min((*t - torso[i]).length());
                }
            }
            c.nearest_other.push(best);
            if ready[i] {
                c.ready += 1.0;
                let mut near_any = false;
                let mut compat = false;
                for (k, t) in torso.iter().enumerate() {
                    if k == i || !willing[k] {
                        continue;
                    }
                    let gap = (*t - torso[i]).length() - cr.radius() - sim.creatures[k].radius();
                    if gap <= p.mate_range {
                        near_any = true;
                        compat |= cr.genome.distance(&sim.creatures[k].genome) < p.species_threshold;
                    }
                }
                c.ready_partner_near += near_any as u8 as f64;
                c.ready_partner_compat += compat as u8 as f64;
            }
            let food = cr.life.food();
            if let Some(&(e0, f0, ch0)) = prev.get(&cr.id)
                && ch0 == cr.life.children
            {
                let basal = p.basal_metabolism * cr.mass.powf(0.75) * (1.0 + 0.5 * cr.genome.traits.armor + 0.3 * cr.genome.traits.breath) * dt;
                let brain = cr.brain_cost(&p, 1.0) * dt;
                let intake = food - f0;
                c.intake += intake as f64;
                c.basal += basal as f64;
                c.brain += brain as f64;
                c.motor_resid += (intake - (cr.energy - e0) - basal - brain) as f64;
            }
            next.insert(cr.id, (cr.energy, food, cr.life.children));
        }
        prev = next;
    }
    let pct = |x: f64| 100.0 * x / c.samples;
    let med = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("over {secs}s: births {} rained {}", sim.stats.births - births0, sim.stats.rained - rained0);
    println!(
        "eating: plant in reach {:.1}% of the time, eat output on {:.1}%, in reach AND eating {:.1}%",
        pct(c.plant_in_reach), pct(c.eat_on), pct(c.reach_and_eat)
    );
    let tot = c.basal + c.brain + c.motor_resid;
    println!(
        "energy budget per creature-second: intake {:.3}, basal {:.3}, brain {:.3}, motors+other {:.3}  (intake covers {:.0}% of spending)",
        c.intake / c.samples / (3.0 / 60.0), c.basal / c.samples / (3.0 / 60.0), c.brain / c.samples / (3.0 / 60.0),
        c.motor_resid / c.samples / (3.0 / 60.0), 100.0 * c.intake / tot
    );
    println!(
        "mating chain: mature {:.0}%, energy>=thresh {:.0}%, cooldown ok {:.0}%, mate output on {:.0}%, READY {:.1}%",
        pct(c.mature), pct(c.energy_ok), pct(c.cool_ok), pct(c.mate_on), pct(c.ready)
    );
    println!(
        "  of ready: a willing partner within range {:.1}%, and compatible {:.1}%",
        100.0 * c.ready_partner_near / c.ready.max(1.0), 100.0 * c.ready_partner_compat / c.ready.max(1.0)
    );
    println!(
        "median: nearest other creature {:.1} m, energy {:.2} of max, ground speed {:.2} m/s",
        med(&mut c.nearest_other), med(&mut c.energy_frac), med(&mut c.speed)
    );
}

#[test]
#[ignore]
fn where_food_and_creatures_are() {
    use evolve_sim::grid::Grid;
    let path = std::env::var("EVOLVE_DIAG_SAVE").expect("set EVOLVE_DIAG_SAVE");
    let mut sim = evolve_sim::save::load(path.as_ref()).unwrap();
    let p = sim.params.clone();
    let land: f32 = {
        let n = 200;
        let mut k = 0;
        for i in 0..n {
            for j in 0..n {
                let x = (j as f32 / n as f32 - 0.5) * sim.terrain.size;
                let z = (i as f32 / n as f32 - 0.5) * sim.terrain.size;
                if sim.terrain.height_at(x, z) > p.water_level {
                    k += 1;
                }
            }
        }
        k as f32 / (n * n) as f32 * sim.terrain.size * sim.terrain.size
    };
    println!("land {:.0} m2, plants {}, density {:.3}/m2 -> uniform expectation {:.1} plants within 5 m", land, sim.plants.len(), sim.plants.len() as f32 / land, sim.plants.len() as f32 / land * 78.5);
    let mut near5 = Vec::new();
    let mut net10 = Vec::new();
    let mut elev = Vec::new();
    let start: Vec<(u64, Vec3)> = sim.creatures.iter().map(|c| (c.id, sim.world_of(c).bodies[c.parts[0].body].translation())).collect();
    for _ in 0..600 {
        sim.step();
    }
    let mut g = Grid::new(sim.terrain.size, 5.0);
    g.rebuild(sim.plants.iter().map(|f| f.pos));
    for c in &sim.creatures {
        let t = sim.world_of(c).bodies[c.parts[0].body].translation();
        let mut n = 0;
        g.for_each_near(t, 5.0, |k| {
            let q = sim.plants[k as usize].pos;
            if (q.x - t.x).powi(2) + (q.z - t.z).powi(2) < 25.0 {
                n += 1;
            }
        });
        near5.push(n as f32);
        elev.push(t.y - p.water_level);
        if let Some((_, s)) = start.iter().find(|s| s.0 == c.id) {
            net10.push(Vec3::new(t.x - s.x, 0.0, t.z - s.z).length());
        }
    }
    let q = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 10], v[v.len() / 2], v[v.len() * 9 / 10])
    };
    println!("plants within 5 m of a creature p10/p50/p90 {:?}", q(&mut near5));
    println!("net displacement in 10 s p10/p50/p90 {:?} m", q(&mut net10));
    println!("torso height above water p10/p50/p90 {:?} m", q(&mut elev));
    let mut fert: Vec<f32> = sim.plants.iter().map(|f| f.pos.y - p.water_level).collect();
    println!("plant height above water p10/p50/p90 {:?} m", q(&mut fert));
    let mut masses: Vec<f32> = sim.creatures.iter().map(|c| c.mass).collect();
    println!("creature mass p10/p50/p90 {:?} kg", q(&mut masses));
}

/// Reads every genome in a zoo directory (read-only) and checks it upgrades and builds a body.
/// `EVOLVE_DIAG_ZOO=~/.local/share/evolve/zoo cargo test --release --test diag zoo_upgrade -- --ignored --nocapture`
#[test]
#[ignore]
fn zoo_upgrade() {
    let Some(dir) = std::env::var_os("EVOLVE_DIAG_ZOO") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mut ok = 0;
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(id) = name.strip_suffix(".genome").and_then(|s| s.parse::<u64>().ok()) else {
            continue;
        };
        let g = evolve_sim::zoo::load_genome(&dir, id).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(g.brain.w_in.0.len(), g.brain.h * evolve_sim::brain::N_IN, "{name}");
        let plan = body::body_plan(&g);
        assert!(plan.len() >= 1);
        ok += 1;
    }
    println!("{ok} zoo genomes upgraded and built");
}

/// Writes a few hand-built and random new-style bodies into a scratch zoo, for looking at.
/// `EVOLVE_DIAG_ZOO_OUT=/tmp/z cargo test --release --test diag sample_zoo -- --ignored`
#[test]
#[ignore]
fn sample_zoo() {
    let Some(dir) = std::env::var_os("EVOLVE_DIAG_ZOO_OUT") else {
        return;
    };
    let p = SimParams::default();
    let mut zoo = evolve_sim::zoo::Zoo::open(std::path::PathBuf::from(dir));
    let mut rng = Rng::new(77);
    let mut kept = 0;
    while kept < 12 {
        let g = Genome::random(&mut rng, &p);
        let plan = body::body_plan(&g);
        let trunk = plan.iter().filter(|q| q.trunk).count();
        let knees = plan.iter().filter(|q| !q.trunk && q.parent.is_some_and(|r| !plan[r].trunk)).count();
        if (kept < 4 && trunk < 4) || plan.len() < 5 || knees < 2 {
            continue;
        }
        zoo.add(&g, 10.0 + kept as f32, 0, "sample", false, 100).unwrap();
        kept += 1;
    }
}

/// How big the lifetime-learned weight offsets get, next to the inherited ones, and on which
/// outputs. `EVOLVE_DIAG_SAVE=world.evo cargo test --release --test diag learned_weights -- --ignored --nocapture`
#[test]
#[ignore]
fn learned_weights() {
    use evolve_sim::brain::{N_OUT, output};
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        return;
    };
    let mut sim = evolve_sim::save::load(std::path::Path::new(&path)).unwrap();
    for _ in 0..(60 * 60 * 5) {
        sim.step();
    }
    sim.sync_brains();
    let (mut eat, mut joints, mut other, mut genetic) = ((0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0));
    for c in &sim.creatures {
        let h = c.genome.brain.h;
        if c.brain.plastic.len() != N_OUT * h {
            continue;
        }
        for k in 0..N_OUT {
            let p: f32 = c.brain.plastic[k * h..(k + 1) * h].iter().map(|v| v.abs()).sum::<f32>() / h as f32;
            let w: f32 = c.genome.brain.w_out.0[k * h..(k + 1) * h].iter().map(|v| v.to_f32().abs()).sum::<f32>() / h as f32;
            let slot = match k {
                output::EAT => &mut eat,
                _ if k < output::EAT => &mut joints,
                _ => &mut other,
            };
            slot.0 += p;
            slot.1 += 1;
            genetic.0 += w;
            genetic.1 += 1;
        }
    }
    let m = |(s, n): (f32, i32)| s / n.max(1) as f32;
    println!(
        "after 5 min, {} creatures: mean |learned| eat {:.4}, joints {:.4}, other {:.4}; mean |inherited| {:.4}",
        sim.creatures.len(),
        m(eat),
        m(joints),
        m(other),
        m(genetic)
    );
}

/// Do creatures get better at feeding as they live? Food eaten per minute in each band of age,
/// over everyone alive, plus how big the learned offsets get. Set EVOLVE_DIAG_SAVE to a world and
/// optionally EVOLVE_DIAG_SET="learning_rate=0,exploration=0".
#[test]
#[ignore]
fn learning_within_life() {
    use evolve_sim::params::ParamRef;
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        return;
    };
    let mut sim = evolve_sim::save::load(std::path::Path::new(&path)).unwrap();
    let mut p = sim.params.clone();
    for kv in std::env::var("EVOLVE_DIAG_SET").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        p.visit(&mut |info, r| {
            if info.name == k
                && let ParamRef::F32(x) = r
            {
                *x = v.parse().unwrap();
            }
        });
    }
    sim.set_params(p);
    let minutes: usize = std::env::var("EVOLVE_DIAG_MINUTES").ok().and_then(|m| m.parse().ok()).unwrap_or(20);
    let bands = [0.0f32, 60.0, 120.0, 180.0, 240.0, 330.0, 500.0];
    let mut food = vec![0.0f32; bands.len() - 1];
    let mut time = vec![0.0f32; bands.len() - 1];
    let mut last: std::collections::HashMap<u64, f32> = Default::default();
    for _ in 0..minutes * 60 {
        for _ in 0..60 {
            sim.step();
        }
        for c in &sim.creatures {
            let f = c.life.food();
            let prev = last.insert(c.id, f).unwrap_or(f);
            if let Some(b) = bands.windows(2).position(|w| c.age >= w[0] && c.age < w[1]) {
                food[b] += f - prev;
                time[b] += 1.0;
            }
        }
    }
    let line: Vec<String> = bands
        .windows(2)
        .zip(food.iter().zip(&time))
        .map(|(w, (f, t))| format!("{:.0}-{:.0}s {:.2}", w[0], w[1], f / t.max(1.0) * 60.0))
        .collect();
    sim.sync_brains();
    let (mut lp, mut lw) = (0.0, 0.0);
    for c in &sim.creatures {
        lp += c.brain.plastic.iter().map(|v| v.abs()).sum::<f32>() / c.brain.plastic.len().max(1) as f32;
        lw += c.genome.brain.w_out.0.iter().map(|v| v.to_f32().abs()).sum::<f32>() / c.genome.brain.w_out.0.len().max(1) as f32;
    }
    println!(
        "food per minute by age: {}; learned {:.0}% of inherited; births {}",
        line.join(", "),
        100.0 * lp / lw.max(1e-6),
        sim.stats.births
    );
}

/// Twins: each of the gene pool's genomes is dropped twice into the same world at once, one
/// learning normally and one with its plasticity gene at 0 (exploring just as much). Compares
/// food eaten early and late in life. EVOLVE_DIAG_SAVE=world.evo, EVOLVE_DIAG_MINUTES (default 8).
#[test]
#[ignore]
fn learning_twins() {
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        return;
    };
    let mut sim = evolve_sim::save::load(std::path::Path::new(&path)).unwrap();
    let mut p = sim.params.clone();
    p.max_population = 1000;
    for kv in std::env::var("EVOLVE_DIAG_SET").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        p.visit(&mut |info, r| {
            if info.name == k
                && let evolve_sim::params::ParamRef::F32(x) = r
            {
                *x = v.parse().unwrap();
            }
        });
    }
    sim.set_params(p.clone());
    let minutes: usize = std::env::var("EVOLVE_DIAG_MINUTES").ok().and_then(|m| m.parse().ok()).unwrap_or(8);
    let genomes: Vec<Genome> = sim.pool.entries.iter().map(|e| e.genome.clone()).collect();
    let mut twins = Vec::new();
    for g in &genomes {
        let at = sim.random_land_point(6.0);
        let learner = sim.spawn_creature(g.clone(), at, 0.3, p.start_energy, 1, None);
        let mut fixed = g.clone();
        fixed.traits.plasticity = 0.0;
        let at = sim.random_land_point(6.0);
        let control = sim.spawn_creature(fixed, at, 0.3, p.start_energy, 1, None);
        twins.push((learner, control));
    }
    let half = minutes * 60 / 2;
    let mut food: std::collections::HashMap<u64, (f32, f32)> = Default::default();
    for s in 0..minutes * 60 {
        for _ in 0..60 {
            sim.step();
        }
        for &(a, b) in &twins {
            for id in [a, b] {
                if let Some(c) = sim.creature(id) {
                    let e = food.entry(id).or_insert((0.0, 0.0));
                    if s < half {
                        e.0 = c.life.food();
                    } else {
                        e.1 = c.life.food() - e.0;
                    }
                }
            }
        }
    }
    let (mut a1, mut b1, mut a2, mut b2, mut wins, mut n) = (0.0, 0.0, 0.0, 0.0, 0, 0);
    for &(a, b) in &twins {
        let fa = food.get(&a).copied().unwrap_or_default();
        let fb = food.get(&b).copied().unwrap_or_default();
        a1 += fa.0;
        b1 += fb.0;
        a2 += fa.1;
        b2 += fb.1;
        if fa.0 + fa.1 > fb.0 + fb.1 {
            wins += 1;
        }
        if fa.0 + fa.1 != fb.0 + fb.1 {
            n += 1;
        }
    }
    let k = twins.len().max(1) as f32;
    println!(
        "{} twin pairs over {minutes} min: food first half learner {:.1} vs control {:.1}; second half {:.1} vs {:.1}; learner ahead in {wins} of {n} pairs that differ",
        twins.len(),
        a1 / k,
        b1 / k,
        a2 / k,
        b2 / k
    );
}

/// How short of breath creatures get on land: the spread of oxygen levels over a few minutes.
/// EVOLVE_DIAG_SAVE=world.evo
#[test]
#[ignore]
fn breathlessness() {
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        return;
    };
    let mut sim = evolve_sim::save::load(std::path::Path::new(&path)).unwrap();
    let mut p = sim.params.clone();
    for kv in std::env::var("EVOLVE_DIAG_SET").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        p.visit(&mut |info, r| {
            if info.name == k
                && let evolve_sim::params::ParamRef::F32(x) = r
            {
                *x = v.parse().unwrap();
            }
        });
    }
    sim.set_params(p);
    let (mut samples, mut low, mut winded, mut sum) = (0u32, 0u32, 0u32, 0.0f32);
    let mut powers = Vec::new();
    for _ in 0..180 {
        for _ in 0..60 {
            sim.step();
        }
        for c in &sim.creatures {
            if c.in_water > 0.5 {
                continue;
            }
            samples += 1;
            sum += c.oxygen;
            powers.push(c.exertion);
            low += (c.oxygen < 0.3) as u32;
            winded += (c.oxygen < 0.7) as u32;
        }
    }
    let f = |k: u32| 100.0 * k as f32 / samples.max(1) as f32;
    powers.sort_by(|a, b| a.total_cmp(b));
    let q = |x: f32| powers.get(((powers.len() as f32 - 1.0) * x) as usize).copied().unwrap_or(0.0);
    println!("muscle power W/kg: median {:.3}, 90% {:.3}, 99% {:.3}, max {:.3}", q(0.5), q(0.9), q(0.99), q(1.0));
    println!(
        "on land: mean oxygen {:.2}; below 70% {:.1}% of the time, below 30% (weakened) {:.1}%",
        sum / samples.max(1) as f32,
        f(winded),
        f(low)
    );
}

/// Where a world's food comes from and who moves its earth: carcasses of creatures that fell
/// from the sky (the population floor) versus those born in the world, what each kind of body
/// eats, how far it goes, and how torn up the land is. Loads `EVOLVE_DIAG_SAVE`, or starts a
/// world seeded from the zoo in `EVOLVE_ZOO_DIR` with `EVOLVE_DIAG_MIN`/`EVOLVE_DIAG_MAX` as the
/// population floor and cap.
/// `EVOLVE_ZOO_DIR=copy cargo test --release --test diag economy -- --ignored --nocapture`
#[test]
#[ignore]
fn economy() {
    use std::collections::HashMap;
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let mut sim = match std::env::var("EVOLVE_DIAG_SAVE") {
        Ok(path) => evolve_sim::save::load(path.as_ref()).unwrap(),
        Err(_) => {
            let p = SimParams {
                min_population: env("EVOLVE_DIAG_MIN", 790.0) as u32,
                max_population: env("EVOLVE_DIAG_MAX", 1000.0) as u32,
                seed: env("EVOLVE_DIAG_SEED", 1.0) as u64,
                ..SimParams::default()
            };
            let mut sim = Sim::new(p);
            let dir = std::env::var("EVOLVE_ZOO_DIR").expect("set EVOLVE_ZOO_DIR to a copy of a zoo");
            evolve_sim::zoo::seed_world(&evolve_sim::zoo::Zoo::shared(dir.into()), &mut sim);
            sim
        }
    };
    let warm = env("EVOLVE_DIAG_WARM", 120.0);
    let secs = env("EVOLVE_DIAG_SECS", 300.0);
    let tps = evolve_sim::sim::TICKS_PER_SECOND as f32;
    for _ in 0..(warm * tps) as usize {
        sim.step();
    }
    let p = sim.params.clone();
    let terrain = |sim: &Sim| {
        let t = &sim.terrain;
        let (n, cell) = (t.res, t.cell());
        let (mut land, mut moved, mut deep, mut steep, mut edges, mut steep0) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                let k = i * n + j;
                if t.heights[k] <= p.water_level {
                    continue;
                }
                land += 1.0;
                let d = (t.heights[k] - t.natural(k)).abs();
                moved += d;
                deep += (d > 0.5) as u8 as f64;
                for k2 in [k + 1, k + n] {
                    edges += 1.0;
                    steep += ((t.heights[k2] - t.heights[k]).abs() > cell) as u8 as f64;
                    steep0 += ((t.ground[k2] - t.ground[k]).abs() > cell) as u8 as f64;
                }
            }
        }
        format!(
            "land points {land:.0}: mean |dug/piled| {:.2} m, {:.1}% moved > 0.5 m; slopes over 45 deg: {:.1}% of edges now, {:.1}% as made",
            moved / land as f32,
            100.0 * deep / land,
            100.0 * steep / edges,
            100.0 * steep0 / edges
        )
    };
    println!("after {warm:.0} s warm-up: {}", terrain(&sim));
    // [rained, born] x [one box, more parts]
    #[derive(Default, Clone, Copy)]
    struct K {
        samples: f64,
        meat: f64,
        plant: f64,
        dist: f64,
        earth: f64,
        kills: f64,
        mass: f64,
    }
    let mut k = [[K::default(); 2]; 2];
    let class = |c: &evolve_sim::sim::Creature| (usize::from(!c.rained), usize::from(c.plan.len() > 1));
    let (mut carcass_meat, mut injected) = ([0.0f64; 2], 0.0f64);
    let (mut deaths, births0, rained0) = ([0u32; 2], sim.stats.births, sim.stats.rained);
    let mut prev: HashMap<u64, (usize, usize, f32, f32, evolve_sim::sim::LifeStats, f32)> = HashMap::new();
    let start: HashMap<u64, Vec3> = sim.creatures.iter().map(|c| (c.id, sim.world_of(c).bodies[c.parts[0].body].translation())).collect();
    let dt = 1.0 / tps;
    for _ in 0..(secs * tps) as usize {
        sim.step();
        let mut next = HashMap::new();
        for c in &sim.creatures {
            let (r, b) = class(c);
            let x = &mut k[r][b];
            x.samples += 1.0;
            x.mass += c.mass as f64;
            match prev.get(&c.id) {
                Some((_, _, _, _, l, soil)) => {
                    x.meat += (c.life.meat_eaten - l.meat_eaten) as f64;
                    x.plant += (c.life.plant_eaten - l.plant_eaten) as f64;
                    x.dist += (c.life.distance - l.distance) as f64;
                    x.kills += (c.life.kills - l.kills) as f64;
                    x.earth += (c.soil - soil).abs() as f64;
                }
                None if c.rained => injected += c.energy as f64,
                None => {}
            }
            next.insert(c.id, (r, b, c.mass, c.energy, c.life.clone(), c.soil));
        }
        for (id, (r, _, mass, energy, _, _)) in &prev {
            if !next.contains_key(id) {
                deaths[*r] += 1;
                // Only a born body leaves meat of its own (see `Sim::process_deaths`).
                let body = if *r == 1 { mass * p.meat_per_mass } else { 0.0 };
                carcass_meat[*r] += (body + energy.max(0.0) * 0.5) as f64;
            }
        }
        prev = next;
    }
    let s = secs as f64;
    println!(
        "over {secs:.0} s: rained {} (energy given {:.0}/s), born {}; deaths: rained {} born {}",
        sim.stats.rained - rained0,
        injected / s,
        sim.stats.births - births0,
        deaths[0],
        deaths[1]
    );
    println!(
        "meat from carcasses per second: of rained {:.0}, of born {:.0}  (plant energy eaten per second {:.0}, meat {:.0})",
        carcass_meat[0] / s,
        carcass_meat[1] / s,
        k.iter().flatten().map(|x| x.plant).sum::<f64>() / s,
        k.iter().flatten().map(|x| x.meat).sum::<f64>() / s
    );
    for (r, rn) in ["rained", "born"].iter().enumerate() {
        for (b, bn) in ["one box", "with parts"].iter().enumerate() {
            let x = k[r][b];
            if x.samples == 0.0 {
                continue;
            }
            let n = x.samples * dt as f64; // creature-seconds
            println!(
                "{rn:>6} {bn:>10}: {:5.0} alive on average, mass {:4.0} kg; per creature-minute: meat {:5.1} plant {:5.1} distance {:5.1} m, earth moved {:.3} m3, kills {:.3}",
                x.samples / (secs * tps) as f64,
                x.mass / x.samples,
                60.0 * x.meat / n,
                60.0 * x.plant / n,
                60.0 * x.dist / n,
                60.0 * x.earth / n,
                60.0 * x.kills / n
            );
        }
    }
    let mut net: Vec<(bool, f32)> = sim
        .creatures
        .iter()
        .filter_map(|c| {
            let a = start.get(&c.id)?;
            let t = sim.world_of(c).bodies[c.parts[0].body].translation();
            Some((c.plan.len() > 1, Vec3::new(t.x - a.x, 0.0, t.z - a.z).length()))
        })
        .collect();
    net.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for parts in [false, true] {
        let v: Vec<f32> = net.iter().filter(|x| x.0 == parts).map(|x| x.1).collect();
        if !v.is_empty() {
            println!(
                "survivors {}: {} of them, net displacement over the run p50 {:.1} m, p90 {:.1} m",
                if parts { "with parts" } else { "one box" },
                v.len(),
                v[v.len() / 2],
                v[v.len() * 9 / 10]
            );
        }
    }
    println!("at the end: {}", terrain(&sim));
}

/// The observatory's report on a world: a save (EVOLVE_DIAG_SAVE), or a new world seeded from a
/// zoo (EVOLVE_ZOO_DIR, with EVOLVE_DIAG_MIN/MAX/SEED), run for EVOLVE_DIAG_SECS after a warm-up
/// of EVOLVE_DIAG_WARM, with a summary line every minute (and a full report every
/// EVOLVE_DIAG_REPORT_EVERY seconds, if set). Extra params: EVOLVE_DIAG_SET="a=1,b=2".
/// `EVOLVE_ZOO_DIR=copy cargo test --release --test diag census -- --ignored --nocapture`
#[test]
#[ignore]
fn census() {
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let set = |p: &mut SimParams| {
        for kv in std::env::var("EVOLVE_DIAG_SET").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("EVOLVE_DIAG_SET takes name=value pairs");
            p.set_by_name(k.trim(), v.trim()).unwrap();
        }
    };
    let mut sim = match std::env::var("EVOLVE_DIAG_SAVE") {
        Ok(path) => {
            let mut sim = evolve_sim::save::load(path.as_ref()).unwrap();
            let mut p = sim.params.clone();
            set(&mut p);
            sim.set_params(p);
            sim
        }
        Err(_) => {
            let mut p = SimParams {
                min_population: env("EVOLVE_DIAG_MIN", 790.0) as u32,
                max_population: env("EVOLVE_DIAG_MAX", 1000.0) as u32,
                seed: env("EVOLVE_DIAG_SEED", 1.0) as u64,
                ..SimParams::default()
            };
            set(&mut p);
            let mut sim = Sim::new(p);
            if let Ok(dir) = std::env::var("EVOLVE_ZOO_DIR") {
                evolve_sim::zoo::seed_world(&evolve_sim::zoo::Zoo::shared(dir.into()), &mut sim);
            }
            sim
        }
    };
    let tps = evolve_sim::sim::TICKS_PER_SECOND as usize;
    let warm = env("EVOLVE_DIAG_WARM", 120.0) as usize;
    let secs = env("EVOLVE_DIAG_SECS", 300.0) as usize;
    // A full report every this many seconds along the way too (0: only at the end).
    let every = env("EVOLVE_DIAG_REPORT_EVERY", 0.0) as usize;
    for _ in 0..warm * tps {
        sim.step();
    }
    for s in 0..secs {
        for _ in 0..tps {
            sim.step();
        }
        if (s + 1) % 60 == 0 {
            println!("{}", sim.summary_line(60.0));
        }
        if every > 0 && (s + 1) % every == 0 && s + 1 < secs {
            println!("\n{}", sim.report(every as f64));
        }
    }
    println!("\n{}", sim.report(secs as f64));
}

/// How the best creatures of a zoo (EVOLVE_ZOO_DIR, any score version) move: the top
/// EVOLVE_DIAG_TOP (5) are dropped into a fresh world, two copies each, settle for 20 s, and each is
/// watched for EVOLVE_DIAG_SECS (15).
/// `EVOLVE_ZOO_DIR=copy cargo test --release --test diag watch_the_zoo -- --ignored --nocapture`
#[test]
#[ignore]
fn watch_the_zoo() {
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let dir = std::env::var("EVOLVE_ZOO_DIR").expect("set EVOLVE_ZOO_DIR to a copy of a zoo");
    let zoo = evolve_sim::zoo::Zoo::open(dir.into());
    let mut best = zoo.entries.clone();
    best.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut sim = Sim::new(SimParams { min_population: 0, parallel_worlds: 0, ..SimParams::default() });
    for e in best.iter().take(env("EVOLVE_DIAG_TOP", 5.0) as usize) {
        let g = zoo.genome(e.id).unwrap();
        println!("{} (score {:.1}): dropped", e.name, e.score);
        sim.drop_genome(&g, e.generation, None, 2);
    }
    for _ in 0..20 * 60 {
        sim.step();
    }
    let secs = env("EVOLVE_DIAG_SECS", 15.0);
    let mut waiting = Vec::new();
    let ids: Vec<u64> = sim.creatures.iter().map(|c| c.id).collect();
    for id in ids {
        let (tx, rx) = std::sync::mpsc::channel();
        sim.watch(id, secs, tx);
        waiting.push(rx);
    }
    for _ in 0..((secs + 1.0) * 60.0) as usize {
        sim.step();
    }
    for rx in waiting {
        if let Ok(Ok(text)) = rx.try_recv() {
            println!("\n{text}");
        }
    }
}
