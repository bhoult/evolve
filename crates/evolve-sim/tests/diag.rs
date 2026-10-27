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

/// Whether creatures sink into the ground: every few steps, how far each creature's parts sit
/// below the terrain surface, and how the deepest ones end. Same world settings as `census`
/// (EVOLVE_DIAG_SAVE, SEED, SET, SECS).
/// `EVOLVE_CPU_BRAINS=1 cargo test --release --test diag sinking -- --ignored --nocapture`
#[test]
#[ignore]
fn sinking() {
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
            let mut p = SimParams { seed: env("EVOLVE_DIAG_SEED", 1.0) as u64, ..SimParams::default() };
            set(&mut p);
            Sim::new(p)
        }
    };
    let tps = evolve_sim::sim::TICKS_PER_SECOND as usize;
    let secs = env("EVOLVE_DIAG_SECS", 120.0) as usize;
    // Deepest a creature's part centers ever went below the ground, and the torso's, by id.
    let mut deepest: std::collections::HashMap<u64, (f32, f32)> = Default::default();
    let mut samples = 0u64;
    let mut under = [0u64; 4];
    let mut shown = 0;
    let glitches0 = sim.stats.deaths_other;
    for step in 0..secs * tps {
        sim.step();
        if step % 6 != 0 {
            continue;
        }
        for c in sim.creatures.iter().filter(|c| c.dead.is_none()) {
            let world = sim.world_of(c);
            let mut worst = f32::MIN;
            let mut torso = 0.0;
            for (k, q) in c.parts.iter().enumerate() {
                let Some(b) = world.bodies.get(q.body) else { continue };
                let t = b.translation();
                let d = sim.terrain.height_at(t.x, t.z) - t.y;
                worst = worst.max(d);
                if k == 0 {
                    torso = d;
                }
            }
            samples += 1;
            for (i, lim) in [0.0, 0.25, 0.5, 1.0].iter().enumerate() {
                if worst > *lim {
                    under[i] += 1;
                }
            }
            let e = deepest.entry(c.id).or_insert((f32::MIN, f32::MIN));
            e.0 = e.0.max(worst);
            e.1 = e.1.max(torso);
            if torso > 0.5 && shown < 15 {
                shown += 1;
                let t = world.bodies[c.parts[0].body].translation();
                let v = world.bodies[c.parts[0].body].linvel();
                println!(
                    "t {:.1}s #{} torso {:.2} m under the ground at ({:.1}, {:.1}, {:.1}), ground {:.2}, velocity ({:.1}, {:.1}, {:.1}), age {:.0} s, in water {:.2}, {} kg",
                    sim.time, c.id, torso, t.x, t.y, t.z, sim.terrain.height_at(t.x, t.z), v.x, v.y, v.z, c.age, c.in_water, c.mass as i32
                );
            }
        }
    }
    let lives = deepest.len().max(1) as f32;
    let deep = |lim: f32, torso: bool| deepest.values().filter(|d| if torso { d.1 > lim } else { d.0 > lim }).count();
    println!(
        "{} creature samples: some part's center under the ground {:.2}%, >0.25 m {:.3}%, >0.5 m {:.3}%, >1 m {:.4}%",
        samples,
        100.0 * under[0] as f64 / samples as f64,
        100.0 * under[1] as f64 / samples as f64,
        100.0 * under[2] as f64 / samples as f64,
        100.0 * under[3] as f64 / samples as f64
    );
    println!(
        "{} lives: a part ever >0.5 m under {:.2}%, >1 m {:.2}%; torso ever >0.5 m under {:.2}%, >1 m {:.2}%, >3 m {:.2}%; deaths as glitches {}",
        deepest.len(),
        100.0 * deep(0.5, false) as f32 / lives,
        100.0 * deep(1.0, false) as f32 / lives,
        100.0 * deep(0.5, true) as f32 / lives,
        100.0 * deep(1.0, true) as f32 / lives,
        100.0 * deep(3.0, true) as f32 / lives,
        sim.stats.deaths_other - glitches0
    );
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

/// How far limbless blocks (which can't move themselves) drift on their own in 12 s at random
/// places on the land (EVOLVE_DIAG_SLOPE=a,b: only where the ground slopes a to b degrees), with
/// `soft_ccd` from EVOLVE_DIAG_SET.
#[test]
#[ignore]
fn block_drift() {
    let mut p = SimParams { min_population: 0, terrain_resolution: 65, weather: false, stones: 0, ..SimParams::default() };
    for kv in std::env::var("EVOLVE_DIAG_SET").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        p.set_by_name(k, v).unwrap();
    }
    let mut sim = Sim::new(p);
    let mut ids = Vec::new();
    for k in 0..12 {
        let mut g = Genome::random(&mut Rng::new(21 + k), &SimParams::default());
        g.spine.segments = 0;
        for l in &mut g.limbs {
            l.present = false;
        }
        // EVOLVE_DIAG_SLOPE=a,b: only places whose ground slopes between a and b degrees.
        let (lo, hi) = std::env::var("EVOLVE_DIAG_SLOPE")
            .ok()
            .and_then(|s| s.split_once(',').map(|(a, b)| (a.parse::<f32>().unwrap(), b.parse::<f32>().unwrap())))
            .unwrap_or((0.0, 90.0));
        let at = loop {
            let at = sim.random_land_point(40.0);
            let slope = sim.terrain.normal_at(at.x, at.z).y.acos().to_degrees();
            if (lo..=hi).contains(&slope) {
                break at;
            }
        };
        ids.push((sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None), at));
    }
    for s in 0..12 {
        for _ in 0..evolve_sim::sim::TICKS_PER_SECOND {
            sim.step();
        }
        if s == 1 || s == 11 {
            let line: Vec<String> = ids
                .iter()
                .filter_map(|&(id, at)| {
                    let c = sim.creature(id)?;
                    let t = sim.torso_of(c)?.translation();
                    let v = sim.torso_of(c)?.linvel();
                    let n = sim.terrain.normal_at(t.x, t.z);
                    Some(format!("{:.1}m (v {:.2}, slope {:.0}°)", (t.x - at.x).hypot(t.z - at.z), v.length(), n.y.acos().to_degrees()))
                })
                .collect();
            println!("t {} s: {}", s + 1, line.join(", "));
        }
    }
}

/// How varied a zoo (EVOLVE_ZOO_DIR, ideally a copy) is: how many groups its genomes fall into
/// at the species distance, how far each is from its nearest neighbour, and the spread of
/// body plans. Only current-score-version entries unless EVOLVE_DIAG_ALL is set.
#[test]
#[ignore]
fn zoo_diversity() {
    let Some(dir) = std::env::var_os("EVOLVE_ZOO_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let zoo = evolve_sim::zoo::Zoo::open(dir.clone());
    let all = std::env::var_os("EVOLVE_DIAG_ALL").is_some();
    let genomes: Vec<(String, Genome)> = zoo
        .entries
        .iter()
        .filter(|m| all || m.score_version == evolve_sim::zoo::SCORE_VERSION)
        .filter_map(|m| Some((m.name.clone(), evolve_sim::zoo::load_genome(&dir, m.id).ok()?)))
        .collect();
    let n = genomes.len();
    for t in [0.05, 0.1, 0.15, 0.2] {
        let mut leaders: Vec<usize> = Vec::new();
        for i in 0..n {
            if !leaders.iter().any(|&l| genomes[l].1.distance(&genomes[i].1) < t) {
                leaders.push(i);
            }
        }
        println!("{} groups at distance {t}", leaders.len());
        if t == 0.1 {
            // Each group's leader's body plan: do the groups differ in shape or only in proportion?
            let mut by_plan: std::collections::BTreeMap<(usize, u8, usize), usize> = Default::default();
            for &l in &leaders {
                let g = &genomes[l].1;
                *by_plan.entry((g.limbs.iter().filter(|x| x.present).count(), g.spine.segments, body::body_plan(g).len())).or_default() += 1;
            }
            let mut v: Vec<_> = by_plan.into_iter().collect();
            v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            println!("  leaders' body plans (limb genes/spine segments/parts): {} kinds; {}", v.len(), v.iter().map(|((l, s, p), c)| format!("{l}/{s}/{p}: {c}")).collect::<Vec<_>>().join(", "));
        }
    }
    let t = SimParams::default().species_threshold;
    // Leader clustering: each genome joins the first group whose leader is within `t`.
    let mut leaders: Vec<usize> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    for i in 0..n {
        match leaders.iter().position(|&l| genomes[l].1.distance(&genomes[i].1) < t) {
            Some(k) => sizes[k] += 1,
            None => {
                leaders.push(i);
                sizes.push(1);
            }
        }
    }
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(sizes[k]));
    let mut nearest: Vec<f32> = (0..n)
        .map(|i| (0..n).filter(|&j| j != i).map(|j| genomes[i].1.distance(&genomes[j].1)).fold(f32::MAX, f32::min))
        .collect();
    nearest.sort_by(f32::total_cmp);
    let q = |f: f32| nearest[((n - 1) as f32 * f) as usize];
    println!("{n} genomes; {} groups at distance {t}; largest {:?}", sizes.len(), order.iter().take(8).map(|&k| sizes[k]).collect::<Vec<_>>());
    println!("nearest neighbour distance p10 {:.3} p50 {:.3} p90 {:.3}", q(0.1), q(0.5), q(0.9));
    let mut plans: std::collections::BTreeMap<(usize, u8, usize), usize> = Default::default();
    for (_, g) in &genomes {
        let limbs = g.limbs.iter().filter(|l| l.present).count();
        *plans.entry((limbs, g.spine.segments, body::body_plan(g).len())).or_default() += 1;
    }
    let mut common: Vec<_> = plans.iter().collect();
    common.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    println!(
        "{} body plans (limb genes, spine segments, parts); commonest: {}",
        plans.len(),
        common.iter().take(8).map(|((l, s, p), c)| format!("{l}/{s}/{p}: {c}")).collect::<Vec<_>>().join(", ")
    );
}

/// How far a coarser physics ground would be from the drawn one: the ground's height sampled
/// every 2nd, 4th and 8th grid point (as the physics pieces would) against the full grid, over
/// random land points and under the body parts of the creatures in a saved world
/// (EVOLVE_DIAG_SAVE).
#[test]
#[ignore]
fn ground_stride_mismatch() {
    let Ok(path) = std::env::var("EVOLVE_DIAG_SAVE") else {
        return;
    };
    let sim = evolve_sim::save::load(path.as_ref()).unwrap();
    let t = &sim.terrain;
    let sea = sim.params.water_level;
    let mut rng = Rng::new(3);
    let land: Vec<(f32, f32)> = (0..200_000)
        .map(|_| (rng.range(-t.half(), t.half()), rng.range(-t.half(), t.half())))
        .filter(|&(x, z)| t.height_at(x, z) > sea)
        .collect();
    let sim = &sim;
    let feet: Vec<(f32, f32)> = sim
        .creatures
        .iter()
        .flat_map(|c| c.parts.iter().filter_map(move |q| sim.world_of(c).bodies.get(q.body)))
        .map(|b| b.translation())
        .filter(|p| p.y - t.height_at(p.x, p.z) < 0.5)
        .map(|p| (p.x, p.z))
        .collect();
    println!("terrain {} a side over {} m ({:.3} m apart); {} land points, {} body parts near the ground", t.res, t.size, t.cell(), land.len(), feet.len());
    for stride in [2usize, 4, 8] {
        let coarse = t.resampled((t.res - 1).div_ceil(stride) + 1);
        let gaps = |pts: &[(f32, f32)]| {
            let mut d: Vec<f32> = pts.iter().map(|&(x, z)| (t.height_at(x, z) - coarse.height_at(x, z)).abs()).collect();
            d.sort_by(f32::total_cmp);
            let q = |f: f32| d[((d.len() - 1) as f32 * f) as usize];
            (q(0.5), q(0.99), d.last().copied().unwrap_or(0.0))
        };
        let (a, b) = (gaps(&land), gaps(&feet));
        println!(
            "every {stride} points ({:.2} m): land gap p50 {:.3} p99 {:.3} max {:.3} m; under bodies p50 {:.3} p99 {:.3} max {:.3} m",
            coarse.cell(),
            a.0,
            a.1,
            a.2,
            b.0,
            b.1,
            b.2
        );
    }
}

/// How worked the ground of a saved world is: what creatures dug or piled, what water cut, how
/// much land is under water, and how bumpy it is underfoot against the ground as made.
/// `EVOLVE_DIAG_SAVE=path.evo cargo test --release --test diag terrain_worked -- --ignored --nocapture`
#[test]
#[ignore]
fn terrain_worked() {
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        println!("set EVOLVE_DIAG_SAVE to a save");
        return;
    };
    let sim = evolve_sim::save::load(std::path::Path::new(&path)).expect("load");
    let t = &sim.terrain;
    let (r, wl) = (t.res, sim.params.water_level);
    let cell = t.size / (r - 1) as f32;
    println!("terrain {r} a side, {cell:.3} m cells, water level {wl}");
    let pct = |v: &mut Vec<f32>, q: f32| {
        v.sort_by(f32::total_cmp);
        v.get(((v.len() - 1) as f32 * q) as usize).copied().unwrap_or(0.0)
    };
    let (mut dug, mut cut, mut wet) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..r * r {
        if t.natural(k) <= wl {
            continue;
        }
        dug.push((t.heights[k] - t.natural(k)).abs());
        cut.push(t.eroded.get(k).copied().unwrap_or(0.0));
        wet.push(t.water.get(k).copied().unwrap_or(0.0));
    }
    let n = dug.len() as f32;
    let share = |v: &[f32], x: f32| v.iter().filter(|&&d| d > x).count() as f32 / n * 100.0;
    let mean = |v: &[f32]| v.iter().sum::<f32>() / n;
    println!("land points {} ({:.0}% of the grid)", dug.len(), n / (r * r) as f32 * 100.0);
    println!(
        "dug or piled: mean {:.3} m, over 10 cm {:.1}%, over 30 cm {:.1}%, p90 {:.3}, p99 {:.3}",
        mean(&dug), share(&dug, 0.1), share(&dug, 0.3), pct(&mut dug.clone(), 0.9), pct(&mut dug, 0.99)
    );
    println!(
        "cut by water: mean {:.3} m, over 10 cm {:.1}%, over 50 cm {:.1}%, p90 {:.3}, p99 {:.3}",
        mean(&cut), share(&cut, 0.1), share(&cut, 0.5), pct(&mut cut.clone(), 0.9), pct(&mut cut, 0.99)
    );
    println!("under water: over 5 cm {:.1}%, over 30 cm {:.1}% of land", share(&wet, 0.05), share(&wet, 0.3));
    // Bumpiness underfoot: height change over about half a meter, now and as made.
    let d = ((0.5 / cell).round() as usize).max(1);
    let (mut now, mut made) = (Vec::new(), Vec::new());
    for i in (0..r).step_by(3) {
        for j in (0..r - d).step_by(3) {
            let (k, k2) = (i * r + j, i * r + j + d);
            if t.natural(k) <= wl || t.natural(k2) <= wl {
                continue;
            }
            now.push((t.heights[k2] - t.heights[k]).abs());
            made.push((t.ground[k2] - t.ground[k]).abs());
        }
    }
    println!(
        "height change over {:.2} m: now mean {:.3} p90 {:.3} p99 {:.3}; as made mean {:.3} p90 {:.3} p99 {:.3}",
        d as f32 * cell,
        now.iter().sum::<f32>() / now.len() as f32, pct(&mut now.clone(), 0.9), pct(&mut now, 0.99),
        made.iter().sum::<f32>() / made.len() as f32, pct(&mut made.clone(), 0.9), pct(&mut made, 0.99)
    );
}

/// How plants lie over the land: how far a random spot on land is from the nearest plant, and how
/// much of the land has plants at all, for a saved world and for a fresh one on the same settings
/// grown for `EVOLVE_DIAG_MINUTES` (default 60) simulated minutes with no creatures, on ground as
/// coarse as an experiment arm's. Plants seed within `plant_spread_radius` of their parents, so a
/// world's plants may bunch up over the hours.
/// `EVOLVE_DIAG_SAVE=path.evo cargo test --release --test diag plant_layout -- --ignored --nocapture`
#[test]
#[ignore]
fn plant_layout() {
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        println!("set EVOLVE_DIAG_SAVE to a save");
        return;
    };
    let saved = evolve_sim::save::load(std::path::Path::new(&path)).expect("load");
    layout("saved", &saved);
    let minutes: f64 = std::env::var("EVOLVE_DIAG_MINUTES").ok().and_then(|m| m.parse().ok()).unwrap_or(60.0);
    let mut fresh = Sim::new(SimParams {
        min_population: 0,
        terrain_resolution: (saved.params.world_size as u32) + 1,
        ..saved.params.clone()
    });
    while fresh.time < minutes * 60.0 {
        fresh.step();
    }
    layout(&format!("fresh, {minutes:.0} min"), &fresh);
}

/// What the land with no plant within 10 m is like, against the land with one: under water, on
/// the shore, trampled, worn by water, steep, poor; and how well a plant would grow there. Says
/// whether plants are missing because the ground is bad or because the plants are elsewhere.
/// `EVOLVE_DIAG_SAVE=<save> cargo test --release --test diag bare_land -- --ignored --nocapture`
#[test]
#[ignore]
fn bare_land() {
    let Some(path) = std::env::var_os("EVOLVE_DIAG_SAVE") else {
        println!("set EVOLVE_DIAG_SAVE to a save");
        return;
    };
    let sim = evolve_sim::save::load(std::path::Path::new(&path)).expect("load");
    let (t, p) = (&sim.terrain, &sim.params);
    let half = t.size * 0.5;
    const CELL: f32 = 10.0;
    let n = (t.size / CELL).ceil() as usize + 1;
    let mut grid = vec![Vec::new(); n * n];
    let at = |x: f32, z: f32| (((x + half) / CELL) as usize).min(n - 1) + (((z + half) / CELL) as usize).min(n - 1) * n;
    for f in &sim.plants {
        grid[at(f.pos.x, f.pos.z)].push((f.pos.x, f.pos.z));
    }
    let near = |x: f32, z: f32| {
        let (ci, cj) = (((x + half) / CELL) as i64, ((z + half) / CELL) as i64);
        (-1..=1i64).any(|di| {
            (-1..=1i64).any(|dj| {
                let (i, j) = (ci + di, cj + dj);
                i >= 0 && j >= 0 && i < n as i64 && j < n as i64 && grid[i as usize + j as usize * n].iter().any(|&(px, pz)| (px - x).powi(2) + (pz - z).powi(2) <= 100.0)
            })
        })
    };
    // Per kind of land (0 bare, 1 with a plant near): spots, wet, shore, suitability sum, poor
    // (suitability under 0.1), worn sum, trampled (worn over 0.3), eroded sum, cut (over 10 cm),
    // flatness sum (normal.y squared), fertility sum.
    let mut s = [[0.0f64; 11]; 2];
    let mut x = -half + 1.25;
    while x < half {
        let mut z = -half + 1.25;
        while z < half {
            let h = t.height_at(x, z);
            if h > p.water_level {
                let k = t.nearest_point(x, z);
                let suit = t.plant_suitability(x, z, p.water_level, p.terrain_height) as f64;
                let worn = t.worn(k) as f64;
                let cut = t.eroded.get(k).copied().unwrap_or(0.0) as f64;
                let flat = (t.normal_at(x, z).y.clamp(0.0, 1.0) as f64).powi(2);
                let row = &mut s[usize::from(near(x, z))];
                for (c, v) in [
                    1.0,
                    f64::from(t.water_at(x, z) > 0.03),
                    f64::from(h - p.water_level < 0.15),
                    suit,
                    f64::from(suit < 0.1),
                    worn,
                    f64::from(worn > 0.3),
                    cut,
                    f64::from(cut > 0.1),
                    flat,
                    t.fertility_at(x, z) as f64,
                ]
                .into_iter()
                .enumerate()
                {
                    row[c] += v;
                }
            }
            z += 2.5;
        }
        x += 2.5;
    }
    let total = s[0][0] + s[1][0];
    for (name, r) in [("no plant within 10 m", &s[0]), ("a plant within 10 m", &s[1])] {
        let (m, pc) = (|c: usize| r[c] / r[0].max(1.0), |c: usize| r[c] / r[0].max(1.0) * 100.0);
        println!(
            "{name}: {:.0}% of land; wet {:.0}%, shore {:.0}%; suitability mean {:.2}, under 0.1 {:.0}%; worn mean {:.2}, over 0.3 {:.0}%; cut by water mean {:.2} m, over 10 cm {:.0}%; flatness {:.2}; fertility {:.2}",
            r[0] / total * 100.0, pc(1), pc(2), m(3), pc(4), m(5), pc(6), m(7), pc(8), m(9), m(10)
        );
    }
}

fn layout(name: &str, sim: &Sim) {
    let t = &sim.terrain;
    let wl = sim.params.water_level;
    let half = t.size * 0.5;
    // Plants bucketed in 5 m cells for nearest-plant searches.
    const CELL: f32 = 5.0;
    let n = (t.size / CELL).ceil() as usize + 1;
    let mut grid = vec![Vec::new(); n * n];
    let at = |x: f32, z: f32| (((x + half) / CELL) as usize).min(n - 1) + (((z + half) / CELL) as usize).min(n - 1) * n;
    for p in &sim.plants {
        grid[at(p.pos.x, p.pos.z)].push((p.pos.x, p.pos.z));
    }
    let (mut spots, mut within) = (0usize, [0usize; 4]);
    let reach = [2.0f32, 5.0, 10.0, 25.0];
    let mut x = -half + 1.25;
    while x < half {
        let mut z = -half + 1.25;
        while z < half {
            if t.height_at(x, z) > wl {
                spots += 1;
                let (ci, cj) = (((x + half) / CELL) as i64, ((z + half) / CELL) as i64);
                let mut best = f32::INFINITY;
                for di in -5..=5i64 {
                    for dj in -5..=5i64 {
                        let (i, j) = (ci + di, cj + dj);
                        if i < 0 || j < 0 || i >= n as i64 || j >= n as i64 {
                            continue;
                        }
                        for &(px, pz) in &grid[i as usize + j as usize * n] {
                            best = best.min(((px - x).powi(2) + (pz - z).powi(2)).sqrt());
                        }
                    }
                }
                for (k, r) in reach.iter().enumerate() {
                    if best <= *r {
                        within[k] += 1;
                    }
                }
            }
            z += 2.5;
        }
        x += 2.5;
    }
    let land_cells = grid.iter().enumerate().filter(|(k, _)| {
        let (i, j) = (k % n, k / n);
        t.height_at(i as f32 * CELL - half + 2.5, j as f32 * CELL - half + 2.5) > wl
    });
    let (mut cells, mut planted) = (0usize, 0usize);
    for (_, c) in land_cells {
        cells += 1;
        planted += usize::from(!c.is_empty());
    }
    let pct = |k: usize| within[k] as f32 / spots.max(1) as f32 * 100.0;
    println!(
        "{name}: {} plants; land spots with a plant within 2 m {:.0}%, 5 m {:.0}%, 10 m {:.0}%, 25 m {:.0}%; 5 m land cells with a plant {:.0}%",
        sim.plants.len(), pct(0), pct(1), pct(2), pct(3), planted as f32 / cells.max(1) as f32 * 100.0
    );
}
