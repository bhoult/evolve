//! Multi-jointed bodies, thirst, and upgrading genomes from before them.

use std::collections::HashSet;

use evolve_sim::body::{self, JointKind, PartShape};
use evolve_sim::brain::{self, BrainGenes, BrainState, HalfVec, N_IN, N_MOTORS, OutputBias, input, output};
use evolve_sim::genome::{Genome, MAX_PARTS};
use evolve_sim::rng::Rng;
use evolve_sim::{Command, Sim, SimParams};
use half::f16;

fn quiet_params() -> SimParams {
    SimParams {
        min_population: 0,
        terrain_resolution: 65,
        weather: false,
        stones: 0,
        ..SimParams::default()
    }
}

#[test]
fn random_bodies_include_snakes_knees_tentacles_and_paddles() {
    let p = SimParams::default();
    let mut rng = Rng::new(5);
    let (mut snakes, mut knees, mut swivels, mut paddles) = (0, 0, 0, 0);
    for _ in 0..1000 {
        let g = Genome::random(&mut rng, &p);
        let plan = body::body_plan(&g);
        assert!(plan.len() <= MAX_PARTS);
        let mut used = HashSet::new();
        for part in &plan[1..] {
            let j = part.joint.as_ref().expect("every part but the torso hangs on a joint");
            for d in 0..j.kind.axes() {
                assert!(j.slot + d < N_MOTORS);
                assert!(used.insert(j.slot + d), "each joint axis has its own motor channel");
            }
            assert!(j.range[0] < 0.0 && j.range[1] > 0.0, "the rest pose is inside the range");
        }
        snakes += (plan.iter().filter(|p| p.trunk).count() >= 3) as u32;
        knees += plan
            .iter()
            .any(|p| !p.trunk && p.parent.is_some_and(|q| !plan[q].trunk))
            as u32;
        swivels += plan
            .iter()
            .any(|p| p.joint.as_ref().is_some_and(|j| j.kind == JointKind::Swivel)) as u32;
        paddles += plan
            .iter()
            .any(|p| !p.trunk && matches!(p.shape, PartShape::Box { .. })) as u32;
    }
    eprintln!("of 1000: {snakes} long bodies, {knees} with knees, {swivels} with swivels, {paddles} with paddles");
    for n in [snakes, knees, swivels, paddles] {
        assert!(n > 30);
    }
}

fn base(seed: u64) -> Genome {
    let mut g = Genome::random(&mut Rng::new(seed), &SimParams::default());
    g.spine.segments = 0;
    for l in &mut g.limbs {
        l.present = false;
    }
    g
}

fn snake() -> Genome {
    let mut g = base(9);
    g.spine.segments = 5;
    g.spine.swivel = false;
    g.spine.axis_angle = 0.0;
    g.spine.limit = 0.6;
    g
}

fn tentacled() -> Genome {
    let mut g = base(10);
    let l = &mut g.limbs[0];
    l.present = true;
    l.parent = 0;
    l.segment = 0;
    l.attach = [0.0, -1.0, 0.6];
    l.mirror = true;
    l.segments = 3;
    l.swivel_root = true;
    l.swivel_knees = true;
    g
}

fn legged() -> Genome {
    let mut g = base(11);
    g.spine.segments = 2;
    for (i, x) in [0usize, 1].into_iter().zip([0.8f32, -0.8]) {
        let l = &mut g.limbs[i];
        l.present = true;
        l.parent = 0;
        l.segment = if x > 0.0 { 0 } else { 2 };
        l.attach = [0.0, -0.5, 1.0];
        l.bend = [0.0, -0.8, 0.0];
        l.mirror = true;
        l.segments = 3;
        l.swivel_root = false;
        l.swivel_knees = false;
        l.curl = 0.5;
        l.paddle = 0.6;
    }
    g
}

#[test]
fn snakes_tentacles_and_legs_stay_stable_and_within_their_range() {
    let mut sim = Sim::new(quiet_params());
    let mut ids = Vec::new();
    for g in [snake(), tentacled(), legged()] {
        let at = sim.random_land_point(20.0);
        ids.push(sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None));
    }
    let mut swing: std::collections::HashMap<(u64, usize, usize), (f32, f32)> = Default::default();
    for _ in 0..1200 {
        sim.step();
        for c in &sim.creatures {
            for (k, j) in c.joints.iter().enumerate() {
                let (angles, _) = body::joint_state(sim.world_of(c), &c.parts, j).unwrap();
                for d in 0..j.kind.axes() {
                    let r = swing.entry((c.id, k, d)).or_insert((f32::MAX, f32::MIN));
                    r.0 = r.0.min(angles[d]);
                    r.1 = r.1.max(angles[d]);
                }
            }
        }
    }
    for (name, id) in ["snake", "tentacles", "legs"].into_iter().zip(ids) {
        let c = sim.creature(id).unwrap_or_else(|| panic!("the {name} creature died"));
        eprintln!("{name}: {} parts, {} joints", c.parts.len(), c.joints.len());
        let mut moved = 0;
        for (k, j) in c.joints.iter().enumerate() {
            for d in 0..j.kind.axes() {
                let (lo, hi) = swing[&(id, k, d)];
                // The solver enforces limits a little softly when a light part carries a
                // heavy body or a light tail whips, so allow some give.
                assert!(
                    lo >= j.range[0] - 0.55 && hi <= j.range[1] + 0.55,
                    "{name} joint {k} axis {d} went {lo:.2}..{hi:.2}, range {:?}",
                    j.range
                );
                moved += (hi - lo > 0.1) as u32;
            }
        }
        assert!(moved >= 2, "{name}'s brain should flex its joints");
    }
}

#[test]
fn creatures_dry_out_on_land_and_drink_at_the_water() {
    let mut p = quiet_params();
    p.thirst_time = 10.0;
    let mut sim = Sim::new(p.clone());
    let at = sim.random_land_point(20.0);
    let id = sim.spawn_creature(legged(), at, 0.3, f32::MAX, 0, None);
    // Drain the sea so there's nothing to drink, wherever it wanders.
    p.water_level = -100.0;
    sim.apply(Command::SetParams(Box::new(p.clone())));
    for _ in 0..300 {
        sim.step();
    }
    let dry = sim.creature(id).unwrap().hydration;
    assert!((0.4..0.6).contains(&dry), "5 s of a 10 s thirst should leave about half: {dry}");
    // Flood the world: now the mouth is in the water.
    p.water_level = 100.0;
    sim.apply(Command::SetParams(Box::new(p)));
    for _ in 0..120 {
        sim.step();
    }
    let wet = sim.creature(id).unwrap().hydration;
    assert!(wet > 0.95, "drinking for 2 s should refill: {wet}");
}

#[test]
fn old_brains_upgrade_without_changing_behavior() {
    const OLD_IN: usize = 47;
    const OLD_OUT: usize = 18;
    let mut rng = Rng::new(3);
    let h = 12;
    let halves = |n: usize, rng: &mut Rng| -> Vec<f16> { (0..n).map(|_| f16::from_f32(rng.normal() * 0.4)).collect() };
    let w_in_old = halves(h * OLD_IN, &mut rng);
    let w_out_old = halves(OLD_OUT * h, &mut rng);
    let w_rec = halves(h * h, &mut rng);
    let bias: Vec<f32> = (0..h).map(|_| rng.normal() * 0.2).collect();
    let b_out_old: Vec<f32> = (0..OLD_OUT).map(|_| rng.normal() * 0.3).collect();
    let mut g = BrainGenes {
        h,
        w_in: HalfVec(w_in_old.clone()),
        w_rec: HalfVec(w_rec),
        bias: bias.clone(),
        w_out: HalfVec(w_out_old.clone()),
        b_out: b_out_old.clone(),
        era: 0,
    };
    assert!(g.upgrade_io(&mut rng));
    assert!(!g.upgrade_io(&mut rng), "upgrading twice does nothing");

    // The old layout: 32 fixed senses, 3 memories, then 12 joint angles (6 limbs, 6 mirrors).
    let joint = |s: usize| brain::limb_motor(s % 6, 0, s >= 6);
    let x_old: Vec<f32> = (0..OLD_IN).map(|_| rng.range(-1.0, 1.0)).collect();
    let mut x_new = [0.0f32; N_IN];
    for (i, &v) in x_old.iter().enumerate() {
        let k = match i {
            0..32 => i,
            32..35 => input::MEMORY + i - 32,
            _ => input::JOINTS + joint(i - 35),
        };
        x_new[k] = v;
    }
    let map_out = |k: usize| match k {
        0..12 => output::JOINTS + joint(k),
        12 => output::EAT,
        13 => output::ATTACK,
        14 => output::MATE,
        _ => output::MEMORY + k - 15,
    };
    // One tick of the old network from rest, by hand.
    let hidden: Vec<f32> = (0..h)
        .map(|i| {
            let s: f32 = (0..OLD_IN).map(|k| w_in_old[i * OLD_IN + k].to_f32() * x_old[k]).sum();
            (bias[i] + s).tanh()
        })
        .collect();
    let mut st = BrainState::new(&g);
    brain::step_cpu(&g, OutputBias::default(), &mut st, &x_new, brain::Learning::OFF);
    for k in 0..OLD_OUT {
        let s: f32 = (0..h).map(|i| w_out_old[k * h + i].to_f32() * hidden[i]).sum();
        let old = (b_out_old[k] + s).tanh();
        let new = st.outputs[map_out(k)];
        assert!((old - new).abs() < 1e-4, "output {k}: old {old}, upgraded {new}");
    }
}

/// An old-layout brain, upgraded, must compute exactly what it used to: its outputs land on
/// today's indices and new senses don't change them.
fn check_upgrade(n_in: usize, n_out: usize, map_in: impl Fn(usize) -> usize, map_out: impl Fn(usize) -> usize) {
    let mut rng = Rng::new(n_in as u64);
    let h = 10;
    let halves = |n: usize, rng: &mut Rng| -> Vec<f16> { (0..n).map(|_| f16::from_f32(rng.normal() * 0.3)).collect() };
    let w_in_old = halves(h * n_in, &mut rng);
    let w_out = halves(n_out * h, &mut rng);
    let w_rec = halves(h * h, &mut rng);
    let bias: Vec<f32> = (0..h).map(|_| rng.normal() * 0.2).collect();
    let b_out: Vec<f32> = (0..n_out).map(|_| rng.normal() * 0.3).collect();
    let mut g = BrainGenes {
        h,
        w_in: HalfVec(w_in_old.clone()),
        w_rec: HalfVec(w_rec),
        bias: bias.clone(),
        w_out: HalfVec(w_out.clone()),
        b_out: b_out.clone(),
        era: 0,
    };
    assert!(g.upgrade_io(&mut rng));
    assert!(!g.upgrade_io(&mut rng));
    let x_old: Vec<f32> = (0..n_in).map(|_| rng.range(-1.0, 1.0)).collect();
    let mut x_new = [0.0f32; N_IN];
    for (i, &v) in x_old.iter().enumerate() {
        x_new[map_in(i)] = v;
    }
    // Senses the old layout didn't have read something, but nothing listens to them yet.
    let mapped: HashSet<usize> = (0..n_in).map(&map_in).collect();
    for k in (0..input::JOINTS).filter(|k| !mapped.contains(k)) {
        x_new[k] = 0.9;
    }
    let hidden: Vec<f32> = (0..h)
        .map(|i| {
            let s: f32 = (0..n_in).map(|k| w_in_old[i * n_in + k].to_f32() * x_old[k]).sum();
            (bias[i] + s).tanh()
        })
        .collect();
    let mut st = BrainState::new(&g);
    brain::step_cpu(&g, OutputBias::default(), &mut st, &x_new, brain::Learning::OFF);
    for k in 0..n_out {
        let s: f32 = (0..h).map(|i| w_out[k * h + i].to_f32() * hidden[i]).sum();
        let old = (b_out[k] + s).tanh();
        let new = st.outputs[map_out(k)];
        assert!((old - new).abs() < 1e-4, "{n_in}-input brain, output {k}: old {old}, upgraded {new}");
    }
    if n_out <= 90 {
        assert!(st.outputs[output::DIG] < 0.2 && st.outputs[output::PILE] < 0.2, "upgraded creatures don't dig");
    }
    if n_in < 127 {
        assert!(st.outputs[output::CALL] < 0.2, "upgraded creatures start silent");
    }
}

#[test]
fn brains_from_before_the_mating_drive_and_voices_upgrade_without_changing_behavior() {
    let out = |k: usize| if k < 85 { k } else { output::MEMORY + k - 85 };
    // Before the mating drive: 36 senses, 3 memories, 82 joint angles.
    check_upgrade(
        121,
        88,
        |i| match i {
            0..36 => i,
            36..39 => input::MEMORY + i - 36,
            _ => input::JOINTS + i - 39,
        },
        out,
    );
    // Before voices: the drive at 36, then memories and joint angles.
    check_upgrade(
        122,
        88,
        |i| match i {
            0..37 => i,
            37..40 => input::MEMORY + i - 37,
            _ => input::JOINTS + i - 40,
        },
        out,
    );
    // Before digging: hearing up to 41, then memories and joint angles; calls at 85-86.
    check_upgrade(
        127,
        90,
        |i| match i {
            0..42 => i,
            42..45 => input::MEMORY + i - 42,
            _ => input::JOINTS + i - 45,
        },
        |k| if k < 87 { k } else { output::MEMORY + k - 87 },
    );
    // Before carrying: the soil sense at 42, then memories and joint angles; digging at 87-88.
    check_upgrade(
        128,
        92,
        |i| match i {
            0..43 => i,
            43..46 => input::MEMORY + i - 43,
            _ => input::JOINTS + i - 46,
        },
        |k| if k < 89 { k } else { output::MEMORY + k - 89 },
    );
    // Before the sense of dread: holding at 46, then memories and joint angles; the actions as
    // they are now.
    check_upgrade(
        132,
        94,
        |i| match i {
            0..47 => i,
            47..50 => input::MEMORY + i - 47,
            _ => input::JOINTS + i - 50,
        },
        |k| k,
    );
}

#[test]
fn the_mating_drive_builds_when_ready_and_fades_when_hungry() {
    let p = quiet_params();
    let mut sim = Sim::new(p.clone());
    let mut ids = Vec::new();
    for _ in 0..2 {
        let at = sim.random_land_point(20.0);
        ids.push(sim.spawn_creature(legged(), at, 0.3, f32::MAX, 0, None));
    }
    for &id in &ids {
        let c = sim.creature_mut(id).unwrap();
        c.age = p.maturity_age + 1.0;
        c.energy = c.max_energy;
    }
    // The second one goes hungry.
    let hungry = ids[1];
    for _ in 0..600 {
        if let Some(c) = sim.creature_mut(hungry) {
            c.energy = c.max_energy * 0.1;
        }
        sim.step();
    }
    let fed = sim.creature(ids[0]).unwrap();
    let libido = fed.genome.traits.libido;
    let expected = (10.0 * 2f32.powf(libido) / p.mate_drive_time).min(1.0);
    assert!(
        (fed.mate_drive - expected).abs() < 0.1 || fed.mate_cooldown > 0.0,
        "10 s ready should build the drive to about {expected}: {}",
        fed.mate_drive
    );
    let starving = sim.creature(hungry).unwrap();
    assert!(starving.mate_drive < 0.01, "a hungry creature feels no urge: {}", starving.mate_drive);
}

#[test]
fn a_creature_dropped_from_the_sky_lands_with_a_thud() {
    use evolve_sim::snapshot::HighlightKind;
    let mut sim = Sim::new(quiet_params());
    let at = sim.random_land_point(20.0);
    let id = sim.spawn_creature(legged(), at, 8.0, f32::MAX, 0, None);
    let mut hardest = 0.0f32;
    for _ in 0..240 {
        sim.step();
        for h in sim.highlights.drain(..) {
            if h.kind == HighlightKind::Impact && h.subject == id {
                hardest = hardest.max(h.strength);
            }
        }
    }
    assert!(hardest > 0.5, "an 8 m fall should land hard: strongest impact {hardest}");
}

/// A genome that always calls (or wants to mate) at full strength.
fn always(mut g: Genome, k: usize) -> Genome {
    let h = g.brain.h;
    g.brain.w_out.0[k * h..(k + 1) * h].fill(f16::ZERO);
    g.brain.b_out[k] = 5.0;
    g
}

#[test]
fn creatures_hear_a_calling_neighbour_and_know_its_voice() {
    let mut sim = Sim::new(quiet_params());
    let at = sim.random_land_point(20.0);
    let caller = sim.spawn_creature(always(legged(), output::CALL), at, 0.3, f32::MAX, 0, None);
    let mut quiet = legged();
    quiet.brain.b_out[output::CALL] = -5.0;
    let listener = sim.spawn_creature(quiet, at + evolve_sim::math::Vec3::new(8.0, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    for _ in 0..90 {
        sim.step();
    }
    let c = sim.creature(caller).unwrap();
    let l = sim.creature(listener).unwrap();
    let heard = &l.brain.inputs;
    assert!(heard[input::HEAR + 2] > 0.3, "the listener should hear the call: {}", heard[input::HEAR + 2]);
    assert!(heard[input::HEAR_LIKE] > 0.99, "same genome, same voice");
    // The direction points from the listener toward the caller.
    let (cp, lp) = (sim.world_of(c).bodies[c.parts[0].body].translation(), sim.world_of(l).bodies[l.parts[0].body].translation());
    let rot = *sim.world_of(l).bodies[l.parts[0].body].rotation();
    let (fwd, side) = evolve_sim::math::heading_frame(rot);
    let to = evolve_sim::math::Vec3::new(cp.x - lp.x, 0.0, cp.z - lp.z).normalize();
    assert!((heard[input::HEAR] - to.dot(fwd)).abs() < 0.3 && (heard[input::HEAR + 1] - to.dot(side)).abs() < 0.3);
    assert!(c.brain.inputs[input::HEAR + 2] == 0.0, "nobody hears their own call");
}

#[test]
fn specks_too_small_to_feed_a_child_cannot_breed() {
    let mut p = quiet_params();
    p.maturity_age = 0.0;
    p.mate_cooldown = 0.0;
    let mut sim = Sim::new(p);
    let mut speck = always(base(12), output::MATE);
    speck.torso.half = [0.12, 0.06, 0.08];
    speck.traits.size = 0.6;
    let at = sim.random_land_point(20.0);
    for k in 0..2 {
        sim.spawn_creature(speck.clone(), at + evolve_sim::math::Vec3::new(k as f32 * 0.4, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    }
    let big = always(legged(), output::MATE);
    let far = at + evolve_sim::math::Vec3::new(15.0, 0.0, 0.0);
    for k in 0..2 {
        sim.spawn_creature(big.clone(), far + evolve_sim::math::Vec3::new(k as f32 * 1.5, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    }
    for _ in 0..240 {
        sim.step();
    }
    let specks: u32 = sim.creatures.iter().filter(|c| c.mass < 5.0).map(|c| c.life.children).sum();
    let bigs: u32 = sim.creatures.iter().filter(|c| c.mass >= 5.0).map(|c| c.life.children).sum();
    assert_eq!(specks, 0, "specks bred");
    assert!(bigs > 0, "a well-fed normal pair should still breed");
}

#[test]
fn a_child_killed_at_birth_leaves_no_more_meat_than_it_cost() {
    use evolve_sim::math::Vec3;
    let mut p = quiet_params();
    p.maturity_age = 0.0;
    p.plant_max_count = 0;
    p.plant_spawn_rate = 0.0;
    let mut sim = Sim::new(p);
    let g = always(legged(), output::MATE);
    let at = sim.random_land_point(20.0);
    sim.spawn_creature(g.clone(), at, 0.3, f32::MAX, 0, None);
    sim.spawn_creature(g, at + Vec3::new(1.5, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    let parents: Vec<u64> = sim.creatures.iter().map(|c| c.id).collect();
    let held = |sim: &Sim| -> f32 { parents.iter().filter_map(|&id| sim.creature(id)).map(|c| c.energy).sum() };
    let mut spent = 0.0;
    for _ in 0..600 {
        let before = held(&sim);
        sim.step();
        if sim.creatures.len() > 2 {
            spent = before - held(&sim);
            break;
        }
    }
    assert_eq!(sim.creatures.len(), 3, "the pair should breed");
    let child = sim.creatures.iter_mut().find(|c| !parents.contains(&c.id)).unwrap();
    child.dead = Some(evolve_sim::sim::DeathCause::Killed);
    let meat_before: f32 = sim.meat.iter().map(|m| m.energy).sum();
    sim.step();
    let meat: f32 = sim.meat.iter().map(|m| m.energy).sum::<f32>() - meat_before;
    assert!(meat > 0.0, "a carcass leaves meat");
    assert!(meat <= spent + 0.1, "the carcass left {meat} of meat, but its parents only spent {spent} on it");
}

/// Meat-eaters that do nothing but breed, bite and eat, in a world with no plants and no rain:
/// with energy conserved they die out, since every birth and death loses some.
#[test]
fn meat_eaters_cannot_live_on_their_own_dead() {
    use evolve_sim::math::Vec3;
    let mut p = quiet_params();
    p.plant_max_count = 0;
    p.plant_spawn_rate = 0.0;
    p.maturity_age = 2.0;
    p.mate_cooldown = 3.0;
    p.lifespan = 40.0;
    p.max_population = 40;
    let mut sim = Sim::new(p);
    let mut g = base(7);
    g.traits.diet = 1.0;
    let g = always(always(always(g, output::MATE), output::EAT), output::ATTACK);
    let at = sim.random_land_point(20.0);
    for k in 0..12 {
        let a = k as f32 * std::f32::consts::TAU / 12.0;
        sim.spawn_creature(g.clone(), at + Vec3::new(a.cos(), 0.0, a.sin()) * 2.0, 0.3, f32::MAX, 0, None);
    }
    let mut births = 0;
    for s in 0..400 {
        for _ in 0..60 {
            sim.step();
        }
        births = sim.stats.births;
        if s % 40 == 0 {
            let held: f32 = sim.creatures.iter().map(|c| c.energy).sum();
            let meat: f32 = sim.meat.iter().map(|m| m.energy).sum();
            eprintln!("{s:4}s alive {:3} births {births:4} energy {held:7.1} meat {meat:7.1}", sim.creatures.len());
        }
        if sim.creatures.is_empty() {
            break;
        }
    }
    assert!(births > 0, "they should breed at first");
    assert!(sim.creatures.is_empty(), "{} still alive after 400 s with nothing to eat but each other ({births} births)", sim.creatures.len());
}

#[test]
fn moving_earth_is_conserved_and_limited() {
    use evolve_sim::terrain::Terrain;
    let mut t = Terrain::generate(&quiet_params(), 1);
    let before = t.heights.clone();
    let dug = t.reshape(3.3, -7.1, -0.4, 3.0, 3.0);
    assert!((dug + 0.4).abs() < 1e-4, "dug {dug}");
    let piled = t.reshape(3.3, -7.1, 0.4, 3.0, 3.0);
    assert!((piled - 0.4).abs() < 1e-4);
    let diff: f32 = t.heights.iter().zip(&before).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff < 1e-3, "piling back what was dug restores the ground");
    // Digging stops at the depth limit.
    let deep = t.reshape(3.3, -7.1, -1000.0, 0.5, 3.0);
    let area = t.cell() * t.cell();
    assert!(deep < 0.0 && deep >= -0.5 * area * 4.0 - 1e-3, "dug {deep}, limit {}", -0.5 * area * 4.0);
}

#[test]
fn a_digging_creature_carries_earth_and_leaves_a_hole() {
    let mut p = quiet_params();
    p.dig_rate = 0.3;
    let mut sim = Sim::new(p);
    let at = sim.random_land_point(20.0);
    // A legless body, so its mouth rests on the ground (tall legs only reach it by flailing).
    let id = sim.spawn_creature(always(base(11), output::DIG), at, 0.3, f32::MAX, 0, None);
    let before = sim.terrain.heights.clone();
    for _ in 0..600 {
        sim.step();
    }
    let c = sim.creature(id).expect("the digger should survive (and not fall through its hole)");
    assert!(c.soil > 0.0, "it should be carrying earth");
    let removed: f32 = before.iter().zip(&sim.terrain.heights).map(|(b, a)| (b - a).max(0.0)).sum::<f32>()
        * sim.terrain.cell()
        * sim.terrain.cell();
    assert!((removed - c.soil).abs() < 0.01 + c.soil * 0.05, "earth removed {removed} vs carried {}", c.soil);
    // The physics ground follows: the torso sits above the terrain surface.
    let t = sim.world_of(c).bodies[c.parts[0].body].translation();
    assert!(t.y > sim.terrain.height_at(t.x, t.z) - 0.2);
}

/// A body with many joints on one part, floating in empty space with its muscles off, given a
/// spin: its motion can only die down. (Separate joint constraints pumped energy into bodies with
/// six or more joints on the torso until they flung themselves about.)
#[test]
fn many_jointed_bodies_never_gain_energy_on_their_own() {
    let mut p = quiet_params();
    p.gravity = 0.0;
    p.motor_torque_per_kg = 0.0;
    // Even calm weather has a breeze, which really would blow a floating body along.
    p.wind_force = 0.0;
    let mut many = legged();
    for i in 2..6 {
        let l = many.limbs[i].clone();
        many.limbs[i] = evolve_sim::genome::LimbGene { present: true, parent: 0, segment: 0, attach: [0.3 - 0.15 * i as f32, -0.4, 1.0], ..l };
    }
    for (name, g) in [("snake", snake()), ("tentacled", tentacled()), ("legged", legged()), ("many-limbed", many)] {
        let mut sim = Sim::new(p.clone());
        let at = sim.random_land_point(20.0);
        let id = sim.spawn_creature(g, at, 40.0, f32::MAX, 0, None);
        let parts = sim.creature(id).unwrap().parts.clone();
        let joints = sim.creature(id).unwrap().joints.len();
        let shard = sim.creature(id).unwrap().shard;
        let mb = body::multibody_mut(&mut sim.shards[shard].world, &parts).expect("a jointed body is a multibody");
        // The root's spin (generalized velocities: the root's linear, then angular, velocity).
        let mut v = mb.generalized_velocity_mut();
        v[3] = 0.5;
        v[4] = 1.0;
        v[5] = 0.3;
        let energy = |sim: &Sim| -> f32 { parts.iter().map(|q| sim.shards[shard].world.bodies.get(q.body).unwrap().kinetic_energy()).sum() };
        sim.step();
        let start = energy(&sim);
        let mut peak = start;
        for _ in 0..20 * 60 {
            sim.step();
            peak = peak.max(energy(&sim));
        }
        eprintln!("{name} ({joints} joints): kinetic energy {start:.3} J, peak {peak:.3} J");
        assert!(start > 0.01, "{name} should have been spinning");
        assert!(peak <= start * 1.05, "{name} gained energy: {start} J -> {peak} J");
    }
}

/// A body moved mid-stride into another physics world carries on as its twin that stayed:
/// same pose, joint angles and velocities at the move, and for a body that isn't chaotic, the
/// same motion after. (The world is split into several physics worlds solved in parallel, and
/// creatures move between them as they meet. A tentacle swinging against its limits is chaotic:
/// a millionth of a radian puts it centimeters off a second later, moved or not.)
#[test]
fn a_body_moved_to_another_world_carries_on_exactly() {
    use rapier3d::prelude::PhysicsWorld;
    let p = quiet_params();
    for (name, g, chaotic) in [("legged", legged(), false), ("tentacled", tentacled(), true)] {
        let plan = body::body_plan(&g);
        let mass = body::plan_mass(&plan, p.body_density);
        let make = || {
            let mut w = PhysicsWorld::new();
            w.integration_parameters.num_solver_iterations = p.solver_iterations as usize;
            w.integration_parameters.num_internal_pgs_iterations = p.solver_passes as usize;
            w
        };
        let root = evolve_sim::math::Pose::from_parts(
            evolve_sim::math::Vec3::new(0.0, 100.0, 0.0),
            evolve_sim::math::Quat::from_rotation_y(0.7),
        );
        let (mut a, mut b) = (make(), make());
        let (pa, ja) = body::spawn_body(&mut a, &plan, root, 1, mass, &p);
        let (mut pb, mut jb) = body::spawn_body(&mut b, &plan, root, 1, mass, &p);
        let drive = |w: &mut PhysicsWorld, joints: &[body::JointRef], t: usize| {
            for (n, j) in joints.iter().enumerate() {
                for &(axis, d) in j.axes() {
                    let phase = (t as f32 * 0.07 + n as f32 * 1.3 + d as f32).sin();
                    let target = 0.5 * (j.range[0] + j.range[1]) + 0.8 * phase * 0.5 * (j.range[1] - j.range[0]);
                    if let Some(data) = body::axis_joint_mut(w, j, d) {
                        let (k, c) = body::servo(&p, data.motors[axis as usize].max_force, j.inertia);
                        data.set_motor_position(axis, target, k, c);
                    }
                }
            }
        };
        for t in 0..30 {
            drive(&mut a, &ja, t);
            drive(&mut b, &jb, t);
            a.step();
            b.step();
        }
        let mut c = make();
        (pb, jb) = body::transplant(&mut b, &mut c, &plan, &pb, &jb, 1, mass, &p).expect("moved");
        assert_eq!(b.bodies.len(), 0, "nothing is left behind");
        let apart = |a: &PhysicsWorld, c: &PhysicsWorld| {
            let mut worst = 0.0f32;
            for (x, y) in pa.iter().zip(&pb) {
                let (x, y) = (a.bodies.get(x.body).unwrap(), c.bodies.get(y.body).unwrap());
                worst = worst.max((x.translation() - y.translation()).length());
                worst = worst.max((x.linvel() - y.linvel()).length() / 10.0);
                worst = worst.max((x.angvel() - y.angvel()).length() / 10.0);
            }
            worst
        };
        let at_move = apart(&a, &c);
        let (ma, mc) = (body::multibody(&a, &pa).unwrap(), body::multibody(&c, &pb).unwrap());
        let gv = ma.generalized_velocity().iter().zip(mc.generalized_velocity().iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(at_move < 1e-5 && gv < 1e-5, "{name} moved inexactly: {at_move} apart, joint speeds {gv} apart");
        let mut worst = 0.0f32;
        for t in 30..90 {
            drive(&mut a, &ja, t);
            drive(&mut c, &jb, t);
            a.step();
            c.step();
            worst = worst.max(apart(&a, &c));
        }
        eprintln!("{name}: apart by {at_move:.1e} at the move, {worst:.1e} at most over the next second");
        if !chaotic {
            assert!(worst < 1e-3, "{name} drifted from its twin by {worst}");
        }
    }
}

/// A genome whose brain never asks for output `k`.
fn never(mut g: Genome, k: usize) -> Genome {
    let h = g.brain.h;
    g.brain.w_out.0[k * h..(k + 1) * h].fill(f16::ZERO);
    g.brain.b_out[k] = -5.0;
    g
}

#[test]
fn totems_make_creatures_breed_fight_and_flee() {
    use evolve_sim::Command;
    use evolve_sim::acts::TotemKind;
    use evolve_sim::math::Vec3;
    let mut p = quiet_params();
    p.maturity_age = 0.0;
    p.plant_max_count = 0;
    p.plant_spawn_rate = 0.0;
    let mut sim = Sim::new(p);
    let run = |sim: &mut Sim, s: u64| {
        for _ in 0..s * evolve_sim::sim::TICKS_PER_SECOND {
            sim.step();
        }
    };
    // Love: a pair that would never mate breeds by the totem.
    let mut prude = never(legged(), output::MATE);
    prude.traits.libido = -1.0;
    let at = sim.random_land_point(40.0);
    let a = sim.spawn_creature(prude.clone(), at, 0.3, f32::MAX, 0, None);
    let b = sim.spawn_creature(prude, at + Vec3::new(1.5, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    run(&mut sim, 5);
    let kids = |sim: &Sim| [a, b].iter().filter_map(|&id| sim.creature(id)).map(|c| c.life.children).sum::<u32>();
    assert_eq!(kids(&sim), 0, "without the totem they don't breed");
    sim.apply(Command::Totem { at: at.to_array(), kind: TotemKind::Love });
    run(&mut sim, 20);
    assert!(kids(&sim) > 0, "by a totem of love they do");
    sim.apply(Command::Totem { at: at.to_array(), kind: TotemKind::Love });
    assert!(sim.acts.totems.is_empty(), "clicking it again takes it down");

    // War: two that never bite fight by the totem (blocks, whose whole body is a head).
    let mut meek = never(base(13), output::ATTACK);
    meek.traits.aggression = -1.0;
    let war = sim.random_land_point(40.0) + Vec3::new(0.0, 0.0, 30.0);
    let c = sim.spawn_creature(meek.clone(), war, 0.3, f32::MAX, 0, None);
    let d = sim.spawn_creature(meek, war + Vec3::new(2.0, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    sim.apply(Command::Totem { at: war.to_array(), kind: TotemKind::War });
    run(&mut sim, 30);
    let bites: u32 = [c, d].iter().filter_map(|&id| sim.creature(id)).map(|c| c.life.attacks).sum();
    assert!(bites > 0, "by a totem of war they bite");
    sim.apply(Command::Totem { at: war.to_array(), kind: TotemKind::War });

    // Fear only makes creatures want to run: a block, which can't move itself, stays where it
    // is, but it hurts by the totem and senses where it stands (getting away would be a relief).
    let fear = sim.random_land_point(40.0);
    let e = sim.spawn_creature(base(21), fear + Vec3::new(3.0, 0.0, 0.0), 0.3, f32::MAX, 0, None);
    run(&mut sim, 2);
    sim.apply(Command::Totem { at: fear.to_array(), kind: TotemKind::Fear });
    let dist = |sim: &Sim| {
        let t = sim.torso_of(sim.creature(e).unwrap()).unwrap().translation();
        (t.x - fear.x).hypot(t.z - fear.z)
    };
    let before = dist(&sim);
    run(&mut sim, 10);
    assert!(dist(&sim) < before + 0.5, "a totem of fear doesn't drive it off: {before:.1} -> {:.1} m", dist(&sim));
    let c = sim.creature(e).unwrap();
    assert_eq!(c.urge_push, Vec3::ZERO, "nor pushes it at all");
    let x = &c.brain.inputs;
    assert!(c.pain > 0.5, "it hurts: {}", c.pain);
    assert!(x[input::DREAD + 2] > 0.7, "it senses the totem close by: {}", x[input::DREAD + 2]);
    let toward = x[input::DREAD].hypot(x[input::DREAD + 1]);
    assert!((toward - 1.0).abs() < 0.01, "and which way it is: {toward}");
    assert!((c.brain.dread - x[input::DREAD + 2]).abs() < 0.05, "and keeps how close it was, for the relief of getting away");
}

#[test]
fn creatures_can_pick_up_and_carry_stones_and_drop_them() {
    use evolve_sim::carry::{Held, stone_shape};
    use evolve_sim::math::Vec3;
    let mut p = quiet_params();
    p.stones = 0;
    p.plant_max_count = 0;
    p.plant_spawn_rate = 0.0;
    p.maturity_age = 1000.0;
    let mut sim = Sim::new(p);
    let at = sim.random_land_point(40.0);
    // A low creature (its mouth near the ground) that always grips, with a stone at its mouth.
    let g = always(base(12), output::GRIP);
    let id = sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None);
    for _ in 0..30 {
        sim.step();
    }
    let mouth = |sim: &Sim| {
        let c = sim.creature(id).unwrap();
        let t = sim.torso_of(c).unwrap();
        t.position().transform_point(Vec3::new(c.torso_half().x, 0.0, 0.0))
    };
    let m = mouth(&sim);
    let stone = sim.add_obstacle(Vec3::new(m.x, 0.0, m.z), stone_shape(0.3), true, 0.0);
    for _ in 0..10 {
        sim.step();
    }
    assert!(
        matches!(sim.creature(id).unwrap().held, Some(Held::Obstacle { id, .. }) if id == stone),
        "it picks up the stone at its mouth: {:?}",
        sim.creature(id).unwrap().held
    );
    // Carried: the stone stays at its mouth as it moves.
    for _ in 0..120 {
        sim.step();
    }
    let o = sim.obstacles.iter().find(|o| o.id == stone).unwrap();
    let pos = sim.shards[o.shard].world.bodies[o.body].translation();
    assert!(pos.distance(mouth(&sim)) < 1.0, "carried at its mouth: {:.2} m away", pos.distance(mouth(&sim)));
    assert_eq!(sim.creature(id).unwrap().brain.inputs[input::HOLDING], 1.0, "and it feels it's holding something");
    // When it's gone, what it held falls and lies on the ground.
    sim.apply(evolve_sim::Command::RemoveCreature(id));
    for _ in 0..180 {
        sim.step();
    }
    let o = sim.obstacles.iter().find(|o| o.id == stone).unwrap();
    let b = &sim.shards[o.shard].world.bodies[o.body];
    assert!(b.is_dynamic(), "set down, it's a loose stone again");
    let ground = sim.terrain.height_at(b.translation().x, b.translation().z);
    assert!(b.translation().y < ground + 0.5, "lying on the ground: {:.2} above it", b.translation().y - ground);
}

#[test]
fn a_plant_carried_off_takes_root_where_it_is_dropped() {
    use evolve_sim::carry::Held;
    use evolve_sim::math::Vec3;
    let mut p = quiet_params();
    p.stones = 0;
    p.plant_max_count = 0;
    p.plant_spawn_rate = 0.0;
    let mut sim = Sim::new(p);
    let at = sim.random_land_point(40.0);
    let id = sim.spawn_creature(always(base(12), output::GRIP), at, 0.3, f32::MAX, 0, None);
    for _ in 0..30 {
        sim.step();
    }
    let c = sim.creature(id).unwrap();
    let mouth = sim.torso_of(c).unwrap().position().transform_point(Vec3::new(c.torso_half().x, 0.0, 0.0));
    sim.add_plant(mouth, 12.0);
    for _ in 0..10 {
        sim.step();
    }
    assert!(matches!(sim.creature(id).unwrap().held, Some(Held::Plant(_))), "it picks the plant up");
    assert!(sim.plants.is_empty(), "so it's no longer in the ground");
    sim.apply(evolve_sim::Command::RemoveCreature(id));
    sim.step();
    assert_eq!(sim.plants.len(), 1, "dropped, it takes root again");
}
