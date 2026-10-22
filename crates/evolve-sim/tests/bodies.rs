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
                let (angles, _) = body::joint_state(&sim.world, &c.parts, j).unwrap();
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
    assert!(st.outputs[output::DIG] < 0.2 && st.outputs[output::PILE] < 0.2, "upgraded creatures don't dig");
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
    let (cp, lp) = (sim.world.bodies[c.parts[0].body].translation(), sim.world.bodies[l.parts[0].body].translation());
    let rot = *sim.world.bodies[l.parts[0].body].rotation();
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
    let t = sim.world.bodies[c.parts[0].body].translation();
    assert!(t.y > sim.terrain.height_at(t.x, t.z) - 0.2);
}
