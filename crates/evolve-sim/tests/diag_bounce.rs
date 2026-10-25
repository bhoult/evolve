//! Diagnostic: how the best zoo creatures get about. Each is run alone on the island and
//! measured: how much of the time it's off the ground, how hard it bounces, how fast its joints
//! swing, and how far it gets.
//!
//!     EVOLVE_DIAG_ZOO=~/.local/share/evolve/zoo cargo test --release --test diag_bounce -- --ignored --nocapture
//!
//! Reads the zoo's index and genome files only (it never opens the zoo for writing).
//! `EVOLVE_DIAG_SET=name=value,...` changes params for a what-if run.

use evolve_sim::brain::BRAIN_ERA;
use evolve_sim::math::Vec3;
use evolve_sim::params::ParamRef;
use evolve_sim::sim::TICKS_PER_SECOND;
use evolve_sim::zoo::{ZooMeta, load_genome};
use evolve_sim::{Sim, SimParams};

fn params() -> SimParams {
    let mut p = SimParams { min_population: 0, weather: false, physics_shards: 1, ..SimParams::default() };
    if let Ok(sets) = std::env::var("EVOLVE_DIAG_SET") {
        for kv in sets.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("name=value");
            p.visit(&mut |info, r| {
                if info.name == k {
                    match r {
                        ParamRef::F32(x) => *x = v.parse().unwrap(),
                        ParamRef::U32(x) => *x = v.parse().unwrap(),
                        ParamRef::U64(x) => *x = v.parse().unwrap(),
                        ParamRef::Bool(x) => *x = v.parse().unwrap(),
                    }
                }
            });
        }
    }
    p
}

#[test]
#[ignore]
fn bounce() {
    let Ok(dir) = std::env::var("EVOLVE_DIAG_ZOO") else {
        println!("set EVOLVE_DIAG_ZOO to a zoo directory");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let index: Vec<ZooMeta> = serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    let mut best = index.clone();
    best.sort_by(|a, b| b.score_version.cmp(&a.score_version).then(b.score.total_cmp(&a.score)));
    let n: usize = std::env::var("EVOLVE_DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    println!("{:<16} {:>6} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7}", "creature", "score", "kg", "air%", "up m/s", "hop m", "joint/s", "moved m");
    let mut totals = [0.0f32; 5];
    let mut count = 0;
    for meta in best.iter().take(n) {
        let Ok(mut genome) = load_genome(&dir, meta.id) else {
            continue;
        };
        genome.brain.era = BRAIN_ERA; // as it was, not doubled
        if std::env::var_os("EVOLVE_DIAG_HINGES").is_some() {
            genome.spine.swivel = false;
            for l in &mut genome.limbs {
                l.swivel_root = false;
                l.swivel_knees = false;
            }
        }
        let mut sim = Sim::new(params());
        let at = sim.random_land_point(30.0);
        let id = sim.spawn_creature(genome, at, 0.3, f32::MAX, 0, None);
        for _ in 0..2 * TICKS_PER_SECOND {
            sim.step();
        }
        let steps = 20 * TICKS_PER_SECOND;
        let (mut air, mut up, mut hop, mut spin) = (0u32, 0.0f32, 0.0f32, 0.0f32);
        let (mut kind_spin, mut kind_n) = ([0.0f32; 2], [0.0f32; 2]);
        let com = |sim: &Sim| -> Option<(Vec3, Vec3, f32)> {
            let c = sim.creature(id)?;
            let (mut p, mut v, mut m) = (Vec3::ZERO, Vec3::ZERO, 0.0);
            for part in &c.parts {
                let b = sim.shards[0].world.bodies.get(part.body)?;
                p += b.center_of_mass() * b.mass();
                v += b.linvel() * b.mass();
                m += b.mass();
            }
            Some((p / m, v / m, m))
        };
        let Some((start, _, mass)) = com(&sim) else {
            continue;
        };
        let mut path = 0.0;
        let mut last = start;
        for _ in 0..steps {
            sim.step();
            let Some(c) = sim.creature(id) else {
                break;
            };
            // Off the ground: no part touching the terrain.
            let touching = sim
                .shards[0]
                .world
                .narrow_phase
                .contact_pairs_with(sim.shards[0].terrain_collider)
                .any(|pair| pair.has_any_active_contact());
            if !touching {
                air += 1;
            }
            let (p, v, _) = com(&sim).unwrap();
            up += v.y.max(0.0);
            hop = hop.max(p.y - sim.terrain.height_at(p.x, p.z));
            path += Vec3::new(p.x - last.x, 0.0, p.z - last.z).length();
            last = p;
            for j in &c.joints {
                let (Some(a), Some(b)) = (sim.world_of(c).bodies.get(c.parts[j.parent].body), sim.world_of(c).bodies.get(c.parts[j.child].body)) else {
                    continue;
                };
                let w = (b.angvel() - a.angvel()).length();
                spin += w;
                kind_spin[(j.kind == evolve_sim::body::JointKind::Swivel) as usize] += w;
                kind_n[(j.kind == evolve_sim::body::JointKind::Swivel) as usize] += 1.0;
            }
            let _ = c;
        }
        let joints = sim.creature(id).map_or(1, |c| c.joints.len().max(1)) as f32;
        let exertion = sim.creature(id).map_or(0.0, |c| c.exertion);
        let row = [
            air as f32 / steps as f32 * 100.0,
            up / steps as f32,
            hop,
            spin / steps as f32 / joints,
            Vec3::new(last.x - start.x, 0.0, last.z - start.z).length(),
        ];
        println!(
            "{:<16} {:>6.1} {:>5.0} {:>7.1} {:>7.2} {:>7.2} {:>7.1} {:>7.1}   (path {:.0} m; hinges {:.1} rad/s, swivels {:.1}; {:.2} W/kg)",
            meta.name, meta.score, mass, row[0], row[1], row[2], row[3], row[4], path,
            kind_spin[0] / kind_n[0].max(1.0), kind_spin[1] / kind_n[1].max(1.0), exertion
        );
        for (t, r) in totals.iter_mut().zip(row) {
            *t += r;
        }
        count += 1;
    }
    let c = count.max(1) as f32;
    println!(
        "{:<16} {:>6} {:>5} {:>7.1} {:>7.2} {:>7.2} {:>7.1} {:>7.1}",
        "mean", "", "", totals[0] / c, totals[1] / c, totals[2] / c, totals[3] / c, totals[4] / c
    );
}

/// against its range.
#[test]
#[ignore]
fn anatomy() {
    let (Ok(dir), Ok(name)) = (std::env::var("EVOLVE_DIAG_ZOO"), std::env::var("EVOLVE_DIAG_NAME")) else {
        println!("set EVOLVE_DIAG_ZOO and EVOLVE_DIAG_NAME");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let index: Vec<ZooMeta> = serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    let meta = index.iter().find(|m| m.name.starts_with(&name)).expect("no such creature");
    let mut genome = load_genome(&dir, meta.id).unwrap();
    genome.brain.era = BRAIN_ERA;
    let mut sim = Sim::new(params());
    let at = sim.random_land_point(30.0);
    let id = sim.spawn_creature(genome, at, 0.3, f32::MAX, 0, None);
    let c = sim.creature(id).unwrap();
    for (k, part) in c.parts.iter().enumerate() {
        let b = sim.shards[0].world.bodies.get(part.body).unwrap();
        println!("part {k:>2}: {:?} mass {:.2} kg", c.plan[k].shape, b.mass());
    }
    let joints = c.joints.clone();
    let parts = c.parts.clone();
    let mut swing = vec![0.0f32; joints.len()];
    let mut off = vec![0.0f32; joints.len()];
    let mut worst_out = vec![0.0f32; joints.len()];
    let steps = 10 * TICKS_PER_SECOND;
    for _ in 0..steps {
        sim.step();
        for (n, j) in joints.iter().enumerate() {
            let Some((angle, rate)) = evolve_sim::body::joint_state(&sim.shards[0].world, &parts, j) else {
                continue;
            };
            let a = sim.shards[0].world.bodies.get(parts[j.parent].body).unwrap();
            let b = sim.shards[0].world.bodies.get(parts[j.child].body).unwrap();
            let total = (b.angvel() - a.angvel()).length();
            swing[n] += rate[0].abs();
            off[n] += (total - rate[0].abs()).max(0.0);
            let out = (j.range[0] - angle[0]).max(angle[0] - j.range[1]).max(0.0);
            worst_out[n] = worst_out[n].max(out);
        }
    }
    for (n, j) in joints.iter().enumerate() {
        println!(
            "joint {n:>2} {:?} parts {}->{} range [{:.2}, {:.2}] strength {:.2}: swing {:.1} rad/s, off-axis {:.1} rad/s, worst past limit {:.2} rad",
            j.kind, j.parent, j.child, j.range[0], j.range[1], j.strength,
            swing[n] / steps as f32, off[n] / steps as f32, worst_out[n]
        );
    }
}

/// spun gently. Kinetic energy over 20 s should only fall.
#[test]
#[ignore]
fn rapier_minimal() {
    use rapier3d::prelude::*;
    for (legs, passes) in [(4usize, 2usize), (5, 2), (6, 2), (6, 1), (8, 2)] {
        let mut w = PhysicsWorld::new();
        w.gravity = Vec3::ZERO;
        w.integration_parameters.num_solver_iterations = 4;
        w.integration_parameters.num_internal_pgs_iterations = passes;
        let torso = w.insert(RigidBodyBuilder::dynamic(), ColliderBuilder::cuboid(0.26, 0.1, 0.1).density(1000.0)).0;
        let mut bodies = vec![torso];
        for k in 0..legs {
            let a = k as f32 * std::f32::consts::TAU / legs as f32;
            let at = Vec3::new(a.cos() * 0.2, -0.1, a.sin() * 0.1);
            let leg = w
                .insert(
                    RigidBodyBuilder::dynamic().translation(at + Vec3::new(0.0, -0.3, 0.0)),
                    ColliderBuilder::capsule_y(0.3, 0.06).density(1000.0),
                )
                .0;
            let joint = GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES)
                .local_anchor1(at)
                .local_anchor2(Vec3::new(0.0, 0.3, 0.0))
                .contacts_enabled(false);
            w.impulse_joints.insert(torso, leg, joint.build(), true);
            bodies.push(leg);
        }
        w.bodies.get_mut(torso).unwrap().set_angvel(Vec3::new(0.5, 1.0, 0.3), true);
        let mut line = format!("{legs} leg(s), {passes} pass(es):");
        for t in 0..=1200 {
            if t % 240 == 0 {
                let e: f32 = bodies.iter().map(|b| w.bodies.get(*b).unwrap().kinetic_energy()).sum();
                line += &format!(" {e:>9.4}");
            }
            w.step();
        }
        println!("{line}   (J at 0,4,8,12,16,20 s)");
    }
}

/// and are motor impulses reported?
#[test]
#[ignore]
fn multibody_check() {
    use rapier3d::prelude::*;
    use evolve_sim::body::{JointKind, PartShape};
    let Ok(dir) = std::env::var("EVOLVE_DIAG_ZOO") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mut index: Vec<ZooMeta> = serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    index.sort_by(|a, b| b.score_version.cmp(&a.score_version).then(b.score.total_cmp(&a.score)));
    let p = SimParams::default();
    let n: usize = std::env::var("EVOLVE_DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    let mut rng = evolve_sim::rng::Rng::new(3);
    for (multi, motors) in [(false, false), (true, false), (false, true), (true, true)] {
        let (mut air, mut spin, mut joints_n, mut impulse_seen, mut steps_total) = (0usize, 0.0f32, 0usize, false, 0usize);
        for meta in index.iter().take(n) {
            let Ok(genome) = load_genome(&dir, meta.id) else { continue };
            let plan = evolve_sim::body::body_plan(&genome);
            let mass = evolve_sim::body::plan_mass(&plan, p.body_density);
            let mut w = PhysicsWorld::new();
            w.integration_parameters.num_solver_iterations = 4;
            w.integration_parameters.num_internal_pgs_iterations = 2;
            let ground = w.colliders.insert(ColliderBuilder::cuboid(100.0, 1.0, 100.0).translation(Vec3::new(0.0, -1.0, 0.0)).friction(p.ground_friction));
            let root = evolve_sim::math::Pose::from_parts(Vec3::new(0.0, evolve_sim::body::plan_depth(&plan) + 0.3, 0.0), evolve_sim::math::Quat::IDENTITY);
            let mut bodies = Vec::new();
            for spec in &plan {
                let col = match spec.shape {
                    PartShape::Box { half } => ColliderBuilder::cuboid(half[0], half[1], half[2]),
                    PartShape::Capsule { half_height, radius } => ColliderBuilder::capsule_y(half_height, radius),
                }
                .density(p.body_density)
                .friction(p.body_friction)
                // Parts of one creature don't collide with each other.
                .collision_groups(InteractionGroups::new(Group::GROUP_2, Group::GROUP_1, InteractionTestMode::And));
                bodies.push(w.insert(RigidBodyBuilder::dynamic().pose(root * spec.rest).angular_damping(0.3).linear_damping(p.air_drag), col).0);
            }
            let _ = ground;
            let mut handles = Vec::new();
            for (k, spec) in plan.iter().enumerate() {
                let (Some(parent), Some(js)) = (spec.parent, &spec.joint) else { continue };
                // A multibody joint can't have exactly two free angles: a swivel becomes a ball
                // joint whose twist is held near zero by a limit.
                let mask = match (js.kind, multi) {
                    (JointKind::Hinge, _) => JointAxesMask::LOCKED_REVOLUTE_AXES,
                    (JointKind::Swivel, false) => JointAxesMask::LIN_AXES | JointAxesMask::ANG_X,
                    (JointKind::Swivel, true) => JointAxesMask::LIN_AXES,
                };
                let axes: &[JointAxis] = match js.kind {
                    JointKind::Hinge => &[JointAxis::AngX],
                    JointKind::Swivel => &[JointAxis::AngY, JointAxis::AngZ],
                };
                let mid = 0.5 * (js.range[0] + js.range[1]);
                let force = if motors { p.motor_torque_per_kg * js.strength * mass } else { 0.0 };
                let mut b = GenericJointBuilder::new(mask).local_frame1(js.frame1).local_frame2(js.frame2).contacts_enabled(false);
                for &axis in axes {
                    b = b.limits(axis, js.range).motor_position(axis, mid, 200.0, 20.0).motor_max_force(axis, force);
                }
                if multi && js.kind == JointKind::Swivel {
                    b = b.limits(JointAxis::AngX, [-0.02, 0.02]);
                }
                if multi {
                    handles.push((Err(w.multibody_joints.insert(bodies[parent], bodies[k], b.build(), true).expect("multibody")), axes, js.range, parent, k));
                } else {
                    handles.push((Ok(w.impulse_joints.insert(bodies[parent], bodies[k], b.build(), true)), axes, js.range, parent, k));
                }
            }
            for step in 0..20 * 60 {
                if motors && step % 2 == 0 {
                    for (h, axes, range, _, _) in &handles {
                        for &axis in axes.iter() {
                            let target = range[0] + (range[1] - range[0]) * rng.f32();
                            match h {
                                Ok(h) => {
                                    w.impulse_joints.get_mut(*h, true).unwrap().data.set_motor_position(axis, target, 200.0, 20.0);
                                }
                                Err(h) => {
                                    let (mb, link) = w.multibody_joints.get_mut(*h).unwrap();
                                    mb.link_mut(link).unwrap().joint.data.set_motor_position(axis, target, 200.0, 20.0);
                                }
                            }
                        }
                    }
                }
                w.step();
                if step >= 120 {
                    steps_total += 1;
                    let touching = w.contact_pairs_with(ground).any(|pair| pair.has_any_active_contact());
                    if !touching {
                        air += 1;
                    }
                    for (h, _, _, parent, k) in &handles {
                        let (a, b) = (w.bodies.get(bodies[*parent]).unwrap(), w.bodies.get(bodies[*k]).unwrap());
                        spin += (b.angvel() - a.angvel()).length();
                        joints_n += 1;
                        let imp = match h {
                            Ok(h) => w.impulse_joints.get(*h).unwrap().data.motors[3].impulse,
                            Err(h) => {
                                let (mb, link) = w.multibody_joints.get(*h).unwrap();
                                mb.link(link).unwrap().joint.data.motors[3].impulse
                            }
                        };
                        impulse_seen |= imp != 0.0;
                    }
                }
            }
            // Does an impulse on a link body stick?
            if multi && meta.id == index[0].id {
                let b = w.bodies.get_mut(bodies[0]).unwrap();
                let before = b.linvel();
                b.apply_impulse(Vec3::new(0.0, 50.0 * mass, 0.0), true);
                let after_apply = w.bodies.get(bodies[0]).unwrap().linvel();
                w.step();
                let after_step = w.bodies.get(bodies[0]).unwrap().linvel();
                println!("multibody torso: vy {:.2} -> after impulse {:.2} -> after a step {:.2}", before.y, after_apply.y, after_step.y);
                let b = w.bodies.get_mut(bodies[0]).unwrap();
                b.add_force(Vec3::new(0.0, 50.0 * mass * 60.0, 0.0), true);
                w.step();
                let with_force = w.bodies.get(bodies[0]).unwrap().linvel();
                w.bodies.get_mut(bodies[0]).unwrap().reset_forces(true);
                println!("multibody torso: a force of 50 g for one step -> vy {:.2}", with_force.y);
            }
        }
        println!(
            "{:<10} motors {:<5}: off the ground {:>5.1}%, joint spin {:>5.1} rad/s, motor impulses reported: {impulse_seen}",
            if multi { "multibody" } else { "impulse" }, motors,
            air as f32 / steps_total.max(1) as f32 * 100.0, spin / joints_n.max(1) as f32
        );
    }
}

/// Prints the first few hinge angles over time, for impulse and multibody joints.
#[test]
#[ignore]
fn motor_tracking() {
    use rapier3d::prelude::*;
    use evolve_sim::body::{JointKind, PartShape};
    let (Ok(dir), Ok(name)) = (std::env::var("EVOLVE_DIAG_ZOO"), std::env::var("EVOLVE_DIAG_NAME")) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let index: Vec<ZooMeta> = serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    let meta = index.iter().find(|m| m.name.starts_with(&name)).unwrap();
    let plan = evolve_sim::body::body_plan(&load_genome(&dir, meta.id).unwrap());
    let p = SimParams::default();
    let mass = evolve_sim::body::plan_mass(&plan, p.body_density);
    for multi in [false, true] {
        let mut w = PhysicsWorld::new();
        w.integration_parameters.num_solver_iterations = 4;
        w.integration_parameters.num_internal_pgs_iterations = 2;
        w.colliders.insert(ColliderBuilder::cuboid(100.0, 1.0, 100.0).translation(Vec3::new(0.0, -1.0, 0.0)).friction(p.ground_friction));
        let root = evolve_sim::math::Pose::from_parts(Vec3::new(0.0, evolve_sim::body::plan_depth(&plan) + 0.3, 0.0), evolve_sim::math::Quat::IDENTITY);
        let mut bodies = Vec::new();
        for spec in &plan {
            let col = match spec.shape {
                PartShape::Box { half } => ColliderBuilder::cuboid(half[0], half[1], half[2]),
                PartShape::Capsule { half_height, radius } => ColliderBuilder::capsule_y(half_height, radius),
            }
            .density(p.body_density)
            .friction(p.body_friction)
            .collision_groups(InteractionGroups::new(Group::GROUP_2, Group::GROUP_1, InteractionTestMode::And));
            bodies.push(w.insert(RigidBodyBuilder::dynamic().pose(root * spec.rest).angular_damping(0.3).linear_damping(p.air_drag), col).0);
        }
        let mut handles = Vec::new();
        for (k, spec) in plan.iter().enumerate() {
            let (Some(parent), Some(js)) = (spec.parent, &spec.joint) else { continue };
            if js.kind != JointKind::Hinge {
                continue;
            }
            let mid = 0.5 * (js.range[0] + js.range[1]);
            let force = p.motor_torque_per_kg * js.strength * mass;
            let b = GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES)
                .local_frame1(js.frame1)
                .local_frame2(js.frame2)
                .contacts_enabled(false)
                .limits(JointAxis::AngX, js.range)
                .motor_position(JointAxis::AngX, mid, 200.0, 20.0)
                .motor_max_force(JointAxis::AngX, force);
            let h = if multi {
                Err(w.multibody_joints.insert(bodies[parent], bodies[k], b.build(), true).unwrap())
            } else {
                Ok(w.impulse_joints.insert(bodies[parent], bodies[k], b.build(), true))
            };
            handles.push((h, js.range, parent, k));
        }
        let mut lines = vec![String::new(); handles.len().min(3)];
        for step in 0..120 {
            let high = step < 60;
            for (h, range, _, _) in &handles {
                let target = if high { range[1] } else { range[0] };
                match h {
                    Ok(h) => {
                        w.impulse_joints.get_mut(*h, true).unwrap().data.set_motor_position(JointAxis::AngX, target, 200.0, 20.0);
                    }
                    Err(h) => {
                        let (mb, link) = w.multibody_joints.get_mut(*h).unwrap();
                        mb.link_mut(link).unwrap().joint.data.set_motor_position(JointAxis::AngX, target, 200.0, 20.0);
                    }
                }
            }
            w.step();
            if step % 10 == 9 {
                for (n, line) in lines.iter_mut().enumerate() {
                    let (h, range, parent, k) = &handles[n];
                    let angle = match h {
                        Ok(h) => {
                            let j = w.impulse_joints.get(*h).unwrap();
                            let r1 = *w.bodies.get(bodies[*parent]).unwrap().rotation() * j.data.local_frame1.rotation;
                            let r2 = *w.bodies.get(bodies[*k]).unwrap().rotation() * j.data.local_frame2.rotation;
                            let e = r1.inverse() * r2;
                            2.0 * e.x.atan2(e.w)
                        }
                        Err(h) => {
                            let (mb, link) = w.multibody_joints.get(*h).unwrap();
                            let j = &mb.link(link).unwrap().joint;
                            let r1 = *w.bodies.get(bodies[*parent]).unwrap().rotation() * j.data.local_frame1.rotation;
                            let r2 = *w.bodies.get(bodies[*k]).unwrap().rotation() * j.data.local_frame2.rotation;
                            let e = r1.inverse() * r2;
                            2.0 * e.x.atan2(e.w)
                        }
                    };
                    if line.is_empty() {
                        *line = format!("  hinge {n} range [{:+.2}, {:+.2}]:", range[0], range[1]);
                    }
                    *line += &format!(" {angle:+.2}");
                }
            }
        }
        println!("{} (target high end for 1 s, then low end; angle every 1/6 s):", if multi { "multibody" } else { "impulse" });
        for l in lines {
            println!("{l}");
        }
    }
}

/// joint limits, for a few solver settings.
#[test]
#[ignore]
fn multibody_snake_energy() {
    use rapier3d::prelude::*;
    use evolve_sim::body::PartShape;
    let mut g = evolve_sim::genome::Genome::random(&mut evolve_sim::rng::Rng::new(9), &SimParams::default());
    g.spine.segments = 5;
    g.spine.swivel = false;
    for l in &mut g.limbs {
        l.present = false;
    }
    let plan = evolve_sim::body::body_plan(&g);
    let p = SimParams::default();
    for (limits, iters, passes, drag) in [(true, 4, 2, 0), (true, 4, 2, 1), (true, 4, 2, 2), (true, 4, 2, 3)] {
        let mut w = PhysicsWorld::new();
        w.gravity = Vec3::ZERO;
        w.integration_parameters.num_solver_iterations = iters;
        w.integration_parameters.num_internal_pgs_iterations = passes;
        let root = evolve_sim::math::Pose::from_parts(Vec3::new(0.0, 50.0, 0.0), evolve_sim::math::Quat::IDENTITY);
        let mut bodies = Vec::new();
        for spec in &plan {
            let col = match spec.shape {
                PartShape::Box { half } => ColliderBuilder::cuboid(half[0], half[1], half[2]),
                PartShape::Capsule { half_height, radius } => ColliderBuilder::capsule_y(half_height, radius),
            }
            .density(p.body_density)
            .collision_groups(InteractionGroups::none());
            bodies.push(w.insert(RigidBodyBuilder::dynamic().can_sleep(false).pose(root * spec.rest), col).0);
        }
        let mut range0 = [0.0; 2];
        for (k, spec) in plan.iter().enumerate() {
            let (Some(parent), Some(js)) = (spec.parent, &spec.joint) else { continue };
            range0 = js.range;
            let mut b = GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES).local_frame1(js.frame1).local_frame2(js.frame2);
            if limits {
                b = b.limits(JointAxis::AngX, js.range);
            }
            w.multibody_joints.insert(bodies[parent], bodies[k], b.build(), true).unwrap();
        }
        {
            let link = *w.multibody_joints.rigid_body_link(bodies[0]).unwrap();
            let mb = w.multibody_joints.get_multibody_mut(link.multibody).unwrap();
            let mut v = mb.generalized_velocity_mut();
            v[3] = 0.5;
            v[4] = 1.0;
            v[5] = 0.3;
        }
        let energy = |w: &PhysicsWorld| -> f32 { bodies.iter().map(|b| w.bodies.get(*b).unwrap().kinetic_energy()).sum() };
        let label = ["no drag", "drag from linvel()", "drag from velocity at COM", "drag, linvel(), as torque-free impulse"][drag];
        let mut line = format!("limits {limits:<5} {iters}x{passes} {label:<38}:");
        let _ = range0;
        for t in 0..=1200 {
            if t % 240 == 0 {
                line += &format!(" {:>7.3}", energy(&w));
            }
            for &h in &bodies {
                let b = w.bodies.get_mut(h).unwrap();
                b.reset_forces(false);
                let v = match drag {
                    1 | 3 => b.linvel(),
                    2 => b.velocity_at_point(b.center_of_mass()),
                    _ => continue,
                };
                let f = -v * v.length() * 2.0;
                if drag == 3 {
                    b.apply_impulse(f / 60.0, false);
                } else {
                    b.add_force(f, false);
                }
            }
            w.step();
        }
        println!("{line}");
    }
}

/// The test snake in the sim, zero gravity, muscles off, spun, with sim forces switched off one
/// at a time: which one adds energy?
#[test]
#[ignore]
fn sim_snake_energy() {
    use evolve_sim::body;
    let mut g = evolve_sim::genome::Genome::random(&mut evolve_sim::rng::Rng::new(9), &SimParams::default());
    g.spine.segments = 5;
    g.spine.swivel = false;
    for l in &mut g.limbs {
        l.present = false;
    }
    for case in ["as is", "no wind force", "no air drag", "no wind, no drag", "no spin cap"] {
        let mut p = SimParams { min_population: 0, weather: false, gravity: 0.0, motor_torque_per_kg: 0.0, ..SimParams::default() };
        match case {
            "no wind force" => p.wind_force = 0.0,
            "no air drag" => p.air_drag = 0.0,
            "no wind, no drag" => {
                p.wind_force = 0.0;
                p.air_drag = 0.0;
            }
            "no spin cap" => p.max_body_spin = 1000.0,
            _ => {}
        }
        let mut sim = Sim::new(p);
        let at = sim.random_land_point(20.0);
        let id = sim.spawn_creature(g.clone(), at, 40.0, f32::MAX, 0, None);
        let parts = sim.creature(id).unwrap().parts.clone();
        {
            let mb = body::multibody_mut(&mut sim.shards[0].world, &parts).unwrap();
            let mut v = mb.generalized_velocity_mut();
            v[3] = 0.5;
            v[4] = 1.0;
            v[5] = 0.3;
        }
        let energy = |sim: &Sim| -> f32 { parts.iter().map(|q| sim.shards[0].world.bodies.get(q.body).unwrap().kinetic_energy()).sum() };
        let mut line = format!("{case:<18}:");
        for t in 0..=1200 {
            if t % 240 == 1 {
                line += &format!(" {:>7.3}", energy(&sim));
            }
            sim.step();
        }
        println!("{line}");
    }
}

/// Where a physics step's time goes, by rapier stage, with N (EVOLVE_DIAG_N) random creatures.
/// Needs rapier's "profiler" feature.
#[test]
#[ignore]
fn physics_stages() {
    let n: usize = std::env::var("EVOLVE_DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let mut sim = Sim::new(SimParams { min_population: 0, weather: false, physics_shards: 1, ..SimParams::default() });
    let mut rng = evolve_sim::rng::Rng::new(4);
    for _ in 0..n {
        let g = evolve_sim::genome::Genome::random(&mut rng, &sim.params);
        let at = sim.random_land_point(10.0);
        sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None);
    }
    for _ in 0..120 {
        sim.step();
    }
    sim.shards[0].world.physics_pipeline.counters.enable();
    let mut acc = [0.0f64; 8];
    let steps = 120;
    for _ in 0..steps {
        sim.step();
        let c = &sim.shards[0].world.physics_pipeline.counters;
        let s = &c.stages;
        for (a, v) in acc.iter_mut().zip([
            c.step_time_ms(),
            s.collision_detection_time.time_ms(),
            s.island_construction_time.time_ms(),
            s.island_constraints_collection_time.time_ms(),
            s.solver_time.time_ms(),
            c.solver.velocity_assembly_time.time_ms(),
            c.solver.velocity_resolution_time.time_ms(),
            c.solver.velocity_update_time.time_ms(),
        ]) {
            *a += v / steps as f64;
        }
    }
    let parts: usize = sim.creatures.iter().map(|c| c.parts.len()).sum();
    println!("{} creatures, {parts} parts: step {:.2} ms = collision {:.2} + islands {:.2} + constraint collection {:.2} + solver {:.2} (assembly {:.2}, resolution {:.2}, update {:.2})",
        sim.creatures.len(), acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
}

/// Would splitting the world into several physics worlds stepped in parallel pay? One world of
/// 8n creatures against 8 worlds of n, on an 8-thread pool (CPU brains, to time the CPU only).
#[test]
#[ignore]
fn shard_scaling() {
    use rayon::prelude::*;
    // Safe: set before any thread reads it.
    unsafe { std::env::set_var("EVOLVE_CPU_BRAINS", "1") };
    let n: usize = std::env::var("EVOLVE_DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let k: usize = std::env::var("EVOLVE_DIAG_K").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(k).build().unwrap();
    let make = |count: usize, seed: u64| {
        let mut sim = Sim::new(SimParams { min_population: 0, weather: false, seed, physics_shards: 1, ..SimParams::default() });
        let mut rng = evolve_sim::rng::Rng::new(seed);
        for _ in 0..count {
            let g = evolve_sim::genome::Genome::random(&mut rng, &sim.params);
            let at = sim.random_land_point(10.0);
            sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None);
        }
        sim
    };
    let steps = 120;
    let mut one = make(k * n, 1);
    let t_one = pool.install(|| {
        for _ in 0..60 {
            one.step();
        }
        let t = std::time::Instant::now();
        for _ in 0..steps {
            one.step();
        }
        t.elapsed().as_secs_f64() * 1000.0 / steps as f64
    });
    let mut many: Vec<Sim> = (0..k).map(|i| make(n, 1 + i as u64)).collect();
    if std::env::var_os("EVOLVE_DIAG_OWN_POOLS").is_some() {
        // Each world's physics on a pool of its own, one thread: rapier's solver workers then
        // never spin waiting on each other across worlds.
        for s in &mut many {
            s.shards[0].world.physics_pipeline.configure_thread_pool(1).unwrap();
        }
    }
    let t_many = pool.install(|| {
        for _ in 0..60 {
            many.par_iter_mut().for_each(|s| s.step());
        }
        let t = std::time::Instant::now();
        for _ in 0..steps {
            many.par_iter_mut().for_each(|s| s.step());
        }
        t.elapsed().as_secs_f64() * 1000.0 / steps as f64
    });
    let mut single = make(n, 9);
    let t_single = pool.install(|| {
        for _ in 0..60 {
            single.step();
        }
        let t = std::time::Instant::now();
        for _ in 0..steps {
            single.step();
        }
        t.elapsed().as_secs_f64() * 1000.0 / steps as f64
    });
    println!(
        "one world of {}: {t_one:.1} ms/step; {k} worlds of {n} in parallel: {t_many:.1} ms/step ({:.1}x faster); one world of {n} alone: {t_single:.1} ms/step",
        k * n, t_one / t_many
    );
}

/// How hard random creatures work and how out of breath they get (EVOLVE_DIAG_SET to vary).
#[test]
#[ignore]
fn breathlessness_now() {
    let mut sim = Sim::new(SimParams { min_population: 0, weather: false, ..params() });
    let mut rng = evolve_sim::rng::Rng::new(5);
    for _ in 0..60 {
        let g = evolve_sim::genome::Genome::random(&mut rng, &sim.params);
        let at = sim.random_land_point(10.0);
        sim.spawn_creature(g, at, 0.3, f32::MAX, 0, None);
    }
    let start: Vec<(u64, Vec3)> = sim.creatures.iter().map(|c| (c.id, sim.world_of(c).bodies.get(c.parts[0].body).unwrap().translation())).collect();
    let (mut lifted, mut samples) = (0usize, 0usize);
    for step in 0..30 * 60 {
        sim.step();
        if step % 30 == 0 {
            for c in &sim.creatures {
                let t = sim.world_of(c).bodies.get(c.parts[0].body).unwrap().translation();
                samples += 1;
                lifted += (t.y - sim.terrain.height_at(t.x, t.z) > c.torso_half().y + 0.15) as usize;
            }
        }
    }
    let mut ex: Vec<f32> = sim.creatures.iter().map(|c| c.exertion).collect();
    ex.sort_by(|a, b| a.total_cmp(b));
    let winded = sim.creatures.iter().filter(|c| c.oxygen < 0.3).count();
    let moved: Vec<f32> = start
        .iter()
        .filter_map(|(id, p)| sim.creature(*id).map(|c| (sim.world_of(c).bodies.get(c.parts[0].body).unwrap().translation() - *p).length()))
        .collect();
    let mut m = moved.clone();
    m.sort_by(|a, b| a.total_cmp(b));
    println!(
        "{} creatures after 30 s: exertion median {:.1} W/kg (90th pct {:.1}); {winded} out of breath (<30%); torso off the ground {:.0}% of samples; moved median {:.1} m, best {:.1} m",
        sim.creatures.len(), ex[ex.len() / 2], ex[ex.len() * 9 / 10], lifted as f32 / samples.max(1) as f32 * 100.0, m[m.len() / 2], m[m.len() - 1]
    );
}

/// How sensitive is the tentacled test body to a millionth of a radian? Two identical worlds;
/// one gets one joint nudged by 1e-6 rad mid-motion; how far apart are they a second later?
#[test]
#[ignore]
fn swivel_sensitivity() {
    use evolve_sim::body;
    use rapier3d::prelude::PhysicsWorld;
    let p = SimParams { min_population: 0, terrain_resolution: 65, weather: false, ..SimParams::default() };
    let mut g = evolve_sim::genome::Genome::random(&mut evolve_sim::rng::Rng::new(10), &SimParams::default());
    g.spine.segments = 0;
    for l in &mut g.limbs {
        l.present = false;
    }
    let l = &mut g.limbs[0];
    l.present = true;
    l.parent = 0;
    l.segment = 0;
    l.attach = [0.0, -1.0, 0.6];
    l.mirror = true;
    l.segments = 3;
    l.swivel_root = true;
    l.swivel_knees = true;
    let plan = body::body_plan(&g);
    let mass = body::plan_mass(&plan, p.body_density);
    let make = || {
        let mut w = PhysicsWorld::new();
        w.integration_parameters.num_solver_iterations = p.solver_iterations as usize;
        w.integration_parameters.num_internal_pgs_iterations = p.solver_passes as usize;
        w
    };
    let root = evolve_sim::math::Pose::from_parts(evolve_sim::math::Vec3::new(0.0, 100.0, 0.0), evolve_sim::math::Quat::from_rotation_y(0.7));
    let (mut a, mut d) = (make(), make());
    let (pa, ja) = body::spawn_body(&mut a, &plan, root, 1, mass, &p);
    let (pd, jd) = body::spawn_body(&mut d, &plan, root, 1, mass, &p);
    let drive = |w: &mut PhysicsWorld, joints: &[body::JointRef], t: usize| {
        for (n, j) in joints.iter().enumerate() {
            for &(axis, dd) in j.axes() {
                let phase = (t as f32 * 0.07 + n as f32 * 1.3 + dd as f32).sin();
                let target = 0.5 * (j.range[0] + j.range[1]) + 0.8 * phase * 0.5 * (j.range[1] - j.range[0]);
                if let Some(data) = body::axis_joint_mut(w, j, dd) {
                    data.set_motor_position(axis, target, 200.0, 20.0);
                }
            }
        }
    };
    for t in 0..30 {
        drive(&mut a, &ja, t);
        drive(&mut d, &jd, t);
        a.step();
        d.step();
    }
    {
        let link = *d.multibody_joints.rigid_body_link(pd[0].body).unwrap();
        let mb = d.multibody_joints.get_multibody_mut(link.multibody).unwrap();
        let mut disp = vec![0.0; mb.ndofs()];
        disp[7] = 1e-6;
        mb.apply_displacements(&disp);
        mb.forward_kinematics(&d.bodies, true);
        mb.update_rigid_bodies(&mut d.bodies, true);
    }
    let mut worst = 0.0f32;
    for t in 30..90 {
        drive(&mut a, &ja, t);
        drive(&mut d, &jd, t);
        a.step();
        d.step();
        for (x, y) in pa.iter().zip(&pd) {
            worst = worst.max((a.bodies.get(x.body).unwrap().translation() - d.bodies.get(y.body).unwrap().translation()).length());
        }
    }
    println!("nudged by 1e-6 rad: apart by {worst:.2e} m a second later");
}

/// How creatures in a saved world (EVOLVE_DIAG_SAVE, read only) bunch up: groups of creatures
/// close enough to touch soon (bodies' reach plus a margin), which must share a physics world.
#[test]
#[ignore]
fn touching_groups() {
    let Ok(path) = std::env::var("EVOLVE_DIAG_SAVE") else {
        return;
    };
    unsafe { std::env::set_var("EVOLVE_CPU_BRAINS", "1") };
    let sim = evolve_sim::save::load(std::path::Path::new(&path)).expect("load");
    let margin: f32 = std::env::var("EVOLVE_DIAG_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
    let spots: Vec<(Vec3, f32)> = sim
        .creatures
        .iter()
        .map(|c| (sim.world_of(c).bodies.get(c.parts[0].body).unwrap().translation(), evolve_sim::body::plan_extent(&c.plan)))
        .collect();
    let n = spots.len();
    let mut group: Vec<usize> = (0..n).collect();
    fn find(g: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while g[r] != r {
            r = g[r];
        }
        g[i] = r;
        r
    }
    for i in 0..n {
        for j in i + 1..n {
            let reach = spots[i].1 + spots[j].1 + margin;
            if (spots[i].0 - spots[j].0).length() < reach {
                let (a, b) = (find(&mut group, i), find(&mut group, j));
                group[a] = b;
            }
        }
    }
    let mut sizes = std::collections::HashMap::new();
    for i in 0..n {
        *sizes.entry(find(&mut group, i)).or_insert(0usize) += 1;
    }
    let mut s: Vec<usize> = sizes.into_values().collect();
    s.sort_unstable_by(|a, b| b.cmp(a));
    println!("{n} creatures at t={:.0}s: {} groups, largest {:?}", sim.time, s.len(), &s[..s.len().min(8)]);
}

/// What limits a creature pushing on the ground: each body is held with every joint at one end
/// of its range, then the other, and the torso's height above the ground is compared, under a
/// few motor settings (and one near-rigid reference that shows what the legs could do).
#[test]
#[ignore]
fn push_ups() {
    use evolve_sim::brain::{N_MOTORS, output};
    use evolve_sim::genome::Genome;
    use evolve_sim::rng::Rng;
    use half::f16;
    fn base(seed: u64) -> Genome {
        let mut g = Genome::random(&mut Rng::new(seed), &SimParams::default());
        g.spine.segments = 0;
        for l in &mut g.limbs {
            l.present = false;
        }
        g
    }
    fn legs(seed: u64, spine: u8, segments: u8, length: f32) -> Genome {
        let mut g = base(seed);
        g.spine.segments = spine;
        for (i, front) in [true, false].into_iter().enumerate() {
            let l = &mut g.limbs[i];
            l.present = true;
            l.parent = 0;
            l.segment = if front { 0 } else { spine };
            l.attach = [0.0, -0.5, 1.0];
            l.bend = [0.0, -0.8, 0.0];
            l.mirror = true;
            l.segments = segments;
            l.length = length;
            l.strength = 1.0;
            l.swivel_root = false;
            l.swivel_knees = false;
            l.curl = 0.5;
        }
        g
    }
    fn hold(mut g: Genome, sign: f32) -> Genome {
        let h = g.brain.h;
        for k in output::JOINTS..output::JOINTS + N_MOTORS {
            g.brain.w_out.0[k * h..(k + 1) * h].fill(f16::ZERO);
            g.brain.b_out[k] = 5.0 * sign;
        }
        g
    }
    let bodies = [("4 legs, 2 segments", legs(11, 1, 2, 0.5)), ("4 legs, 3 segments", legs(12, 2, 3, 0.45)), ("4 short stiff legs", legs(13, 0, 1, 0.35))];
    // (label, N·m per kg, reach, damping ratio)
    let configs: [(&str, f32, f32, f32); 5] = [
        ("servos, 6 N·m/kg", 6.0, 0.2, 1.0),
        ("servos, 3 N·m/kg", 3.0, 0.2, 1.0),
        ("servos, 12 N·m/kg", 12.0, 0.2, 1.0),
        ("servos, reach 0.5", 6.0, 0.5, 1.0),
        ("near-rigid reference", 1000.0, 0.02, 1.0),
    ];
    for (name, g) in &bodies {
        println!("{name}:");
        for (label, torque, k, c) in configs {
            let mut heights = [0.0f32; 2];
            for (s, sign) in [-1.0f32, 1.0].into_iter().enumerate() {
                let mut p = params();
                p.exploration = 0.0;
                p.motor_torque_per_kg = torque;
                p.motor_reach = k;
                p.motor_damping_ratio = c;
                p.thirst_time = 0.0;
                let mut sim = Sim::new(p);
                let at = sim.random_land_point(30.0);
                let id = sim.spawn_creature(hold(g.clone(), sign), at, 0.2, f32::MAX, 0, None);
                let (mut sum, mut n) = (0.0, 0);
                for step in 0..(5 * TICKS_PER_SECOND) {
                    sim.step();
                    if step >= 4 * TICKS_PER_SECOND {
                        let c = sim.creature(id).unwrap();
                        let t = sim.torso_of(c).unwrap().translation();
                        sum += t.y - sim.terrain.height_at(t.x, t.z);
                        n += 1;
                    }
                }
                heights[s] = sum / n as f32;
            }
            let mass = sim_mass(g);
            println!(
                "  {label:28} torso {:.2} m / {:.2} m above ground (lift {:+.2} m), {:.0} kg",
                heights[0],
                heights[1],
                (heights[1] - heights[0]).abs(),
                mass
            );
        }
    }
    fn sim_mass(g: &Genome) -> f32 {
        evolve_sim::body::plan_mass(&evolve_sim::body::body_plan(g), SimParams::default().body_density)
    }
}

/// How hard muscles work in a world of random creatures: how often a motor is at its strength
/// limit, muscle power, breath, and how fast bodies and parts move (flinging shows as part speeds
/// far above torso speeds).
#[test]
#[ignore]
fn motor_load() {
    use evolve_sim::body;
    let mut p = params();
    p.min_population = 40;
    p.rain_from_world = 0.0;
    p.rain_from_zoo = 0.0;
    p.thirst_time = 0.0;
    let mut sim = Sim::new(p);
    for _ in 0..(10 * TICKS_PER_SECOND) {
        sim.step();
    }
    let (mut axes, mut saturated) = (0u64, 0u64);
    let (mut breathless, mut samples) = (0u64, 0u64);
    let mut exertion = Vec::new();
    let mut torso_speed = Vec::new();
    let mut part_speed = Vec::new();
    for step in 0..(60 * TICKS_PER_SECOND) {
        sim.step();
        if step % 6 != 0 {
            continue;
        }
        for c in &sim.creatures {
            let w = sim.world_of(c);
            for j in &c.joints {
                let Some((angle, rate)) = body::joint_state(w, &c.parts, j) else { continue };
                for &(axis, d) in j.axes() {
                    let Some(data) = body::axis_joint(w, j, d) else { continue };
                    let m = &data.motors[axis as usize];
                    let ask = m.stiffness * (m.target_pos - angle[d]) + m.damping * (m.target_vel - rate[d]);
                    axes += 1;
                    if ask.abs() >= m.max_force * 0.98 {
                        saturated += 1;
                    }
                }
            }
            samples += 1;
            if c.oxygen < 0.3 {
                breathless += 1;
            }
            exertion.push(c.exertion);
            if let Some(t) = sim.torso_of(c) {
                let v = t.linvel();
                torso_speed.push((v.x * v.x + v.z * v.z).sqrt());
            }
            for q in &c.parts {
                if let Some(b) = w.bodies.get(q.body) {
                    part_speed.push(b.linvel().length());
                }
            }
        }
    }
    let pct = |v: &mut Vec<f32>, q: f32| {
        v.sort_by(f32::total_cmp);
        v.get(((v.len() as f32 * q) as usize).min(v.len().saturating_sub(1))).copied().unwrap_or(0.0)
    };
    println!(
        "motors at their limit {:.1}% of the time; out of breath {:.1}%; muscle power median {:.1} W/kg, 95th {:.1}",
        100.0 * saturated as f64 / axes.max(1) as f64,
        100.0 * breathless as f64 / samples.max(1) as f64,
        pct(&mut exertion, 0.5),
        pct(&mut exertion, 0.95)
    );
    println!(
        "torso speed median {:.2} m/s, 95th {:.2}; part speed median {:.2}, 95th {:.2}, 99.9th {:.2}",
        pct(&mut torso_speed, 0.5),
        pct(&mut torso_speed, 0.95),
        pct(&mut part_speed, 0.5),
        pct(&mut part_speed, 0.95),
        pct(&mut part_speed, 0.999)
    );
    println!("deaths: {:?}", (sim.stats.deaths_starved, sim.stats.deaths_drowned, sim.stats.deaths_other));
}

/// Where a spring's water goes: a spring in a hollow beside an earthquake crack, run for a while
/// without rain. Prints the pond, the water in the crack and the stream between.
#[test]
#[ignore]
fn spring_beside_a_crack() {
    use evolve_sim::Command;
    let mut p = params();
    p.terrain_resolution = SimParams::default().terrain_resolution;
    p.min_population = 0;
    p.weather = false;
    let mut sim = Sim::new(p);
    let at = sim.random_land_point(40.0);
    // A hollow, and a crack running past it 8 m away.
    sim.apply(Command::Terraform { at: at.to_array(), radius: 5.0, amount: -2.5 });
    let crack: Vec<[f32; 2]> = (-6..=6).map(|k| [at.x + k as f32 * 4.0, at.z + 8.0]).collect();
    sim.apply(Command::Earthquake { path: crack.clone() });
    sim.apply(Command::Spring { at: at.to_array() });
    let t = |sim: &Sim| sim.terrain.clone();
    for minute in 0..=10 {
        if minute > 0 {
            for _ in 0..(60 * TICKS_PER_SECOND) {
                sim.step();
            }
        }
        let tr = t(&sim);
        let pond = tr.water_at(at.x, at.z);
        let crack_water: Vec<String> = (-6..=6).map(|k| format!("{:.2}", tr.water_at(at.x + k as f32 * 4.0, at.z + 8.0))).collect();
        let between: Vec<String> = (0..=8).map(|k| format!("{:.2}", tr.water_at(at.x, at.z + k as f32))).collect();
        println!("min {minute:2}: pond {pond:.2} m | between {} | crack {}", between.join(" "), crack_water.join(" "));
    }
    let tr = t(&sim);
    let h: Vec<String> = (0..=10).map(|k| format!("{:.1}", tr.height_at(at.x, at.z + k as f32))).collect();
    println!("ground from the spring to the crack: {}", h.join(" "));
    let rim: Vec<String> = (0..16)
        .map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 16.0;
            format!("{:.1}", tr.height_at(at.x + a.cos() * 5.5, at.z + a.sin() * 5.5))
        })
        .collect();
    println!("pond rim heights: {}  (pond surface {:.1})", rim.join(" "), tr.height_at(at.x, at.z) + tr.water_at(at.x, at.z));
}

/// Which creature and part spike in speed in the stability test's world, and what its body is.
#[test]
#[ignore]
fn speed_spike() {
    use evolve_sim::genome::Genome;
    use evolve_sim::rng::Rng;
    let p = SimParams { min_population: 0, terrain_resolution: 65, stones: 0, weather: true, physics_shards: 12, ..params() };
    let seed: u64 = std::env::var("EVOLVE_DIAG_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(99);
    let mut sim = Sim::new(SimParams { seed, ..p });
    let mut rng = Rng::new(seed);
    for _ in 0..60 {
        let g = Genome::random(&mut rng, &SimParams::default());
        let at = sim.random_land_point(10.0);
        sim.spawn_creature(g, at, 0.5, f32::MAX, 0, None);
    }
    for step in 0..(60 * 60) {
        sim.step();
        for c in &sim.creatures {
            for (k, q) in c.parts.iter().enumerate() {
                let b = &sim.world_of(c).bodies[q.body];
                let v = b.linvel().length();
                if v > 40.0 {
                    let kinds: Vec<String> = c.joints.iter().map(|j| format!("{:?}", j.kind)).collect();
                    println!(
                        "step {step}: creature #{} part {k}/{} at {v:.0} m/s, mass {:.0} kg, part mass {:.2} kg, shape {:?}, joints {:?}, held {:?}, in water {:.2}",
                        c.id,
                        c.parts.len(),
                        c.mass,
                        b.mass(),
                        c.plan[k].shape,
                        kinds,
                        c.held,
                        c.in_water
                    );
                    return;
                }
            }
        }
    }
    println!("no spike");
}
