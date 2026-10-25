//! Turning a genome into a body: rest poses relative to the torso, and the rapier bodies,
//! colliders and motorized joints that make it physical.

use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use crate::brain::{limb_motor, spine_motor};
use crate::genome::{Genome, LimbGene, MAX_LIMB_GENES, MAX_PARTS, MAX_SPINE};
use crate::math::{Pose, Quat, Vec3, reflect_pose_z};
use crate::params::SimParams;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum PartShape {
    Box { half: [f32; 3] },
    /// Along the part's local y axis.
    Capsule { half_height: f32, radius: f32 },
}

impl PartShape {
    /// Its longest extent.
    pub fn length(&self) -> f32 {
        match *self {
            PartShape::Box { half } => 2.0 * half[0].max(half[1]).max(half[2]),
            PartShape::Capsule { half_height, radius } => 2.0 * (half_height + radius),
        }
    }

    pub fn volume(&self) -> f32 {
        match *self {
            PartShape::Box { half } => 8.0 * half[0] * half[1] * half[2],
            PartShape::Capsule { half_height, radius } => {
                let pi = std::f32::consts::PI;
                pi * radius * radius * 2.0 * half_height + 4.0 / 3.0 * pi * radius.powi(3)
            }
        }
    }

    /// Projected area facing each local axis (x, y, z), for water drag.
    pub fn areas(&self) -> [f32; 3] {
        match *self {
            PartShape::Box { half: [x, y, z] } => [4.0 * y * z, 4.0 * x * z, 4.0 * x * y],
            PartShape::Capsule { half_height, radius } => {
                let side = 2.0 * radius * 2.0 * (half_height + radius);
                let end = std::f32::consts::PI * radius * radius;
                [side, end, side]
            }
        }
    }

    pub fn bounding_radius(&self) -> f32 {
        match *self {
            PartShape::Box { half } => Vec3::from(half).length(),
            PartShape::Capsule { half_height, radius } => half_height + radius,
        }
    }

    /// Half thickness, used to judge how deep a part sits in water.
    pub fn vertical_radius(&self) -> f32 {
        match *self {
            PartShape::Box { half } => half[1].max(half[0].min(half[2])),
            PartShape::Capsule { radius, half_height } => radius + half_height * 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointKind {
    /// Bends about one axis (the joint frame's x). Cheap to drive.
    Hinge,
    /// Bends about two axes (the frame's y and z) with twist (about x, along the bone) locked,
    /// like a tentacle. Each unit of work costs `swivel_energy_cost` times a hinge's.
    Swivel,
}

impl JointKind {
    /// Motor channels it uses.
    pub fn axes(self) -> usize {
        match self {
            JointKind::Hinge => 1,
            JointKind::Swivel => 2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointSpec {
    /// Joint frame in the parent's local space.
    pub frame1: Pose,
    /// Joint frame in the child's local space.
    pub frame2: Pose,
    pub kind: JointKind,
    /// Range of motion on each axis, radians from the rest pose.
    pub range: [f32; 2],
    pub strength: f32,
    /// Motor channel of the first axis (a swivel's second axis uses the next one).
    pub slot: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartSpec {
    pub shape: PartShape,
    pub parent: Option<usize>,
    /// Rest pose relative to the torso.
    pub rest: Pose,
    pub joint: Option<JointSpec>,
    /// Part of the trunk (torso or spine) rather than a limb.
    #[serde(default)]
    pub trunk: bool,
}

/// Rotation taking local y to `dir` and local x to `x` (made perpendicular to `dir`).
fn frame_toward(dir: Vec3, x: Vec3) -> Quat {
    let q = Quat::from_rotation_arc(Vec3::Y, dir);
    let x = (x - dir * x.dot(dir)).try_normalize().unwrap_or_else(|| dir.any_orthonormal_vector());
    let x1 = q * Vec3::X;
    let angle = dir.dot(x1.cross(x)).atan2(x1.dot(x));
    Quat::from_axis_angle(dir, angle) * q
}

/// Turns a part's local frame (x along the hinge axis, y along the bone) into a swivel joint's
/// frame (x along the bone, y along the old hinge axis).
fn swivel_frame(rot: Quat) -> Quat {
    rot * Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), std::f32::consts::PI)
}

pub fn body_plan(g: &Genome) -> Vec<PartSpec> {
    let s = g.traits.size;
    let th = Vec3::from(g.torso.half) * s;
    let mut parts = vec![PartSpec {
        shape: PartShape::Box { half: th.to_array() },
        parent: None,
        rest: Pose::IDENTITY,
        joint: None,
        trunk: true,
    }];

    // The spine trails behind the torso along -x, each segment joined to the one in front.
    let sp = &g.spine;
    let mut trunk = vec![0usize];
    let hinge_axis = Quat::from_axis_angle(Vec3::X, sp.axis_angle) * Vec3::Y;
    let mut back = -th.x;
    for k in 0..(sp.segments as usize).min(MAX_SPINE) {
        if parts.len() >= MAX_PARTS {
            break;
        }
        let t = sp.taper.powi(k as i32 + 1);
        let half = Vec3::new(
            (th.x * sp.length * t).max(0.04),
            (th.y * t).max(0.03),
            (th.z * t).max(0.03),
        );
        let gap = 0.02 * s;
        let anchor = Vec3::new(back - gap * 0.5, 0.0, 0.0);
        let center = Vec3::new(back - gap - half.x, 0.0, 0.0);
        // Local frame: x along the hinge axis, y pointing back along the spine.
        let rot = frame_toward(-Vec3::X, hinge_axis);
        let (kind, frame) = if sp.swivel {
            (JointKind::Swivel, swivel_frame(rot))
        } else {
            (JointKind::Hinge, rot)
        };
        let parent = *trunk.last().unwrap();
        let i = push_part(
            &mut parts,
            parent,
            Pose::from_parts(center, Quat::IDENTITY),
            Pose::from_parts(anchor, frame),
            PartShape::Box { half: half.to_array() },
            JointSpec {
                frame1: Pose::IDENTITY,
                frame2: Pose::IDENTITY,
                kind,
                range: [-sp.limit, sp.limit],
                strength: sp.strength,
                slot: spine_motor(k),
            },
        );
        parts[i].trunk = true;
        trunk.push(i);
        back = center.x - half.x;
    }

    // For each limb gene: the (primary, mirror) parts at its tip.
    let mut tips: [Option<(usize, Option<usize>)>; MAX_LIMB_GENES] = [None; MAX_LIMB_GENES];
    for (gi, lg) in g.limbs.iter().enumerate() {
        if !lg.present {
            continue;
        }
        let (ppart, pmirror, on_trunk) = if lg.parent == 0 {
            (trunk[(lg.segment as usize).min(trunk.len() - 1)], None, true)
        } else {
            let pg = lg.parent as usize - 1;
            if pg >= gi {
                continue;
            }
            match tips[pg] {
                Some((p, m)) => (p, m, false),
                None => continue,
            }
        };
        // Mirror onto the trunk's other side, or onto the parent's own mirror image. A mirror
        // hanging off an unmirrored limb would float in mid-air, so it's skipped.
        let mirror_parent = match (on_trunk, pmirror) {
            (_, Some(pm)) => Some(pm),
            (true, None) if lg.mirror => Some(ppart),
            _ => None,
        };
        if let Some(tip) = grow_limb(&mut parts, g, gi, lg, ppart, mirror_parent, on_trunk) {
            tips[gi] = Some(tip);
        }
    }
    parts
}

/// Adds a limb's segments (and their mirror images). Returns the tip parts, or None if there was
/// no room for even one segment.
fn grow_limb(
    parts: &mut Vec<PartSpec>,
    g: &Genome,
    gi: usize,
    lg: &LimbGene,
    parent: usize,
    mirror_parent: Option<usize>,
    on_trunk: bool,
) -> Option<(usize, Option<usize>)> {
    let s = g.traits.size;
    let bend = Vec3::from(lg.bend) * 0.8;
    let (anchor_local, dir_local) = match parts[parent].shape {
        PartShape::Box { half } if on_trunk => {
            let mut a = Vec3::from(lg.attach);
            if a.length() < 1e-3 {
                a = Vec3::X;
            }
            let a = a / a.abs().max_element();
            let normal = if a.x.abs() >= a.y.abs() && a.x.abs() >= a.z.abs() {
                Vec3::X * a.x.signum()
            } else if a.y.abs() >= a.z.abs() {
                Vec3::Y * a.y.signum()
            } else {
                Vec3::Z * a.z.signum()
            };
            (a * Vec3::from(half), (normal + bend).try_normalize().unwrap_or(normal))
        }
        // Chained limbs grow from the tip of the parent limb.
        shape => {
            let tip = match shape {
                PartShape::Box { half } => half[1],
                PartShape::Capsule { half_height, .. } => half_height,
            };
            (Vec3::new(0.0, tip, 0.0), (Vec3::Y + bend).try_normalize().unwrap_or(Vec3::Y))
        }
    };
    let ppose = parts[parent].rest;
    let mut anchor = ppose.transform_point(anchor_local);
    let mut dir = (ppose.rotation * dir_local).normalize();
    let perp = dir_local.any_orthonormal_vector();
    let axis = (ppose.rotation * (Quat::from_axis_angle(dir_local, lg.axis_angle) * perp)).normalize();
    let segments = (lg.segments as usize).clamp(1, crate::genome::MAX_LIMB_SEGMENTS);
    let (mut len, mut rad) = (lg.length * s, lg.radius * s);
    let mut prev = (parent, mirror_parent);
    let mut tip = None;
    for k in 0..segments {
        let needed = 1 + mirror_parent.is_some() as usize;
        if parts.len() + needed > MAX_PARTS {
            break;
        }
        if k > 0 {
            // The knee: the chain bends about the hinge axis by the resting curl.
            dir = (Quat::from_axis_angle(axis, lg.curl * 0.8) * dir).normalize();
            len *= lg.taper;
            rad *= lg.taper;
        }
        let hh = (len * 0.5).max(0.03);
        let radius = rad.min(hh * 1.5).max(0.015);
        let center = anchor + dir * hh;
        // Local frame: y along the bone, x along the hinge axis.
        let rot = frame_toward(dir, axis);
        let shape = if k == segments - 1 && lg.paddle >= 0.2 {
            // Broad across the hinge axis, thin in the direction it sweeps: an oar blade.
            PartShape::Box { half: [radius * (1.0 + 3.0 * lg.paddle), hh, (radius * 0.35).max(0.02)] }
        } else {
            PartShape::Capsule { half_height: hh, radius }
        };
        let swivel = if k == 0 { lg.swivel_root } else { lg.swivel_knees };
        let (kind, frame) = if swivel {
            (JointKind::Swivel, swivel_frame(rot))
        } else {
            (JointKind::Hinge, rot)
        };
        let range = if k == 0 {
            [-lg.limit, lg.limit]
        } else if swivel {
            [-lg.knee_limit.min(0.9), lg.knee_limit.min(0.9)]
        } else if lg.curl >= 0.0 {
            // Knees bend mostly one way.
            [-0.25 * lg.knee_limit, lg.knee_limit]
        } else {
            [-lg.knee_limit, 0.25 * lg.knee_limit]
        };
        let strength = lg.strength * if k == 0 { 1.0 } else { 0.7 };
        let rest = Pose::from_parts(center, rot);
        let jframe = Pose::from_parts(anchor, frame);
        let spec = |slot, range| JointSpec {
            frame1: Pose::IDENTITY,
            frame2: Pose::IDENTITY,
            kind,
            range,
            strength,
            slot,
        };
        let primary = push_part(parts, prev.0, rest, jframe, shape, spec(limb_motor(gi, k, false), range));
        // The mirror image bends the opposite way about its (reflected) axis.
        let mirror = prev.1.map(|mp| {
            push_part(
                parts,
                mp,
                reflect_pose_z(rest),
                reflect_pose_z(jframe),
                shape,
                spec(limb_motor(gi, k, true), [-range[1], -range[0]]),
            )
        });
        prev = (primary, mirror);
        tip = Some(prev);
        anchor = center + dir * hh;
    }
    tip
}

fn push_part(
    parts: &mut Vec<PartSpec>,
    parent: usize,
    rest: Pose,
    joint_frame: Pose,
    shape: PartShape,
    joint: JointSpec,
) -> usize {
    let parent_rest = parts[parent].rest;
    parts.push(PartSpec {
        shape,
        parent: Some(parent),
        rest,
        joint: Some(JointSpec {
            frame1: parent_rest.inv_mul(&joint_frame),
            frame2: rest.inv_mul(&joint_frame),
            ..joint
        }),
        trunk: false,
    });
    parts.len() - 1
}

pub fn plan_mass(plan: &[PartSpec], density: f32) -> f32 {
    plan.iter().map(|p| p.shape.volume() * density).sum()
}

/// Lowest point of the body below the torso center, in the rest pose (positive number).
pub fn plan_depth(plan: &[PartSpec]) -> f32 {
    plan.iter()
        .map(|p| -(p.rest.translation.y) + p.shape.bounding_radius())
        .fold(0.0, f32::max)
}

pub fn plan_extent(plan: &[PartSpec]) -> f32 {
    plan.iter()
        .map(|p| p.rest.translation.length() + p.shape.bounding_radius())
        .fold(0.0, f32::max)
}

/// What a collider belongs to, packed into rapier's `user_data`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    None,
    Terrain,
    Wall,
    Creature { id: u64, part: u16 },
    Obstacle { id: u64 },
    /// A tree's trunk, in every physics world like the ground.
    Tree { id: u64 },
}

impl Tag {
    pub fn encode(self) -> u128 {
        match self {
            Tag::None => 0,
            Tag::Terrain => 1 << 120,
            Tag::Wall => 2 << 120,
            Tag::Creature { id, part } => (3 << 120) | ((id as u128) << 16) | part as u128,
            Tag::Obstacle { id } => (4 << 120) | ((id as u128) << 16),
            Tag::Tree { id } => (5 << 120) | ((id as u128) << 16),
        }
    }

    pub fn decode(v: u128) -> Tag {
        let id = ((v >> 16) & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        match v >> 120 {
            1 => Tag::Terrain,
            2 => Tag::Wall,
            3 => Tag::Creature {
                id,
                part: (v & 0xFFFF) as u16,
            },
            4 => Tag::Obstacle { id },
            5 => Tag::Tree { id },
            _ => Tag::None,
        }
    }

    pub fn creature_id(self) -> Option<u64> {
        match self {
            Tag::Creature { id, .. } => Some(id),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartRef {
    pub body: RigidBodyHandle,
    pub collider: ColliderHandle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointRef {
    /// A link of the creature's multibody: an articulation solved in joint coordinates, so the
    /// joints can't drift apart and pump energy into the body as separate constraints did.
    pub handle: MultibodyJointHandle,
    pub parent: usize,
    pub child: usize,
    pub kind: JointKind,
    pub slot: usize,
    pub range: [f32; 2],
    pub strength: f32,
    /// Moment of inertia of everything beyond the joint about it, kg·m² (for muscle work).
    #[serde(default)]
    pub inertia: f32,
    /// A swivel is two hinges in series (a universal joint, which can't twist) through a small
    /// knuckle body with no collider: `handle` turns about the joint's y axis, `second` about
    /// its z axis. (A multibody's own ball joint tracks its angles by summing its spin, which
    /// drifts, so its limits would drift too.)
    #[serde(default)]
    pub second: Option<MultibodyJointHandle>,
    #[serde(default)]
    pub knuckle: Option<RigidBodyHandle>,
}

impl JointRef {
    /// The rapier axes it drives, with their motor channel offsets.
    pub fn axes(&self) -> &'static [(JointAxis, usize)] {
        axes_of(self.kind)
    }

    /// Middle and half-width of the range of motion.
    pub fn span(&self) -> (f32, f32) {
        let [lo, hi] = self.range;
        (0.5 * (lo + hi), (0.5 * (hi - lo)).max(1e-3))
    }
}

/// Inserts the creature's bodies and joints. `root` is the torso's world pose.
pub fn spawn_body(
    world: &mut PhysicsWorld,
    plan: &[PartSpec],
    root: Pose,
    id: u64,
    mass: f32,
    p: &SimParams,
) -> (Vec<PartRef>, Vec<JointRef>) {
    let mut parts = Vec::with_capacity(plan.len());
    for (k, spec) in plan.iter().enumerate() {
        let tag = Tag::Creature { id, part: k as u16 }.encode();
        let rb = RigidBodyBuilder::dynamic()
            .pose(root * spec.rest)
            .linear_damping(p.air_drag)
            .angular_damping(0.3)
            .soft_ccd_prediction(p.soft_ccd)
            .can_sleep(false)
            .user_data(tag);
        let col = match spec.shape {
            PartShape::Box { half } => ColliderBuilder::cuboid(half[0], half[1], half[2]),
            PartShape::Capsule { half_height, radius } => ColliderBuilder::capsule_y(half_height, radius),
        }
        .density(p.body_density)
        .friction(p.body_friction)
        .user_data(tag)
        .active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS);
        let (body, collider) = world.insert(rb, col);
        parts.push(PartRef { body, collider });
    }
    let mut joints = Vec::new();
    for (k, spec) in plan.iter().enumerate() {
        let (Some(parent), Some(js)) = (spec.parent, &spec.joint) else {
            continue;
        };
        let mid = 0.5 * (js.range[0] + js.range[1]);
        let force = p.motor_torque_per_kg * js.strength * mass;
        let inertia = subtree_inertia(plan, k, (root * spec.rest * js.frame2).translation, root, p.body_density);
        let (stiffness, damping) = servo(p, force, inertia);
        // One free angle per joint: a hinge's x, or a swivel's y then z (see `JointRef::second`).
        let hinge = |free: JointAxis, f1: Pose, f2: Pose| {
            let locked = JointAxesMask::LOCKED_SPHERICAL_AXES
                | match free {
                    JointAxis::AngX => JointAxesMask::ANG_Y | JointAxesMask::ANG_Z,
                    JointAxis::AngY => JointAxesMask::ANG_X | JointAxesMask::ANG_Z,
                    _ => JointAxesMask::ANG_X | JointAxesMask::ANG_Y,
                };
            GenericJointBuilder::new(locked)
                .local_frame1(f1)
                .local_frame2(f2)
                .contacts_enabled(false)
                .limits(free, js.range)
                .motor_model(free, MotorModel::ForceBased)
                .motor_position(free, mid, stiffness, damping)
                .motor_max_force(free, force)
                .build()
        };
        let mut attach = |a: RigidBodyHandle, b: RigidBodyHandle, joint: GenericJoint| {
            world.multibody_joints.insert(a, b, joint, true).expect("a body plan is a tree, rooted at the torso")
        };
        let (handle, second, knuckle) = match js.kind {
            JointKind::Hinge => (attach(parts[parent].body, parts[k].body, hinge(JointAxis::AngX, js.frame1, js.frame2)), None, None),
            JointKind::Swivel => {
                let km = (spec.shape.volume() * p.body_density * 0.02).max(0.01);
                let rb = RigidBodyBuilder::dynamic()
                    .pose(root * parts_rest(plan, parent) * js.frame1)
                    .can_sleep(false)
                    .additional_mass_properties(MassProperties::new(Vec3::ZERO, km, Vec3::splat(km * 4e-4)));
                let knuckle = world.bodies.insert(rb);
                let first = attach(parts[parent].body, knuckle, hinge(JointAxis::AngY, js.frame1, Pose::IDENTITY));
                let second = attach(knuckle, parts[k].body, hinge(JointAxis::AngZ, Pose::IDENTITY, js.frame2));
                (first, Some(second), Some(knuckle))
            }
        };
        joints.push(JointRef {
            handle,
            parent,
            child: k,
            kind: js.kind,
            slot: js.slot,
            range: js.range,
            strength: js.strength,
            inertia,
            second,
            knuckle,
        });
    }
    (parts, joints)
}

/// A joint motor's spring and damper. The motors are servos with a torque limit: rapier's
/// default (acceleration-based) spring scales with the limb's own inertia, so a light leg planted
/// on the ground could push with only a sliver of its strength however strong it was. This one
/// pushes with its full `max_force` once the joint is `motor_reach` off target, and is damped for
/// the limb it swings (`inertia`, about the joint).
pub fn servo(p: &SimParams, max_force: f32, inertia: f32) -> (f32, f32) {
    let stiffness = max_force / p.motor_reach.max(1e-3);
    let damping = 2.0 * p.motor_damping_ratio * (stiffness * inertia.max(1e-4)).sqrt();
    (stiffness, damping)
}

fn parts_rest(plan: &[PartSpec], k: usize) -> Pose {
    plan[k].rest
}

fn axes_of(kind: JointKind) -> &'static [(JointAxis, usize)] {
    match kind {
        JointKind::Hinge => &[(JointAxis::AngX, 0)],
        JointKind::Swivel => &[(JointAxis::AngY, 0), (JointAxis::AngZ, 1)],
    }
}

/// Moment of inertia about `anchor` of part `k` and everything hanging off it (as point masses
/// at their centers, plus a rod's own inertia for each), kg·m².
fn subtree_inertia(plan: &[PartSpec], k: usize, anchor: Vec3, root: Pose, density: f32) -> f32 {
    let in_subtree = |mut i: usize| loop {
        if i == k {
            return true;
        }
        match plan[i].parent {
            Some(p) => i = p,
            None => return false,
        }
    };
    plan.iter()
        .enumerate()
        .filter(|(i, _)| in_subtree(*i))
        .map(|(_, s)| {
            let m = s.shape.volume() * density;
            let r = (root * s.rest).translation.distance(anchor);
            let len = s.shape.length();
            m * (r * r + len * len / 12.0)
        })
        .sum::<f32>()
        .max(1e-4)
}

/// The multibody a creature's parts form, if it has joints.
pub fn multibody<'a>(world: &'a PhysicsWorld, parts: &[PartRef]) -> Option<&'a Multibody> {
    let link = world.multibody_joints.rigid_body_link(parts.first()?.body)?;
    world.multibody_joints.get_multibody(link.multibody)
}

pub fn multibody_mut<'a>(world: &'a mut PhysicsWorld, parts: &[PartRef]) -> Option<&'a mut Multibody> {
    let link = *world.multibody_joints.rigid_body_link(parts.first()?.body)?;
    world.multibody_joints.get_multibody_mut(link.multibody)
}

fn link_joint(world: &PhysicsWorld, h: MultibodyJointHandle) -> Option<&GenericJoint> {
    let (mb, link) = world.multibody_joints.get(h)?;
    Some(&mb.link(link)?.joint.data)
}

/// The joint (frames, limit, motor) that drives axis `d` of a creature joint.
pub fn axis_joint<'a>(world: &'a PhysicsWorld, j: &JointRef, d: usize) -> Option<&'a GenericJoint> {
    link_joint(world, if d == 0 { j.handle } else { j.second? })
}

pub fn axis_joint_mut<'a>(world: &'a mut PhysicsWorld, j: &JointRef, d: usize) -> Option<&'a mut GenericJoint> {
    let (mb, link) = world.multibody_joints.get_mut(if d == 0 { j.handle } else { j.second? })?;
    Some(&mut mb.link_mut(link)?.joint.data)
}

/// Mechanical work a joint's muscles do in `dt`, J: the torque each motor applies (its spring and
/// damper's pull, up to the motor's strength) times how fast the joint turns. Rapier doesn't
/// report a multibody's motor impulses.
pub fn muscle_work(world: &PhysicsWorld, parts: &[PartRef], j: &JointRef, dt: f32) -> f32 {
    let Some((angle, rate)) = joint_state(world, parts, j) else {
        return 0.0;
    };
    let inertia = if j.inertia > 0.0 { j.inertia } else { 0.05 };
    axes_of(j.kind)
        .iter()
        .filter_map(|&(axis, d)| Some((axis, d, axis_joint(world, j, d)?)))
        .map(|(axis, d, data)| {
            let m = &data.motors[axis as usize];
            let ask = m.stiffness * (m.target_pos - angle[d]) + m.damping * (m.target_vel - rate[d]);
            // An acceleration-based motor (a joint from an older save, until its first brain
            // tick) asks for an acceleration of the limb rather than a torque.
            let torque = match m.model {
                MotorModel::ForceBased => ask.abs(),
                MotorModel::AccelerationBased => inertia * ask.abs(),
            };
            torque.min(m.max_force) * rate[d].abs() * dt
        })
        .sum()
}

/// Current angles of a joint's axes (a hinge only uses the first), and the relative angular
/// speeds about them.
pub fn joint_state(world: &PhysicsWorld, parts: &[PartRef], j: &JointRef) -> Option<([f32; 2], [f32; 2])> {
    // The angle of one hinge, about `axis`, and how fast it's turning.
    let hinge = |h: MultibodyJointHandle, a: RigidBodyHandle, b: RigidBodyHandle, axis: Vec3, pick: fn(Quat) -> f32| {
        let data = link_joint(world, h)?;
        let (b1, b2) = (world.bodies.get(a)?, world.bodies.get(b)?);
        let jr1 = *b1.rotation() * data.local_frame1.rotation;
        let err = jr1.inverse() * (*b2.rotation() * data.local_frame2.rotation);
        let flip = if err.w < 0.0 { -1.0 } else { 1.0 };
        let angle = pick(err).clamp(-1.0, 1.0).asin() * 2.0 * flip;
        Some((angle, (b2.angvel() - b1.angvel()).dot(jr1 * axis)))
    };
    let (parent, child) = (parts[j.parent].body, parts[j.child].body);
    Some(match (j.kind, j.second, j.knuckle) {
        (JointKind::Swivel, Some(second), Some(knuckle)) => {
            let (ay, wy) = hinge(j.handle, parent, knuckle, Vec3::Y, |q| q.y)?;
            let (az, wz) = hinge(second, knuckle, child, Vec3::Z, |q| q.z)?;
            ([ay, az], [wy, wz])
        }
        _ => {
            let (ax, wx) = hinge(j.handle, parent, child, Vec3::X, |q| q.x)?;
            ([ax, 0.0], [wx, 0.0])
        }
    })
}

/// Moves a creature's body from one physics world to another just as it is: where it stands,
/// its joint angles, its motors' settings and every velocity. (Rapier builds a multibody's joints
/// at their rest angles, so the angles are put back as displacements.)
#[allow(clippy::too_many_arguments)]
pub fn transplant(
    from: &mut PhysicsWorld,
    to: &mut PhysicsWorld,
    plan: &[PartSpec],
    parts: &[PartRef],
    joints: &[JointRef],
    id: u64,
    mass: f32,
    p: &SimParams,
) -> Option<(Vec<PartRef>, Vec<JointRef>)> {
    let root = from.bodies.get(parts.first()?.body)?;
    let (root_pose, root_vel) = (*root.position(), (root.linvel(), root.angvel()));
    let angles: Vec<[f32; 2]> = joints.iter().map(|j| joint_state(from, parts, j).map_or([0.0; 2], |(a, _)| a)).collect();
    let motors: Vec<[Option<GenericJoint>; 2]> =
        joints.iter().map(|j| [axis_joint(from, j, 0).copied(), axis_joint(from, j, 1).copied()]).collect();
    let velocities: Option<Vec<f32>> = multibody(from, parts).map(|mb| mb.generalized_velocity().iter().copied().collect());
    // Each body's own velocity too: rapier only refreshes a link's from the joint velocities
    // during a step, and the first step needs them right.
    let body_vel = |w: &PhysicsWorld, h: RigidBodyHandle| w.bodies.get(h).map(|b| (b.linvel(), b.angvel()));
    let part_vels: Vec<_> = parts.iter().map(|q| body_vel(from, q.body)).collect();
    let knuckle_vels: Vec<_> = joints.iter().map(|j| j.knuckle.and_then(|k| body_vel(from, k))).collect();
    remove_body(from, parts, joints);

    let (new_parts, new_joints) = spawn_body(to, plan, root_pose * plan[0].rest.inverse(), id, mass, p);
    // The whole joint, motors and limits with their impulses from the last step (the solver
    // starts each step from them).
    for (j, m) in new_joints.iter().zip(&motors) {
        for (d, data) in m.iter().enumerate() {
            if let (Some(data), Some(new)) = (data, axis_joint_mut(to, j, d)) {
                *new = *data;
            }
        }
    }
    let Some(link) = to.multibody_joints.rigid_body_link(new_parts[0].body).copied() else {
        if let Some(b) = to.bodies.get_mut(new_parts[0].body) {
            b.set_linvel(root_vel.0, true);
            b.set_angvel(root_vel.1, true);
        }
        return Some((new_parts, new_joints));
    };
    // Each joint's angle at its place among the multibody's coordinates.
    let slots: Vec<(usize, f32)> = new_joints
        .iter()
        .zip(&angles)
        .flat_map(|(j, a)| {
            let handles = [Some(j.handle), j.second];
            (0..j.kind.axes()).filter_map(move |d| Some((handles[d]?, a[d])))
        })
        .filter_map(|(h, angle)| {
            let (mb, l) = to.multibody_joints.get(h)?;
            Some((mb.link(l)?.assembly_id(), angle))
        })
        .collect();
    let mb = to.multibody_joints.get_multibody_mut(link.multibody)?;
    let mut disp = vec![0.0; mb.ndofs()];
    for (at, angle) in slots {
        disp[at] = angle;
    }
    mb.apply_displacements(&disp);
    mb.forward_kinematics(&to.bodies, true);
    mb.update_rigid_bodies(&mut to.bodies, true);
    if let Some(v) = velocities.filter(|v| v.len() == mb.ndofs()) {
        mb.generalized_velocity_mut().copy_from_slice(&v);
    }
    let set = |w: &mut PhysicsWorld, h: RigidBodyHandle, v: Option<(Vec3, Vec3)>| {
        if let (Some(b), Some((lin, ang))) = (w.bodies.get_mut(h), v) {
            b.set_linvel(lin, true);
            b.set_angvel(ang, true);
        }
    };
    for (q, v) in new_parts.iter().zip(part_vels) {
        set(to, q.body, v);
    }
    for (j, v) in new_joints.iter().zip(knuckle_vels) {
        if let Some(k) = j.knuckle {
            set(to, k, v);
        }
    }
    Some((new_parts, new_joints))
}

pub fn remove_body(world: &mut PhysicsWorld, parts: &[PartRef], joints: &[JointRef]) {
    for p in parts {
        world.remove_body(p.body);
    }
    for k in joints.iter().filter_map(|j| j.knuckle) {
        world.remove_body(k);
    }
}

/// Same-creature parts never collide with each other (collision groups can't express this for
/// hundreds of creatures, so it's a contact filter).
pub struct ContactFilter;

impl PhysicsHooks for ContactFilter {
    fn filter_contact_pair(&self, ctx: &PairFilterContext) -> Option<SolverFlags> {
        let a = ctx.colliders.get(ctx.collider1)?;
        let b = ctx.colliders.get(ctx.collider2)?;
        match (
            Tag::decode(a.user_data).creature_id(),
            Tag::decode(b.user_data).creature_id(),
        ) {
            (Some(x), Some(y)) if x == y => None,
            _ => Some(SolverFlags::COMPUTE_RIGID_IMPULSES),
        }
    }
}
