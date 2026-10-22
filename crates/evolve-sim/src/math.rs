pub use rapier3d::math::{Pose, Real};

pub type Vec3 = rapier3d::math::Vector;
pub type Quat = rapier3d::math::Rotation;

/// Mirror across the creature's sagittal plane (z -> -z).
pub fn reflect_z(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, -v.z)
}

/// The rotation conjugated by the z-reflection, which is again a proper rotation.
pub fn reflect_quat_z(q: Quat) -> Quat {
    Quat::from_xyzw(-q.x, -q.y, q.z, q.w)
}

pub fn reflect_pose_z(p: Pose) -> Pose {
    Pose::from_parts(reflect_z(p.translation), reflect_quat_z(p.rotation))
}

/// Yaw-only frame of a body: forward and right vectors flattened onto the ground plane.
pub fn heading_frame(rot: Quat) -> (Vec3, Vec3) {
    let f = rot * Vec3::X;
    let mut fwd = Vec3::new(f.x, 0.0, f.z);
    if fwd.length_squared() < 1e-6 {
        // Pointing straight up or down: fall back to the body's up vector for heading.
        let u = rot * Vec3::Y;
        fwd = Vec3::new(-u.x, 0.0, -u.z);
    }
    let fwd = fwd.try_normalize().unwrap_or(Vec3::X);
    let side = Vec3::new(-fwd.z, 0.0, fwd.x);
    (fwd, side)
}
