//! Orbit camera: right-drag to orbit, middle-drag to pan, scroll to zoom, WASD/QE to move,
//! F to follow the selected creature.

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::light::ShadowFilteringMethod;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::{ColorGrading, Hdr, Msaa};
use bevy_egui::input::{egui_wants_any_keyboard_input, egui_wants_any_pointer_input};

use crate::tools::ToolState;
use crate::{Args, Snap};

pub struct CameraPlugin;

#[derive(Component)]
pub struct MainCamera;

#[derive(Resource)]
pub struct OrbitCam {
    pub focus: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub follow: bool,
    /// How fast the focus is moving while following a creature (see `spring`).
    pub follow_vel: Vec3,
    /// How hard the view is shaking (a meteor strike nearby), m; it dies away by itself.
    pub shake: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            yaw: 0.6,
            pitch: 0.75,
            distance: 150.0,
            follow: false,
            follow_vel: Vec3::ZERO,
            shake: 0.0,
        }
    }
}

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OrbitCam>()
            .add_systems(Startup, spawn_camera)
            .add_systems(
                Update,
                (
                    mouse_control.run_if(not(egui_wants_any_pointer_input)),
                    key_control.run_if(not(egui_wants_any_keyboard_input)),
                    follow_selected,
                    apply_camera,
                )
                    .chain(),
            );
    }
}

fn spawn_camera(mut commands: Commands, args: Res<Args>, mut cam: ResMut<OrbitCam>) {
    if let Some([x, z, d, yaw, pitch]) = args.camera {
        *cam = OrbitCam {
            focus: Vec3::new(x, 0.0, z),
            yaw,
            pitch,
            distance: d,
            follow: false,
            follow_vel: Vec3::ZERO,
            shake: 0.0,
        };
    }
    // HDR so the sun, lightning and highlights can bloom; env.rs drives the fog every frame.
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Tonemapping::TonyMcMapface,
        Msaa::Sample4,
        ShadowFilteringMethod::Gaussian,
        ColorGrading::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 50f32.to_radians(),
            near: 0.1,
            far: 20_000.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 80.0, 120.0).looking_at(Vec3::ZERO, Vec3::Y),
        DistanceFog {
            color: crate::scene::SKY,
            falloff: FogFalloff::ExponentialSquared { density: 0.0004 },
            ..default()
        },
        MainCamera,
    ));
}

/// Soft bloom: enough for the sun, lightning and highlights to glow without hazing the scene.
pub fn bloom() -> Bloom {
    Bloom {
        intensity: 0.07,
        ..Bloom::NATURAL
    }
}

fn mouse_control(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    tools: Res<ToolState>,
    mut cam: ResMut<OrbitCam>,
) {
    if buttons.pressed(MouseButton::Right) {
        cam.yaw -= motion.delta.x * 0.005;
        cam.pitch = (cam.pitch + motion.delta.y * 0.005).clamp(0.05, 1.55);
    }
    if buttons.pressed(MouseButton::Middle) {
        let (fwd, right) = ground_axes(cam.yaw);
        let k = cam.distance * 0.0015;
        let d = -right * motion.delta.x * k + fwd * motion.delta.y * k;
        cam.focus += d;
        cam.follow = false;
    }
    // While dragging a creature, the wheel lifts it instead of zooming.
    if scroll.delta.y != 0.0 && tools.drag.is_none() {
        let lines = match scroll.unit {
            MouseScrollUnit::Line => scroll.delta.y,
            MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
        };
        cam.distance = (cam.distance * (1.0 - lines * 0.1)).clamp(2.0, 800.0);
    }
}

/// Forward and right along the ground for a camera yaw.
fn ground_axes(yaw: f32) -> (Vec3, Vec3) {
    let fwd = Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
    let right = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
    (fwd, right)
}

fn key_control(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut cam: ResMut<OrbitCam>) {
    let (fwd, right) = ground_axes(cam.yaw);
    let mut v = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        v += fwd;
    }
    if keys.pressed(KeyCode::KeyS) {
        v -= fwd;
    }
    if keys.pressed(KeyCode::KeyD) {
        v += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        v -= right;
    }
    let dt = time.delta_secs();
    if v != Vec3::ZERO {
        let fast = if keys.pressed(KeyCode::ShiftLeft) { 3.0 } else { 1.0 };
        let speed = (10.0 + cam.distance * 0.8) * fast * dt;
        cam.focus += v.normalize() * speed;
        cam.follow = false;
    }
    if keys.pressed(KeyCode::KeyQ) {
        cam.yaw += 1.5 * dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        cam.yaw -= 1.5 * dt;
    }
    if keys.just_pressed(KeyCode::KeyF) {
        cam.follow = !cam.follow;
    }
}

fn follow_selected(snap: Res<Snap>, time: Res<Time>, mut cam: ResMut<OrbitCam>) {
    if !cam.follow {
        return;
    }
    let Some(sel) = &snap.0.selected else {
        return;
    };
    let Some(c) = snap.0.creatures.iter().find(|c| c.id == sel.id) else {
        return;
    };
    let cam = &mut *cam;
    spring(&mut cam.focus, &mut cam.follow_vel, center_of_mass(c), FOLLOW_RATE, time.delta_secs());
}

/// How quickly a following camera catches up, in 1/s. Low enough that a creature thrashing
/// several times a second barely moves the view.
pub const FOLLOW_RATE: f32 = 3.0;

/// Where a creature's mass is. Limbs flailing about move it far less than they move the torso.
pub fn center_of_mass(c: &evolve_sim::snapshot::CreatureView) -> Vec3 {
    let (mut sum, mut total) = (Vec3::ZERO, 0.0);
    for p in &c.parts {
        let v = p.shape.volume();
        sum += Vec3::from_array(p.pos) * v;
        total += v;
    }
    if total > 0.0 { sum / total } else { Vec3::from_array(c.parts[0].pos) }
}

/// Moves `pos` toward `target` as a critically damped spring of rate `omega`: no overshoot,
/// and shaking faster than `omega` is smoothed away far better than by easing alone. Exact for
/// a still target, so it's stable at any frame time.
pub fn spring(pos: &mut Vec3, vel: &mut Vec3, target: Vec3, omega: f32, dt: f32) {
    let x = *pos - target;
    let e = (-omega * dt).exp();
    let k = (*vel + x * omega) * dt;
    *pos = target + (x + k) * e;
    *vel = (*vel - k * omega) * e;
}

pub fn apply_camera(
    snap: Res<Snap>,
    env: Res<crate::env::Env>,
    time: Res<Time>,
    mut cam: ResMut<OrbitCam>,
    mut tf: Single<&mut Transform, With<MainCamera>>,
) {
    let t = &snap.0.terrain;
    if t.res >= 3 {
        let half = t.size * 0.5 + 20.0;
        cam.focus.x = cam.focus.x.clamp(-half, half);
        cam.focus.z = cam.focus.z.clamp(-half, half);
        if !cam.follow {
            // Keep the focus resting on the ground so orbiting feels anchored.
            let ground = t.height_at(cam.focus.x, cam.focus.z).max(snap.0.params.water_level);
            cam.focus.y += (ground - cam.focus.y) * 0.2;
        }
    }
    let dir = Vec3::new(
        cam.yaw.sin() * cam.pitch.cos(),
        cam.pitch.sin(),
        cam.yaw.cos() * cam.pitch.cos(),
    );
    let mut pos = cam.focus + dir * cam.distance;
    if t.res >= 3 {
        let ground = t.height_at(pos.x, pos.z) + 0.5;
        pos.y = pos.y.max(ground);
    }
    // Stay above the highest crests the waves can make.
    pos.y = pos.y.max(snap.0.params.water_level + 0.6 + env.wave_crest);
    // A shake jolts the eye more than what it looks at, so the view jars rather than slides.
    let mut focus = cam.focus;
    if cam.shake > 0.001 {
        let t = time.elapsed_secs();
        let jolt = Vec3::new((t * 53.0).sin(), (t * 67.0 + 1.0).sin(), (t * 71.0 + 2.0).sin()) * cam.shake;
        pos += jolt;
        focus += jolt * 0.3;
        cam.shake *= (-time.delta_secs() * 3.5).exp();
    }
    **tf = Transform::from_translation(pos).looking_at(focus, Vec3::Y);
}
