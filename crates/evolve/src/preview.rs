//! The zoo's creature viewer. A stored genome's body plan is laid out in its rest pose, drawn by
//! a camera of its own into a texture that the Zoo window shows, and spun by dragging.

use std::sync::Arc;

use bevy::camera::RenderTarget;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};
use evolve_sim::body::{self, JointKind, PartShape};
use evolve_sim::genome::Genome;

pub struct PreviewPlugin;

impl Plugin for PreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZooPreview>()
            .add_systems(Update, (wake_preview, rebuild_preview, spin_preview).chain());
    }
}

/// A body plan in plain numbers (the sim's math types aren't Bevy's), ready to draw.
pub struct Sketch {
    pub parts: Vec<SketchPart>,
    /// Joint positions, and whether each is a two-axis swivel.
    pub joints: Vec<([f32; 3], bool)>,
    pub extent: f32,
    /// How far the lowest point hangs below the torso's center.
    pub depth: f32,
    pub trunk: usize,
    pub limbs: usize,
    pub hinges: usize,
    pub swivels: usize,
}

pub struct SketchPart {
    pub shape: PartShape,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub trunk: bool,
}

pub fn sketch(g: &Genome) -> Sketch {
    let plan = body::body_plan(g);
    let mut joints = Vec::new();
    let (mut hinges, mut swivels) = (0, 0);
    for p in &plan {
        if let Some(j) = &p.joint {
            joints.push(((p.rest * j.frame2).translation.to_array(), j.kind == JointKind::Swivel));
            match j.kind {
                JointKind::Hinge => hinges += 1,
                JointKind::Swivel => swivels += 1,
            }
        }
    }
    Sketch {
        parts: plan
            .iter()
            .map(|p| SketchPart {
                shape: p.shape,
                pos: p.rest.translation.to_array(),
                rot: p.rest.rotation.to_array(),
                trunk: p.trunk,
            })
            .collect(),
        joints,
        extent: body::plan_extent(&plan),
        depth: body::plan_depth(&plan),
        trunk: plan.iter().filter(|p| p.trunk).count(),
        limbs: plan.iter().filter(|p| !p.trunk).count(),
        hinges,
        swivels,
    }
}

/// The Zoo window asks for a creature here and shows `texture`.
#[derive(Resource, Default)]
pub struct ZooPreview {
    /// Zoo id, body and color of the creature to show.
    pub want: Option<(u64, Arc<Sketch>, [f32; 3])>,
    pub yaw: f32,
    /// Set by the window each frame it shows the picture; the camera sleeps otherwise.
    pub visible: bool,
    pub dragging: bool,
    pub texture: Option<egui::TextureId>,
    image: Handle<Image>,
    camera: Option<Entity>,
    shown: Option<u64>,
    root: Option<Entity>,
}

pub const PREVIEW_SIZE: [u32; 2] = [520, 340];
const LAYER: usize = 3;
/// Far below the world, so nothing of it can show up anywhere else.
const ORIGIN: Vec3 = Vec3::new(0.0, -3000.0, 0.0);

/// Creates the render target and camera the first time the picture is wanted. Spawning the
/// camera late keeps egui's main context on the main camera, which exists by then.
#[allow(clippy::too_many_arguments)]
fn wake_preview(
    mut commands: Commands,
    mut pv: ResMut<ZooPreview>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut textures: ResMut<EguiUserTextures>,
    mut cameras: Query<&mut Camera>,
) {
    let visible = std::mem::take(&mut pv.visible) && pv.want.is_some();
    match pv.camera {
        Some(e) => {
            if let Ok(mut cam) = cameras.get_mut(e) {
                cam.is_active = visible;
            }
        }
        None if visible || pv.want.is_some() => {
            let [w, h] = PREVIEW_SIZE;
            let image = images.add(Image::new_target_texture(
                w,
                h,
                TextureFormat::Rgba8Unorm,
                Some(TextureFormat::Rgba8UnormSrgb),
            ));
            pv.texture = Some(textures.add_image(EguiTextureHandle::Strong(image.clone())));
            pv.image = image.clone();
            let layer = RenderLayers::layer(LAYER);
            pv.camera = Some(
                commands
                    .spawn((
                        Camera3d::default(),
                        Camera {
                            order: -1,
                            clear_color: Color::srgb(0.11, 0.12, 0.14).into(),
                            ..default()
                        },
                        RenderTarget::Image(image.into()),
                        // Lit the same by day or night in the world outside.
                        AmbientLight {
                            color: Color::WHITE,
                            brightness: 400.0,
                            ..default()
                        },
                        Transform::from_translation(ORIGIN + Vec3::new(0.0, 1.0, 4.0)).looking_at(ORIGIN, Vec3::Y),
                        layer.clone(),
                    ))
                    .id(),
            );
            commands.spawn((
                DirectionalLight {
                    illuminance: 8000.0,
                    shadows_enabled: true,
                    ..default()
                },
                Transform::from_translation(ORIGIN + Vec3::new(2.0, 4.0, 3.0)).looking_at(ORIGIN, Vec3::Y),
                layer.clone(),
            ));
            commands.spawn((
                DirectionalLight {
                    illuminance: 2500.0,
                    color: Color::srgb(0.7, 0.8, 1.0),
                    ..default()
                },
                Transform::from_translation(ORIGIN + Vec3::new(-3.0, 1.0, -2.0)).looking_at(ORIGIN, Vec3::Y),
                layer.clone(),
            ));
            // A floor to stand on and cast a shadow onto.
            commands.spawn((
                Mesh3d(meshes.add(Circle::new(1.0))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.24, 0.26, 0.28),
                    perceptual_roughness: 0.9,
                    ..default()
                })),
                Transform::from_translation(ORIGIN)
                    .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
                    .with_scale(Vec3::splat(4.0)),
                layer,
                PreviewFloor,
            ));
        }
        None => {}
    }
}

#[derive(Component)]
struct PreviewFloor;

#[allow(clippy::too_many_arguments)]
fn rebuild_preview(
    mut commands: Commands,
    mut pv: ResMut<ZooPreview>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut camera: Query<&mut Transform, (With<Camera>, Without<PreviewFloor>)>,
    mut floor: Query<&mut Transform, With<PreviewFloor>>,
) {
    let Some((id, sk, color)) = pv.want.clone() else {
        return;
    };
    if pv.shown == Some(id) || pv.camera.is_none() {
        return;
    }
    pv.shown = Some(id);
    if let Some(old) = pv.root.take() {
        commands.entity(old).despawn();
    }
    let layer = RenderLayers::layer(LAYER);
    let trunk_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(color[0], color[1], color[2]),
        perceptual_roughness: 0.55,
        ..default()
    });
    let limb_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(color[0] * 0.8, color[1] * 0.8, color[2] * 0.8),
        perceptual_roughness: 0.55,
        ..default()
    });
    let eye_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.03, 0.03, 0.03),
        perceptual_roughness: 0.2,
        ..default()
    });
    let marker = |materials: &mut Assets<StandardMaterial>, c: Color| {
        materials.add(StandardMaterial {
            base_color: c,
            unlit: true,
            ..default()
        })
    };
    let hinge_mat = marker(&mut materials, Color::srgb(1.0, 0.85, 0.2));
    let swivel_mat = marker(&mut materials, Color::srgb(0.95, 0.35, 0.9));
    let cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let ball = meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap());
    // Stand the body on the floor.
    let lift = Vec3::Y * sk.depth;
    let root = commands
        .spawn((Transform::from_translation(ORIGIN + lift), Visibility::default(), layer.clone()))
        .with_children(|b| {
            for (k, p) in sk.parts.iter().enumerate() {
                let at = Transform::from_translation(Vec3::from_array(p.pos)).with_rotation(Quat::from_array(p.rot));
                let mat = if p.trunk { trunk_mat.clone() } else { limb_mat.clone() };
                match p.shape {
                    PartShape::Box { half } => {
                        let h = Vec3::from_array(half);
                        b.spawn((Mesh3d(cube.clone()), MeshMaterial3d(mat), at.with_scale(h * 2.0), layer.clone()));
                        if k == 0 {
                            let r = (h.y.min(h.z) * 0.45).clamp(0.035, 0.09);
                            for side in [-1.0, 1.0] {
                                b.spawn((
                                    Mesh3d(ball.clone()),
                                    MeshMaterial3d(eye_mat.clone()),
                                    Transform::from_xyz(h.x, h.y * 0.4, side * h.z * 0.45).with_scale(Vec3::splat(r)),
                                    layer.clone(),
                                ));
                            }
                        }
                    }
                    PartShape::Capsule { half_height, radius } => {
                        b.spawn((
                            Mesh3d(meshes.add(Capsule3d::new(radius, half_height * 2.0))),
                            MeshMaterial3d(mat),
                            at,
                            layer.clone(),
                        ));
                    }
                }
            }
            let r = (sk.extent * 0.06).clamp(0.05, 0.12);
            for &(pos, swivel) in &sk.joints {
                b.spawn((
                    Mesh3d(ball.clone()),
                    MeshMaterial3d(if swivel { swivel_mat.clone() } else { hinge_mat.clone() }),
                    Transform::from_translation(Vec3::from_array(pos)).with_scale(Vec3::splat(r)),
                    layer.clone(),
                ));
            }
        })
        .id();
    pv.root = Some(root);
    // Frame the whole body.
    let d = sk.extent * 2.6 + 0.6;
    if let Some(cam) = pv.camera
        && let Ok(mut tf) = camera.get_mut(cam)
    {
        let target = ORIGIN + lift * 0.6;
        *tf = Transform::from_translation(target + Vec3::new(0.0, d * 0.45, d)).looking_at(target, Vec3::Y);
    }
    for mut tf in &mut floor {
        tf.scale = Vec3::splat(sk.extent * 2.2 + 0.5);
    }
}

fn spin_preview(time: Res<Time>, mut pv: ResMut<ZooPreview>, mut roots: Query<&mut Transform>) {
    if !pv.dragging {
        pv.yaw += time.delta_secs() * 0.5;
    }
    if let Some(root) = pv.root
        && let Ok(mut tf) = roots.get_mut(root)
    {
        tf.rotation = Quat::from_rotation_y(pv.yaw);
    }
}
