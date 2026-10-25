//! Plants drawn as trees and bushes instead of cones, bending in the wind: grass as low bushes,
//! tree seedlings and trees as conifers, broadleaves and poplars (which, by the plant's id), at
//! the size sync.rs gives them. sync.rs owns the plant entities; this swaps their mesh and
//! material as they appear.

use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Buffer};
use bevy::shader::ShaderRef;

use crate::env::{EnvGpu, shader};
use crate::sync::{PlantLook, PlantOf};

pub struct TreesPlugin;

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct TreeExt {
    #[storage(100, read_only, buffer)]
    env: Buffer,
}

impl MaterialExtension for TreeExt {
    fn vertex_shader() -> ShaderRef {
        shader("tree.wgsl")
    }

    // Shadows are drawn with the prepass pipeline; without this they wouldn't sway.
    fn prepass_vertex_shader() -> ShaderRef {
        shader("tree.wgsl")
    }
}

type TreeMaterial = ExtendedMaterial<StandardMaterial, TreeExt>;

#[derive(Resource)]
struct TreeAssets {
    meshes: Vec<Handle<Mesh>>,
    material: Handle<TreeMaterial>,
}

impl Plugin for TreesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TreeMaterial>::default())
            .add_systems(Startup, setup)
            .add_systems(PostUpdate, dress_plants);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TreeMaterial>>,
    gpu: Res<EnvGpu>,
) {
    let shapes = [conifer(), broadleaf(), bush(), poplar()];
    let mut handles = Vec::new();
    for turn in [0.0, 2.1] {
        for s in &shapes {
            handles.push(meshes.add(s.clone().transformed_by(Transform::from_rotation(Quat::from_rotation_y(turn)))));
        }
    }
    let material = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.8,
            reflectance: 0.25,
            ..default()
        },
        extension: TreeExt { env: gpu.buffer.clone() },
    });
    commands.insert_resource(TreeAssets {
        meshes: handles,
        material,
    });
}

/// Mesh indices (see `setup`: four shapes, in two turns).
const BUSHES: [usize; 2] = [2, 6];
const TREES: [usize; 6] = [0, 1, 3, 4, 5, 7];

fn dress_plants(mut commands: Commands, added: Query<(Entity, &PlantOf, &PlantLook), Added<PlantOf>>, assets: Res<TreeAssets>) {
    let _cost = crate::FrameCost::start("dress_plants");
    for (e, p, look) in &added {
        let h = (p.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize;
        let kinds: &[usize] = if *look == PlantLook::Grass { &BUSHES } else { &TREES };
        let mesh = assets.meshes[kinds[h % kinds.len()]].clone();
        commands
            .entity(e)
            .remove::<MeshMaterial3d<StandardMaterial>>()
            .insert((Mesh3d(mesh), MeshMaterial3d(assets.material.clone())));
    }
}

fn lin(c: [f32; 3]) -> Vec3 {
    let c = Color::srgb(c[0], c[1], c[2]).to_linear();
    Vec3::new(c.red, c.green, c.blue)
}

/// One part of a plant: a primitive moved into place and colored, darker toward its bottom.
/// Alpha 1 marks foliage, which the shader tints per plant.
fn part(mesh: Mesh, tf: Transform, color: Vec3, foliage: bool) -> Mesh {
    let mut m = mesh.transformed_by(tf);
    m.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    m.remove_attribute(Mesh::ATTRIBUTE_TANGENT);
    let colors: Vec<[f32; 4]> = match m.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(bevy::mesh::VertexAttributeValues::Float32x3(ps)) => {
            let lo = ps.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            let hi = ps.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
            ps.iter()
            .map(|p| {
                let t = ((p[1] - lo) / (hi - lo).max(1e-3)).clamp(0.0, 1.0);
                let c = color * (0.62 + 0.45 * t);
                [c.x, c.y, c.z, foliage as u32 as f32]
            })
            .collect()
        }
        _ => Vec::new(),
    };
    m.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    m
}

fn merge(parts: Vec<Mesh>) -> Mesh {
    let mut it = parts.into_iter();
    let mut out = it.next().expect("a plant has parts");
    for p in it {
        out.merge(&p).expect("plant parts share attributes");
    }
    out
}

fn trunk(height: f32, radius: f32) -> Mesh {
    let color = lin([0.36, 0.25, 0.16]);
    part(
        Cylinder::new(radius, height).mesh().resolution(6).build(),
        Transform::from_xyz(0.0, -0.45 + height * 0.5, 0.0),
        color,
        false,
    )
}

fn blob(at: Vec3, r: f32, color: Vec3) -> Mesh {
    part(Sphere::new(1.0).mesh().ico(1).unwrap(), Transform::from_translation(at).with_scale(Vec3::splat(r)), color, true)
}

fn conifer() -> Mesh {
    let c = lin([0.16, 0.36, 0.19]);
    let tier = |base: f32, r: f32, h: f32| {
        part(
            Cone::new(r, h).mesh().resolution(8).build(),
            Transform::from_xyz(0.0, base + h * 0.5, 0.0),
            c,
            true,
        )
    };
    merge(vec![
        trunk(0.22, 0.035),
        tier(-0.33, 0.33, 0.36),
        tier(-0.13, 0.26, 0.36),
        tier(0.08, 0.18, 0.37),
    ])
}

fn broadleaf() -> Mesh {
    let c = lin([0.27, 0.50, 0.17]);
    merge(vec![
        trunk(0.42, 0.045),
        blob(Vec3::new(0.0, 0.1, 0.0), 0.26, c),
        blob(Vec3::new(0.13, 0.0, 0.07), 0.19, c),
        blob(Vec3::new(-0.11, 0.02, -0.09), 0.2, c),
        blob(Vec3::new(0.02, 0.27, -0.02), 0.17, c),
    ])
}

fn bush() -> Mesh {
    let c = lin([0.22, 0.44, 0.15]);
    let mut parts = vec![
        blob(Vec3::new(0.0, -0.24, 0.0), 0.25, c),
        blob(Vec3::new(0.16, -0.29, 0.08), 0.17, c),
        blob(Vec3::new(-0.14, -0.28, -0.1), 0.18, c),
        blob(Vec3::new(0.03, -0.06, 0.01), 0.19, c),
    ];
    let flower = lin([0.95, 0.75, 0.85]);
    for k in 0..7 {
        let a = k as f32 * 2.4;
        let y = -0.12 + 0.05 * (k as f32 * 1.7).sin();
        let p = Vec3::new(a.cos() * 0.2, y, a.sin() * 0.2);
        parts.push(part(
            Sphere::new(1.0).mesh().ico(0).unwrap(),
            Transform::from_translation(p).with_scale(Vec3::splat(0.035)),
            flower,
            false,
        ));
    }
    merge(parts)
}

fn poplar() -> Mesh {
    let c = lin([0.40, 0.54, 0.20]);
    merge(vec![
        trunk(0.32, 0.03),
        part(
            Sphere::new(1.0).mesh().ico(1).unwrap(),
            Transform::from_xyz(0.0, 0.14, 0.0).with_scale(Vec3::new(0.17, 0.31, 0.17)),
            c,
            true,
        ),
    ])
}
