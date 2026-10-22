//! The terrain mesh and its material. The sky, sun, ocean and weather live in their own modules.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Buffer};
use bevy::shader::ShaderRef;
use evolve_sim::terrain::Terrain;

use crate::Snap;
use crate::env::{EnvGpu, shader};

/// Fallback clear color before the sky has drawn.
pub const SKY: Color = Color::srgb(0.62, 0.75, 0.88);

pub struct ScenePlugin;

#[derive(Component)]
pub struct TerrainMesh;

/// Beaches, rock, grass, sea floor and cloud shadows, all from world position in the shader, so
/// changing the water level needs no rebuild.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct TerrainExt {
    #[storage(100, read_only, buffer)]
    env: Buffer,
}

impl MaterialExtension for TerrainExt {
    fn fragment_shader() -> ShaderRef {
        shader("terrain.wgsl")
    }
}

type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExt>;

#[derive(Resource, Default)]
struct TerrainState {
    /// The terrain the mesh was built from. Compared by identity, not version number: every
    /// freshly generated world starts at version 1.
    built_from: Option<std::sync::Arc<Terrain>>,
    material: Option<Handle<TerrainMaterial>>,
    mesh: Option<Handle<Mesh>>,
}

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default())
            .init_resource::<TerrainState>()
            .add_systems(Update, update_terrain);
    }
}

fn update_terrain(
    mut commands: Commands,
    snap: Res<Snap>,
    gpu: Res<EnvGpu>,
    mut state: ResMut<TerrainState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    old: Query<Entity, With<TerrainMesh>>,
) {
    let t = &snap.0.terrain;
    if t.res < 3 || state.built_from.as_ref().is_some_and(|b| std::sync::Arc::ptr_eq(b, t)) {
        return;
    }
    // The same ground reshaped by creatures (same grid): update the mesh in place.
    let same_grid = state.built_from.as_ref().is_some_and(|b| b.res == t.res && b.size == t.size && b.ground == t.ground);
    if same_grid
        && let Some(mesh) = state.mesh.as_ref().and_then(|h| meshes.get_mut(h))
    {
        let (positions, normals, colors) = terrain_attributes(t);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        state.built_from = Some(t.clone());
        return;
    }
    for e in &old {
        commands.entity(e).despawn();
    }
    let material = state
        .material
        .get_or_insert_with(|| {
            materials.add(ExtendedMaterial {
                base: StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 0.9,
                    reflectance: 0.3,
                    ..default()
                },
                extension: TerrainExt { env: gpu.buffer.clone() },
            })
        })
        .clone();
    let mesh = meshes.add(terrain_mesh(t));
    state.mesh = Some(mesh.clone());
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), Transform::default(), TerrainMesh));
    state.built_from = Some(t.clone());
}

/// Per-vertex positions, normals and colors (fertility in red, moved earth in green).
#[allow(clippy::type_complexity)]
fn terrain_attributes(t: &Terrain) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 4]>) {
    let n = t.res;
    let mut positions = Vec::with_capacity(n * n);
    let mut normals = Vec::with_capacity(n * n);
    let mut colors = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let x = (j as f32 / (n - 1) as f32 - 0.5) * t.size;
            let z = (i as f32 / (n - 1) as f32 - 0.5) * t.size;
            let k = i * n + j;
            positions.push([x, t.heights[k], z]);
            normals.push(t.normal_at(x, z).to_array());
            colors.push([t.fertility[k], (t.disturbance(k).abs() / 0.6).min(1.0), 0.0, 1.0]);
        }
    }
    (positions, normals, colors)
}

/// Triangulated exactly like the physics heightfield (diagonal from (x0, z1) to (x1, z0)).
/// Fertility rides along in the vertex color's red channel.
fn terrain_mesh(t: &Terrain) -> Mesh {
    let n = t.res;
    let (positions, normals, colors) = terrain_attributes(t);
    let idx = |i: usize, j: usize| (i * n + j) as u32;
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            indices.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i, j + 1)]);
            indices.extend_from_slice(&[idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}
