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

/// Water deeper than this gets a surface of its own (the lake mesh) above the ground; shallower
/// water is painted on the ground.
const SURFACE_DEPTH: f32 = 0.1;

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
    /// The surface of deep water: the same material, blended, drawing its own branch.
    lake_material: Option<Handle<TerrainMaterial>>,
    lake_mesh: Option<Handle<Mesh>>,
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
    // Ground reshaped by creatures (or any terrain on the same grid, so the same triangles and
    // much the same bounds): the mesh is updated in place, only where the ground moved if it's
    // the same world.
    if let Some(old) = state.built_from.clone().filter(|b| b.res == t.res && b.size == t.size)
        && let Some(handle) = state.mesh.clone()
    {
        let same_world = old.ground == t.ground && old.fertility == t.fertility;
        // Points whose shape changed, or whose look did (trampling, water, wet soil): those
        // change a little everywhere every update, so only a visible step counts.
        let moved: Vec<usize> = if same_world {
            (0..t.heights.len())
                .filter(|&k| old.heights[k] != t.heights[k] || surface_step(&old, k) != surface_step(t, k))
                .collect()
        } else {
            Vec::new()
        };
        if same_world && moved.is_empty() {
            state.built_from = Some(t.clone());
            return;
        }
        if let Some(mesh) = meshes.get_mut(&handle) {
            if !(same_world && patch_terrain(mesh, t, &moved)) {
                set_attributes(mesh, terrain_attributes(t));
            }
            if let Some(lake) = state.lake_mesh.clone()
                && let Some(mesh) = meshes.get_mut(&lake)
            {
                set_attributes(mesh, lake_attributes(t));
            }
            state.built_from = Some(t.clone());
            return;
        }
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
    let mesh = meshes.add(grid_mesh(t, terrain_attributes(t)));
    state.mesh = Some(mesh.clone());
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), Transform::default(), TerrainMesh));
    let lake_material = state
        .lake_material
        .get_or_insert_with(|| {
            materials.add(ExtendedMaterial {
                base: StandardMaterial {
                    base_color: Color::WHITE,
                    alpha_mode: AlphaMode::Blend,
                    ..default()
                },
                extension: TerrainExt { env: gpu.buffer.clone() },
            })
        })
        .clone();
    let lake = meshes.add(grid_mesh(t, lake_attributes(t)));
    state.lake_mesh = Some(lake.clone());
    commands.spawn((
        Mesh3d(lake),
        MeshMaterial3d(lake_material),
        Transform::default(),
        TerrainMesh,
        bevy::light::NotShadowCaster,
    ));
    state.built_from = Some(t.clone());
}

type Attributes = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 2]>);

fn set_attributes(mesh: &mut Mesh, (positions, normals, colors, surfaces): Attributes) {
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, surfaces);
}

/// The surface of water deeper than `SURFACE_DEPTH`, on the terrain's grid: the ground plus the
/// water there, or tucked just under the ground where there's none (so it meets the bank in
/// between). Its UV is (depth, -1), the -1 telling the shader it's a water surface. Lakes and
/// ponds lie level; deep streams follow their bed.
fn lake_attributes(t: &Terrain) -> Attributes {
    let count = t.res * t.res;
    let mut positions = Vec::with_capacity(count);
    let mut normals = Vec::with_capacity(count);
    let mut surfaces = Vec::with_capacity(count);
    let n = t.res;
    let surface = |k: usize| {
        let water = t.water.get(k).copied().unwrap_or(0.0);
        (water > SURFACE_DEPTH).then(|| t.heights[k] + water)
    };
    for k in 0..count {
        let (x, z) = grid_xz(t, k);
        let water = t.water.get(k).copied().unwrap_or(0.0);
        let pooled = t.pooled.get(k).copied().unwrap_or(0.0);
        let deep = water > SURFACE_DEPTH;
        // Next to water, a dry point whose ground rises above it keeps the water's level, so
        // the surface meets the bank where the bank crosses it (the ground hides the rest)
        // instead of slanting up the bank in a bright sliver. Elsewhere it sits under the ground.
        let height = if deep {
            t.heights[k] + water
        } else {
            let (i, j) = (k / n, k % n);
            let mut level = f32::MIN;
            for ii in i.saturating_sub(1)..(i + 2).min(n) {
                for jj in j.saturating_sub(1)..(j + 2).min(n) {
                    if let Some(l) = surface(ii * n + jj) {
                        level = level.max(l);
                    }
                }
            }
            if level > f32::MIN && t.heights[k] >= level { level } else { t.heights[k] - 0.3 }
        };
        positions.push([x, height, z]);
        normals.push(if pooled >= water - 1e-3 { [0.0, 1.0, 0.0] } else { t.normal_at(x, z).to_array() });
        surfaces.push([if deep { water } else { 0.0 }, -1.0]);
    }
    (positions, normals, vec![[0.0; 4]; count], surfaces)
}

/// Where grid point `k` sits on the ground plane.
fn grid_xz(t: &Terrain, k: usize) -> (f32, f32) {
    let n = t.res;
    let x = ((k % n) as f32 / (n - 1) as f32 - 0.5) * t.size;
    let z = ((k / n) as f32 / (n - 1) as f32 - 0.5) * t.size;
    (x, z)
}

/// Fertility in red, moved earth in green, blue set where it was dug rather than piled, and
/// how trampled the ground is in alpha.
fn vertex_color(t: &Terrain, k: usize) -> [f32; 4] {
    let d = t.disturbance(k);
    [t.fertility[k], (d.abs() / 0.6).min(1.0), if d < 0.0 { 1.0 } else { 0.0 }, t.worn(k)]
}

/// Water standing or running on the ground (m) and how wet the soil is, for the UV channel (unused otherwise: the material has no textures).
fn vertex_surface(t: &Terrain, k: usize) -> [f32; 2] {
    [t.water.get(k).copied().unwrap_or(0.0), t.moisture.get(k).copied().unwrap_or(0.0)]
}

/// The look of a point in coarse steps, to tell when it has visibly changed.
fn surface_step(t: &Terrain, k: usize) -> [u16; 3] {
    let [water, wet] = vertex_surface(t, k);
    [(t.worn(k) * 64.0) as u16, (water * 200.0).min(4000.0) as u16, (wet * 32.0) as u16]
}

/// Per-vertex positions, normals, colors and surface water.
fn terrain_attributes(t: &Terrain) -> Attributes {
    let count = t.res * t.res;
    let mut positions = Vec::with_capacity(count);
    let mut normals = Vec::with_capacity(count);
    let mut colors = Vec::with_capacity(count);
    let mut surfaces = Vec::with_capacity(count);
    for k in 0..count {
        let (x, z) = grid_xz(t, k);
        positions.push([x, t.heights[k], z]);
        normals.push(t.normal_at(x, z).to_array());
        colors.push(vertex_color(t, k));
        surfaces.push(vertex_surface(t, k));
    }
    (positions, normals, colors, surfaces)
}

/// Updates the vertices at `moved` grid points, and the normals around them, in a mesh built
/// from an earlier copy of the same terrain. False if the mesh lacks the expected attributes.
fn patch_terrain(mesh: &mut Mesh, t: &Terrain, moved: &[usize]) -> bool {
    use bevy::mesh::VertexAttributeValues::{Float32x2, Float32x3, Float32x4};
    let n = t.res;
    let mut near = vec![false; n * n];
    for &k in moved {
        let (i, j) = (k / n, k % n);
        for ii in i.saturating_sub(1)..(i + 2).min(n) {
            for jj in j.saturating_sub(1)..(j + 2).min(n) {
                near[ii * n + jj] = true;
            }
        }
    }
    let Some(Float32x3(positions)) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) else {
        return false;
    };
    for &k in moved {
        positions[k][1] = t.heights[k];
    }
    let Some(Float32x4(colors)) = mesh.attribute_mut(Mesh::ATTRIBUTE_COLOR) else {
        return false;
    };
    for &k in moved {
        colors[k] = vertex_color(t, k);
    }
    let Some(Float32x2(surfaces)) = mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0) else {
        return false;
    };
    for &k in moved {
        surfaces[k] = vertex_surface(t, k);
    }
    let Some(Float32x3(normals)) = mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL) else {
        return false;
    };
    for (k, _) in near.iter().enumerate().filter(|(_, n)| **n) {
        let (x, z) = grid_xz(t, k);
        normals[k] = t.normal_at(x, z).to_array();
    }
    true
}

/// A mesh on the terrain's grid, triangulated exactly like the physics heightfield (diagonal
/// from (x0, z1) to (x1, z0)).
fn grid_mesh(t: &Terrain, (positions, normals, colors, surfaces): Attributes) -> Mesh {
    let n = t.res;
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
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, surfaces)
        .with_inserted_indices(Indices::U32(indices))
}
