//! Ray-traced lighting (`--raytracing`, in builds with the `raytracing` feature): Bevy Solari
//! lights the scene, sun and sky shadows and light bounced off the ground included, by tracing
//! rays through copies of its meshes.
//!
//! Solari traces only meshes in one layout (positions, normals, uvs and tangents, u32 indices)
//! with a plain `StandardMaterial`, so each mesh that casts a shadow gets a copy in that layout,
//! kept up to date as the original changes. The ground and the trees have materials of their own:
//! their copies hang on a child with a plain material of about their colour, seen only by the
//! rays. Water, rain, the sky and the lights of miracles aren't traced.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::camera::CameraMainTextureUsages;
use bevy::camera::visibility::{RenderLayers, VisibilitySystems};
use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass};
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::render_resource::TextureUsages;
use bevy::render::view::Msaa;
use bevy::solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins};

use crate::camera::MainCamera;
use crate::scene::TerrainMaterial;
use crate::trees::TreeMaterial;

pub struct RaytracingPlugin;

impl Plugin for RaytracingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SolariPlugins)
            .init_resource::<Copies>()
            .add_systems(Startup, stand_in_materials)
            .add_systems(
                PostUpdate,
                (cameras, refresh_copies, trace_new_meshes, follow_visibility)
                    .chain()
                    .after(VisibilitySystems::VisibilityPropagate),
            );
    }
}

/// The traceable copy of each mesh that has one, by the original's id.
#[derive(Resource, Default)]
struct Copies(HashMap<AssetId<Mesh>, Handle<Mesh>>);

/// What the rays see in place of the ground's and the trees' own materials.
#[derive(Resource)]
struct StandIns {
    ground: Handle<StandardMaterial>,
    tree: Handle<StandardMaterial>,
}

/// The copy an entity is traced through, on the entity itself or (for the ground and trees) on
/// a child of it. Traced while the entity is visible.
#[derive(Component)]
struct Traced(Handle<Mesh>);

fn stand_in_materials(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let plain = |r: f32, g: f32, b: f32| StandardMaterial {
        base_color: Color::srgb(r, g, b),
        perceptual_roughness: 0.9,
        reflectance: 0.3,
        ..default()
    };
    commands.insert_resource(StandIns {
        ground: materials.add(plain(0.38, 0.34, 0.24)),
        tree: materials.add(plain(0.24, 0.30, 0.14)),
    });
}

/// The main camera lit by Solari (which wants no MSAA and a main texture it can write to); any
/// other camera (the creature preview) draws with Bevy's own deferred lighting, since nothing
/// opaque is drawn forward any more.
fn cameras(
    mut commands: Commands,
    main: Query<Entity, Added<MainCamera>>,
    others: Query<Entity, (Added<Camera3d>, Without<MainCamera>)>,
) {
    for e in &main {
        commands.entity(e).insert((
            Msaa::Off,
            CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
            SolariLighting::default(),
        ));
    }
    for e in &others {
        commands.entity(e).insert((Msaa::Off, DepthPrepass, DeferredPrepass));
    }
}

/// A copy of `mesh` that Solari can trace, or None if it isn't made of triangles.
fn traceable(mesh: &Mesh) -> Option<Mesh> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return None;
    }
    let Some(VertexAttributeValues::Float32x3(positions)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        return None;
    };
    let n = positions.len();
    let indices: Vec<u32> = match mesh.indices() {
        Some(i) => i.iter().map(|i| i as u32).collect(),
        None => (0..n as u32).collect(),
    };
    let mut copy = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone())
        .with_inserted_indices(Indices::U32(indices));
    match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(normals)) => copy.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals.clone()),
        _ => copy.compute_normals(),
    }
    let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(VertexAttributeValues::Float32x2(uvs)) => uvs.clone(),
        _ => vec![[0.0, 0.0]; n],
    };
    copy.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    // Only the shading of what a ray hits uses them, and none of these materials has a normal map.
    copy.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; n]);
    Some(copy)
}

/// The copy of a mesh, made the first time it's asked for. None for meshes that can't be traced
/// or whose data lives only on the GPU (the water and effects).
fn copy_of(id: AssetId<Mesh>, copies: &mut Copies, meshes: &mut Assets<Mesh>) -> Option<Handle<Mesh>> {
    if let Some(h) = copies.0.get(&id) {
        return Some(h.clone());
    }
    let copy = traceable(meshes.get(id)?)?;
    let h = meshes.add(copy);
    copies.0.insert(id, h.clone());
    Some(h)
}

/// Remakes the copies of meshes that changed (the ground is rebuilt as it's dug and as the
/// camera moves) and forgets those of meshes that are gone.
fn refresh_copies(mut events: MessageReader<AssetEvent<Mesh>>, mut copies: ResMut<Copies>, mut meshes: ResMut<Assets<Mesh>>) {
    for event in events.read() {
        match *event {
            AssetEvent::Modified { id } => {
                let Some(h) = copies.0.get(&id).cloned() else { continue };
                if let Some(copy) = meshes.get(id).and_then(traceable) {
                    let _ = meshes.insert(&h, copy);
                }
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                copies.0.remove(&id);
            }
            _ => {}
        }
    }
}

/// Gives each new mesh that casts a shadow in the world (not the creature preview's, on a layer
/// of its own) a copy to be traced through.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn trace_new_meshes(
    mut commands: Commands,
    mut copies: ResMut<Copies>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    stand_ins: Res<StandIns>,
    plain: Query<(Entity, &Mesh3d, &MeshMaterial3d<StandardMaterial>, Option<&RenderLayers>), (Changed<Mesh3d>, Without<NotShadowCaster>)>,
    ground: Query<(Entity, &Mesh3d), (Changed<Mesh3d>, With<MeshMaterial3d<TerrainMaterial>>, Without<NotShadowCaster>)>,
    trees: Query<(Entity, &Mesh3d), (Changed<Mesh3d>, With<MeshMaterial3d<TreeMaterial>>, Without<NotShadowCaster>)>,
) {
    for (e, mesh, material, layers) in &plain {
        let solid = materials
            .get(&material.0)
            .is_some_and(|m| !m.unlit && matches!(m.alpha_mode, AlphaMode::Opaque | AlphaMode::Mask(_)));
        let in_world = layers.is_none_or(|l| l.intersects(&RenderLayers::default()));
        match copy_of(mesh.id(), &mut copies, &mut meshes).filter(|_| solid && in_world) {
            Some(h) => commands.entity(e).insert(Traced(h)),
            None => commands.entity(e).remove::<(Traced, RaytracingMesh3d)>(),
        };
    }
    let stood_in = ground.iter().map(|(e, m)| (e, m, &stand_ins.ground)).chain(trees.iter().map(|(e, m)| (e, m, &stand_ins.tree)));
    for (e, mesh, material) in stood_in {
        let Some(h) = copy_of(mesh.id(), &mut copies, &mut meshes) else { continue };
        commands.entity(e).with_child((Traced(h), MeshMaterial3d(material.clone()), Transform::default(), Visibility::Inherited));
    }
}

/// Traces what's visible and only that (Solari would trace hidden entities too).
#[allow(clippy::type_complexity)]
fn follow_visibility(
    mut commands: Commands,
    traced: Query<(Entity, &Traced, &InheritedVisibility, Has<RaytracingMesh3d>), Or<(Changed<Traced>, Changed<InheritedVisibility>)>>,
) {
    for (e, t, visible, tracing) in &traced {
        if visible.get() {
            commands.entity(e).insert(RaytracingMesh3d(t.0.clone()));
        } else if tracing {
            commands.entity(e).remove::<RaytracingMesh3d>();
        }
    }
}
