//! The ocean: one camera-centered polar grid out to the horizon, displaced by Gerstner waves and
//! shaded against the terrain heightfield for depth color, transparency and surf.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Buffer, BufferDescriptor, BufferUsages, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::shader::ShaderRef;
use evolve_sim::terrain::Terrain;

use crate::Snap;
use crate::camera::MainCamera;
use crate::env::{EnvGpu, EnvSystems, heights_buffer, heights_bytes, shader};

pub struct WaterPlugin;

/// Keeps the heights buffer current and the ocean under the camera.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct WaterSystems;

/// The terrain heights on the GPU. Marked changed only when the buffer itself is replaced.
#[derive(Resource)]
pub struct Heights {
    pub buffer: Buffer,
    from: Option<Arc<Terrain>>,
}

/// Where creatures are stirring the surface, written each frame by splash.rs.
#[derive(Resource)]
pub struct RippleBuffer(pub Buffer);

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WaterMaterial {
    #[storage(100, read_only, buffer)]
    env: Buffer,
    #[storage(101, read_only, buffer)]
    heights: Buffer,
    #[storage(102, read_only, buffer)]
    ripples: Buffer,
}

impl Material for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        shader("water.wgsl")
    }

    fn fragment_shader() -> ShaderRef {
        shader("water.wgsl")
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _: &MaterialPipeline,
        desc: &mut RenderPipelineDescriptor,
        _: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        desc.primitive.cull_mode = None;
        Ok(())
    }
}

#[derive(Component)]
struct Ocean(Handle<WaterMaterial>);

// Ring radii are RING_A * (exp(RING_B * i) - 1): about 0.4 m apart under the camera, growing
// with distance, out past 8 km. Must match water.wgsl.
const RINGS: usize = 200;
const SEGMENTS: usize = 256;
const RING_A: f32 = 12.0;
const RING_B: f32 = 0.033;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .add_systems(PreStartup, create_heights)
            .add_systems(Startup, spawn_ocean)
            .add_systems(Update, (update_heights, follow_camera).chain().in_set(WaterSystems).after(EnvSystems));
    }
}

fn create_heights(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    commands.insert_resource(Heights {
        buffer: heights_buffer(&device, &Terrain::default()),
        from: None,
    });
    let ripples = device.create_buffer(&BufferDescriptor {
        label: Some("evolve ripples"),
        size: crate::splash::RIPPLE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // No ripples yet.
    queue.write_buffer(&ripples, 0, &[0; 32]);
    commands.insert_resource(RippleBuffer(ripples));
}

fn spawn_ocean(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    gpu: Res<EnvGpu>,
    heights: Res<Heights>,
    ripples: Res<RippleBuffer>,
) {
    let mat = materials.add(WaterMaterial {
        env: gpu.buffer.clone(),
        heights: heights.buffer.clone(),
        ripples: ripples.0.clone(),
    });
    commands.spawn((
        Mesh3d(meshes.add(polar_grid())),
        MeshMaterial3d(mat.clone()),
        Transform::default(),
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        Ocean(mat),
    ));
}

fn polar_grid() -> Mesh {
    let mut positions = vec![[0.0f32, 0.0, 0.0]];
    for i in 1..=RINGS {
        let r = RING_A * ((RING_B * i as f32).exp() - 1.0);
        for s in 0..SEGMENTS {
            let a = s as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            positions.push([r * a.cos(), 0.0, r * a.sin()]);
        }
    }
    let at = |ring: usize, s: usize| (1 + (ring - 1) * SEGMENTS + s % SEGMENTS) as u32;
    let mut indices = Vec::with_capacity(RINGS * SEGMENTS * 6);
    for s in 0..SEGMENTS {
        indices.extend_from_slice(&[0, at(1, s + 1), at(1, s)]);
    }
    for ring in 1..RINGS {
        for s in 0..SEGMENTS {
            let (a, b, c, d) = (at(ring, s), at(ring, s + 1), at(ring + 1, s), at(ring + 1, s + 1));
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_indices(Indices::U32(indices))
}

/// Copies changed heights into the buffer (only the rows that changed, for the same world: a
/// terrain can be 100 MB of heights), or makes a new buffer (and tells the materials) when the
/// terrain changes size.
fn update_heights(
    snap: Res<Snap>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut heights: ResMut<Heights>,
    ocean: Single<&Ocean>,
    mut materials: ResMut<Assets<WaterMaterial>>,
) {
    let t = &snap.0.terrain;
    if t.res < 3 || heights.from.as_ref().is_some_and(|f| Arc::ptr_eq(f, t)) {
        return;
    }
    let n = t.res;
    if let Some(old) = heights.from.clone().filter(|o| o.res == n && o.heights.len() == t.heights.len())
        && heights.buffer.size() == (n * n * 4) as u64
    {
        let h = heights.bypass_change_detection();
        let (was, now) = (&old.heights, &t.heights);
        for (start, end) in crate::scene::changed_rows(n, |i| was[i * n..(i + 1) * n] != now[i * n..(i + 1) * n]) {
            queue.write_buffer(&h.buffer, (start * n * 4) as u64, bytemuck::cast_slice(&t.heights[start * n..end * n]));
        }
        h.from = Some(t.clone());
        return;
    }
    let bytes = heights_bytes(t);
    if heights.buffer.size() == bytes.len() as u64 {
        // Same buffer, so nothing that binds it needs to hear about it.
        let h = heights.bypass_change_detection();
        queue.write_buffer(&h.buffer, 0, &bytes);
        h.from = Some(t.clone());
        return;
    }
    heights.buffer = heights_buffer(&device, t);
    heights.from = Some(t.clone());
    if let Some(m) = materials.get_mut(&ocean.0) {
        m.heights = heights.buffer.clone();
    }
}

/// Keep the grid centered under the camera, at the water level so it sorts sensibly.
fn follow_camera(
    snap: Res<Snap>,
    cam: Single<&Transform, (With<MainCamera>, Without<Ocean>)>,
    mut ocean: Single<&mut Transform, With<Ocean>>,
) {
    let p = cam.translation;
    ocean.translation = Vec3::new(p.x, snap.0.params.water_level, p.z);
}
