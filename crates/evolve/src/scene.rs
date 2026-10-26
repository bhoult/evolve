//! The terrain: the ground drawn as square chunks of the heightfield, each at a level of detail
//! for its distance from the camera, with the water on it painted from a buffer the shader
//! samples smoothly per pixel, and a second surface over deep water. The sky, sun, ocean and
//! weather live in their own modules.
//!
//! A terrain can be millions of points a side's square (25 million at 5000 a side), so nothing
//! here walks the whole grid on the main thread when it can be helped: changes are found chunk
//! by chunk in parallel, only the chunks that changed are rebuilt, only the rows of the water
//! buffer that changed are sent, and chunk meshes are built a frame's budget at a time, nearest
//! first.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Buffer, BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::shader::ShaderRef;
use evolve_sim::terrain::Terrain;
use rayon::prelude::*;

use crate::Snap;
use crate::camera::MainCamera;
use crate::env::{EnvGpu, GraphicsSettings, shader};

/// Fallback clear color before the sky has drawn.
pub const SKY: Color = Color::srgb(0.62, 0.75, 0.88);

pub struct ScenePlugin;

#[derive(Component)]
pub struct TerrainMesh;

/// Water deeper than this gets a surface of its own (the lake mesh) above the ground; shallower
/// water is painted on the ground.
const SURFACE_DEPTH: f32 = 0.1;

/// Cells along each side of a terrain chunk.
const CHUNK: usize = 128;
/// The coarsest a distant chunk gets: every this many grid points.
const MAX_STRIDE: usize = 16;
/// Terrains up to this many points a side are drawn at full detail everywhere: a few hundred
/// thousand vertices cost nothing, and every close-up keeps exactly the physics surface.
const LOD_FROM: usize = 513;
/// A chunk is drawn coarser while one of its cells, at that stride, spans less than this much of
/// the view (its width over its distance, in radians; a pixel is about 0.0005).
const LOD_ANGLE: f32 = 0.012;
/// Vertices built per frame at most (nearest chunks first), so a new world or a camera jump
/// spreads its work over a few frames instead of stalling one.
const VERTEX_BUDGET: usize = 600_000;
/// And time spent building chunks per frame at most, ms, once the first is built: on a big
/// terrain with hundreds of creatures digging and trampling everywhere, chunks change faster
/// than that, and building the whole vertex budget every frame took 30-150 ms a frame.
const BUILD_MS: f32 = 4.0;

/// Beaches, rock, grass, sea floor and cloud shadows, all from world position in the shader, so
/// changing the water level needs no rebuild.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct TerrainExt {
    #[storage(100, read_only, buffer)]
    env: Buffer,
    /// Water depth and soil moisture at every grid point (see `pack_surface`), which the shader
    /// interpolates smoothly per pixel: interpolated per vertex across triangles, stream banks
    /// zigzagged with the triangulation.
    #[storage(101, read_only, buffer)]
    surface: Buffer,
    /// Where creatures stir the water (splash.rs), for ripples on lakes, ponds and streams.
    #[storage(102, read_only, buffer)]
    ripples: Buffer,
}

impl MaterialExtension for TerrainExt {
    fn fragment_shader() -> ShaderRef {
        shader("terrain.wgsl")
    }

    // Used only when the scene is drawn deferred (ray-traced lighting, see raytracing.rs).
    fn deferred_fragment_shader() -> ShaderRef {
        shader("terrain.wgsl")
    }
}

pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExt>;

/// A square of the terrain, drawn as one mesh (and another over its deep water, if it has any).
struct Chunk {
    /// Its first grid row and column, and how many cells it spans each way.
    i0: usize,
    j0: usize,
    ni: usize,
    nj: usize,
    /// The stride its ground mesh was built at, 0 before it has been.
    stride: usize,
    /// The ground (heights, dug earth, trampling) or the deep water changed since it was built.
    stale: bool,
    lake_stale: bool,
    ground: Option<(Entity, Handle<Mesh>)>,
    lake: Option<(Entity, Handle<Mesh>)>,
}

#[derive(Resource, Default)]
struct TerrainState {
    /// The terrain the chunks were built from. Compared by identity, not version number: every
    /// freshly generated world starts at version 1.
    built_from: Option<Arc<Terrain>>,
    epoch: u64,
    chunks: Vec<Chunk>,
    material: Option<Handle<TerrainMaterial>>,
    /// Ground drawn tinted by its detail level (see `GraphicsSettings::lod_tint`).
    lod_tint: bool,
    /// The surface of deep water: the same material, blended, drawing its own branch.
    lake_material: Option<Handle<TerrainMaterial>>,
    surface: Option<Buffer>,
    /// A comparison of `built_from` with a newer copy of the same world, running on a thread of
    /// its own (over a big terrain it takes tens of milliseconds): the newer copy, and where the
    /// answer comes back.
    scan: Option<(Arc<Terrain>, std::sync::Mutex<std::sync::mpsc::Receiver<Scan>>)>,
}

/// What changed between two copies of the same world: per chunk whether its ground and its deep
/// water did, and the runs of rows of water and wetness to send.
struct Scan {
    chunks: Vec<(bool, bool)>,
    rows: Vec<(usize, usize)>,
}

/// A chunk's first row and column and its size in cells, for a scan away from the chunks.
type Rect = (usize, usize, usize, usize);

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default())
            .init_resource::<TerrainState>()
            .add_systems(Update, update_terrain);
    }
}

#[allow(clippy::too_many_arguments)]
fn update_terrain(
    mut commands: Commands,
    snap: Res<Snap>,
    gpu: Res<EnvGpu>,
    ripples: Res<crate::water::RippleBuffer>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    settings: Res<GraphicsSettings>,
    mut state: ResMut<TerrainState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    camera: Query<&GlobalTransform, With<MainCamera>>,
) {
    let _cost = crate::FrameCost::start("update_terrain");
    let t = snap.0.terrain.clone();
    if t.res < 3 {
        return;
    }
    let state = &mut *state;
    // A finished scan: mark the chunks that changed and send the water rows that did.
    if let Some((newer, rx)) = &state.scan {
        let got = rx.lock().map_or(Err(std::sync::mpsc::TryRecvError::Disconnected), |rx| rx.try_recv());
        match got {
            Ok(scan) => {
                let newer = newer.clone();
                apply_scan(state, &newer, scan, &queue);
                state.scan = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => state.scan = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }
    if !state.built_from.as_ref().is_some_and(|b| Arc::ptr_eq(b, &t)) && state.scan.is_none() {
        take_in(&mut commands, state, &t, snap.0.world_epoch, &device, &queue, &gpu, &ripples.0, &mut materials);
    }
    if state.lod_tint != settings.lod_tint {
        state.lod_tint = settings.lod_tint;
        for c in &mut state.chunks {
            c.stale = true;
        }
    }
    let Some(cam) = camera.iter().next() else {
        return;
    };
    build_chunks(&mut commands, state, &t, cam.translation(), &mut meshes);
}

/// Takes in a new copy of the terrain: a new world gets a fresh set of chunks; the same world
/// marks the chunks whose ground or deep water changed, and sends the rows of water that did
/// (found on a thread of its own for a big terrain, and taken in when it's done).
#[allow(clippy::too_many_arguments)]
fn take_in(
    commands: &mut Commands,
    state: &mut TerrainState,
    t: &Arc<Terrain>,
    epoch: u64,
    device: &RenderDevice,
    queue: &RenderQueue,
    gpu: &EnvGpu,
    ripples: &Buffer,
    materials: &mut Assets<TerrainMaterial>,
) {
    let same = state
        .built_from
        .clone()
        .filter(|o| o.res == t.res && o.size == t.size && state.epoch == epoch && same_ground(o, t));
    state.epoch = epoch;
    match same {
        Some(old) => {
            let rects: Vec<Rect> = state.chunks.iter().map(|c| (c.i0, c.j0, c.ni, c.nj)).collect();
            if t.res > LOD_FROM {
                let (tx, rx) = std::sync::mpsc::channel();
                let newer = t.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(scan(&old, &newer, &rects));
                });
                state.scan = Some((t.clone(), std::sync::Mutex::new(rx)));
            } else {
                let found = scan(&old, t, &rects);
                apply_scan(state, t, found, queue);
            }
            return;
        }
        None => {
            state.built_from = Some(t.clone());
            state.scan = None;
            for c in state.chunks.drain(..) {
                for (e, _) in c.ground.into_iter().chain(c.lake) {
                    commands.entity(e).despawn();
                }
            }
            state.chunks = layout(t.res);
            let bytes = surface_bytes(t, 0, t.res);
            let fits = state.surface.as_ref().is_some_and(|b| b.size() == bytes.len() as u64);
            if fits {
                queue.write_buffer(state.surface.as_ref().unwrap(), 0, &bytes);
            } else {
                // A new size: a new buffer, which the materials have to be told about.
                let buf = device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("evolve terrain surface"),
                    contents: &bytes,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });
                state.surface = Some(buf.clone());
                let handles = state.material.iter().chain(&state.lake_material);
                for h in handles {
                    if let Some(m) = materials.get_mut(h) {
                        m.extension.surface = buf.clone();
                    }
                }
            }
        }
    }
    let surface = state.surface.clone().expect("made above for a new world");
    let ext = || TerrainExt { env: gpu.buffer.clone(), surface: surface.clone(), ripples: ripples.clone() };
    if state.material.is_none() {
        state.material = Some(materials.add(ExtendedMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.9,
                reflectance: 0.3,
                ..default()
            },
            extension: ext(),
        }));
    }
    if state.lake_material.is_none() {
        state.lake_material = Some(materials.add(ExtendedMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                alpha_mode: AlphaMode::Blend,
                ..default()
            },
            extension: ext(),
        }));
    }
}

/// What changed between two copies of the same world (see `Scan`).
fn scan(old: &Terrain, t: &Terrain, rects: &[Rect]) -> Scan {
    Scan {
        chunks: rects.par_iter().map(|&r| chunk_changes(old, t, r)).collect(),
        rows: surface_changes(old, t),
    }
}

/// Marks the chunks a scan found changed and sends the water rows it found; the chunks now
/// stand for `newer`.
fn apply_scan(state: &mut TerrainState, newer: &Arc<Terrain>, scan: Scan, queue: &RenderQueue) {
    for (c, (ground, lake)) in state.chunks.iter_mut().zip(scan.chunks) {
        c.stale |= ground;
        c.lake_stale |= lake;
    }
    if let Some(buf) = &state.surface {
        let n = newer.res;
        for (start, end) in scan.rows {
            queue.write_buffer(buf, (start * n * 4) as u64, &surface_bytes(newer, start, end));
        }
    }
    state.built_from = Some(newer.clone());
}

/// Whether two copies of a terrain on the same grid are the same world, judged from a few
/// thousand points of the untouched ground (a regenerated terrain differs nearly everywhere).
fn same_ground(a: &Terrain, b: &Terrain) -> bool {
    let n = a.ground.len().min(b.ground.len());
    if a.ground.len() != b.ground.len() {
        return false;
    }
    let step = (n / 4096).max(1);
    (0..n).step_by(step).all(|k| a.ground[k] == b.ground[k] && a.fertility[k] == b.fertility[k])
}

/// The chunks covering a grid of `res` points a side, row by row.
fn layout(res: usize) -> Vec<Chunk> {
    let cells = res - 1;
    let starts: Vec<(usize, usize)> = (0..cells).step_by(CHUNK).map(|a| (a, CHUNK.min(cells - a))).collect();
    let mut chunks = Vec::with_capacity(starts.len() * starts.len());
    for &(i0, ni) in &starts {
        for &(j0, nj) in &starts {
            chunks.push(Chunk {
                i0,
                j0,
                ni,
                nj,
                stride: 0,
                stale: true,
                lake_stale: true,
                ground: None,
                lake: None,
            });
        }
    }
    chunks
}

/// Whether a grid point's ground visibly changed between two copies of the same world: its
/// height, the earth dug or piled there (which only moves with the height or with erosion), or
/// how trampled it looks (trampling fades a little everywhere every update, so only a visible
/// step counts). Cheap comparisons first: this runs over every point of a big terrain.
fn ground_changed(old: &Terrain, t: &Terrain, k: usize) -> bool {
    let layer = |l: &[f32], k: usize| l.get(k).copied().unwrap_or(0.0);
    if old.heights[k] != t.heights[k] || layer(&old.eroded, k) != layer(&t.eroded, k) {
        return true;
    }
    let (a, b) = (layer(&old.wear, k), layer(&t.wear, k));
    a != b && (old.worn(k) * 64.0) as i32 != (t.worn(k) * 64.0) as i32
}

/// Whether a point's water visibly changed where it's deep enough to have a surface of its own
/// (or nearly): what the lake mesh shows.
fn lake_changed(old: &Terrain, t: &Terrain, k: usize) -> bool {
    let (a, b) = (old.water.get(k).copied().unwrap_or(0.0), t.water.get(k).copied().unwrap_or(0.0));
    let step = |w: f32| if w > SURFACE_DEPTH * 0.5 { (w * 200.0) as i32 } else { 0 };
    a != b && step(a) != step(b)
}

/// Whether a chunk's ground (heights or look) and its deep water changed between two copies of
/// the same world. Its points and the ring around them, whose heights shape its normals.
fn chunk_changes(old: &Terrain, t: &Terrain, (i0, j0, ni, nj): Rect) -> (bool, bool) {
    let n = t.res;
    let (i_lo, i_hi) = (i0.saturating_sub(1), (i0 + ni + 1).min(n - 1));
    let (j_lo, j_hi) = (j0.saturating_sub(1), (j0 + nj + 1).min(n - 1));
    let (mut ground, mut lake) = (false, false);
    for i in i_lo..=i_hi {
        let row = i * n;
        for k in row + j_lo..=row + j_hi {
            ground = ground || ground_changed(old, t, k);
            lake = lake || lake_changed(old, t, k);
        }
        if ground && lake {
            break;
        }
    }
    (ground, lake)
}

/// Water depth and soil moisture of one grid point as two half floats in a u32, the layout the
/// shader unpacks with `unpack2x16float`.
fn pack_surface(t: &Terrain, k: usize) -> u32 {
    let w = half::f16::from_f32(t.water.get(k).copied().unwrap_or(0.0)).to_bits() as u32;
    let m = half::f16::from_f32(t.moisture.get(k).copied().unwrap_or(0.0)).to_bits() as u32;
    w | (m << 16)
}

/// The surface buffer's bytes for grid rows `i0..i1`.
fn surface_bytes(t: &Terrain, i0: usize, i1: usize) -> Vec<u8> {
    let n = t.res;
    let packed: Vec<u32> = (i0 * n..i1 * n).into_par_iter().map(|k| pack_surface(t, k)).collect();
    bytemuck::cast_slice(&packed).to_vec()
}

/// The rows of an `n`-row grid for which `row_changed` holds, as runs of rows `[start, end)`.
pub(crate) fn changed_rows(n: usize, row_changed: impl Fn(usize) -> bool + Sync + Send) -> Vec<(usize, usize)> {
    let flags: Vec<bool> = (0..n).into_par_iter().map(row_changed).collect();
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        if flags[i] {
            let start = i;
            while i < n && flags[i] {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    runs
}

/// The rows of water and wetness that visibly changed between two copies of the same world
/// (wetness drifts a little everywhere every update; a coarse step of it is what shows), as runs
/// of rows. All of them if a layer has just appeared (an older save's first hydrology).
fn surface_changes(old: &Terrain, t: &Terrain) -> Vec<(usize, usize)> {
    let n = t.res;
    let (w0, w1, m0, m1) = (&old.water, &t.water, &old.moisture, &t.moisture);
    if w0.len() != w1.len() || m0.len() != m1.len() || w1.len() < n * n || m1.len() < n * n {
        return vec![(0, n)];
    }
    changed_rows(n, |i| {
        (i * n..(i + 1) * n).any(|k| {
            (w0[k] != w1[k] && (w0[k] * 400.0) as i32 != (w1[k] * 400.0) as i32)
                || (m0[k] != m1[k] && (m0[k] * 64.0) as i32 != (m1[k] * 64.0) as i32)
        })
    })
}

/// The stride a chunk at `distance` should be drawn at, sticking with `current` near the
/// thresholds so a chunk on the boundary doesn't flip back and forth as the camera drifts.
fn stride_for(cell: f32, distance: f32, current: usize) -> usize {
    let pick = |angle: f32| {
        let mut s = 1;
        while s * 2 <= MAX_STRIDE && cell * (s * 2) as f32 / distance.max(1e-3) < angle {
            s *= 2;
        }
        s
    };
    let (fine, coarse) = (pick(LOD_ANGLE * 0.8), pick(LOD_ANGLE * 1.25));
    if current >= fine && current <= coarse { current } else { pick(LOD_ANGLE) }
}

/// Rebuilds the chunks that changed or want another level of detail, nearest first, within the
/// frame's vertex budget.
fn build_chunks(commands: &mut Commands, state: &mut TerrainState, t: &Terrain, eye: Vec3, meshes: &mut Assets<Mesh>) {
    let lod = t.res > LOD_FROM;
    let cell = t.cell();
    let mut todo: Vec<(f32, usize, usize)> = state
        .chunks
        .iter()
        .enumerate()
        .filter_map(|(c, ch)| {
            let distance = chunk_distance(t, ch, eye);
            let want = if lod { stride_for(cell, distance, ch.stride) } else { 1 };
            (ch.stride != want || ch.stale || ch.lake_stale).then_some((distance, c, want))
        })
        .collect();
    todo.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut budget = VERTEX_BUDGET;
    let started = std::time::Instant::now();
    for (_, c, want) in todo {
        let ch = &state.chunks[c];
        let verts = (ch.ni / want + 2) * (ch.nj / want + 2);
        if budget < VERTEX_BUDGET && (verts > budget || started.elapsed().as_secs_f32() * 1000.0 > BUILD_MS) {
            break;
        }
        budget = budget.saturating_sub(verts);
        // The deep water's surface sits on the ground, so it's rebuilt with it.
        let reground = ch.stride != want || ch.stale;
        if reground {
            let mesh = ground_mesh(t, ch, want, state.lod_tint);
            let material = state.material.clone().expect("made with the first terrain");
            let ch = &mut state.chunks[c];
            match &ch.ground {
                Some((_, h)) => {
                    let _ = meshes.insert(h, mesh);
                }
                None => {
                    let h = meshes.add(mesh);
                    let e = commands.spawn((Mesh3d(h.clone()), MeshMaterial3d(material), Transform::default(), TerrainMesh)).id();
                    ch.ground = Some((e, h));
                }
            }
            ch.stride = want;
            ch.stale = false;
        }
        let ch = &state.chunks[c];
        if reground || ch.lake_stale {
            let mesh = lake_mesh(t, ch, want);
            let lake_material = state.lake_material.clone().expect("made with the first terrain");
            let ch = &mut state.chunks[c];
            match (mesh, &ch.lake) {
                (Some(mesh), Some((_, h))) => {
                    let _ = meshes.insert(h, mesh);
                }
                (Some(mesh), None) => {
                    let h = meshes.add(mesh);
                    let e = commands
                        .spawn((
                            Mesh3d(h.clone()),
                            MeshMaterial3d(lake_material),
                            Transform::default(),
                            TerrainMesh,
                            bevy::light::NotShadowCaster,
                        ))
                        .id();
                    ch.lake = Some((e, h));
                }
                (None, Some((e, _))) => {
                    commands.entity(*e).despawn();
                    ch.lake = None;
                }
                (None, None) => {}
            }
            ch.lake_stale = false;
        }
    }
}

/// How far the camera is from a chunk's square (in the plane, and in height from its middle).
fn chunk_distance(t: &Terrain, c: &Chunk, eye: Vec3) -> f32 {
    let (x0, z0) = grid_point_xz(t, c.i0, c.j0);
    let (x1, z1) = grid_point_xz(t, c.i0 + c.ni, c.j0 + c.nj);
    let dx = (x0 - eye.x).max(eye.x - x1).max(0.0);
    let dz = (z0 - eye.z).max(eye.z - z1).max(0.0);
    let mid = t.heights[(c.i0 + c.ni / 2) * t.res + c.j0 + c.nj / 2];
    Vec3::new(dx, eye.y - mid, dz).length()
}

/// The grid lines a chunk is drawn along at `stride`: every `stride`-th from its first, and its
/// last whether or not the stride lands on it (so neighbouring chunks share their edges).
fn lines(start: usize, cells: usize, stride: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (start..start + cells).step_by(stride).collect();
    v.push(start + cells);
    v
}

/// Where grid point (i, j) sits on the ground plane.
fn grid_point_xz(t: &Terrain, i: usize, j: usize) -> (f32, f32) {
    let n = (t.res - 1) as f32;
    ((j as f32 / n - 0.5) * t.size, (i as f32 / n - 0.5) * t.size)
}

/// The ground's normal at a spot, from heights `e` m either side: a cell either side at full
/// detail, wider for a coarser stride so distant chunks don't shimmer with detail they can't show.
fn normal(t: &Terrain, x: f32, z: f32, e: f32) -> [f32; 3] {
    let dx = t.height_at(x + e, z) - t.height_at(x - e, z);
    let dz = t.height_at(x, z + e) - t.height_at(x, z - e);
    Vec3::new(-dx, 2.0 * e, -dz).normalize().to_array()
}

/// Fertility in red, moved earth in green, blue set where it was dug rather than piled, and
/// how trampled the ground is in alpha.
fn vertex_color(t: &Terrain, k: usize) -> [f32; 4] {
    let d = t.disturbance(k);
    [t.fertility[k], (d.abs() / 0.6).min(1.0), if d < 0.0 { 1.0 } else { 0.0 }, t.worn(k)]
}

/// How far a chunk's skirt must hang below its edges to cover the gap to a neighbour drawn at
/// another detail level: the most its edges' heights stray from the straight lines between the
/// points the coarsest stride keeps, and a little more.
fn skirt_depth(t: &Terrain, c: &Chunk) -> f32 {
    let n = t.res;
    let h = |i: usize, j: usize| t.heights[i * n + j];
    let mut most = 0.0f32;
    let mut edge = |len: usize, at: &dyn Fn(usize) -> f32| {
        let keep = lines(0, len, MAX_STRIDE);
        for w in keep.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (ha, hb) = (at(a), at(b));
            for k in a..=b {
                let f = (k - a) as f32 / (b - a).max(1) as f32;
                most = most.max((at(k) - (ha + (hb - ha) * f)).abs());
            }
        }
    };
    edge(c.nj, &|k| h(c.i0, c.j0 + k));
    edge(c.nj, &|k| h(c.i0 + c.ni, c.j0 + k));
    edge(c.ni, &|k| h(c.i0 + k, c.j0));
    edge(c.ni, &|k| h(c.i0 + k, c.j0 + c.nj));
    most + 0.05 + t.cell() * 0.5
}

/// A chunk's ground mesh at `stride`, triangulated exactly like the physics heightfield
/// (diagonal from (x0, z1) to (x1, z0)), so at full detail it is the physics surface. A skirt
/// hangs down from its edges (both faces drawn) to hide the cracks where a neighbour drawn at
/// another stride has its edge vertices elsewhere. With `tint`, its detail level goes in UV x for
/// the shader to tint (debugging).
fn ground_mesh(t: &Terrain, c: &Chunk, stride: usize, tint: bool) -> Mesh {
    let n = t.res;
    let rows = lines(c.i0, c.ni, stride);
    let cols = lines(c.j0, c.nj, stride);
    let (nr, nc) = (rows.len(), cols.len());
    let e = t.cell() * 0.5 * stride as f32;
    let mut positions = Vec::with_capacity(nr * nc + 2 * (nr + nc));
    let mut normals = Vec::with_capacity(positions.capacity());
    let mut colors = Vec::with_capacity(positions.capacity());
    for &i in &rows {
        for &j in &cols {
            let k = i * n + j;
            let (x, z) = grid_point_xz(t, i, j);
            positions.push([x, t.heights[k], z]);
            normals.push(normal(t, x, z, e));
            colors.push(vertex_color(t, k));
        }
    }
    let idx = |a: usize, b: usize| (a * nc + b) as u32;
    let mut indices = Vec::with_capacity((nr - 1) * (nc - 1) * 6 + 24 * (nr + nc));
    for a in 0..nr - 1 {
        for b in 0..nc - 1 {
            indices.extend_from_slice(&[idx(a, b), idx(a + 1, b), idx(a, b + 1)]);
            indices.extend_from_slice(&[idx(a + 1, b), idx(a + 1, b + 1), idx(a, b + 1)]);
        }
    }
    // The skirt, only where a neighbour may be drawn at another stride (large terrains).
    if n > LOD_FROM {
        let depth = skirt_depth(t, c);
        let edges: [Vec<u32>; 4] = [
            (0..nc).map(|b| idx(0, b)).collect(),
            (0..nc).map(|b| idx(nr - 1, b)).collect(),
            (0..nr).map(|a| idx(a, 0)).collect(),
            (0..nr).map(|a| idx(a, nc - 1)).collect(),
        ];
        for edge in edges {
            let first = positions.len() as u32;
            for &v in &edge {
                let [x, y, z] = positions[v as usize];
                positions.push([x, y - depth, z]);
                normals.push(normals[v as usize]);
                colors.push(colors[v as usize]);
            }
            for w in 0..edge.len() - 1 {
                let (t0, t1, b0, b1) = (edge[w], edge[w + 1], first + w as u32, first + w as u32 + 1);
                indices.extend_from_slice(&[t0, t1, b0, t1, b1, b0, t0, b0, t1, t1, b0, b1]);
            }
        }
    }
    let count = positions.len();
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices));
    if tint {
        mesh.with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[stride.trailing_zeros() as f32, 0.0f32]; count])
    } else {
        mesh
    }
}

/// The surface of a chunk's water deeper than `SURFACE_DEPTH`, on its grid at `stride`: the
/// ground plus the water there, or tucked just under the ground where there's none (so it meets
/// the bank in between). Its UV is (depth, -1), the -1 telling the shader it's a water surface.
/// Lakes and ponds lie level; deep streams follow their bed. None if the chunk has no deep water.
fn lake_mesh(t: &Terrain, c: &Chunk, stride: usize) -> Option<Mesh> {
    let n = t.res;
    let water = |k: usize| t.water.get(k).copied().unwrap_or(0.0);
    let (i_lo, i_hi) = (c.i0.saturating_sub(1), (c.i0 + c.ni + 1).min(n - 1));
    let (j_lo, j_hi) = (c.j0.saturating_sub(1), (c.j0 + c.nj + 1).min(n - 1));
    let any = (i_lo..=i_hi).any(|i| (j_lo..=j_hi).any(|j| water(i * n + j) > SURFACE_DEPTH));
    if !any {
        return None;
    }
    let surface = |k: usize| (water(k) > SURFACE_DEPTH).then(|| t.heights[k] + water(k));
    let rows = lines(c.i0, c.ni, stride);
    let cols = lines(c.j0, c.nj, stride);
    let (nr, nc) = (rows.len(), cols.len());
    let mut positions = Vec::with_capacity(nr * nc);
    let mut normals = Vec::with_capacity(nr * nc);
    let mut uvs = Vec::with_capacity(nr * nc);
    let e = t.cell() * 0.5 * stride as f32;
    for &i in &rows {
        for &j in &cols {
            let k = i * n + j;
            let (x, z) = grid_point_xz(t, i, j);
            let w = water(k);
            let pooled = t.pooled.get(k).copied().unwrap_or(0.0);
            let deep = w > SURFACE_DEPTH;
            // Next to water, a dry point whose ground rises above it keeps the water's level, so
            // the surface meets the bank where the bank crosses it (the ground hides the rest)
            // instead of slanting up the bank in a bright sliver. Elsewhere it sits under the
            // ground. (Neighbours at full detail, whatever the stride.)
            let height = if deep {
                t.heights[k] + w
            } else {
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
            normals.push(if pooled >= w - 1e-3 { [0.0, 1.0, 0.0] } else { normal(t, x, z, e) });
            uvs.push([if deep { w } else { 0.0 }, -1.0]);
        }
    }
    let idx = |a: usize, b: usize| (a * nc + b) as u32;
    let mut indices = Vec::with_capacity((nr - 1) * (nc - 1) * 6);
    for a in 0..nr - 1 {
        for b in 0..nc - 1 {
            indices.extend_from_slice(&[idx(a, b), idx(a + 1, b), idx(a, b + 1)]);
            indices.extend_from_slice(&[idx(a + 1, b), idx(a + 1, b + 1), idx(a, b + 1)]);
        }
    }
    Some(
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0f32; 4]; nr * nc])
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(indices)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use evolve_sim::SimParams;
    use std::time::Instant;

    fn terrain(res: u32, size: f32) -> Terrain {
        let p = SimParams { terrain_resolution: res, world_size: size, ..SimParams::default() };
        Terrain::generate(&p, 1)
    }

    /// The chunks tile the grid exactly, and at full detail a chunk's triangles are the physics
    /// surface's: same points, same diagonal.
    #[test]
    fn chunks_tile_the_grid_and_match_the_physics_surface() {
        let t = terrain(177, 220.0);
        let chunks = layout(t.res);
        let cells: usize = chunks.iter().map(|c| c.ni * c.nj).sum();
        assert_eq!(cells, (t.res - 1) * (t.res - 1));
        let c = &chunks[0];
        let m = ground_mesh(&t, c, 1, false);
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) = m.attribute(Mesh::ATTRIBUTE_POSITION) else {
            panic!("positions")
        };
        let Some(Indices::U32(ix)) = m.indices() else { panic!("indices") };
        // The surface's height at the middle of each triangle is the physics surface's.
        for tri in ix.chunks(3).take(2000) {
            let v: Vec<Vec3> = tri.iter().map(|&k| Vec3::from_array(p[k as usize])).collect();
            let mid = (v[0] + v[1] + v[2]) / 3.0;
            assert!((t.height_at(mid.x, mid.z) - mid.y).abs() < 1e-3, "{mid} vs {}", t.height_at(mid.x, mid.z));
        }
        // Small terrains have no skirts.
        assert_eq!(p.len(), (c.ni + 1) * (c.nj + 1));
    }

    #[test]
    fn distant_chunks_get_coarser_and_stay_put_near_the_thresholds() {
        let cell = 0.16;
        assert_eq!(stride_for(cell, 5.0, 0), 1);
        assert_eq!(stride_for(cell, 1000.0, 0), MAX_STRIDE);
        let s = stride_for(cell, 60.0, 0);
        assert_eq!(stride_for(cell, 60.0 * 1.1, s), s, "a little farther keeps it");
        assert_eq!(stride_for(cell, 60.0 / 1.1, s), s, "a little nearer too");
    }

    #[test]
    fn packed_water_unpacks_to_what_it_was() {
        let mut t = terrain(65, 100.0);
        t.water = vec![0.0; t.res * t.res].into();
        t.moisture = vec![0.0; t.res * t.res].into();
        t.water[10] = 0.37;
        t.moisture[10] = 0.8;
        let v = pack_surface(&t, 10);
        let w = half::f16::from_bits((v & 0xffff) as u16).to_f32();
        let m = half::f16::from_bits((v >> 16) as u16).to_f32();
        assert!((w - 0.37).abs() < 1e-3 && (m - 0.8).abs() < 1e-3);
        let runs = changed_rows(10, |i| i == 2 || i == 3 || i == 7);
        assert_eq!(runs, vec![(2, 4), (7, 8)]);
    }

    /// What the terrain costs to build and keep current at three sizes: every chunk at full
    /// detail, the chunks a camera above the middle wants, a change scan between two copies, and
    /// packing the water buffer.
    /// `cargo test --release -p evolve terrain_costs -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn terrain_costs() {
        for (res, size) in [(177u32, 220.0f32), (1025, 800.0), (5000, 800.0)] {
            let started = Instant::now();
            let t = terrain(res, size);
            let made = started.elapsed().as_secs_f32();
            let chunks = layout(t.res);
            let eye = Vec3::new(0.0, t.heights[t.res * t.res / 2 + t.res / 2] + 40.0, 0.0);
            let full_verts: usize = chunks.iter().map(|c| (c.ni + 1) * (c.nj + 1)).sum();
            let lod = t.res > LOD_FROM;
            let started = Instant::now();
            let mut verts = 0;
            for c in &chunks {
                let s = if lod { stride_for(t.cell(), chunk_distance(&t, c, eye), 0) } else { 1 };
                let m = ground_mesh(&t, c, s, false);
                verts += m.count_vertices();
            }
            let built = started.elapsed().as_secs_f32();
            // A dig in the middle, and the drift of a water update: wetness a little different
            // everywhere, one stream point deeper.
            let count = t.res * t.res;
            let mid = count / 2;
            let mut was = t.clone();
            was.moisture = vec![0.3005; count].into();
            was.water = vec![0.0; count].into();
            let mut changed = t.clone();
            changed.heights[mid] += 0.1;
            changed.moisture = vec![0.3; count].into();
            changed.water = vec![0.0; count].into();
            changed.water[mid + 5] = 0.2;
            let started = Instant::now();
            let marks: Vec<(bool, bool)> = chunks.par_iter().map(|c| chunk_changes(&was, &changed, (c.i0, c.j0, c.ni, c.nj))).collect();
            let scanned = started.elapsed().as_secs_f32();
            let started = Instant::now();
            let rows = surface_changes(&was, &changed);
            let rows_scanned = started.elapsed().as_secs_f32();
            let started = Instant::now();
            let n = t.res;
            let (a, b) = (&was.heights, &changed.heights);
            let height_rows = changed_rows(n, |i| a[i * n..(i + 1) * n] != b[i * n..(i + 1) * n]);
            println!("  height row scan {:.1} ms ({} runs)", started.elapsed().as_secs_f32() * 1e3, height_rows.len());
            let started = Instant::now();
            let bytes = surface_bytes(&t, 0, t.res);
            let packed = started.elapsed().as_secs_f32();
            println!(
                "{res} a side ({size} m, made in {made:.1} s): {} chunks; {full_verts} vertices at full detail, {verts} as drawn from 40 m over the middle, built in {:.0} ms; chunk change scan {:.1} ms ({} ground, {} lake changed); water row scan {:.1} ms ({} rows to send); whole water buffer {:.0} MB packed in {:.0} ms",
                chunks.len(),
                built * 1e3,
                scanned * 1e3,
                marks.iter().filter(|m| m.0).count(),
                marks.iter().filter(|m| m.1).count(),
                rows_scanned * 1e3,
                rows.iter().map(|r| r.1 - r.0).sum::<usize>(),
                bytes.len() as f32 / 1e6,
                packed * 1e3
            );
        }
    }
}
