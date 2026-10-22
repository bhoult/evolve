//! Rain streaks around the camera and lightning in storms.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::diagnostic::FrameCount;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Buffer, RenderPipelineDescriptor, SpecializedMeshPipelineError};
use bevy::shader::ShaderRef;

use crate::camera::{MainCamera, OrbitCam};
use crate::env::{Env, EnvGpu, EnvSystems, GraphicsSettings, VisualOverride, shader};
use crate::water::Heights;
use crate::{Args, Snap};

pub struct WeatherFxPlugin;

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct RainMaterial {
    #[storage(100, read_only, buffer)]
    env: Buffer,
    #[storage(101, read_only, buffer)]
    heights: Buffer,
}

impl Material for RainMaterial {
    fn vertex_shader() -> ShaderRef {
        shader("rain.wgsl")
    }

    fn fragment_shader() -> ShaderRef {
        shader("rain.wgsl")
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
struct Rain(Handle<RainMaterial>);

#[derive(Component)]
struct Bolt;

const STREAKS: usize = 12_000;

/// A small xorshift generator; lightning only needs to look random.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
}

/// A lightning strike just happened here (for the thunder).
#[derive(Message, Clone, Copy)]
pub struct LightningStrike {
    pub at: Vec3,
}

/// The current strike: a few return strokes that flicker and fade.
#[derive(Resource)]
struct Lightning {
    rng: Rng,
    /// Wall time of the strike and of each stroke after it.
    start: f32,
    strokes: Vec<(f32, f32)>,
    material: Handle<StandardMaterial>,
}

impl Plugin for WeatherFxPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<RainMaterial>::default())
            .add_message::<LightningStrike>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    lightning.before(EnvSystems).after(crate::camera::apply_camera),
                    update_rain.after(EnvSystems).after(crate::water::WaterSystems),
                ),
            );
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rain_mats: ResMut<Assets<RainMaterial>>,
    mut std_mats: ResMut<Assets<StandardMaterial>>,
    gpu: Res<EnvGpu>,
    heights: Res<Heights>,
) {
    let mat = rain_mats.add(RainMaterial {
        env: gpu.buffer.clone(),
        heights: heights.buffer.clone(),
    });
    commands.spawn((
        Mesh3d(meshes.add(rain_mesh())),
        MeshMaterial3d(mat.clone()),
        Transform::default(),
        Visibility::Hidden,
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        Rain(mat),
    ));
    let material = std_mats.add(StandardMaterial {
        base_color: Color::BLACK,
        unlit: true,
        fog_enabled: false,
        alpha_mode: AlphaMode::Add,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default()))),
        MeshMaterial3d(material.clone()),
        Transform::default(),
        Visibility::Hidden,
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        Bolt,
    ));
    commands.insert_resource(Lightning {
        rng: Rng(0x9E37_79B9_7F4A_7C15),
        start: -100.0,
        strokes: Vec::new(),
        material,
    });
}

fn rain_mesh() -> Mesh {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut positions = Vec::with_capacity(STREAKS * 4);
    let mut uvs = Vec::with_capacity(STREAKS * 4);
    let mut indices = Vec::with_capacity(STREAKS * 6);
    for k in 0..STREAKS {
        let seed = [rng.next(), rng.next(), rng.next()];
        for uv in [[-1.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]] {
            positions.push(seed);
            uvs.push(uv);
        }
        let b = (k * 4) as u32;
        indices.extend_from_slice(&[b, b + 1, b + 2, b + 1, b + 3, b + 2]);
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

fn update_rain(
    env: Res<Env>,
    settings: Res<GraphicsSettings>,
    heights: Res<Heights>,
    cam: Single<&Transform, (With<MainCamera>, Without<Rain>)>,
    rain: Single<(&Rain, &mut Visibility, &mut Transform)>,
    mut materials: ResMut<Assets<RainMaterial>>,
) {
    let (r, mut vis, mut tf) = rain.into_inner();
    let show = settings.rain && env.rain > 0.01;
    let want = if show { Visibility::Inherited } else { Visibility::Hidden };
    if *vis != want {
        *vis = want;
    }
    tf.translation = cam.translation;
    if heights.is_changed()
        && let Some(m) = materials.get_mut(&r.0)
    {
        m.heights = heights.buffer.clone();
    }
}

#[allow(clippy::too_many_arguments)]
fn lightning(
    time: Res<Time>,
    frames: Res<FrameCount>,
    args: Res<Args>,
    over: Res<VisualOverride>,
    settings: Res<GraphicsSettings>,
    snap: Res<Snap>,
    orbit: Res<OrbitCam>,
    cam: Single<&Transform, (With<MainCamera>, Without<Bolt>)>,
    mut env: ResMut<Env>,
    mut state: ResMut<Lightning>,
    mut strikes: MessageWriter<LightningStrike>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    bolt: Single<(&Mesh3d, &mut Visibility, &mut Transform), With<Bolt>>,
) {
    let (bolt_mesh_handle, mut vis, mut bolt_tf) = bolt.into_inner();
    let now = time.elapsed_secs();
    let dt = time.delta_secs().min(0.1);
    let rate = if settings.lightning {
        ((env.storm - 0.4) / 0.6).clamp(0.0, 1.0).powf(1.5) * 0.4
    } else {
        0.0
    };
    let forced = over.lightning && args.screenshot.is_some() && frames.0 + 2 == args.frames;
    let s = &mut *state;
    if forced || (now - s.start > 1.0 && s.rng.next() < rate * dt) {
        s.start = now;
        s.strokes.clear();
        let n = 2 + (s.rng.next() * 3.0) as usize;
        let mut t = 0.0;
        for i in 0..n {
            s.strokes.push((t, if i == 0 { 1.0 } else { s.rng.range(0.4, 0.9) }));
            t += s.rng.range(0.05, 0.12);
        }
        let cam_pos = cam.translation;
        let fwd = cam.forward().as_vec3();
        let heading = Vec2::new(fwd.x, fwd.z).normalize_or(Vec2::X);
        let ang = s.rng.range(-0.7, 0.7);
        let dir = Vec2::from_angle(ang).rotate(heading);
        let dist = (orbit.distance * 0.8 + s.rng.range(120.0, 400.0)).min(900.0);
        let foot = Vec2::new(cam_pos.x, cam_pos.z) + dir * dist;
        let t = &snap.0.terrain;
        let wl = snap.0.params.water_level;
        let ground = if t.res >= 3 && t.contains(foot.x, foot.y, 0.0) {
            t.height_at(foot.x, foot.y).max(wl)
        } else {
            wl
        };
        let bottom = Vec3::new(foot.x, ground, foot.y);
        let top = bottom + Vec3::new(s.rng.range(-60.0, 60.0), 650.0, s.rng.range(-60.0, 60.0));
        let width = (dist * 0.004).max(0.8);
        // Built around the camera so it sorts after the water.
        let mesh = bolt_mesh(&mut s.rng, top - cam_pos, bottom - cam_pos, width, Vec3::ZERO);
        if let Some(m) = meshes.get_mut(&bolt_mesh_handle.0) {
            *m = mesh;
        }
        bolt_tf.translation = cam_pos;
        env.flash_dir = (top * 0.3 + bottom * 0.7 - cam_pos).normalize_or(Vec3::Y);
        strikes.write(LightningStrike { at: bottom });
    }
    let age = now - s.start;
    let mut flash = 0.0;
    for &(t0, a) in &s.strokes {
        if age >= t0 {
            flash += a * (-(age - t0) / 0.07).exp();
        }
    }
    let visible = age < 0.45 && !s.strokes.is_empty();
    let want = if visible { Visibility::Inherited } else { Visibility::Hidden };
    if *vis != want {
        *vis = want;
    }
    if visible && let Some(m) = materials.get_mut(&s.material) {
        let b = 3.0 + 9.0 * flash.min(1.5);
        m.base_color = Color::linear_rgb(0.85 * b, 0.9 * b, 1.0 * b);
    }
    // Smooth the flash a hair so a single frame never pops to full white.
    env.flash += (flash.min(1.5) - env.flash) * (1.0 - (-dt / 0.015).exp());
}

/// A jagged bolt with a few branches, as camera-facing ribbons, plus a faint wide glow around
/// the main channel.
fn bolt_mesh(rng: &mut Rng, top: Vec3, bottom: Vec3, width: f32, cam: Vec3) -> Mesh {
    let main = jagged(rng, top, bottom, 7);
    let mut strands = vec![(main.clone(), width * 7.0, 0.07), (main.clone(), width, 1.0)];
    for _ in 0..3 {
        let i = (rng.range(0.1, 0.6) * main.len() as f32) as usize;
        let from = main[i];
        let drop = (from.y - bottom.y) * rng.range(0.2, 0.45);
        let to = from + Vec3::new(rng.range(-0.6, 0.6) * drop, -drop, rng.range(-0.6, 0.6) * drop);
        strands.push((jagged(rng, from, to, 5), width * 0.45, 0.8));
    }
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    for (pts, w, glow) in &strands {
        // Three vertices across each segment; the wide glow fades to nothing at its edges.
        let edge = if *glow < 0.5 { 0.0 } else { *glow };
        for seg in pts.windows(2) {
            let (a, b) = (seg[0], seg[1]);
            let mid = (a + b) * 0.5;
            let view = (cam - mid).normalize_or(Vec3::Z);
            let side = (b - a).cross(view).normalize_or(Vec3::X) * (*w * 0.5);
            let base = positions.len() as u32;
            for (p, c) in [(a - side, edge), (a, *glow), (a + side, edge), (b - side, edge), (b, *glow), (b + side, edge)] {
                positions.push(p.to_array());
                normals.push(view.to_array());
                colors.push([c, c, c, 1.0]);
            }
            for (i, j) in [(0, 1), (1, 2)] {
                let (p0, p1, q0, q1) = (base + i, base + j, base + 3 + i, base + 3 + j);
                indices.extend_from_slice(&[p0, p1, q0, p1, q1, q0]);
            }
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

fn jagged(rng: &mut Rng, a: Vec3, b: Vec3, levels: u32) -> Vec<Vec3> {
    let mut pts = vec![a, b];
    let mut disp = (a - b).length() * 0.16;
    for _ in 0..levels {
        let mut next = Vec::with_capacity(pts.len() * 2);
        for w in pts.windows(2) {
            next.push(w[0]);
            let off = Vec3::new(rng.range(-1.0, 1.0), rng.range(-0.25, 0.25), rng.range(-1.0, 1.0));
            next.push((w[0] + w[1]) * 0.5 + off * disp);
        }
        next.push(pts[pts.len() - 1]);
        pts = next;
        disp *= 0.55;
    }
    pts
}
