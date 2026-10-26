//! Creatures in the water, the sea or a lake, pond or stream: rings spreading on the surface where
//! they move through it (drawn by the ocean and terrain shaders from a short list of ripple
//! sources) and spray thrown up where they hit it or burst out of it. Worked out from the
//! snapshots: each body part against the water surface (moving with the waves at sea), and its
//! speed from one snapshot to the next.

use std::collections::{HashMap, HashSet};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::renderer::RenderQueue;
use evolve_sim::body::PartShape;

use crate::Snap;
use crate::camera::MainCamera;
use crate::env::{Env, EnvSystems};
use crate::water::RippleBuffer;

pub struct SplashPlugin;

/// Ripple sources the ocean shader reads. Must match water.wgsl.
pub const MAX_RIPPLES: usize = 64;
/// Seconds a ripple spreads before it's gone. Must match water.wgsl.
const RIPPLE_LIFE: f32 = 2.6;
/// Bytes of the ripple buffer: a count, the box they all fit in, then each ripple as two vec4s.
pub const RIPPLE_BYTES: u64 = 32 + MAX_RIPPLES as u64 * 32;
/// Most droplets in the air at once.
const MAX_DROPS: usize = 800;
/// New ripples a frame, the strongest and nearest first.
const RIPPLES_PER_FRAME: usize = 16;

#[derive(Clone, Copy)]
struct Ripple {
    at: Vec2,
    /// Env clock when it started.
    t0: f32,
    strength: f32,
    /// Size of what made it, m: bigger things make wider, longer rings.
    size: f32,
    /// White water where the surface was broken hard.
    foam: f32,
}

struct Drop {
    pos: Vec3,
    vel: Vec3,
    size: f32,
    /// The sea surface where it was thrown up; it's gone once it falls back below.
    floor: f32,
    age: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Wet {
    Land,
    Above,
    /// Straddling the surface.
    Surface,
    Under,
}

struct PartTrack {
    pos: Vec3,
    wet: Wet,
    /// Env clock when this part may start its next ripple.
    next_ripple: f32,
}

#[derive(Resource)]
struct Splashes {
    epoch: u64,
    /// Sim time of the snapshot last looked at.
    sim_time: f64,
    parts: HashMap<u64, Vec<PartTrack>>,
    ripples: Vec<Ripple>,
    drops: Vec<Drop>,
    rng: u64,
    /// The droplet mesh and its buffers, reused every frame.
    mesh: Handle<Mesh>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl Splashes {
    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand()
    }

    /// Adds a ripple, pushing out whichever has the least left to show if the list is full.
    fn add_ripple(&mut self, r: Ripple, now: f32) {
        if self.ripples.len() < MAX_RIPPLES {
            self.ripples.push(r);
            return;
        }
        let left = |x: &Ripple| x.strength * (1.0 - (now - x.t0) / RIPPLE_LIFE).max(0.0);
        if let Some((i, weakest)) = self
            .ripples
            .iter()
            .enumerate()
            .map(|(i, x)| (i, left(x)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            && r.strength > weakest
        {
            self.ripples[i] = r;
        }
    }

    /// Throws `n` droplets up from a ring of `radius` around `at`, on a surface at `floor`.
    #[allow(clippy::too_many_arguments)]
    fn spray(&mut self, at: Vec3, floor: f32, radius: f32, n: usize, up: f32, out: f32, carry: Vec3, size: f32) {
        for _ in 0..n {
            if self.drops.len() >= MAX_DROPS {
                return;
            }
            let a = self.range(0.0, std::f32::consts::TAU);
            let dir = Vec3::new(a.cos(), 0.0, a.sin());
            let r = radius * self.range(0.3, 1.0);
            let vel = dir * out * self.range(0.3, 1.0) + Vec3::Y * up * self.range(0.35, 1.0) + carry * 0.3;
            let size = size * self.range(0.6, 1.3);
            self.drops.push(Drop {
                pos: Vec3::new(at.x, floor + 0.02, at.z) + dir * r,
                vel,
                size,
                floor,
                age: 0.0,
            });
        }
    }
}

impl Plugin for SplashPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, (track, draw_drops).chain().after(EnvSystems));
    }
}

#[derive(Component)]
struct DropMesh;

/// A soft round dot, so droplets aren't squares.
fn dot_image() -> Image {
    const N: usize = 32;
    let mut data = Vec::with_capacity(N * N * 4);
    for y in 0..N {
        for x in 0..N {
            let dx = (x as f32 + 0.5) / N as f32 * 2.0 - 1.0;
            let dy = (y as f32 + 0.5) / N as f32 * 2.0 - 1.0;
            let r = (dx * dx + dy * dy).sqrt();
            let a = ((1.0 - r) / 0.35).clamp(0.0, 1.0);
            data.extend_from_slice(&[255, 255, 255, (a * a * (3.0 - 2.0 * a) * 255.0) as u8]);
        }
    }
    Image::new(
        Extent3d {
            width: N as u32,
            height: N as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let mesh = meshes.add(Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default()));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.82, 0.9, 0.95, 0.6),
        base_color_texture: Some(images.add(dot_image())),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.2,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh.clone()),
        MeshMaterial3d(material),
        Transform::default(),
        Visibility::Hidden,
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        DropMesh,
    ));
    commands.insert_resource(Splashes {
        epoch: u64::MAX,
        sim_time: f64::NAN,
        parts: HashMap::new(),
        ripples: Vec::new(),
        drops: Vec::new(),
        rng: 0x9E37_79B9_7F4A_7C15,
        mesh,
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
    });
}

/// How far from its center a ripple shows, `age` seconds in. Must match water.wgsl.
fn ripple_reach(size: f32, age: f32) -> f32 {
    let lambda = 0.22 + 0.2 * size;
    let front = size * 0.5 + (0.45 + 0.35 * size) * age;
    front + lambda * 2.0 + 0.05
}

/// How far a part reaches below and above its center, and across.
fn extents(shape: &PartShape, rot: Quat) -> (f32, f32) {
    match *shape {
        PartShape::Box { half } => {
            let h = Vec3::from_array(half);
            let (x, y, z) = (rot * Vec3::X, rot * Vec3::Y, rot * Vec3::Z);
            let vertical = x.y.abs() * h.x + y.y.abs() * h.y + z.y.abs() * h.z;
            (vertical, h.max_element())
        }
        PartShape::Capsule { half_height, radius } => {
            let axis = rot * Vec3::Y;
            (radius + half_height * axis.y.abs(), radius + half_height)
        }
    }
}

/// Follows every body part against the sea surface and makes ripples and spray.
fn track(
    snap: Res<Snap>,
    env: Res<Env>,
    time: Res<Time>,
    queue: Res<RenderQueue>,
    ripple_buf: Res<RippleBuffer>,
    cam: Single<&Transform, With<MainCamera>>,
    mut st: ResMut<Splashes>,
) {
    let st = &mut *st;
    let now = env.time;
    let dt = time.delta_secs().min(0.1);
    let snap = &*snap.0;
    if snap.world_epoch != st.epoch {
        st.epoch = snap.world_epoch;
        st.parts.clear();
        st.ripples.clear();
        st.drops.clear();
        st.sim_time = f64::NAN;
    }

    // The clock restarts with each new world and wraps after hours: drop what that orphaned.
    st.ripples.retain(|r| {
        let age = now - r.t0;
        (-0.5..RIPPLE_LIFE).contains(&age)
    });
    for d in &mut st.drops {
        d.vel.y -= 9.81 * dt;
        d.vel *= 1.0 - 0.4 * dt;
        d.pos += d.vel * dt;
        d.age += dt;
    }
    st.drops.retain(|d| d.age < 3.0 && !(d.pos.y < d.floor && d.vel.y < 0.0));

    // A paused world republishes the same moment; only a new one moves anything.
    let t = &snap.terrain;
    if snap.time != st.sim_time && t.res >= 3 {
        let dt_sim = (snap.time - st.sim_time) as f32;
        st.sim_time = snap.time;
        // Speeds only between snapshots close together in sim time; a big jump (max speed, a
        // loaded world) would make anything look still, which is the safe way to be wrong.
        let speeds = dt_sim > 1e-4 && dt_sim < 0.5;
        let wl = snap.params.water_level;
        let cam_pos = cam.translation;
        let mut wanted: Vec<(f32, Ripple)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::with_capacity(snap.creatures.len());
        let mut parts = std::mem::take(&mut st.parts);
        for c in &snap.creatures {
            seen.insert(c.id);
            let tracks = parts.entry(c.id).or_default();
            let known = tracks.len() == c.parts.len();
            if !known {
                tracks.clear();
            }
            for (k, p) in c.parts.iter().enumerate() {
                let pos = Vec3::from_array(p.pos);
                let rot = Quat::from_array(p.rot);
                let (vertical, size) = extents(&p.shape, rot);
                // The sea, or water standing on the land above it (only the sea has waves).
                let ground = t.height_at(pos.x, pos.z);
                let still = t.water_surface(pos.x, pos.z, wl);
                let depth = still - ground;
                let surface = if still <= wl && depth > 0.05 { wl + env.wave_height(Vec2::new(pos.x, pos.z), depth) } else { still };
                let wet = if depth <= 0.05 {
                    Wet::Land
                } else if pos.y - vertical > surface {
                    Wet::Above
                } else if pos.y + vertical < surface {
                    Wet::Under
                } else {
                    Wet::Surface
                };
                if !known {
                    tracks.push(PartTrack { pos, wet, next_ripple: now });
                    continue;
                }
                let prev = &mut tracks[k];
                let (was, from) = (prev.wet, prev.pos);
                prev.pos = pos;
                prev.wet = wet;
                if !speeds || wet == Wet::Land {
                    continue;
                }
                let v = ((pos - from) / dt_sim).clamp_length_max(25.0);
                let speed = v.length();
                let at = Vec3::new(pos.x, surface, pos.z);
                let dist = (at - cam_pos).length();
                // Hitting the water.
                if was == Wet::Above && wet != Wet::Above && v.y < -1.0 {
                    let hit = -v.y;
                    let n = ((hit - 0.5) * size * 25.0).clamp(3.0, 60.0) as usize;
                    let drop_size = 0.025 + 0.03 * size.min(1.5);
                    st.spray(at, surface, size, n, hit * 0.55 + 0.6, 0.6 + hit * 0.2, v.with_y(0.0), drop_size);
                    let r = Ripple {
                        at: at.xz(),
                        t0: now,
                        strength: (hit * 0.3 * (0.5 + size)).min(2.5),
                        size,
                        foam: ((hit - 1.0) * 0.3 * size.sqrt()).clamp(0.0, 1.0),
                    };
                    wanted.push((f32::MAX, r));
                    prev.next_ripple = now + 0.15;
                    continue;
                }
                // Bursting out of it: water running off.
                if was != Wet::Above && was != Wet::Land && wet == Wet::Above && v.y > 1.2 {
                    let n = (v.y * size * 10.0).clamp(2.0, 20.0) as usize;
                    st.spray(at, surface, size * 0.7, n, v.y * 0.5, 0.4, v.with_y(0.0), 0.025);
                }
                // Thrashing fast at the surface throws a little spray, more the faster.
                if wet == Wet::Surface && speed > 3.5 {
                    let expected = (speed - 3.0) * size * 20.0 * dt_sim;
                    let n = expected.floor() as usize + usize::from(st.rand() < expected.fract());
                    st.spray(at, surface, size * 0.6, n.min(6), 0.6 + speed * 0.25, 0.3 + speed * 0.1, v, 0.02);
                }
                // Rings wherever it moves through the surface (or just under it), a new one every
                // quarter meter or so; something floating still makes none.
                let near_surface = wet == Wet::Surface || (wet == Wet::Under && pos.y + vertical > surface - 0.4);
                if near_surface && speed > 0.12 && now >= prev.next_ripple {
                    prev.next_ripple = now + (0.25 / speed).clamp(0.1, 0.6);
                    let strength = (speed * (0.25 + 0.6 * size)).min(1.5);
                    let r = Ripple {
                        at: at.xz(),
                        t0: now,
                        strength,
                        size,
                        foam: ((speed - 3.0) * 0.08).clamp(0.0, 0.3),
                    };
                    wanted.push((strength / (1.0 + dist / 25.0), r));
                }
            }
        }
        parts.retain(|id, _| seen.contains(id));
        st.parts = parts;
        wanted.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, r) in wanted.into_iter().take(RIPPLES_PER_FRAME) {
            st.add_ripple(r, now);
        }
    }

    // Upload the ripple list, with how far each reaches right now and the box they all fit in,
    // so the shader can pass over the rest of the sea quickly.
    let n = st.ripples.len().min(MAX_RIPPLES);
    let mut list = Vec::with_capacity(n * 8);
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for r in &st.ripples[..n] {
        let age = (now - r.t0).max(0.0);
        let reach = ripple_reach(r.size, age);
        lo = lo.min(r.at - reach);
        hi = hi.max(r.at + reach);
        list.extend_from_slice(&[r.at.x, r.at.y, r.t0, r.strength, r.size, r.foam, reach * reach, 0.0]);
    }
    let mut bytes = Vec::with_capacity(RIPPLE_BYTES as usize);
    for x in [n as f32, 0.0, 0.0, 0.0, lo.x, lo.y, hi.x, hi.y].into_iter().chain(list) {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    queue.write_buffer(&ripple_buf.0, 0, &bytes);
}

/// Draws the droplets as small dots facing the camera, stretched along their motion.
fn draw_drops(
    mut st: ResMut<Splashes>,
    mut meshes: ResMut<Assets<Mesh>>,
    cam: Single<&Transform, (With<MainCamera>, Without<DropMesh>)>,
    drop_ent: Single<(&mut Transform, &mut Visibility), With<DropMesh>>,
) {
    let st = &mut *st;
    let (mut tf, mut vis) = drop_ent.into_inner();
    let want = if st.drops.is_empty() { Visibility::Hidden } else { Visibility::Inherited };
    if *vis != want {
        *vis = want;
    }
    if st.drops.is_empty() {
        return;
    }
    let cam_pos = cam.translation;
    let fwd = cam.forward().as_vec3();
    let cam_up = cam.up().as_vec3();
    // Lit as if facing up and toward the camera, so the sky lights them by day and they go dark
    // at night.
    let normal = (-fwd + Vec3::Y * 0.8).normalize().to_array();
    // Built around the camera so they sort after the water.
    tf.translation = cam_pos;
    st.positions.clear();
    st.normals.clear();
    st.uvs.clear();
    st.indices.clear();
    for d in &st.drops {
        let flat = d.vel - fwd * d.vel.dot(fwd);
        let len = flat.length();
        let along = if len > 0.5 { flat / len } else { cam_up };
        let across = fwd.cross(along).normalize_or(Vec3::X) * d.size;
        // A little motion blur.
        let along = along * (d.size + len * 0.012);
        let c = d.pos - cam_pos;
        let b = st.positions.len() as u32;
        for (p, uv) in [
            (c - across - along, [0.0, 1.0]),
            (c + across - along, [1.0, 1.0]),
            (c - across + along, [0.0, 0.0]),
            (c + across + along, [1.0, 0.0]),
        ] {
            st.positions.push(p.to_array());
            st.normals.push(normal);
            st.uvs.push(uv);
        }
        st.indices.extend_from_slice(&[b, b + 1, b + 2, b + 1, b + 3, b + 2]);
    }
    if let Some(m) = meshes.get_mut(&st.mesh) {
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, st.positions.clone());
        m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, st.normals.clone());
        m.insert_attribute(Mesh::ATTRIBUTE_UV_0, st.uvs.clone());
        m.insert_indices(Indices::U32(st.indices.clone()));
    }
}
