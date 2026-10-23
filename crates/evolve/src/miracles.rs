//! Miracles: when the hand of god (you) touches a creature, it shows. Feeding it is a blessing
//! (a shaft of golden light from the sky, a halo, rising motes and a choir). Killing it is a smite
//! (lightning out of a clear sky, then fire where it fell). Removing it takes it up (a white shaft
//! and a rush of light upward). Drawn as one mesh of glowing, camera-facing shafts and sparks,
//! with a real light at each.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::view::{ColorGrading, ColorGradingGlobal};
use evolve_sim::snapshot::HighlightKind;

use crate::Snap;
use crate::camera::MainCamera;
use crate::env::EnvSystems;
use crate::weather_fx::CallLightning;

pub struct MiraclesPlugin;

impl Plugin for MiraclesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, (watch, animate, draw).chain().after(EnvSystems))
            .add_systems(Update, scripted);
    }
}

/// Most sparks in the air at once.
const MAX_SPARKS: usize = 4000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Blessing,
    Smite,
    Ascension,
}

impl Kind {
    fn length(self) -> f32 {
        match self {
            Kind::Blessing => 4.5,
            Kind::Smite => 6.0,
            Kind::Ascension => 3.5,
        }
    }
}

struct Miracle {
    kind: Kind,
    subject: u64,
    /// Where it is (it follows the creature while the creature lives).
    at: Vec3,
    /// The creature's size (its bounding radius), m.
    size: f32,
    /// Wall time it began.
    t0: f32,
    /// Sparks owed to the emitters (fractions carried over between frames).
    owed: [f32; 3],
    light: Entity,
}

struct Spark {
    pos: Vec3,
    vel: Vec3,
    /// Upward acceleration (negative falls).
    lift: f32,
    /// Fraction of velocity kept per second.
    drag: f32,
    age: f32,
    life: f32,
    size: [f32; 2],
    /// Linear color with alpha, at birth and at death.
    color: [Vec4; 2],
}

#[derive(Resource)]
struct Miracles {
    epoch: u64,
    last_seq: u64,
    miracles: Vec<Miracle>,
    sparks: Vec<Spark>,
    rng: u64,
    mesh: Handle<Mesh>,
}

impl Miracles {
    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand()
    }

    /// A random point in a disc of radius `r` around `at`.
    fn around(&mut self, at: Vec3, r: f32) -> Vec3 {
        let a = self.range(0.0, std::f32::consts::TAU);
        let d = r * self.rand().sqrt();
        at + Vec3::new(a.cos() * d, 0.0, a.sin() * d)
    }

    fn spark(&mut self, s: Spark) {
        if self.sparks.len() < MAX_SPARKS {
            self.sparks.push(s);
        }
    }
}

#[derive(Component)]
struct MiracleMesh;

/// A soft round dot.
fn dot_image() -> Image {
    const N: usize = 32;
    let mut data = Vec::with_capacity(N * N * 4);
    for y in 0..N {
        for x in 0..N {
            let dx = (x as f32 + 0.5) / N as f32 * 2.0 - 1.0;
            let dy = (y as f32 + 0.5) / N as f32 * 2.0 - 1.0;
            let r = (dx * dx + dy * dy).sqrt();
            let a = (1.0 - r).clamp(0.0, 1.0);
            data.extend_from_slice(&[255, 255, 255, (a * a * 255.0) as u8]);
        }
    }
    Image::new(
        Extent3d { width: N as u32, height: N as u32, depth_or_array_layers: 1 },
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
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(dot_image())),
        unlit: true,
        fog_enabled: false,
        alpha_mode: AlphaMode::Add,
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
        MiracleMesh,
    ));
    commands.insert_resource(Miracles {
        epoch: u64::MAX,
        last_seq: 0,
        miracles: Vec::new(),
        sparks: Vec::new(),
        rng: 0x2545_F491_4F6C_DD1D,
        mesh,
    });
}

/// Starts a miracle for each new act of god in the snapshot.
fn watch(
    mut commands: Commands,
    snap: Res<Snap>,
    time: Res<Time>,
    mut st: ResMut<Miracles>,
    mut lightning: MessageWriter<CallLightning>,
) {
    let snap = &*snap.0;
    let st = &mut *st;
    if snap.world_epoch != st.epoch {
        st.epoch = snap.world_epoch;
        st.last_seq = snap.highlights.last().map(|h| h.seq).unwrap_or(0);
        return;
    }
    let now = time.elapsed_secs();
    for h in snap.highlights.iter() {
        if h.seq <= st.last_seq {
            continue;
        }
        st.last_seq = h.seq;
        let kind = match h.kind {
            HighlightKind::Blessing => Kind::Blessing,
            HighlightKind::Smite => Kind::Smite,
            HighlightKind::Ascension => Kind::Ascension,
            _ => continue,
        };
        let at = Vec3::from_array(h.pos);
        if kind == Kind::Smite {
            lightning.write(CallLightning { at });
        }
        let light = commands
            .spawn((
                PointLight { intensity: 0.0, range: 30.0, shadows_enabled: false, ..default() },
                Transform::from_translation(at + Vec3::Y),
            ))
            .id();
        st.miracles.push(Miracle { kind, subject: h.subject, at, size: h.strength.max(0.3), t0: now, owed: [0.0; 3], light });
    }
}

/// `--miracle`: works one on the selected creature a set number of frames before the screenshot.
fn scripted(args: Res<crate::Args>, frames: Res<bevy::diagnostic::FrameCount>, snap: Res<Snap>, link: Res<crate::SimLink>) {
    let Some((kind, lead)) = &args.miracle else {
        return;
    };
    if frames.0 + lead != args.frames {
        return;
    }
    let Some(id) = snap.0.selected.as_ref().map(|c| c.id) else {
        return;
    };
    link.cmd(match kind.as_str() {
        "smite" => evolve_sim::Command::KillCreature(id),
        "take" => evolve_sim::Command::RemoveCreature(id),
        _ => evolve_sim::Command::FeedCreature(id),
    });
}

/// Night cools and greys the whole picture (see env.rs); a miracle keeps its own colors, so they
/// are warmed and saturated by about as much (roughly: the grading is applied after tonemapping).
fn ungrade(c: Vec3, g: &ColorGradingGlobal) -> Vec3 {
    let c = c * Vec3::new(1.0 - 2.5 * g.temperature, 1.0, 1.0 + 2.5 * g.temperature);
    let luma = c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    (Vec3::splat(luma) + (c - luma) / g.post_saturation.clamp(0.5, 1.0)).max(Vec3::ZERO)
}

/// Rises over `rise` seconds, holds, and fades out over the last `fade` seconds of `length`.
fn envelope(age: f32, rise: f32, fade: f32, length: f32) -> f32 {
    let up = (age / rise.max(1e-3)).clamp(0.0, 1.0);
    let down = ((length - age) / fade.max(1e-3)).clamp(0.0, 1.0);
    up.min(down)
}

/// Moves the miracles with their creatures, gives off sparks, ages the sparks and sets the lights.
fn animate(
    mut commands: Commands,
    snap: Res<Snap>,
    time: Res<Time>,
    mut st: ResMut<Miracles>,
    mut lights: Query<(&mut PointLight, &mut Transform)>,
    grading: Single<&ColorGrading, With<MainCamera>>,
) {
    let snap = &*snap.0;
    let st = &mut *st;
    let now = time.elapsed_secs();
    let dt = time.delta_secs().min(0.1);
    let mut miracles = std::mem::take(&mut st.miracles);
    miracles.retain_mut(|m| {
        let age = now - m.t0;
        if age > m.kind.length() {
            commands.entity(m.light).despawn();
            return false;
        }
        // A blessing stays with its creature.
        if let Some(c) = snap.creatures.iter().find(|c| c.id == m.subject)
            && let Some(p) = c.parts.first()
        {
            m.at = Vec3::from_array(p.pos);
        }
        let (s, at) = (m.size, m.at);
        let emit = |m: &mut Miracle, k: usize, rate: f32| -> usize {
            m.owed[k] += rate * dt;
            let n = m.owed[k].floor();
            m.owed[k] -= n;
            n as usize
        };
        let (light_color, light_power) = match m.kind {
            Kind::Blessing => {
                let e = envelope(age, 0.4, 2.0, m.kind.length());
                // Golden motes rising around it, and a few falling down the shaft from above.
                for _ in 0..emit(m, 0, 70.0 * e) {
                    let pos = st.around(at - Vec3::Y * s * 0.5, s * 1.3);
                    let vel = Vec3::new(st.range(-0.2, 0.2), st.range(0.6, 1.6), st.range(-0.2, 0.2));
                    let life = st.range(1.4, 2.6);
                    let size = st.range(0.05, 0.11);
                    st.spark(Spark {
                        pos,
                        vel,
                        lift: 0.3,
                        drag: 0.7,
                        age: 0.0,
                        life,
                        size: [size, size * 0.4],
                        color: [Vec4::new(4.0, 3.2, 1.4, 1.0), Vec4::new(2.5, 1.6, 0.5, 0.0)],
                    });
                }
                for _ in 0..emit(m, 1, 18.0 * e) {
                    let high = st.range(12.0, 30.0);
                    let pos = st.around(at + Vec3::Y * high, s * 0.8 + 0.4);
                    let fall = st.range(6.0, 10.0);
                    st.spark(Spark {
                        pos,
                        vel: Vec3::new(0.0, -fall, 0.0),
                        lift: 0.0,
                        drag: 1.0,
                        age: 0.0,
                        life: 3.0,
                        size: [0.09, 0.05],
                        color: [Vec4::new(4.0, 3.6, 2.2, 0.9), Vec4::new(3.0, 2.4, 1.0, 0.0)],
                    });
                }
                (Color::srgb(1.0, 0.82, 0.45), 600_000.0 * e)
            }
            Kind::Smite => {
                // Sparks thrown out by the strike, then fire and embers where it fell.
                if age < dt * 1.5 {
                    for _ in 0..90 {
                        let dir = Vec3::new(st.range(-1.0, 1.0), st.range(0.2, 1.0), st.range(-1.0, 1.0)).normalize_or(Vec3::Y);
                        let speed = st.range(3.0, 11.0);
                        let life = st.range(0.5, 1.3);
                        st.spark(Spark {
                            pos: at,
                            vel: dir * speed,
                            lift: -9.8,
                            drag: 0.4,
                            age: 0.0,
                            life,
                            size: [0.07, 0.03],
                            color: [Vec4::new(6.0, 5.0, 3.0, 1.0), Vec4::new(3.0, 0.8, 0.1, 0.0)],
                        });
                    }
                }
                let burn = (1.0 - age / (m.kind.length() - 0.5)).clamp(0.0, 1.0);
                for _ in 0..emit(m, 0, 140.0 * burn * (0.6 + s)) {
                    let pos = st.around(at - Vec3::Y * s * 0.6, s * 0.9 + 0.2);
                    let vel = Vec3::new(st.range(-0.3, 0.3), st.range(1.0, 2.6), st.range(-0.3, 0.3));
                    let life = st.range(0.45, 0.95);
                    let size = st.range(0.22, 0.4) * (0.6 + s);
                    st.spark(Spark {
                        pos,
                        vel,
                        lift: 1.5,
                        drag: 0.8,
                        age: 0.0,
                        life,
                        size: [size, size * 0.25],
                        color: [Vec4::new(5.0, 2.0, 0.35, 0.9), Vec4::new(2.0, 0.25, 0.03, 0.0)],
                    });
                }
                for _ in 0..emit(m, 1, 30.0 * burn) {
                    let pos = st.around(at, s * 0.8 + 0.2);
                    let vel = Vec3::new(st.range(-0.6, 0.6), st.range(1.0, 3.0), st.range(-0.6, 0.6));
                    let life = st.range(1.5, 3.0);
                    st.spark(Spark {
                        pos,
                        vel,
                        lift: 0.4,
                        drag: 0.6,
                        age: 0.0,
                        life,
                        size: [0.045, 0.02],
                        color: [Vec4::new(5.0, 1.8, 0.3, 1.0), Vec4::new(2.0, 0.3, 0.05, 0.0)],
                    });
                }
                let flicker = 0.75 + 0.25 * (age * 23.0).sin() * (age * 7.3 + 1.0).sin();
                (Color::srgb(1.0, 0.5, 0.18), 500_000.0 * burn.sqrt() * flicker)
            }
            Kind::Ascension => {
                let e = envelope(age, 0.15, 2.2, m.kind.length());
                if age < dt * 1.5 {
                    for _ in 0..160 {
                        let rise = st.range(-s, s);
                        let pos = st.around(at + Vec3::Y * rise, s * 1.1);
                        let life = st.range(1.2, 2.4);
                        let vel = Vec3::new(st.range(-0.4, 0.4), st.range(2.0, 7.0), st.range(-0.4, 0.4));
                        let lift = st.range(4.0, 9.0);
                        st.spark(Spark {
                            pos,
                            vel,
                            lift,
                            drag: 0.9,
                            age: 0.0,
                            life,
                            size: [0.1, 0.03],
                            color: [Vec4::new(3.5, 3.8, 4.5, 1.0), Vec4::new(1.5, 1.8, 3.0, 0.0)],
                        });
                    }
                }
                (Color::srgb(0.85, 0.9, 1.0), 450_000.0 * e)
            }
        };
        if let Ok((mut light, mut tf)) = lights.get_mut(m.light) {
            let c = ungrade(Vec3::from_array(light_color.to_linear().to_f32_array_no_alpha()), &grading.global);
            light.color = Color::linear_rgb(c.x, c.y, c.z);
            light.intensity = light_power;
            tf.translation = at + Vec3::Y * (1.0 + s);
        }
        true
    });
    st.miracles = miracles;
    for p in &mut st.sparks {
        p.age += dt;
        p.vel.y += p.lift * dt;
        p.vel *= p.drag.powf(dt);
        p.pos += p.vel * dt;
    }
    st.sparks.retain(|p| p.age < p.life);
}

type CameraView = (With<MainCamera>, Without<MiracleMesh>);

/// Builds the mesh: each shaft as a camera-facing ribbon from high in the sky down to its
/// creature, soft at the edges and fading with height, a halo at the creature, and every spark
/// as a small glowing dot.
fn draw(
    mut st: ResMut<Miracles>,
    mut meshes: ResMut<Assets<Mesh>>,
    time: Res<Time>,
    cam: Single<(&Transform, &ColorGrading), CameraView>,
    ent: Single<(&mut Transform, &mut Visibility), With<MiracleMesh>>,
) {
    let (cam, grading) = cam.into_inner();
    let st = &mut *st;
    let (mut tf, mut vis) = ent.into_inner();
    let empty = st.miracles.is_empty() && st.sparks.is_empty();
    let want = if empty { Visibility::Hidden } else { Visibility::Inherited };
    if *vis != want {
        *vis = want;
    }
    if empty {
        return;
    }
    let now = time.elapsed_secs();
    let cam_pos = cam.translation;
    let right = cam.right().as_vec3();
    let up = cam.up().as_vec3();
    // Built around the camera so it sorts after the water.
    tf.translation = cam_pos;
    let mut buf = Buf { origin: cam_pos, right, up, ..default() };
    buf.grading = grading.global.clone();
    for m in &st.miracles {
        let age = now - m.t0;
        let (shaft, halo, color, height) = match m.kind {
            Kind::Blessing => {
                let e = envelope(age, 0.4, 2.0, m.kind.length());
                let pulse = 1.0 + 0.15 * (age * 5.0).sin();
                (e, e * pulse, Vec3::new(2.4, 1.6, 0.45), 160.0)
            }
            Kind::Ascension => {
                let e = envelope(age, 0.15, 2.2, m.kind.length());
                (e, (1.0 - age / 1.0).max(0.0) * 2.0, Vec3::new(2.2, 2.4, 2.9), 200.0)
            }
            // The bolt is the lightning's own; here, the glow of the fire.
            Kind::Smite => {
                let burn = (1.0 - age / (m.kind.length() - 0.5)).clamp(0.0, 1.0);
                (0.0, burn * 0.6, Vec3::new(2.5, 0.9, 0.2), 0.0)
            }
        };
        if halo > 0.0 {
            buf.quad(m.at, m.size * 2.2 + 0.8, (color * 0.6).extend(halo.min(1.5) * 0.55));
        }
        if shaft <= 0.0 {
            continue;
        }
        // The shaft: columns of (edge, middle, edge) vertices going up, widening and fading.
        let base = m.at - Vec3::Y * m.size;
        let w0 = m.size * 1.4 + 0.6;
        const ROWS: usize = 10;
        let start = buf.positions.len() as u32;
        for i in 0..=ROWS {
            let f = i as f32 / ROWS as f32;
            let h = height * f * f;
            let p = base + Vec3::Y * h;
            let to_cam = (cam_pos - p).normalize_or(Vec3::Z);
            let side = Vec3::Y.cross(to_cam).normalize_or(Vec3::X) * w0 * (1.0 + 1.5 * f);
            // Bright low down, gone high up; a little soft at the very bottom.
            let a = shaft * (1.0 - f).powf(1.5) * (f * 12.0).clamp(0.35, 1.0) * 0.7;
            for (q, edge) in [(p - side, 0.0), (p, 1.0), (p + side, 0.0)] {
                buf.vertex(q, color.extend(a * edge), [0.5, 0.5]);
            }
        }
        for i in 0..ROWS as u32 {
            let r = start + i * 3;
            for (a, b) in [(0, 1), (1, 2)] {
                buf.indices.extend_from_slice(&[r + a, r + b, r + 3 + a, r + b, r + 3 + b, r + 3 + a]);
            }
        }
    }
    for p in &st.sparks {
        let u = (p.age / p.life).clamp(0.0, 1.0);
        let size = p.size[0] + (p.size[1] - p.size[0]) * u;
        let color = p.color[0].lerp(p.color[1], u);
        buf.quad(p.pos, size, color);
    }
    if let Some(m) = meshes.get_mut(&st.mesh) {
        let n = buf.positions.len();
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, buf.positions);
        m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; n]);
        m.insert_attribute(Mesh::ATTRIBUTE_COLOR, buf.colors);
        m.insert_attribute(Mesh::ATTRIBUTE_UV_0, buf.uvs);
        m.insert_indices(Indices::U32(buf.indices));
    }
}

/// Vertices for the miracle mesh, relative to `origin` (the camera).
#[derive(Default)]
struct Buf {
    origin: Vec3,
    right: Vec3,
    up: Vec3,
    /// The camera's color grading, undone for the miracle's colors.
    grading: ColorGradingGlobal,
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl Buf {
    fn vertex(&mut self, at: Vec3, color: Vec4, uv: [f32; 2]) {
        let c = ungrade(color.truncate(), &self.grading);
        self.positions.push((at - self.origin).to_array());
        self.colors.push(c.extend(color.w).to_array());
        self.uvs.push(uv);
    }

    /// A glowing dot facing the camera.
    fn quad(&mut self, c: Vec3, half: f32, color: Vec4) {
        let b = self.positions.len() as u32;
        for (dx, dy, uv) in [(-1.0, -1.0, [0.0, 1.0]), (1.0, -1.0, [1.0, 1.0]), (-1.0, 1.0, [0.0, 0.0]), (1.0, 1.0, [1.0, 0.0])] {
            self.vertex(c + (self.right * dx + self.up * dy) * half, color, uv);
        }
        self.indices.extend_from_slice(&[b, b + 1, b + 2, b + 1, b + 3, b + 2]);
    }
}
