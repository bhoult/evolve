//! Miracles and acts of god. When the hand of god (you) touches a creature, it shows: feeding it
//! is a blessing (a shaft of golden light from the sky, a halo, rising motes and a choir), killing
//! it a smite (lightning out of a clear sky, then fire where it fell), removing it takes it up (a
//! white shaft and a rush of light upward). A meteor called down streaks in trailing fire and
//! smoke, and its impact throws up a fireball, a mushroom of smoke, a ring of dust and flying
//! rock, then burns. Drawn as two meshes of camera-facing shafts and sparks: one glowing (added
//! to what's behind), one of smoke, dust and debris (blended over it), with a real light at each.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::view::{ColorGrading, ColorGradingGlobal};
use evolve_sim::acts::TotemKind;
use evolve_sim::snapshot::HighlightKind;

use crate::audio::{PlaySound, Sfx};
use crate::camera::{MainCamera, OrbitCam};
use crate::env::EnvSystems;
use crate::tools::{Tool, ToolState};
use crate::weather_fx::CallLightning;
use crate::{SimLink, Snap};

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
    /// A meteor's impact (`size` is how far its blast reached).
    Impact,
    /// A chunk of land landing (`size` is its radius).
    Landfall,
    /// Manna falling (`size` is how wide).
    Manna,
}

impl Kind {
    fn length(self) -> f32 {
        match self {
            Kind::Blessing => 4.5,
            Kind::Smite => 6.0,
            Kind::Ascension => 3.5,
            Kind::Impact => 10.0,
            Kind::Landfall => 4.0,
            Kind::Manna => 4.0,
        }
    }
}

/// A bit of a whirlwind's funnel: dust (or a scrap of debris) circling its axis and rising.
struct Swirl {
    /// The whirlwind it belongs to (its seed).
    whirl: u32,
    angle: f32,
    height: f32,
    /// Its distance from the axis, relative to the funnel's width at its height.
    out: f32,
    rise: f32,
    age: f32,
    life: f32,
    size: f32,
    color: Vec4,
}

/// How wide a whirlwind's funnel is at a height, m: narrow at the ground, opening upward.
fn funnel(height: f32) -> f32 {
    1.2 + 0.22 * height
}

/// Seconds a meteor takes to come down.
const FALL: f32 = 1.8;

/// Seconds a chunk of land takes to come down.
const LAND_FALL: f32 = 2.4;

/// A meteor, or a chunk of land (with its mesh: earth and a grassy top), on its way down.
struct Flight {
    from: Vec3,
    to: Vec3,
    t0: f32,
    /// A meteor's glow (a chunk of land has none).
    light: Option<Entity>,
    chunk: Option<[Entity; 2]>,
}

impl Flight {
    fn length(&self) -> f32 {
        if self.chunk.is_some() { LAND_FALL } else { FALL }
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
    /// Whether its opening burst has gone off.
    burst: bool,
    /// Its light (a landfall gives none).
    light: Option<Entity>,
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
    flights: Vec<Flight>,
    swirls: Vec<Swirl>,
    /// Funnel bits owed to each whirlwind (by seed), carried between frames.
    swirl_owed: std::collections::HashMap<u32, f32>,
    /// Glowing sparks, and smoke, dust and debris.
    sparks: Vec<Spark>,
    smoke: Vec<Spark>,
    rng: u64,
    mesh: Handle<Mesh>,
    smoke_mesh: Handle<Mesh>,
    /// A chunk of land: earth below, grass on top.
    chunk: [(Handle<Mesh>, Handle<StandardMaterial>); 2],
    /// Bubbles owed to the springs, carried between frames.
    spring_owed: f32,
    /// The tsunami's wall of water (shaped once; moved, turned and scaled to the wave each frame).
    wave: Option<Entity>,
    wave_look: (Handle<Mesh>, Handle<StandardMaterial>),
    /// The totems standing: where, what kind, and their pole, head and light.
    totems: Vec<(Vec3, TotemKind, [Entity; 3])>,
    /// Pole, then a head for each kind (love, war, fear).
    totem_look: (Handle<Mesh>, Handle<Mesh>, Handle<StandardMaterial>, [Handle<StandardMaterial>; 3]),
}

/// A totem's glow: love pink, war red, fear violet.
fn totem_color(kind: TotemKind) -> Vec3 {
    match kind {
        TotemKind::Love => Vec3::new(1.0, 0.35, 0.65),
        TotemKind::War => Vec3::new(1.0, 0.2, 0.08),
        TotemKind::Fear => Vec3::new(0.55, 0.25, 1.0),
    }
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

    /// How many to give off this frame at `rate` a second: on average the rate, whatever the
    /// frame rate.
    fn count(&mut self, rate: f32, dt: f32) -> usize {
        (rate * dt + self.rand()).floor() as usize
    }

    /// Drops everything under way (the world was replaced): the totems and the wave go too, and
    /// come back from the new world's snapshot if it has them.
    fn clear(&mut self, commands: &mut Commands) {
        for f in self.flights.drain(..) {
            for e in f.light.into_iter().chain(f.chunk.into_iter().flatten()) {
                commands.entity(e).despawn();
            }
        }
        for m in self.miracles.drain(..) {
            if let Some(e) = m.light {
                commands.entity(e).despawn();
            }
        }
        for (_, _, ents) in self.totems.drain(..) {
            for e in ents {
                commands.entity(e).despawn();
            }
        }
        if let Some(e) = self.wave.take() {
            commands.entity(e).despawn();
        }
        self.sparks.clear();
        self.smoke.clear();
        self.swirls.clear();
        self.swirl_owed.clear();
        self.spring_owed = 0.0;
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

    fn puff(&mut self, s: Spark) {
        if self.smoke.len() < MAX_SPARKS {
            self.smoke.push(s);
        }
    }

    /// A random direction, at least `min_up` upward (0 level, 1 straight up).
    fn direction(&mut self, min_up: f32) -> Vec3 {
        let a = self.range(0.0, std::f32::consts::TAU);
        let y = self.range(min_up, 1.0);
        let r = (1.0 - y * y).max(0.0).sqrt();
        Vec3::new(a.cos() * r, y, a.sin() * r)
    }
}

#[derive(Component)]
struct MiracleMesh;

#[derive(Component)]
struct SmokeMesh;

/// How long the wall of water is across its front, m (more than the world's diagonal).
const WAVE_SPAN: f32 = 420.0;

/// The tsunami's water as a sheet, in its own frame: x along its crest, z the way it travels
/// (0 at the crest), y its height at full strength. Foam white along the crest, deep green-blue
/// on its face and behind.
fn wave_mesh() -> Mesh {
    use evolve_sim::acts::{TSUNAMI_HEIGHT, tsunami_profile};
    let cols = 140;
    let back = evolve_sim::acts::TSUNAMI_BACK;
    let rows: Vec<f32> = (0..=30).map(|k| -back + (back + 6.0) * (k as f32 / 30.0).powf(0.7)).collect();
    let (mut positions, mut normals, mut colors, mut indices) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &q in &rows {
        let h = TSUNAMI_HEIGHT * tsunami_profile(q);
        let slope = TSUNAMI_HEIGHT * (tsunami_profile(q + 0.5) - tsunami_profile(q - 0.5));
        let n = Vec3::new(0.0, 1.0, -slope).normalize();
        let foam = (1.0 - (q / 3.0).abs()).clamp(0.0, 1.0);
        let body = if q > 0.0 { Vec3::new(0.08, 0.3, 0.36) } else { Vec3::new(0.1, 0.36, 0.42) };
        let c = body.lerp(Vec3::new(0.92, 0.96, 1.0), foam);
        let alpha = if h < 0.05 { 0.0 } else { 0.92 };
        for i in 0..=cols {
            let x = (i as f32 / cols as f32 - 0.5) * WAVE_SPAN;
            positions.push([x, h, q]);
            normals.push(n.to_array());
            colors.push([c.x, c.y, c.z, alpha]);
        }
    }
    let w = cols as u32 + 1;
    for r in 0..rows.len() as u32 - 1 {
        for i in 0..cols as u32 {
            let (a, b) = (r * w + i, (r + 1) * w + i);
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

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
    // Smoke, dust and debris are lit by the sky like everything else, and darken what's behind.
    let smoke_mesh = meshes.add(Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default()));
    let smoke_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(dot_image())),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 1.0,
        reflectance: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(smoke_mesh.clone()),
        MeshMaterial3d(smoke_material),
        Transform::default(),
        Visibility::Hidden,
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        SmokeMesh,
    ));
    let chunk = [
        (
            meshes.add(Sphere::new(1.0).mesh().ico(2).expect("a small icosphere")),
            materials.add(StandardMaterial { base_color: Color::srgb(0.36, 0.27, 0.18), perceptual_roughness: 1.0, ..default() }),
        ),
        (
            meshes.add(Cylinder::new(1.0, 1.0)),
            materials.add(StandardMaterial { base_color: Color::srgb(0.3, 0.5, 0.2), perceptual_roughness: 1.0, ..default() }),
        ),
    ];
    let wood = materials.add(StandardMaterial { base_color: Color::srgb(0.28, 0.2, 0.13), perceptual_roughness: 0.9, ..default() });
    let heads = [TotemKind::Love, TotemKind::War, TotemKind::Fear].map(|k| {
        let c = totem_color(k);
        materials.add(StandardMaterial {
            base_color: Color::srgb(c.x * 0.5, c.y * 0.5, c.z * 0.5),
            emissive: LinearRgba::rgb(c.x * 2.5, c.y * 2.5, c.z * 2.5),
            ..default()
        })
    });
    let totem_look = (
        meshes.add(Cylinder::new(0.22, 1.0)),
        meshes.add(Sphere::new(0.55).mesh().ico(2).expect("a small icosphere")),
        wood,
        heads,
    );
    let wave_look = (
        meshes.add(wave_mesh()),
        materials.add(StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.12,
            reflectance: 0.5,
            double_sided: true,
            cull_mode: None,
            ..default()
        }),
    );
    commands.insert_resource(Miracles {
        wave: None,
        wave_look,
        totems: Vec::new(),
        totem_look,
        chunk,
        spring_owed: 0.0,
        epoch: u64::MAX,
        last_seq: 0,
        miracles: Vec::new(),
        flights: Vec::new(),
        swirls: Vec::new(),
        swirl_owed: Default::default(),
        sparks: Vec::new(),
        smoke: Vec::new(),
        rng: 0x2545_F491_4F6C_DD1D,
        mesh,
        smoke_mesh,
    });
}

/// Starts a miracle for each new act of god in the snapshot.
#[allow(clippy::too_many_arguments)]
fn watch(
    mut commands: Commands,
    snap: Res<Snap>,
    time: Res<Time>,
    mut st: ResMut<Miracles>,
    mut lightning: MessageWriter<CallLightning>,
    mut sounds: MessageWriter<PlaySound>,
    mut tools: ResMut<ToolState>,
    mut cam: ResMut<OrbitCam>,
    eye: Single<&Transform, CameraView>,
    link: Res<SimLink>,
) {
    let snap = &*snap.0;
    let st = &mut *st;
    // A new world (or one loaded): what was under way in the old one is gone.
    let fresh = snap.world_epoch != st.epoch;
    if fresh {
        st.epoch = snap.world_epoch;
        st.last_seq = snap.highlights.last().map(|h| h.seq).unwrap_or(0);
        st.clear(&mut commands);
    }
    // An earthquake: the crack runs across the view through where it was called, jagged.
    if let Some((Tool::Quake, at)) = tools.summon {
        tools.summon = None;
        let fwd = eye.forward().as_vec3();
        let across = Vec3::new(-fwd.z, 0.0, fwd.x).normalize_or(Vec3::X);
        let along = Vec3::new(across.z, 0.0, -across.x);
        let tilt = Quat::from_rotation_y(st.range(-0.4, 0.4));
        let (across, along) = (tilt * across, tilt * along);
        let mut path = Vec::new();
        for k in -6..=6 {
            let p = at + across * (k as f32 * 4.5) + along * if k == 0 { 0.0 } else { st.range(-2.2, 2.2) };
            path.push([p.x, p.z]);
        }
        link.cmd(evolve_sim::Command::Earthquake { path: path.clone() });
        sounds.write(PlaySound { sfx: Sfx::Rumble, at, gain: 1.0, floor: 0.4 });
        let d = eye.translation.distance(at);
        cam.shake = cam.shake.max(2.0 * (1.0 - d / 250.0).clamp(0.0, 1.0));
        // Dust bursting out of the crack, and earth thrown up along it.
        for w in path.windows(2) {
            let (a, b) = (Vec3::new(w[0][0], 0.0, w[0][1]), Vec3::new(w[1][0], 0.0, w[1][1]));
            for _ in 0..14 {
                let p = a.lerp(b, st.rand());
                let ground = snap.terrain.height_at(p.x, p.z);
                let life = st.range(2.0, 4.5);
                let size = st.range(0.8, 1.6);
                let vel = Vec3::new(st.range(-0.8, 0.8), st.range(1.0, 4.0), st.range(-0.8, 0.8));
                st.puff(Spark {
                    pos: Vec3::new(p.x, ground + 0.2, p.z),
                    vel,
                    lift: 0.3,
                    drag: 0.4,
                    age: 0.0,
                    life,
                    size: [size, size * 3.0],
                    color: [Vec4::new(0.5, 0.44, 0.36, 0.55), Vec4::new(0.55, 0.5, 0.43, 0.0)],
                });
            }
            for _ in 0..6 {
                let p = a.lerp(b, st.rand());
                let ground = snap.terrain.height_at(p.x, p.z);
                let dir = st.direction(0.5);
                let speed = st.range(3.0, 7.0);
                let life = st.range(0.8, 1.6);
                let size = st.range(0.1, 0.25);
                st.puff(Spark {
                    pos: Vec3::new(p.x, ground + 0.2, p.z),
                    vel: dir * speed,
                    lift: -9.8,
                    drag: 0.9,
                    age: 0.0,
                    life,
                    size: [size, size],
                    color: [Vec4::new(0.24, 0.19, 0.14, 1.0), Vec4::new(0.24, 0.19, 0.14, 0.8)],
                });
            }
        }
    }
    // A tsunami: the wave rolls in from the sea beyond the spot, the ground roars.
    if let Some((Tool::Tsunami, at)) = tools.summon {
        tools.summon = None;
        let toward = eye.forward().as_vec3();
        link.cmd(evolve_sim::Command::Tsunami { at: at.to_array(), toward: toward.to_array() });
        sounds.write(PlaySound { sfx: Sfx::Rumble, at, gain: 1.0, floor: 0.5 });
    }
    // A chunk of land: it falls from high above, tumbling, straight down onto the spot.
    if let Some((Tool::Landfall, to)) = tools.summon {
        tools.summon = None;
        let from = to + Vec3::new(st.range(-30.0, 30.0), 380.0, st.range(-30.0, 30.0));
        let [(earth_mesh, earth_mat), (grass_mesh, grass_mat)] = st.chunk.clone();
        let r = evolve_sim::acts::LANDFALL_RADIUS;
        let earth = commands
            .spawn((Mesh3d(earth_mesh), MeshMaterial3d(earth_mat), Transform::from_translation(from).with_scale(Vec3::new(r, r * 0.45, r))))
            .id();
        let grass = commands
            .spawn((Mesh3d(grass_mesh), MeshMaterial3d(grass_mat), Transform::from_translation(from).with_scale(Vec3::new(r * 0.92, 0.6, r * 0.92))))
            .id();
        st.flights.push(Flight { from, to, t0: time.elapsed_secs(), light: None, chunk: Some([earth, grass]) });
        sounds.write(PlaySound { sfx: Sfx::Whoosh, at: to, gain: 0.8, floor: 0.3 });
    }
    // A meteor called down: it comes in from beyond where it will land, as seen from the camera,
    // so it streaks down across the sky in view.
    if let Some((Tool::Meteor, to)) = tools.summon.take() {
        let away = Vec3::new(to.x - eye.translation.x, 0.0, to.z - eye.translation.z).normalize_or(Vec3::X);
        let dir = Quat::from_rotation_y(st.range(-0.6, 0.6)) * away;
        let from = to + (dir * 1.3 + Vec3::Y).normalize() * 420.0;
        let light = commands
            .spawn((
                PointLight { color: Color::srgb(1.0, 0.6, 0.3), intensity: 0.0, range: 80.0, shadows_enabled: false, ..default() },
                Transform::from_translation(from),
            ))
            .id();
        st.flights.push(Flight { from, to, t0: time.elapsed_secs(), light: Some(light), chunk: None });
        sounds.write(PlaySound { sfx: Sfx::Whoosh, at: to, gain: 1.0, floor: 0.35 });
    }
    if fresh {
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
            HighlightKind::Meteor => Kind::Impact,
            HighlightKind::Landfall => Kind::Landfall,
            HighlightKind::Manna => Kind::Manna,
            _ => continue,
        };
        let at = Vec3::from_array(h.pos);
        if kind == Kind::Smite {
            lightning.write(CallLightning { at });
        }
        if matches!(kind, Kind::Impact | Kind::Landfall) {
            // The ground jolts the eye, less the farther it is.
            let d = eye.translation.distance(at);
            cam.shake = cam.shake.max(2.5 * (1.0 - d / 300.0).clamp(0.0, 1.0));
        }
        let light = (kind != Kind::Landfall).then(|| {
            commands
                .spawn((
                    PointLight { intensity: 0.0, range: if kind == Kind::Impact { 120.0 } else { 30.0 }, shadows_enabled: false, ..default() },
                    Transform::from_translation(at + Vec3::Y),
                ))
                .id()
        });
        st.miracles.push(Miracle { kind, subject: h.subject, at, size: h.strength.max(0.3), t0: now, owed: [0.0; 3], burst: false, light });
    }
}

/// `--miracle`: works one on the selected creature a set number of frames before the screenshot
/// (a meteor lands 8 m from it).
fn scripted(
    args: Res<crate::Args>,
    frames: Res<bevy::diagnostic::FrameCount>,
    snap: Res<Snap>,
    link: Res<crate::SimLink>,
    mut tools: ResMut<ToolState>,
) {
    let Some((kind, lead)) = &args.miracle else {
        return;
    };
    if frames.0 + lead != args.frames {
        return;
    }
    let Some(id) = snap.0.selected.as_ref().map(|c| c.id) else {
        return;
    };
    let near = |dx: f32| {
        snap.0.creatures.iter().find(|c| c.id == id).and_then(|c| c.parts.first()).map(|p| {
            let at = Vec3::from_array(p.pos) + Vec3::new(dx, 0.0, 0.0);
            Vec3::new(at.x, snap.0.terrain.height_at(at.x, at.z), at.z)
        })
    };
    match kind.as_str() {
        "quake" => tools.summon = near(5.0).map(|at| (Tool::Quake, at)),
        "land" => tools.summon = near(14.0).map(|at| (Tool::Landfall, at)),
        "spring" => {
            if let Some(at) = near(4.0) {
                link.cmd(evolve_sim::Command::Spring { at: at.to_array() });
            }
        }
        "tsunami" => tools.summon = near(0.0).map(|at| (Tool::Tsunami, at)),
        "manna" => {
            if let Some(at) = near(5.0) {
                link.cmd(evolve_sim::Command::Manna { at: at.to_array() });
            }
        }
        "love" | "war" | "fear" => {
            let totem = match kind.as_str() {
                "love" => TotemKind::Love,
                "war" => TotemKind::War,
                _ => TotemKind::Fear,
            };
            if let Some(at) = near(4.0) {
                link.cmd(evolve_sim::Command::Totem { at: at.to_array(), kind: totem });
            }
        }
        _ => {}
    }
    if matches!(kind.as_str(), "quake" | "land" | "spring" | "manna" | "love" | "war" | "fear" | "tsunami") {
        return;
    }
    if kind == "whirl" {
        if let Some(c) = snap.0.creatures.iter().find(|c| c.id == id)
            && let Some(p) = c.parts.first()
        {
            link.cmd(evolve_sim::Command::Whirlwind { at: (Vec3::from_array(p.pos) + Vec3::new(3.0, 0.0, 0.0)).to_array() });
        }
        return;
    }
    if kind == "meteor" {
        if let Some(c) = snap.0.creatures.iter().find(|c| c.id == id)
            && let Some(p) = c.parts.first()
        {
            let at = Vec3::from_array(p.pos) + Vec3::new(8.0, 0.0, 0.0);
            let ground = snap.0.terrain.height_at(at.x, at.z);
            tools.summon = Some((Tool::Meteor, Vec3::new(at.x, ground, at.z)));
        }
        return;
    }
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
#[allow(clippy::too_many_arguments)]
fn animate(
    mut commands: Commands,
    snap: Res<Snap>,
    time: Res<Time>,
    link: Res<SimLink>,
    mut st: ResMut<Miracles>,
    mut lights: Query<(&mut PointLight, &mut Transform)>,
    mut chunks: Query<&mut Transform, Without<PointLight>>,
    grading: Single<&ColorGrading, With<MainCamera>>,
    mut cam: ResMut<OrbitCam>,
) {
    let snap = &*snap.0;
    let st = &mut *st;
    let now = time.elapsed_secs();
    let dt = time.delta_secs().min(0.1);
    // Meteors coming down, trailing fire and smoke; on arrival the sim does the rest.
    let mut flights = std::mem::take(&mut st.flights);
    flights.retain(|f| {
        let u = ((now - f.t0) / f.length()).clamp(0.0, 1.0);
        if u >= 1.0 {
            match f.chunk {
                Some(chunk) => {
                    link.cmd(evolve_sim::Command::Landfall { at: f.to.to_array() });
                    for e in chunk {
                        commands.entity(e).despawn();
                    }
                }
                None => link.cmd(evolve_sim::Command::Meteor { at: f.to.to_array() }),
            }
            if let Some(e) = f.light {
                commands.entity(e).despawn();
            }
            return false;
        }
        let pos = f.from.lerp(f.to, u.powf(1.3));
        let back = (f.from - f.to).normalize();
        if let Some(chunk) = f.chunk {
            // A tumbling clod of earth trailing dust.
            let r = evolve_sim::acts::LANDFALL_RADIUS;
            let spin = Quat::from_rotation_y(u * 2.0) * Quat::from_rotation_x((u * 7.0).sin() * 0.15);
            if let Ok(mut tf) = chunks.get_mut(chunk[0]) {
                tf.translation = pos;
                tf.rotation = spin;
            }
            if let Ok(mut tf) = chunks.get_mut(chunk[1]) {
                tf.translation = pos + spin * Vec3::Y * (r * 0.4);
                tf.rotation = spin;
            }
            for _ in 0..st.count(90.0, dt) {
                let off =Vec3::new(st.range(-1.0, 1.0), st.range(-0.3, 0.3), st.range(-1.0, 1.0)) * r;
                let life = st.range(1.5, 3.0);
                let size = st.range(1.0, 2.0);
                let behind = st.range(2.0, 8.0);
                st.puff(Spark {
                    pos: pos + off + back * behind,
                    vel: Vec3::ZERO,
                    lift: 0.2,
                    drag: 0.5,
                    age: 0.0,
                    life,
                    size: [size, size * 3.0],
                    color: [Vec4::new(0.45, 0.38, 0.3, 0.6), Vec4::new(0.5, 0.45, 0.38, 0.0)],
                });
            }
            return true;
        }
        for _ in 0..st.count(400.0, dt) {
            let jitter = Vec3::new(st.range(-1.0, 1.0), st.range(-1.0, 1.0), st.range(-1.0, 1.0));
            let life = st.range(0.25, 0.6);
            let size = st.range(0.6, 1.4);
            let behind = st.range(0.0, 6.0);
            st.spark(Spark {
                pos: pos + back * behind,
                vel: jitter * 2.0,
                lift: 0.0,
                drag: 0.3,
                age: 0.0,
                life,
                size: [size, size * 0.3],
                color: [Vec4::new(8.0, 5.0, 2.0, 1.0), Vec4::new(3.0, 0.6, 0.08, 0.0)],
            });
        }
        for _ in 0..st.count(120.0, dt) {
            let jitter = Vec3::new(st.range(-1.0, 1.0), st.range(-0.5, 1.0), st.range(-1.0, 1.0));
            let life = st.range(2.0, 4.0);
            let size = st.range(0.8, 1.6);
            let behind = st.range(2.0, 10.0);
            st.puff(Spark {
                pos: pos + back * behind,
                vel: jitter,
                lift: 0.3,
                drag: 0.5,
                age: 0.0,
                life,
                size: [size, size * 3.0],
                color: [Vec4::new(0.25, 0.23, 0.22, 0.7), Vec4::new(0.5, 0.48, 0.46, 0.0)],
            });
        }
        if let Some(Ok((mut light, mut tf))) = f.light.map(|e| lights.get_mut(e)) {
            light.intensity = 4.0e6 * u;
            tf.translation = pos;
        }
        true
    });
    st.flights = flights;
    let mut miracles = std::mem::take(&mut st.miracles);
    miracles.retain_mut(|m| {
        let age = now - m.t0;
        if age > m.kind.length() {
            if let Some(e) = m.light {
                commands.entity(e).despawn();
            }
            return false;
        }
        // The burst that opens it, once.
        let first = !std::mem::replace(&mut m.burst, true);
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
                if first {
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
                if first {
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
            Kind::Manna => {
                let e = envelope(age, 0.3, 1.5, m.kind.length());
                // Golden flakes drifting down over the spot.
                for _ in 0..emit(m, 0, 140.0 * (1.0 - age / 2.5).max(0.0)) {
                    let high = st.range(15.0, 30.0);
                    let pos = st.around(at + Vec3::Y * high, s);
                    let fall = st.range(3.0, 6.0);
                    let drift = Vec3::new(st.range(-0.6, 0.6), -fall, st.range(-0.6, 0.6));
                    let life = st.range(3.0, 5.0);
                    st.spark(Spark {
                        pos,
                        vel: drift,
                        lift: 0.0,
                        drag: 1.0,
                        age: 0.0,
                        life,
                        size: [0.12, 0.08],
                        color: [Vec4::new(4.0, 3.4, 1.8, 1.0), Vec4::new(3.0, 2.4, 1.0, 0.0)],
                    });
                }
                (Color::srgb(1.0, 0.85, 0.5), 500_000.0 * e)
            }
            Kind::Landfall => {
                if first {
                    // A ring of dust racing out from under it, and earth flung up.
                    for _ in 0..220 {
                        let a = st.range(0.0, std::f32::consts::TAU);
                        let out = Vec3::new(a.cos(), 0.0, a.sin());
                        let speed = st.range(0.8, 1.6) * s;
                        let life = st.range(2.0, 3.5);
                        let size = st.range(1.2, 2.0);
                        let rise = st.range(0.0, 2.0);
                        st.puff(Spark {
                            pos: at + out * s * 0.9 - Vec3::Y * 2.0,
                            vel: out * speed + Vec3::Y * rise,
                            lift: 0.2,
                            drag: 0.15,
                            age: 0.0,
                            life,
                            size: [size, size * 3.5],
                            color: [Vec4::new(0.5, 0.44, 0.35, 0.6), Vec4::new(0.55, 0.5, 0.43, 0.0)],
                        });
                    }
                    for _ in 0..120 {
                        let dir = st.direction(0.4);
                        let speed = st.range(5.0, 14.0);
                        let life = st.range(1.0, 2.2);
                        let size = st.range(0.15, 0.35);
                        let pos = st.around(at, s * 0.8);
                        st.puff(Spark {
                            pos,
                            vel: dir * speed,
                            lift: -9.8,
                            drag: 0.9,
                            age: 0.0,
                            life,
                            size: [size, size],
                            color: [Vec4::new(0.26, 0.2, 0.14, 1.0), Vec4::new(0.26, 0.2, 0.14, 0.8)],
                        });
                    }
                }
                (Color::WHITE, 0.0)
            }
            Kind::Impact => {
                let (r, crater) = (s, evolve_sim::acts::CRATER);
                if first {
                    // The fireball.
                    for _ in 0..260 {
                        let dir = st.direction(-0.1);
                        let speed = st.range(4.0, 18.0);
                        let life = st.range(0.6, 1.5);
                        let size = st.range(1.5, 2.5);
                        st.spark(Spark {
                            pos: at,
                            vel: dir * speed,
                            lift: 2.5,
                            drag: 0.08,
                            age: 0.0,
                            life,
                            size: [size, size * 1.6],
                            color: [Vec4::new(10.0, 6.0, 2.0, 1.0), Vec4::new(4.0, 0.8, 0.1, 0.0)],
                        });
                    }
                    // A mushroom of smoke.
                    for _ in 0..110 {
                        let pos = st.around(at, crater * 0.7);
                        let out = Vec3::new(pos.x - at.x, 0.0, pos.z - at.z);
                        let vel = Vec3::Y * st.range(4.0, 11.0) + out * st.range(0.3, 1.0);
                        let life = st.range(4.0, 8.0);
                        let size = st.range(1.5, 2.5);
                        st.puff(Spark {
                            pos,
                            vel,
                            lift: 0.6,
                            drag: 0.35,
                            age: 0.0,
                            life,
                            size: [size, size * 3.0],
                            color: [Vec4::new(0.16, 0.14, 0.13, 0.85), Vec4::new(0.45, 0.43, 0.41, 0.0)],
                        });
                    }
                    // Rock and earth flung out.
                    for _ in 0..150 {
                        let dir = st.direction(0.35);
                        let speed = st.range(7.0, 20.0);
                        let life = st.range(1.2, 2.6);
                        let size = st.range(0.12, 0.35);
                        st.puff(Spark {
                            pos: at + Vec3::Y * 0.5,
                            vel: dir * speed,
                            lift: -9.8,
                            drag: 0.9,
                            age: 0.0,
                            life,
                            size: [size, size],
                            color: [Vec4::new(0.24, 0.19, 0.14, 1.0), Vec4::new(0.24, 0.19, 0.14, 0.8)],
                        });
                    }
                    // A ring of dust racing out along the ground.
                    for _ in 0..150 {
                        let a = st.range(0.0, std::f32::consts::TAU);
                        let out = Vec3::new(a.cos(), 0.0, a.sin());
                        let speed = st.range(0.6, 1.3) * r;
                        let life = st.range(1.8, 3.2);
                        let size = st.range(1.0, 1.6);
                        let rise = st.range(0.0, 1.5);
                        st.puff(Spark {
                            pos: at + out * crater + Vec3::Y * 0.4,
                            vel: out * speed + Vec3::Y * rise,
                            lift: 0.2,
                            drag: 0.15,
                            age: 0.0,
                            life,
                            size: [size, size * 3.5],
                            color: [Vec4::new(0.55, 0.48, 0.38, 0.6), Vec4::new(0.6, 0.55, 0.48, 0.0)],
                        });
                    }
                }
                // Then the crater burns.
                let burn = ((m.kind.length() - 1.5 - age) / (m.kind.length() - 2.0)).clamp(0.0, 1.0);
                for _ in 0..emit(m, 0, 220.0 * burn) {
                    let pos = st.around(at + Vec3::Y * 0.2, crater * 0.8);
                    let vel = Vec3::new(st.range(-0.3, 0.3), st.range(1.5, 3.5), st.range(-0.3, 0.3));
                    let life = st.range(0.5, 1.1);
                    let size = st.range(0.4, 0.8);
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
                for _ in 0..emit(m, 1, 25.0 * burn) {
                    let pos = st.around(at + Vec3::Y * 0.5, crater * 0.6);
                    let life = st.range(3.0, 6.0);
                    let size = st.range(0.8, 1.4);
                    let rise = st.range(1.0, 2.5);
                    st.puff(Spark {
                        pos,
                        vel: Vec3::Y * rise,
                        lift: 0.3,
                        drag: 0.6,
                        age: 0.0,
                        life,
                        size: [size, size * 4.0],
                        color: [Vec4::new(0.12, 0.11, 0.1, 0.6), Vec4::new(0.4, 0.39, 0.38, 0.0)],
                    });
                }
                let flash = 1.2e7 * (-age / 0.08).exp();
                let flicker = 0.75 + 0.25 * (age * 23.0).sin() * (age * 7.3 + 1.0).sin();
                (Color::srgb(1.0, 0.55, 0.2), flash + 1.5e6 * burn.sqrt() * flicker)
            }
        };
        if let Some(Ok((mut light, mut tf))) = m.light.map(|e| lights.get_mut(e)) {
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
    for p in &mut st.smoke {
        p.age += dt;
        p.vel.y += p.lift * dt;
        p.vel *= p.drag.powf(dt);
        p.pos += p.vel * dt;
        // Debris and dust stop at the ground.
        let ground = snap.terrain.height_at(p.pos.x, p.pos.z);
        if p.pos.y < ground && snap.terrain.res >= 3 {
            p.pos.y = ground;
            p.vel = Vec3::ZERO;
        }
    }
    st.smoke.retain(|p| p.age < p.life);

    // Totems: put up and taken down with the world's, each glowing, with what it stirs drifting
    // round its head and its reach marked faintly on the ground. (Matched by where they stand,
    // so one stays put while the ground under it is raised or lowered.)
    let flat = |a: Vec3, b: Vec3| (a.x - b.x).hypot(a.z - b.z) < 0.5;
    st.totems.retain_mut(|(pos, kind, ents)| {
        let Some(&(p, _)) = snap.totems.iter().find(|&&(p, k)| k == *kind && flat(Vec3::from_array(p), *pos)) else {
            for &e in ents.iter() {
                commands.entity(e).despawn();
            }
            return false;
        };
        let rise = p[1] - pos.y;
        if rise.abs() > 0.01 {
            pos.y = p[1];
            for &e in &ents[..2] {
                if let Ok(mut tf) = chunks.get_mut(e) {
                    tf.translation.y += rise;
                }
            }
            if let Ok((_, mut tf)) = lights.get_mut(ents[2]) {
                tf.translation.y += rise;
            }
        }
        true
    });
    for &(p, kind) in &snap.totems {
        let pos = Vec3::from_array(p);
        if st.totems.iter().any(|&(q, k, _)| k == kind && flat(q, pos)) {
            continue;
        }
        let (pole, head, wood, heads) = st.totem_look.clone();
        let c = totem_color(kind);
        let ents = [
            commands.spawn((Mesh3d(pole), MeshMaterial3d(wood), Transform::from_translation(pos + Vec3::Y * 2.0).with_scale(Vec3::new(1.0, 4.0, 1.0)))).id(),
            commands.spawn((Mesh3d(head), MeshMaterial3d(heads[kind as usize].clone()), Transform::from_translation(pos + Vec3::Y * 4.4))).id(),
            commands
                .spawn((
                    PointLight { color: Color::srgb(c.x, c.y, c.z), intensity: 250_000.0, range: 25.0, shadows_enabled: false, ..default() },
                    Transform::from_translation(pos + Vec3::Y * 4.6),
                ))
                .id(),
        ];
        st.totems.push((pos, kind, ents));
    }
    let totems: Vec<(Vec3, TotemKind)> = st.totems.iter().map(|&(p, k, _)| (p, k)).collect();
    for (pos, kind) in totems {
        let c = totem_color(kind);
        let head = pos + Vec3::Y * 4.4;
        for _ in 0..st.count(20.0, dt) {
            let a = st.range(0.0, std::f32::consts::TAU);
            let ring = Vec3::new(a.cos(), 0.0, a.sin()) * evolve_sim::acts::TOTEM_REACH;
            let ground = snap.terrain.height_at(pos.x + ring.x, pos.z + ring.z);
            st.spark(Spark {
                pos: Vec3::new(pos.x + ring.x, ground + 0.3, pos.z + ring.z),
                vel: Vec3::Y * 0.3,
                lift: 0.0,
                drag: 1.0,
                age: 0.0,
                life: 1.5,
                size: [0.3, 0.1],
                color: [(c * 2.0).extend(0.5), (c * 2.0).extend(0.0)],
            });
        }
        for _ in 0..st.count(14.0, dt) {
            let around = st.around(head, 1.2);
            let life = st.range(1.5, 2.5);
            match kind {
                TotemKind::Fear => {
                    let vel = Vec3::new(st.range(-0.6, 0.6), st.range(0.2, 0.8), st.range(-0.6, 0.6));
                    st.puff(Spark {
                        pos: around,
                        vel,
                        lift: 0.1,
                        drag: 0.6,
                        age: 0.0,
                        life,
                        size: [0.4, 1.2],
                        color: [Vec4::new(0.2, 0.08, 0.3, 0.6), Vec4::new(0.12, 0.05, 0.2, 0.0)],
                    });
                }
                _ => {
                    let vel = Vec3::new(st.range(-0.3, 0.3), st.range(0.5, 1.5), st.range(-0.3, 0.3));
                    st.spark(Spark {
                        pos: around,
                        vel,
                        lift: 0.2,
                        drag: 0.7,
                        age: 0.0,
                        life,
                        size: [0.12, 0.05],
                        color: [(c * 4.0).extend(1.0), (c * 2.0).extend(0.0)],
                    });
                }
            }
        }
    }
    // The tsunami's wall of water, and foam and spray off its crest wherever it's above ground.
    match snap.tsunami {
        Some((dir, front, height)) => {
            let way = Vec3::new(dir[0], 0.0, dir[1]);
            let origin = way * front + Vec3::Y * snap.params.water_level;
            let rotation = Quat::from_rotation_arc(Vec3::Z, way);
            let scale = Vec3::new(1.0, (height / evolve_sim::acts::TSUNAMI_HEIGHT).max(0.01), 1.0);
            let place = Transform { translation: origin, rotation, scale };
            match st.wave {
                Some(e) => {
                    if let Ok(mut tf) = chunks.get_mut(e) {
                        *tf = place;
                    }
                }
                None => {
                    let (mesh, material) = st.wave_look.clone();
                    st.wave = Some(commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), place, NotShadowCaster)).id());
                }
            }
            let along = Vec3::new(-way.z, 0.0, way.x);
            let half = snap.terrain.half();
            for _ in 0..st.count(400.0, dt) {
                let x = st.range(-half * 1.4, half * 1.4);
                let p = origin + along * x;
                if p.x.abs() > half || p.z.abs() > half {
                    continue;
                }
                let crest = p.y + height;
                if crest < snap.terrain.height_at(p.x, p.z) + 0.3 {
                    continue;
                }
                let spray = st.rand() < 0.4;
                let life = st.range(0.8, 1.8);
                let size = if spray { st.range(0.15, 0.3) } else { st.range(1.2, 2.8) };
                let vel = way * st.range(6.0, 11.0) + Vec3::Y * st.range(1.0, if spray { 7.0 } else { 3.0 });
                st.puff(Spark {
                    pos: Vec3::new(p.x, crest, p.z),
                    vel,
                    lift: if spray { -9.8 } else { 0.0 },
                    drag: 0.5,
                    age: 0.0,
                    life,
                    size: [size, size * if spray { 0.6 } else { 1.8 }],
                    color: [Vec4::new(0.95, 0.97, 1.0, 0.85), Vec4::new(0.9, 0.95, 1.0, 0.0)],
                });
            }
        }
        None => {
            if let Some(e) = st.wave.take() {
                commands.entity(e).despawn();
            }
        }
    }
    // While the ground shakes, so does the view.
    if snap.quaking {
        cam.shake = cam.shake.max(0.35);
    }
    // Springs bubble where they rise (past a dozen, they share the bubbles, each at random).
    let t = &snap.terrain;
    let n = t.springs.len();
    st.spring_owed += 45.0 * n.min(12) as f32 * dt;
    let bubbles = st.spring_owed.floor();
    st.spring_owed -= bubbles;
    for _ in 0..bubbles as usize {
        let Some(&k) = t.springs.get((st.rand() * n as f32) as usize) else {
            break;
        };
        let (i, j) = (k as usize / t.res, k as usize % t.res);
        let at = |v: usize| (v as f32 / (t.res - 1) as f32 - 0.5) * t.size;
        let (x, z) = (at(j), at(i));
        let surface = t.height_at(x, z) + t.water_at(x, z);
        let froth = st.rand() < 0.3;
        let pos = st.around(Vec3::new(x, surface + 0.05, z), if froth { 1.0 } else { 0.6 });
        let vel = if froth { Vec3::ZERO } else { Vec3::new(st.range(-0.5, 0.5), st.range(1.5, 3.0), st.range(-0.5, 0.5)) };
        let life = st.range(0.6, 1.2);
        let (size, alpha) = if froth { ([0.4, 0.7], 0.55) } else { ([0.18, 0.1], 0.95) };
        st.puff(Spark {
            pos,
            vel,
            lift: if froth { 0.0 } else { -9.8 },
            drag: 0.9,
            age: 0.0,
            life,
            size,
            color: [Vec4::new(0.9, 0.95, 1.0, alpha), Vec4::new(0.85, 0.92, 1.0, 0.0)],
        });
    }
    // Whirlwinds: their funnels, debris and the dust at their feet.
    for &([x, z], s, seed) in &snap.whirlwinds {
        if s <= 0.0 {
            continue;
        }
        let ground = snap.terrain.height_at(x, z).max(snap.params.water_level);
        let owed = st.swirl_owed.entry(seed).or_insert(0.0);
        *owed += 320.0 * s * dt;
        let n = owed.floor();
        *owed -= n;
        for _ in 0..n as usize {
            let debris = st.rand() < 0.2;
            let height = if debris { st.range(0.0, 8.0) } else { st.range(0.0, 1.0).powi(2) * 32.0 };
            let (size, color) = if debris {
                let leaf = st.rand() < 0.3;
                (st.range(0.1, 0.25), if leaf { Vec4::new(0.25, 0.45, 0.18, 1.0) } else { Vec4::new(0.24, 0.19, 0.14, 1.0) })
            } else {
                (st.range(1.0, 2.6), Vec4::new(0.5, 0.45, 0.38, 0.32))
            };
            let swirl = Swirl {
                whirl: seed,
                angle: st.range(0.0, std::f32::consts::TAU),
                height,
                out: if debris { st.range(0.6, 1.6) } else { st.range(0.75, 1.1) },
                rise: st.range(2.0, 5.0),
                age: 0.0,
                life: st.range(2.0, 4.0),
                size,
                color,
            };
            if st.swirls.len() < MAX_SPARKS {
                st.swirls.push(swirl);
            }
        }
        // Dust kicked up round its foot.
        for _ in 0..st.count(40.0 * s, dt) {
            let a = st.range(0.0, std::f32::consts::TAU);
            let r = st.range(2.0, 6.0);
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let life = st.range(1.5, 3.0);
            let size = st.range(1.2, 2.2);
            st.puff(Spark {
                pos: Vec3::new(x, ground + 0.3, z) + out * r,
                vel: out * 2.0 + Vec3::new(-out.z, 0.0, out.x) * 6.0,
                lift: 0.5,
                drag: 0.4,
                age: 0.0,
                life,
                size: [size, size * 2.5],
                color: [Vec4::new(0.52, 0.46, 0.37, 0.4), Vec4::new(0.55, 0.5, 0.42, 0.0)],
            });
        }
    }
    let live: std::collections::HashSet<u32> = snap.whirlwinds.iter().map(|w| w.2).collect();
    st.swirl_owed.retain(|k, _| live.contains(k));
    for p in &mut st.swirls {
        // Spinning fastest low down, where the funnel is narrow; a whirlwind gone, it fades fast.
        let r = funnel(p.height) * p.out;
        p.angle += (22.0 / r.max(1.0)).min(9.0) * dt;
        p.height += p.rise * dt;
        p.age += dt * if live.contains(&p.whirl) { 1.0 } else { 4.0 };
    }
    st.swirls.retain(|p| p.age < p.life && p.height < 40.0);
}

type CameraView = (With<MainCamera>, Without<MiracleMesh>, Without<SmokeMesh>);
type GlowView = (With<MiracleMesh>, Without<SmokeMesh>);
type SmokeView = (With<SmokeMesh>, Without<MiracleMesh>);

/// Builds the mesh: each shaft as a camera-facing ribbon from high in the sky down to its
/// creature, soft at the edges and fading with height, a halo at the creature, and every spark
/// as a small glowing dot.
fn draw(
    mut st: ResMut<Miracles>,
    mut meshes: ResMut<Assets<Mesh>>,
    time: Res<Time>,
    snap: Res<Snap>,
    cam: Single<(&Transform, &ColorGrading), CameraView>,
    ent: Single<(&mut Transform, &mut Visibility), GlowView>,
    smoke_ent: Single<(&mut Transform, &mut Visibility), SmokeView>,
) {
    let (cam, grading) = cam.into_inner();
    let st = &mut *st;
    let (mut tf, mut vis) = ent.into_inner();
    let snap_whirls = &snap.0.whirlwinds;
    let ground_at = |x: f32, z: f32| snap.0.terrain.height_at(x, z).max(snap.0.params.water_level);
    let cam_pos = cam.translation;
    let right = cam.right().as_vec3();
    let up = cam.up().as_vec3();
    // Smoke, dust and debris, lit as if facing up and toward the camera.
    {
        let (mut stf, mut svis) = smoke_ent.into_inner();
        let want = if st.smoke.is_empty() && st.swirls.is_empty() { Visibility::Hidden } else { Visibility::Inherited };
        if *svis != want {
            *svis = want;
        }
        if want == Visibility::Inherited {
            stf.translation = cam_pos;
            let mut buf = Buf { origin: cam_pos, right, up, ..default() };
            for p in &st.smoke {
                let u = (p.age / p.life).clamp(0.0, 1.0);
                let size = p.size[0] + (p.size[1] - p.size[0]) * u;
                buf.quad(p.pos, size, p.color[0].lerp(p.color[1], u));
            }
            for p in &st.swirls {
                let Some(&([x, z], s, _)) = snap_whirls.iter().find(|w| w.2 == p.whirl) else {
                    continue;
                };
                let ground = ground_at(x, z);
                let r = funnel(p.height) * p.out;
                let pos = Vec3::new(x + p.angle.cos() * r, ground + p.height, z + p.angle.sin() * r);
                let u = (p.age / p.life).clamp(0.0, 1.0);
                let fade = (u * 6.0).min(1.0) * (1.0 - u) * s.max(0.2);
                buf.quad(pos, p.size, p.color * Vec4::new(1.0, 1.0, 1.0, fade));
            }
            let normal = (-cam.forward().as_vec3() + Vec3::Y * 0.8).normalize().to_array();
            if let Some(m) = meshes.get_mut(&st.smoke_mesh) {
                buf.write(m, normal);
            }
        }
    }
    let empty = st.miracles.is_empty() && st.sparks.is_empty() && st.flights.iter().all(|f| f.chunk.is_some());
    let want = if empty { Visibility::Hidden } else { Visibility::Inherited };
    if *vis != want {
        *vis = want;
    }
    if empty {
        return;
    }
    let now = time.elapsed_secs();
    // Built around the camera so it sorts after the water.
    tf.translation = cam_pos;
    let mut buf = Buf { origin: cam_pos, right, up, ..default() };
    buf.grading = Some(grading.global.clone());
    // Each meteor's blazing head.
    for f in st.flights.iter().filter(|f| f.chunk.is_none()) {
        let u = ((now - f.t0) / FALL).clamp(0.0, 1.0);
        let pos = f.from.lerp(f.to, u.powf(1.3));
        buf.quad(pos, 2.5, Vec4::new(8.0, 7.0, 5.0, 1.0));
        buf.quad(pos, 7.0, Vec4::new(3.0, 1.6, 0.6, 0.5));
    }
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
            Kind::Landfall => (0.0, 0.0, Vec3::ZERO, 0.0),
            Kind::Manna => (envelope(age, 0.3, 1.5, m.kind.length()) * 0.4, 0.0, Vec3::new(2.4, 1.9, 0.9), 120.0),
            // The flash of the impact, a moment wide.
            Kind::Impact => {
                buf.quad(m.at + Vec3::Y * 2.0, 10.0, Vec4::new(9.0, 7.5, 5.0, 0.7 * (1.0 - age / 0.25).max(0.0)));
                (0.0, 0.0, Vec3::ZERO, 0.0)
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
        // (Manna's size is the spread of its food; its shaft is narrower.)
        let w0 = if m.kind == Kind::Manna { 2.5 } else { m.size * 1.4 + 0.6 };
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
        buf.write(m, [0.0, 1.0, 0.0]);
    }
}

/// Vertices for the miracle mesh, relative to `origin` (the camera).
#[derive(Default)]
struct Buf {
    origin: Vec3,
    right: Vec3,
    up: Vec3,
    /// The camera's color grading, undone for glowing colors (smoke takes the grading).
    grading: Option<ColorGradingGlobal>,
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl Buf {
    fn vertex(&mut self, at: Vec3, color: Vec4, uv: [f32; 2]) {
        let c = match &self.grading {
            Some(g) => ungrade(color.truncate(), g),
            None => color.truncate(),
        };
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

    fn write(self, m: &mut Mesh, normal: [f32; 3]) {
        let n = self.positions.len();
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![normal; n]);
        m.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        m.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        m.insert_indices(Indices::U32(self.indices));
    }
}
