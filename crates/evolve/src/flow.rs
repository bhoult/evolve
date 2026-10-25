//! Flecks of foam riding the streams, so running water looks like it runs: a few thousand small
//! streaks lying on the water, carried down the slope of its surface (faster where it's deeper and
//! falls more steeply) and pulled toward the middle of the channel. They're born where water is
//! running near what the camera looks at, and fade out where it pools, runs dry or they've had
//! their few seconds. Drawn as one mesh rebuilt every frame.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use evolve_sim::terrain::Terrain;

use crate::Snap;
use crate::camera::OrbitCam;
use crate::env::GraphicsSettings;

pub struct FlowPlugin;

/// Most flecks at once.
const MAX_FLECKS: usize = 3000;
/// Water shallower than this carries no flecks, m.
const MIN_DEPTH: f32 = 0.02;
/// Water whose surface falls less steeply than this (m per m) is still: a pond or a lake.
const MIN_FALL: f32 = 0.004;
/// Places tried per frame to find running water for new flecks.
const TRIES: usize = 2500;
/// Manning's roughness of a stream bed: speed = depth^(2/3) * sqrt(fall) / this, m/s.
const ROUGHNESS: f32 = 0.035;
/// Fastest a fleck goes, m/s.
const TOP_SPEED: f32 = 3.0;

struct Fleck {
    pos: Vec2,
    vel: Vec2,
    age: f32,
    life: f32,
    /// Its width, m (flecks vary).
    width: f32,
}

#[derive(Resource)]
struct Flow {
    flecks: Vec<Fleck>,
    rng: u64,
    mesh: Handle<Mesh>,
    entity: Entity,
}

impl Flow {
    /// xorshift: plenty for scattering flecks.
    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl Plugin for FlowPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_flow)
            .add_systems(Update, update_flow.after(crate::env::EnvSystems));
    }
}

fn spawn_flow(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mesh = meshes.add(flecks_mesh(Vec::new(), Vec::new(), Vec::new()));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.93, 0.96, 0.96),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.5,
        reflectance: 0.2,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let entity = commands
        .spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material),
            Transform::default(),
            Visibility::Hidden,
            NoFrustumCulling,
            NotShadowCaster,
            NotShadowReceiver,
        ))
        .id();
    commands.insert_resource(Flow {
        flecks: Vec::with_capacity(MAX_FLECKS),
        rng: 0x9E37_79B9_7F4A_7C15,
        mesh,
        entity,
    });
}

/// The water at a spot: how deep it is, which way and how steeply its surface falls (m per m,
/// pointing downhill), and which way it gets deeper (toward the middle of a channel).
fn water_here(t: &Terrain, p: Vec2) -> (f32, Vec2, Vec2) {
    let e = t.cell() * 0.75;
    let surface = |x: f32, z: f32| t.height_at(x, z) + t.water_at(x, z);
    let fall = Vec2::new(surface(p.x - e, p.y) - surface(p.x + e, p.y), surface(p.x, p.y - e) - surface(p.x, p.y + e)) / (2.0 * e);
    let deeper = Vec2::new(t.water_at(p.x + e, p.y) - t.water_at(p.x - e, p.y), t.water_at(p.x, p.y + e) - t.water_at(p.x, p.y - e)) / (2.0 * e);
    (t.water_at(p.x, p.y), fall, deeper)
}

/// How fast running water `depth` m deep goes down a surface falling `fall` m per m.
fn stream_speed(depth: f32, fall: f32) -> f32 {
    (depth.max(0.0).powf(2.0 / 3.0) * fall.max(0.0).sqrt() / ROUGHNESS).min(TOP_SPEED)
}

fn update_flow(
    time: Res<Time>,
    snap: Res<Snap>,
    settings: Res<GraphicsSettings>,
    cam: Res<OrbitCam>,
    mut flow: ResMut<Flow>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut visibility: Query<&mut Visibility>,
) {
    let t = &snap.0.terrain;
    let on = settings.flow && t.res >= 3 && t.water.len() == t.heights.len();
    if !on {
        flow.flecks.clear();
    }
    if let Ok(mut v) = visibility.get_mut(flow.entity) {
        let want = if on && !flow.flecks.is_empty() { Visibility::Inherited } else { Visibility::Hidden };
        v.set_if_neq(want);
    }
    if !on {
        return;
    }
    let dt = time.delta_secs().min(0.1);
    let half = t.half();
    let sea = snap.0.params.water_level;

    // Carry the flecks along; let them go where the water stops running.
    let k = 1.0 - (-dt / 0.3).exp();
    flow.flecks.retain_mut(|f| {
        let (depth, fall, deeper) = water_here(t, f.pos);
        let steep = fall.length();
        let target = fall.normalize_or_zero() * stream_speed(depth, steep) + deeper.clamp_length_max(1.0) * 0.8;
        f.vel += (target - f.vel) * k;
        f.pos += f.vel * dt;
        // Still water lets them go sooner.
        f.age += dt * if steep < MIN_FALL { 3.0 } else { 1.0 };
        f.age < f.life && depth >= MIN_DEPTH && f.pos.x.abs() < half && f.pos.y.abs() < half
    });

    // New ones where water runs, near where the camera looks.
    let radius = (cam.distance * 1.2).clamp(25.0, 250.0);
    let center = Vec2::new(cam.focus.x, cam.focus.z);
    for _ in 0..TRIES {
        if flow.flecks.len() >= MAX_FLECKS {
            break;
        }
        let (a, r) = (flow.rand() * std::f32::consts::TAU, radius * flow.rand().sqrt());
        let pos = center + Vec2::new(a.cos(), a.sin()) * r;
        if pos.x.abs() >= half || pos.y.abs() >= half || t.height_at(pos.x, pos.y) <= sea {
            continue;
        }
        let (depth, fall, _) = water_here(t, pos);
        if depth < MIN_DEPTH * 1.5 || fall.length() < MIN_FALL {
            continue;
        }
        let life = 4.0 + 5.0 * flow.rand();
        let width = 0.05 + 0.06 * flow.rand();
        flow.flecks.push(Fleck { pos, vel: fall.normalize_or_zero() * stream_speed(depth, fall.length()), age: 0.0, life, width });
    }

    // Each a streak lying on the water along its way, raised a little at both ends to follow the
    // surface down a falling stream.
    let mut positions = Vec::with_capacity(flow.flecks.len() * 4);
    let mut colors = Vec::with_capacity(flow.flecks.len() * 4);
    let mut indices = Vec::with_capacity(flow.flecks.len() * 6);
    let surface = |p: Vec2| t.height_at(p.x, p.y) + t.water_at(p.x, p.y) + 0.015;
    for f in &flow.flecks {
        let speed = f.vel.length();
        let along = if speed > 1e-3 { f.vel / speed } else { Vec2::X };
        let side = Vec2::new(-along.y, along.x);
        let len = 0.2 + 0.3 * speed.min(2.0);
        let fade = smoothstep(0.0, 0.6, f.age) * (1.0 - smoothstep(f.life - 1.2, f.life, f.age));
        let alpha = 0.75 * fade * (t.water_at(f.pos.x, f.pos.y) / 0.05).clamp(0.0, 1.0);
        let base = positions.len() as u32;
        for (u, v) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            let p = f.pos + along * len * u + side * f.width * v;
            positions.push([p.x, surface(p), p.y]);
            colors.push([1.0, 1.0, 1.0, alpha]);
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    let _ = meshes.insert(&flow.mesh, flecks_mesh(positions, colors, indices));
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn flecks_mesh(positions: Vec<[f32; 3]>, colors: Vec<[f32; 4]>, indices: Vec<u32>) -> Mesh {
    let normals = vec![[0.0f32, 1.0, 0.0]; positions.len()];
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deeper_and_steeper_water_runs_faster_but_not_without_limit() {
        assert_eq!(stream_speed(0.0, 0.1), 0.0);
        assert_eq!(stream_speed(0.1, 0.0), 0.0);
        assert!(stream_speed(0.2, 0.05) > stream_speed(0.1, 0.05));
        assert!(stream_speed(0.1, 0.1) > stream_speed(0.1, 0.02));
        assert_eq!(stream_speed(5.0, 1.0), TOP_SPEED);
        // A 10 cm stream falling 5 cm a metre runs at about a metre a second.
        let v = stream_speed(0.1, 0.05);
        assert!(v > 0.8 && v < 1.6, "{v}");
    }
}
