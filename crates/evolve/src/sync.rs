//! Mirrors the simulation snapshot into Bevy entities: one entity per body part, plant, piece of
//! meat and obstacle, created and destroyed as things are born and die.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use evolve_sim::body::PartShape;
use evolve_sim::genome::hsv_to_rgb;
use evolve_sim::sim::ObstacleShape;
use evolve_sim::snapshot::CreatureView;

use crate::{Interp, Snap};

pub struct SyncPlugin;

/// A clickable creature body part.
#[derive(Component)]
pub struct PartOf {
    pub creature: u64,
    pub part: u16,
}

#[derive(Component)]
pub struct PlantOf(pub u64);

#[derive(Component)]
pub struct MeatOf(pub u64);

#[derive(Component)]
pub struct ObstacleOf(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ColorMode {
    #[default]
    Lineage,
    Diet,
    Energy,
    Generation,
    BrainSize,
}

impl ColorMode {
    pub const ALL: [ColorMode; 5] = [
        ColorMode::Lineage,
        ColorMode::Diet,
        ColorMode::Energy,
        ColorMode::Generation,
        ColorMode::BrainSize,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Lineage => "Lineage",
            ColorMode::Diet => "Diet (green plant, red meat)",
            ColorMode::Energy => "Energy",
            ColorMode::Generation => "Generation",
            ColorMode::BrainSize => "Brain size",
        }
    }
}

#[derive(Resource, Default)]
pub struct ViewSettings {
    pub color_mode: ColorMode,
}

#[derive(Resource)]
struct Shared3d {
    cube: Handle<Mesh>,
    sphere: Handle<Mesh>,
    eye: Handle<Mesh>,
    plant: Handle<Mesh>,
    plant_mat: Handle<StandardMaterial>,
    meat_mat: Handle<StandardMaterial>,
    eye_mat: Handle<StandardMaterial>,
    rock_mat: Handle<StandardMaterial>,
    boulder_mat: Handle<StandardMaterial>,
    capsules: HashMap<(u32, u32), Handle<Mesh>>,
}

struct CreatureEnt {
    /// Root entity of each part, positioned from the physics pose.
    parts: Vec<Entity>,
    material: Handle<StandardMaterial>,
    look: ([f32; 3], [f32; 3]),
}

#[derive(Resource, Default)]
struct Mirror {
    /// Ids are only unique within one world, so everything is rebuilt when the world changes.
    epoch: u64,
    creatures: HashMap<u64, CreatureEnt>,
    plants: HashMap<u64, Entity>,
    meat: HashMap<u64, Entity>,
    obstacles: HashMap<u64, Entity>,
}

impl Plugin for SyncPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Mirror>()
            .init_resource::<ViewSettings>()
            .add_systems(Startup, setup_assets)
            .add_systems(
                Update,
                (reset_on_new_world, (sync_creatures, sync_food, sync_obstacles)).chain(),
            );
    }
}

fn setup_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mat = |materials: &mut Assets<StandardMaterial>, c: Color, rough: f32| {
        materials.add(StandardMaterial {
            base_color: c,
            perceptual_roughness: rough,
            ..default()
        })
    };
    commands.insert_resource(Shared3d {
        cube: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        sphere: meshes.add(Sphere::new(0.5).mesh().ico(3).unwrap()),
        eye: meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap()),
        plant: meshes.add(Cone::new(0.35, 0.9)),
        plant_mat: mat(&mut materials, Color::srgb(0.18, 0.55, 0.14), 0.9),
        meat_mat: mat(&mut materials, Color::srgb(0.55, 0.12, 0.10), 0.6),
        eye_mat: mat(&mut materials, Color::srgb(0.03, 0.03, 0.03), 0.2),
        rock_mat: mat(&mut materials, Color::srgb(0.42, 0.41, 0.40), 0.95),
        boulder_mat: mat(&mut materials, Color::srgb(0.55, 0.45, 0.35), 0.85),
        capsules: HashMap::new(),
    });
}

fn creature_look(c: &CreatureView, mode: ColorMode, selected: bool) -> ([f32; 3], [f32; 3]) {
    let color = match mode {
        ColorMode::Lineage => c.color,
        ColorMode::Diet => {
            let d = c.diet;
            [0.2 + 0.7 * d, 0.75 - 0.6 * d, 0.2]
        }
        ColorMode::Energy => {
            let e = c.energy;
            [0.15 + 0.85 * e, 0.15 + 0.75 * e, 0.2 + 0.1 * e]
        }
        ColorMode::Generation => hsv_to_rgb((c.generation as f32 * 0.07).fract(), 0.7, 0.9),
        ColorMode::BrainSize => {
            let n = (c.neurons as f32 / 48.0).min(1.0);
            [0.2 + 0.7 * n, 0.3, 0.9 - 0.5 * n]
        }
    };
    let emissive = if selected {
        [0.45, 0.35, 0.02]
    } else if c.attacking {
        [2.0, 0.1, 0.05]
    } else if c.mating {
        [1.5, 0.3, 0.9]
    } else if c.drowning {
        [0.1, 0.3, 1.2]
    } else {
        [0.0; 3]
    };
    (color, emissive)
}

#[allow(clippy::too_many_arguments)]
fn sync_creatures(
    mut commands: Commands,
    snap: Res<Snap>,
    view: Res<ViewSettings>,
    mut mirror: ResMut<Mirror>,
    mut shared: ResMut<Shared3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut transforms: Query<&mut Transform>,
    interp: Res<Interp>,
    time: Res<Time>,
) {
    let snap = &snap.0;
    let alpha = interp.alpha(time.elapsed_secs_f64());
    let selected = snap.selected.as_ref().map(|s| s.id);
    let mut alive = HashSet::with_capacity(snap.creatures.len());
    for c in &snap.creatures {
        alive.insert(c.id);
        let look = creature_look(c, view.color_mode, selected == Some(c.id));
        if let Some(ent) = mirror.creatures.get_mut(&c.id)
            && ent.parts.len() == c.parts.len()
        {
            let prev = interp
                .prev_index
                .get(&c.id)
                .map(|&i| &interp.prev.creatures[i].parts)
                .filter(|p| p.len() == c.parts.len());
            for (k, (e, p)) in ent.parts.iter().zip(&c.parts).enumerate() {
                let (mut pos, mut rot) = (Vec3::from_array(p.pos), Quat::from_array(p.rot));
                if let Some(prev) = prev {
                    pos = Vec3::from_array(prev[k].pos).lerp(pos, alpha);
                    rot = Quat::from_array(prev[k].rot).slerp(rot, alpha);
                }
                if let Ok(mut tf) = transforms.get_mut(*e) {
                    tf.translation = pos;
                    tf.rotation = rot;
                }
            }
            if ent.look != look {
                ent.look = look;
                if let Some(m) = materials.get_mut(&ent.material) {
                    m.base_color = Color::srgb(look.0[0], look.0[1], look.0[2]);
                    m.emissive = LinearRgba::rgb(look.1[0], look.1[1], look.1[2]);
                }
            }
            continue;
        }
        if let Some(old) = mirror.creatures.remove(&c.id) {
            for e in old.parts {
                commands.entity(e).despawn();
            }
        }
        let material = materials.add(StandardMaterial {
            base_color: Color::srgb(look.0[0], look.0[1], look.0[2]),
            emissive: LinearRgba::rgb(look.1[0], look.1[1], look.1[2]),
            perceptual_roughness: 0.55,
            ..default()
        });
        let mut parts = Vec::with_capacity(c.parts.len());
        for (k, p) in c.parts.iter().enumerate() {
            let root = Transform::from_translation(Vec3::from_array(p.pos))
                .with_rotation(Quat::from_array(p.rot));
            let tag = PartOf {
                creature: c.id,
                part: k as u16,
            };
            let e = match p.shape {
                // The torso (part 0) is the head end and gets eyes; other boxes are spine
                // segments and paddles.
                PartShape::Box { half } if k == 0 => {
                    let h = Vec3::from_array(half);
                    let eye_r = (h.y.min(h.z) * 0.45).clamp(0.035, 0.09);
                    let eye_at = |side: f32| {
                        Transform::from_xyz(h.x, h.y * 0.4, side * h.z * 0.45)
                            .with_scale(Vec3::splat(eye_r))
                    };
                    commands
                        .spawn((root, Visibility::default()))
                        .with_children(|b| {
                            b.spawn((
                                Mesh3d(shared.cube.clone()),
                                MeshMaterial3d(material.clone()),
                                Transform::from_scale(h * 2.0),
                                tag,
                            ));
                            for side in [-1.0, 1.0] {
                                b.spawn((
                                    Mesh3d(shared.eye.clone()),
                                    MeshMaterial3d(shared.eye_mat.clone()),
                                    eye_at(side),
                                ));
                            }
                        })
                        .id()
                }
                PartShape::Box { half } => commands
                    .spawn((root, Visibility::default()))
                    .with_children(|b| {
                        b.spawn((
                            Mesh3d(shared.cube.clone()),
                            MeshMaterial3d(material.clone()),
                            Transform::from_scale(Vec3::from_array(half) * 2.0),
                            tag,
                        ));
                    })
                    .id(),
                PartShape::Capsule { half_height, radius } => {
                    let key = ((half_height * 200.0).round() as u32, (radius * 200.0).round() as u32);
                    let mesh = shared
                        .capsules
                        .entry(key)
                        .or_insert_with(|| {
                            meshes.add(Capsule3d::new(key.1 as f32 / 200.0, key.0 as f32 / 100.0))
                        })
                        .clone();
                    commands
                        .spawn((root, Mesh3d(mesh), MeshMaterial3d(material.clone()), tag))
                        .id()
                }
            };
            parts.push(e);
        }
        mirror.creatures.insert(
            c.id,
            CreatureEnt {
                parts,
                material,
                look,
            },
        );
    }
    let gone: Vec<u64> = mirror
        .creatures
        .keys()
        .filter(|id| !alive.contains(id))
        .copied()
        .collect();
    for id in gone {
        if let Some(ent) = mirror.creatures.remove(&id) {
            for e in ent.parts {
                commands.entity(e).despawn();
            }
            materials.remove(&ent.material);
        }
    }
}

/// A loaded or new world reuses ids, so drop every mirrored entity and start over.
fn reset_on_new_world(
    mut commands: Commands,
    snap: Res<Snap>,
    mut mirror: ResMut<Mirror>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if snap.0.world_epoch == mirror.epoch {
        return;
    }
    mirror.epoch = snap.0.world_epoch;
    for (_, c) in mirror.creatures.drain() {
        for e in c.parts {
            commands.entity(e).despawn();
        }
        materials.remove(&c.material);
    }
    let mirror = &mut *mirror;
    for (_, e) in mirror
        .plants
        .drain()
        .chain(mirror.meat.drain())
        .chain(mirror.obstacles.drain())
    {
        commands.entity(e).despawn();
    }
}

fn plant_transform(pos: [f32; 3], energy: f32, max: f32) -> Transform {
    let s = 0.3 + 0.7 * (energy / max.max(1e-3)).clamp(0.0, 1.0).sqrt();
    // `pos` is the lowest ground under the plant; sink the base a little more so it is planted.
    Transform::from_xyz(pos[0], pos[1] + 0.45 * s - 0.04, pos[2]).with_scale(Vec3::splat(s))
}

fn meat_transform(pos: [f32; 3], energy: f32) -> Transform {
    let s = (0.35 + (energy / 40.0).sqrt() * 0.5).clamp(0.3, 1.4);
    Transform::from_xyz(pos[0], pos[1] + 0.15 * s, pos[2]).with_scale(Vec3::new(s, s * 0.6, s))
}

fn sync_food(
    mut commands: Commands,
    snap: Res<Snap>,
    shared: Res<Shared3d>,
    mut mirror: ResMut<Mirror>,
    mut transforms: Query<&mut Transform>,
) {
    let snap = &snap.0;
    let max = snap.params.plant_max_energy;
    let mirror = &mut *mirror;
    for (list, map, is_meat) in [
        (&snap.plants, &mut mirror.plants, false),
        (&snap.meat, &mut mirror.meat, true),
    ] {
        let mut alive = HashSet::with_capacity(list.len());
        for f in list {
            alive.insert(f.id);
            let tf = if is_meat {
                meat_transform(f.pos, f.energy)
            } else {
                plant_transform(f.pos, f.energy, max)
            };
            match map.get(&f.id) {
                Some(&e) => {
                    if let Ok(mut t) = transforms.get_mut(e)
                        && ((t.scale - tf.scale).length_squared() > 1e-4
                            || (t.translation - tf.translation).length_squared() > 1e-6)
                    {
                        *t = tf;
                    }
                }
                None => {
                    let e = if is_meat {
                        commands
                            .spawn((Mesh3d(shared.sphere.clone()), MeshMaterial3d(shared.meat_mat.clone()), tf, MeatOf(f.id)))
                            .id()
                    } else {
                        commands
                            .spawn((Mesh3d(shared.plant.clone()), MeshMaterial3d(shared.plant_mat.clone()), tf, PlantOf(f.id)))
                            .id()
                    };
                    map.insert(f.id, e);
                }
            }
        }
        map.retain(|id, e| {
            let keep = alive.contains(id);
            if !keep {
                commands.entity(*e).despawn();
            }
            keep
        });
    }
}

fn sync_obstacles(
    mut commands: Commands,
    snap: Res<Snap>,
    shared: Res<Shared3d>,
    mut mirror: ResMut<Mirror>,
    mut transforms: Query<&mut Transform>,
) {
    let mut alive = HashSet::new();
    for o in &snap.0.obstacles {
        alive.insert(o.id);
        let (mesh, scale) = match o.shape {
            ObstacleShape::Box { half } => (shared.cube.clone(), Vec3::from_array(half) * 2.0),
            ObstacleShape::Ball { radius } => (shared.sphere.clone(), Vec3::splat(radius * 2.0)),
        };
        let tf = Transform::from_translation(Vec3::from_array(o.pos))
            .with_rotation(Quat::from_array(o.rot))
            .with_scale(scale);
        match mirror.obstacles.get(&o.id) {
            Some(&e) => {
                if let Ok(mut t) = transforms.get_mut(e) {
                    *t = tf;
                }
            }
            None => {
                let mat = if o.dynamic { shared.boulder_mat.clone() } else { shared.rock_mat.clone() };
                let e = commands
                    .spawn((Mesh3d(mesh), MeshMaterial3d(mat), tf, ObstacleOf(o.id)))
                    .id();
                mirror.obstacles.insert(o.id, e);
            }
        }
    }
    mirror.obstacles.retain(|id, e| {
        let keep = alive.contains(id);
        if !keep {
            commands.entity(*e).despawn();
        }
        keep
    });
}
