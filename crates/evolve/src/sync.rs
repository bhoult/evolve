//! Mirrors the simulation snapshot into Bevy entities: one entity per body part, plant, piece of
//! meat and obstacle, created as things are born and destroyed as they die (a dead creature
//! fades out where it fell first).

use std::collections::{HashMap, HashSet};

use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::render_resource::Face;
use evolve_sim::body::PartShape;
use evolve_sim::genome::hsv_to_rgb;
use evolve_sim::sim::ObstacleShape;
use evolve_sim::sim::PlantKind;
use evolve_sim::snapshot::{CreatureView, FruitView, TreeView};

use crate::{Interp, Snap};

pub struct SyncPlugin;

/// A clickable creature body part.
#[derive(Component)]
pub struct PartOf {
    pub creature: u64,
    pub part: u16,
}

/// Grass, a seedling, a tree or a piece of fruit (all removed by the same tool).
#[derive(Component)]
pub struct PlantOf(pub u64);

/// How trees.rs dresses a plant entity.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum PlantLook {
    Grass,
    /// A tree seedling or a tree.
    Tree,
}

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
    fruit_mat: Handle<StandardMaterial>,
    eye_mat: Handle<StandardMaterial>,
    rock_mat: Handle<StandardMaterial>,
    boulder_mat: Handle<StandardMaterial>,
    outline_mat: Handle<StandardMaterial>,
    capsules: HashMap<(u32, u32), Handle<Mesh>>,
}

struct CreatureEnt {
    /// Root entity of each part, positioned from the physics pose.
    parts: Vec<Entity>,
    /// The entities carrying `PartOf`, which clicks find, and the eyes.
    tagged: Vec<Entity>,
    eyes: Vec<Entity>,
    material: Handle<StandardMaterial>,
    look: ([f32; 3], [f32; 3]),
}

/// Seconds a dead creature's body stays, greying and fading, before it's gone.
const DEATH_FADE: f64 = 3.0;

/// The body of a creature that just died, left where it fell while it fades out.
struct Dying {
    parts: Vec<Entity>,
    material: Handle<StandardMaterial>,
    color: [f32; 3],
    since: f64,
}

#[derive(Resource, Default)]
struct Mirror {
    /// Ids are only unique within one world, so everything is rebuilt when the world changes.
    epoch: u64,
    creatures: HashMap<u64, CreatureEnt>,
    dying: Vec<Dying>,
    plants: HashMap<u64, Entity>,
    trees: HashMap<u64, Entity>,
    fruit: HashMap<u64, Entity>,
    meat: HashMap<u64, Entity>,
    obstacles: HashMap<u64, Entity>,
    outline: Option<Outline>,
}

/// The gold rim round the selected creature: a shell on each body part (see `outline`).
struct Outline {
    creature: u64,
    /// The creature's first part when the shells were made: a new one means the body was rebuilt.
    root: Entity,
    shells: Vec<Entity>,
}

/// How thick the selected creature's rim is, against the size of its biggest part.
const OUTLINE: f32 = 0.15;

impl Plugin for SyncPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Mirror>()
            .init_resource::<ViewSettings>()
            .add_systems(Startup, setup_assets)
            .add_systems(
                Update,
                (reset_on_new_world, (sync_creatures, sync_food, sync_trees, sync_obstacles)).chain(),
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
        fruit_mat: mat(&mut materials, Color::srgb(0.92, 0.42, 0.12), 0.45),
        eye_mat: mat(&mut materials, Color::srgb(0.03, 0.03, 0.03), 0.2),
        rock_mat: mat(&mut materials, Color::srgb(0.42, 0.41, 0.40), 0.95),
        boulder_mat: mat(&mut materials, Color::srgb(0.55, 0.45, 0.35), 0.85),
        // Unlit, and bright enough to bloom a little; only its inside faces are drawn.
        outline_mat: materials.add(StandardMaterial {
            base_color: Color::linear_rgb(1.6, 1.15, 0.15),
            unlit: true,
            cull_mode: Some(Face::Front),
            ..default()
        }),
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
    let emissive = if c.heaven {
        // The blessing of heaven: a golden glow (blooming in the light of the HDR camera).
        [2.4, 1.8, 0.55]
    } else if selected {
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
    let _cost = crate::FrameCost::start("sync_creatures");
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
        let (mut tagged, mut eyes) = (Vec::with_capacity(c.parts.len()), Vec::new());
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
                            tagged.push(
                                b.spawn((
                                    Mesh3d(shared.cube.clone()),
                                    MeshMaterial3d(material.clone()),
                                    Transform::from_scale(h * 2.0),
                                    tag,
                                ))
                                .id(),
                            );
                            for side in [-1.0, 1.0] {
                                eyes.push(
                                    b.spawn((
                                        Mesh3d(shared.eye.clone()),
                                        MeshMaterial3d(shared.eye_mat.clone()),
                                        eye_at(side),
                                    ))
                                    .id(),
                                );
                            }
                        })
                        .id()
                }
                PartShape::Box { half } => commands
                    .spawn((root, Visibility::default()))
                    .with_children(|b| {
                        tagged.push(
                            b.spawn((
                                Mesh3d(shared.cube.clone()),
                                MeshMaterial3d(material.clone()),
                                Transform::from_scale(Vec3::from_array(half) * 2.0),
                                tag,
                            ))
                            .id(),
                        );
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
                    let e = commands
                        .spawn((root, Mesh3d(mesh), MeshMaterial3d(material.clone()), tag))
                        .id();
                    tagged.push(e);
                    e
                }
            };
            parts.push(e);
        }
        mirror.creatures.insert(
            c.id,
            CreatureEnt {
                parts,
                tagged,
                eyes,
                material,
                look,
            },
        );
    }
    // A long-running world sees many body sizes come and go: forget the capsules nothing uses.
    if shared.capsules.len() > 1024 {
        shared
            .capsules
            .retain(|_, h| matches!(h, Handle::Strong(a) if std::sync::Arc::strong_count(a) > 1));
    }
    let gone: Vec<u64> = mirror
        .creatures
        .keys()
        .filter(|id| !alive.contains(id))
        .copied()
        .collect();
    // A dead creature doesn't just vanish: its body stays where it fell, no longer clickable,
    // and greys and fades out.
    let now = time.elapsed_secs_f64();
    for id in gone {
        if let Some(ent) = mirror.creatures.remove(&id) {
            for &e in &ent.eyes {
                commands.entity(e).despawn();
            }
            for &e in &ent.tagged {
                commands.entity(e).remove::<PartOf>();
            }
            if let Some(m) = materials.get_mut(&ent.material) {
                m.emissive = LinearRgba::BLACK;
                m.alpha_mode = AlphaMode::Blend;
            }
            mirror.dying.push(Dying {
                parts: ent.parts,
                material: ent.material,
                color: ent.look.0,
                since: now,
            });
        }
    }
    outline(&mut commands, &mut mirror, &shared, &mut meshes, &snap.creatures, selected);
    mirror.dying.retain(|d| {
        let t = ((now - d.since) / DEATH_FADE) as f32;
        if t >= 1.0 {
            for &e in &d.parts {
                commands.entity(e).despawn();
            }
            materials.remove(&d.material);
            return false;
        }
        if let Some(m) = materials.get_mut(&d.material) {
            // Grey within the first third, transparent by the end.
            let [r, g, b] = d.color;
            let grey = 0.5 * (0.3 * r + 0.59 * g + 0.11 * b);
            let k = (3.0 * t).min(1.0);
            let mix = |c: f32| c + (grey - c) * k;
            m.base_color = Color::srgba(mix(r), mix(g), mix(b), 1.0 - t);
        }
        true
    });
}

/// Rings the selected creature in gold: each body part gets a slightly bigger copy of itself with
/// only its inside drawn, so a rim shows round the body's outline (and at its joints). Unlit and
/// casting no shadow, it shows in shade and at night. Made again when the selection changes or the
/// body is rebuilt, and gone when the creature dies.
fn outline(
    commands: &mut Commands,
    mirror: &mut Mirror,
    shared: &Shared3d,
    meshes: &mut Assets<Mesh>,
    creatures: &[CreatureView],
    selected: Option<u64>,
) {
    let want = selected.and_then(|id| Some((id, *mirror.creatures.get(&id)?.parts.first()?)));
    if mirror.outline.as_ref().map(|o| (o.creature, o.root)) == want {
        return;
    }
    if let Some(old) = mirror.outline.take() {
        // Already gone if the body they hung on was despawned.
        for e in old.shells {
            commands.entity(e).try_despawn();
        }
    }
    let Some((id, root)) = want else { return };
    let Some(c) = creatures.iter().find(|c| c.id == id) else { return };
    let roots = &mirror.creatures[&id].parts;
    let w = (c.parts.iter().map(|p| p.shape.bounding_radius()).fold(0.0, f32::max) * OUTLINE).clamp(0.025, 0.1);
    let mut shells = Vec::with_capacity(roots.len());
    for (p, &part) in c.parts.iter().zip(roots) {
        let (mesh, tf) = match p.shape {
            PartShape::Box { half } => (shared.cube.clone(), Transform::from_scale(Vec3::from_array(half) * 2.0 + 2.0 * w)),
            PartShape::Capsule { half_height, radius } => {
                (meshes.add(Capsule3d::new(radius + w, half_height * 2.0)), Transform::IDENTITY)
            }
        };
        let shell = commands
            .spawn((Mesh3d(mesh), MeshMaterial3d(shared.outline_mat.clone()), tf, NotShadowCaster, NotShadowReceiver))
            .id();
        commands.entity(part).add_child(shell);
        shells.push(shell);
    }
    mirror.outline = Some(Outline { creature: id, root, shells });
}

/// A loaded or new world reuses ids, so drop every mirrored entity and start over.
fn reset_on_new_world(
    mut commands: Commands,
    snap: Res<Snap>,
    mut mirror: ResMut<Mirror>,
    mut shared: ResMut<Shared3d>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if snap.0.world_epoch == mirror.epoch {
        return;
    }
    mirror.epoch = snap.0.world_epoch;
    mirror.outline = None;
    // Every body size ever seen has a capsule mesh cached; start the new world's set afresh (the
    // meshes are freed once the despawned parts below let go of them).
    shared.capsules.clear();
    for (_, c) in mirror.creatures.drain() {
        for e in c.parts {
            commands.entity(e).despawn();
        }
        materials.remove(&c.material);
    }
    for d in mirror.dying.drain(..) {
        for e in d.parts {
            commands.entity(e).despawn();
        }
        materials.remove(&d.material);
    }
    let mirror = &mut *mirror;
    for (_, e) in mirror
        .plants
        .drain()
        .chain(mirror.trees.drain())
        .chain(mirror.fruit.drain())
        .chain(mirror.meat.drain())
        .chain(mirror.obstacles.drain())
    {
        commands.entity(e).despawn();
    }
}

fn plant_transform(pos: [f32; 3], energy: f32, max: f32, kind: PlantKind) -> Transform {
    let grown = (energy / max.max(1e-3)).clamp(0.0, 1.0).sqrt();
    // Grass stays knee-high; a seedling reaches a meter before it turns woody.
    let s = match kind {
        PlantKind::Grass => 0.2 + 0.4 * grown,
        PlantKind::Seedling => 0.25 + 0.85 * grown,
    };
    // `pos` is the lowest ground under the plant; sink the base a little more so it is planted.
    Transform::from_xyz(pos[0], pos[1] + 0.45 * s - 0.04, pos[2]).with_scale(Vec3::splat(s))
}

/// The tree meshes are 0.9 units tall, centered. A tree that died topples over about three
/// seconds, away from where its id points, about the foot of its trunk.
fn tree_transform(t: &TreeView) -> Transform {
    let s = t.height / 0.9;
    let base = Vec3::from_array(t.pos) - Vec3::Y * 0.1;
    let fall = (t.fallen / 3.0).clamp(0.0, 1.0);
    let toward = (t.id.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 * 0.001;
    let axis = Vec3::new(toward.cos(), 0.0, toward.sin());
    let rot = Quat::from_axis_angle(axis, fall * fall * std::f32::consts::FRAC_PI_2 * 0.97);
    Transform::from_translation(base + rot * (Vec3::Y * 0.45 * s)).with_rotation(rot).with_scale(Vec3::splat(s))
}

fn fruit_transform(f: &FruitView, full: f32) -> Transform {
    let s = 0.16 + 0.2 * (f.energy / full.max(1e-3)).clamp(0.0, 1.0).sqrt();
    let lift = if f.hanging { 0.0 } else { 0.4 * s };
    Transform::from_xyz(f.pos[0], f.pos[1] + lift, f.pos[2]).with_scale(Vec3::splat(s))
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
    let _cost = crate::FrameCost::start("sync_food");
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
                plant_transform(f.pos, f.energy, max, f.kind)
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
                        let look = if f.kind == PlantKind::Grass { PlantLook::Grass } else { PlantLook::Tree };
                        commands
                            .spawn((Mesh3d(shared.plant.clone()), MeshMaterial3d(shared.plant_mat.clone()), tf, PlantOf(f.id), look))
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

/// Trees (drawn by trees.rs once they appear here) and their fruit, hanging and fallen.
fn sync_trees(
    mut commands: Commands,
    snap: Res<Snap>,
    shared: Res<Shared3d>,
    mut mirror: ResMut<Mirror>,
    mut transforms: Query<&mut Transform>,
) {
    let snap = &snap.0;
    let mirror = &mut *mirror;
    let mut place = |map: &mut HashMap<u64, Entity>, id: u64, tf: Transform, spawn: &mut dyn FnMut(Transform) -> Entity| {
        match map.get(&id) {
            Some(&e) => {
                if let Ok(mut t) = transforms.get_mut(e)
                    && ((t.scale - tf.scale).length_squared() > 1e-4
                        || (t.translation - tf.translation).length_squared() > 1e-6
                        || t.rotation.angle_between(tf.rotation) > 1e-3)
                {
                    *t = tf;
                }
            }
            None => {
                map.insert(id, spawn(tf));
            }
        }
    };
    let mut alive = HashSet::with_capacity(snap.trees.len());
    for t in &snap.trees {
        alive.insert(t.id);
        place(&mut mirror.trees, t.id, tree_transform(t), &mut |tf| {
            commands
                .spawn((Mesh3d(shared.plant.clone()), MeshMaterial3d(shared.plant_mat.clone()), tf, PlantOf(t.id), PlantLook::Tree))
                .id()
        });
    }
    let gone = |map: &mut HashMap<u64, Entity>, alive: &HashSet<u64>, commands: &mut Commands| {
        map.retain(|id, e| {
            let keep = alive.contains(id);
            if !keep {
                commands.entity(*e).despawn();
            }
            keep
        });
    };
    gone(&mut mirror.trees, &alive, &mut commands);
    let full = snap.params.fruit_energy;
    let mut alive = HashSet::with_capacity(snap.fruit.len());
    for f in &snap.fruit {
        alive.insert(f.id);
        place(&mut mirror.fruit, f.id, fruit_transform(f, full), &mut |tf| {
            commands
                .spawn((Mesh3d(shared.sphere.clone()), MeshMaterial3d(shared.fruit_mat.clone()), tf, PlantOf(f.id)))
                .id()
        });
    }
    gone(&mut mirror.fruit, &alive, &mut commands);
}

fn sync_obstacles(
    mut commands: Commands,
    snap: Res<Snap>,
    shared: Res<Shared3d>,
    mut mirror: ResMut<Mirror>,
    mut transforms: Query<&mut Transform>,
) {
    let _cost = crate::FrameCost::start("sync_obstacles");
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
