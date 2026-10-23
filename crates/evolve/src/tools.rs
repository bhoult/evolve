//! Mouse tools: select and drag creatures, plant food, drop meat, place rocks and boulders,
//! drop creatures, and remove things.

use bevy::input::mouse::{AccumulatedMouseScroll, MouseScrollUnit};
use bevy::picking::mesh_picking::ray_cast::RayMeshHit;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::input::{EguiWantsInput, egui_wants_any_keyboard_input};
use evolve_sim::Command;
use evolve_sim::sim::ObstacleShape;

use crate::camera::MainCamera;
use crate::scene::TerrainMesh;
use crate::sync::{MeatOf, ObstacleOf, PartOf, PlantOf};
use crate::{SimLink, Snap};

pub struct ToolsPlugin;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Tool {
    #[default]
    Select,
    Plant,
    Meat,
    Rock,
    Boulder,
    Creature,
    Remove,
    Meteor,
    Whirlwind,
    Raise,
    Lower,
    Spring,
    Quake,
    Landfall,
    Manna,
    Totem,
    Tsunami,
}

impl Tool {
    pub const ALL: [Tool; 17] = [
        Tool::Select,
        Tool::Plant,
        Tool::Meat,
        Tool::Rock,
        Tool::Boulder,
        Tool::Creature,
        Tool::Remove,
        Tool::Meteor,
        Tool::Whirlwind,
        Tool::Raise,
        Tool::Lower,
        Tool::Spring,
        Tool::Quake,
        Tool::Landfall,
        Tool::Manna,
        Tool::Totem,
        Tool::Tsunami,
    ];

    /// The acts of god, shown apart from the everyday tools.
    pub fn is_power(self) -> bool {
        !matches!(self, Tool::Select | Tool::Plant | Tool::Meat | Tool::Rock | Tool::Boulder | Tool::Creature | Tool::Remove)
    }

    pub fn label(self) -> &'static str {
        match self {
            Tool::Select => "Select / drag",
            Tool::Plant => "Plant food",
            Tool::Meat => "Meat",
            Tool::Rock => "Rock",
            Tool::Boulder => "Boulder",
            Tool::Creature => "Drop creature",
            Tool::Remove => "Remove",
            Tool::Meteor => "Meteor",
            Tool::Whirlwind => "Whirlwind",
            Tool::Raise => "Raise land",
            Tool::Lower => "Lower land",
            Tool::Spring => "Spring",
            Tool::Quake => "Earthquake",
            Tool::Landfall => "Land from the sky",
            Tool::Manna => "Manna",
            Tool::Totem => "Totem",
            Tool::Tsunami => "Tsunami",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tool::Select => "Click a creature to inspect it; hold and drag to carry it (scroll lifts it). Boulders can be dragged too.",
            Tool::Plant => "Click or hold and sweep over the ground to plant food.",
            Tool::Meat => "Click to drop a chunk of meat.",
            Tool::Rock => "Click to place a fixed rock.",
            Tool::Boulder => "Click to place a boulder that can roll and be pushed.",
            Tool::Creature => "Click to drop a new creature from the sky (bred from the gene pool or random).",
            Tool::Remove => "Click a creature, plant, meat or obstacle to remove it.",
            Tool::Meteor => "Click to call a meteor down there: a crater, a blast that throws and hurts what's near and kills what it lands on, and fire.",
            Tool::Whirlwind => "Click to set a whirlwind down there: for 25 seconds it wanders the island, lifting and flinging creatures and tearing up plants.",
            Tool::Raise => "Hold and sweep to raise the land (the brush sets how wide). Out of the sea, it makes new land.",
            Tool::Lower => "Hold and sweep to lower the land (the brush sets how wide). Below the sea, it lets the sea in.",
            Tool::Spring => "Click to make an eternal spring there: water wells up forever, runs downhill and fills the hollows it reaches. Click a spring to stop it.",
            Tool::Quake => "Click to split the land with an earthquake there: a crack opens across your view, one side heaves up, and everything nearby shakes.",
            Tool::Landfall => "Click to drop a chunk of land from the sky there: a hill rises where it lands (out of the sea, new land), crushing what's under it.",
            Tool::Manna => "Click to let food fall from heaven there: a ring of full-grown plants.",
            Tool::Tsunami => "Click a shore and a great wave rolls in from the open sea beyond it, across the island: it floods the low ground, carries creatures off, holds them under, and sweeps away the plants it covers.",
            Tool::Totem => "Click to raise a totem (love, war or fear: pick beside the tools); within 16 m it drives creatures to breed, to fight, or to run. Click a totem to take it down.",
        }
    }
}

pub struct Drag {
    pub height: f32,
}

/// Scripted pointer for `--pick-test`: drives the real tool code without a mouse.
#[derive(Resource, Default)]
pub struct FakePointer {
    pub pos: Option<Vec2>,
    pub just_pressed: bool,
    pub pressed: bool,
    pub just_released: bool,
}

#[derive(Resource)]
pub struct ToolState {
    pub tool: Tool,
    pub brush: f32,
    pub obstacle_size: f32,
    pub drag: Option<Drag>,
    /// A zoo creature waiting to be placed with the next click: (zoo id, name, copies).
    pub zoo_place: Option<(u64, String, u32)>,
    /// A power just called down at a spot, for miracles.rs to carry out.
    pub summon: Option<(Tool, Vec3)>,
    /// What the next totem stirs.
    pub totem: evolve_sim::acts::TotemKind,
    paint_timer: f32,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            tool: Tool::Select,
            brush: 3.0,
            obstacle_size: 2.0,
            drag: None,
            zoo_place: None,
            summon: None,
            totem: evolve_sim::acts::TotemKind::Love,
            paint_timer: 0.0,
        }
    }
}

enum Hit {
    Creature { id: u64, part: u16 },
    Plant(u64),
    Meat(u64),
    Obstacle(u64),
    Terrain,
}

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ToolState>()
            .init_resource::<FakePointer>()
            .add_systems(
            Update,
            (
                pick_test.before(use_tool),
                use_tool,
                hotkeys.run_if(not(egui_wants_any_keyboard_input)),
                select_first_for_screenshot,
            ),
        );
    }
}

fn cursor_ray(cursor: Option<Vec2>, camera: (&Camera, &GlobalTransform)) -> Option<Ray3d> {
    camera.0.viewport_to_world(camera.1, cursor?).ok()
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn use_tool(
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    buttons: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    egui_wants: Res<EguiWantsInput>,
    time: Res<Time>,
    mut ray_cast: MeshRayCast,
    pickable: Query<(), Or<(With<PartOf>, With<PlantOf>, With<MeatOf>, With<ObstacleOf>, With<TerrainMesh>)>>,
    parts: Query<&PartOf>,
    plants: Query<&PlantOf>,
    meat: Query<&MeatOf>,
    obstacles: Query<&ObstacleOf>,
    snap: Res<Snap>,
    link: Res<SimLink>,
    mut tools: ResMut<ToolState>,
    fake: Res<FakePointer>,
) {
    let scripted = fake.pos.is_some();
    let cursor = fake.pos.or_else(|| window.cursor_position());
    let ray = cursor_ray(cursor, (camera.0, camera.1));
    let just_pressed = buttons.just_pressed(MouseButton::Left) || fake.just_pressed;
    let held = buttons.pressed(MouseButton::Left) || fake.pressed;
    let just_released = buttons.just_released(MouseButton::Left) || fake.just_released;

    // Dragging continues even over UI panels, and always ends on release.
    if tools.drag.is_some() {
        if just_released {
            tools.drag = None;
            link.cmd(Command::Release);
            return;
        }
        if scroll.delta.y != 0.0 {
            let lines = match scroll.unit {
                MouseScrollUnit::Line => scroll.delta.y,
                MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
            };
            if let Some(d) = &mut tools.drag {
                d.height += lines * 0.5;
            }
        }
        if let (Some(ray), Some(d)) = (ray, &tools.drag) {
            let dir = *ray.direction;
            if dir.y.abs() > 1e-4 {
                let t = (d.height - ray.origin.y) / dir.y;
                if t > 0.0 {
                    let mut p = ray.origin + dir * t;
                    let half = (snap.0.terrain.size * 0.5 - 1.0).max(1.0);
                    p.x = p.x.clamp(-half, half);
                    p.z = p.z.clamp(-half, half);
                    link.cmd(Command::DragTo(p.to_array()));
                }
            }
        }
        return;
    }

    if egui_wants.wants_any_pointer_input() && !scripted {
        return;
    }
    let pressed = just_pressed;
    let painting = matches!(tools.tool, Tool::Plant | Tool::Raise | Tool::Lower) && held;
    if !pressed && !painting {
        return;
    }
    let Some(ray) = ray else {
        return;
    };
    let filter = |e: Entity| pickable.contains(e);
    let settings = MeshRayCastSettings::default().with_filter(&filter);
    let hits: &[(Entity, RayMeshHit)] = ray_cast.cast_ray(ray, &settings);
    let Some((entity, hit)) = hits.first() else {
        if pressed && tools.tool == Tool::Select {
            link.cmd(Command::Select(None));
        }
        return;
    };
    let kind = if let Ok(p) = parts.get(*entity) {
        Hit::Creature {
            id: p.creature,
            part: p.part,
        }
    } else if let Ok(p) = plants.get(*entity) {
        Hit::Plant(p.0)
    } else if let Ok(m) = meat.get(*entity) {
        Hit::Meat(m.0)
    } else if let Ok(o) = obstacles.get(*entity) {
        Hit::Obstacle(o.0)
    } else {
        Hit::Terrain
    };
    let point = hit.point;
    if scripted {
        let what = match kind {
            Hit::Creature { id, part } => format!("creature #{id} part {part}"),
            Hit::Plant(id) => format!("plant #{id}"),
            Hit::Meat(id) => format!("meat #{id}"),
            Hit::Obstacle(id) => format!("obstacle #{id}"),
            Hit::Terrain => "terrain".into(),
        };
        println!("pick-test: pointer hit {what} at {point:.2}");
    }

    if painting {
        tools.paint_timer -= time.delta_secs();
        if !pressed && tools.paint_timer > 0.0 {
            return;
        }
        tools.paint_timer = 0.08;
        match tools.tool {
            Tool::Raise | Tool::Lower => link.cmd(Command::Terraform {
                at: point.to_array(),
                radius: tools.brush,
                amount: if tools.tool == Tool::Raise { 0.15 } else { -0.15 },
            }),
            _ => {
                let count = (tools.brush * 0.6).ceil().max(1.0) as u32;
                link.cmd(Command::SpawnPlants {
                    center: point.to_array(),
                    radius: tools.brush,
                    count,
                });
            }
        }
        return;
    }

    let rng_yaw = (time.elapsed_secs() * 7.31).sin() * std::f32::consts::PI;
    let size = tools.obstacle_size;
    match (tools.tool, kind) {
        (Tool::Select, Hit::Creature { id, part }) => {
            link.cmd(Command::Select(Some(id)));
            let local = snap
                .0
                .creatures
                .iter()
                .find(|c| c.id == id)
                .and_then(|c| c.parts.get(part as usize))
                .map(|p| Quat::from_array(p.rot).inverse() * (point - Vec3::from_array(p.pos)))
                .unwrap_or(Vec3::ZERO);
            link.cmd(Command::GrabCreature {
                creature: id,
                part: part as usize,
                local: local.to_array(),
                target: (point + Vec3::Y * 0.5).to_array(),
            });
            tools.drag = Some(Drag {
                height: point.y + 0.5,
            });
        }
        (Tool::Select, Hit::Obstacle(id)) => {
            if let Some(o) = snap.0.obstacles.iter().find(|o| o.id == id && o.dynamic) {
                let local = Quat::from_array(o.rot).inverse() * (point - Vec3::from_array(o.pos));
                link.cmd(Command::GrabObstacle {
                    obstacle: id,
                    local: local.to_array(),
                    target: (point + Vec3::Y * 0.5).to_array(),
                });
                tools.drag = Some(Drag {
                    height: point.y + 0.5,
                });
            }
        }
        (Tool::Select, _) => link.cmd(Command::Select(None)),
        (Tool::Meat, _) => link.cmd(Command::SpawnMeat {
            pos: point.to_array(),
        }),
        (Tool::Rock, _) => {
            let s = size * (0.75 + 0.25 * rng_yaw.cos().abs());
            link.cmd(Command::SpawnObstacle {
                pos: point.to_array(),
                shape: ObstacleShape::Box {
                    half: [s * 0.5, s * 0.35, s * 0.4],
                },
                dynamic: false,
                yaw: rng_yaw,
            });
        }
        (Tool::Boulder, _) => link.cmd(Command::SpawnObstacle {
            pos: point.to_array(),
            shape: ObstacleShape::Ball { radius: size * 0.4 },
            dynamic: true,
            yaw: rng_yaw,
        }),
        (Tool::Creature, _) => match tools.zoo_place.take() {
            Some((id, _, count)) => link.0.send(evolve_sim::Control::ZooDrop {
                id,
                target: evolve_sim::ZooTarget::This,
                pos: Some(point.to_array()),
                count,
            }),
            None => link.cmd(Command::SpawnCreature {
                pos: Some(point.to_array()),
                from: None,
                count: 1,
            }),
        },
        (Tool::Remove, Hit::Creature { id, .. }) => link.cmd(Command::RemoveCreature(id)),
        (Tool::Remove, Hit::Plant(id)) => link.cmd(Command::RemovePlant(id)),
        (Tool::Remove, Hit::Meat(id)) => link.cmd(Command::RemoveMeat(id)),
        (Tool::Remove, Hit::Obstacle(id)) => link.cmd(Command::RemoveObstacle(id)),
        (Tool::Meteor, _) => tools.summon = Some((tools.tool, point)),
        (Tool::Whirlwind, _) => link.cmd(Command::Whirlwind { at: point.to_array() }),
        (Tool::Spring, _) => link.cmd(Command::Spring { at: point.to_array() }),
        (Tool::Manna, _) => link.cmd(Command::Manna { at: point.to_array() }),
        (Tool::Tsunami, _) => tools.summon = Some((tools.tool, point)),
        (Tool::Totem, _) => link.cmd(Command::Totem { at: point.to_array(), kind: tools.totem }),
        (Tool::Quake | Tool::Landfall, _) => tools.summon = Some((tools.tool, point)),
        _ => {}
    }
}

fn hotkeys(keys: Res<ButtonInput<KeyCode>>, snap: Res<Snap>, link: Res<SimLink>, mut tools: ResMut<ToolState>) {
    let digits = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    for (k, t) in digits.iter().zip(Tool::ALL) {
        if keys.just_pressed(*k) {
            tools.tool = t;
        }
    }
    if keys.just_pressed(KeyCode::Delete)
        && let Some(sel) = &snap.0.selected
    {
        link.cmd(Command::RemoveCreature(sel.id));
    }
    if keys.just_pressed(KeyCode::Escape) {
        link.cmd(Command::Select(None));
        tools.zoo_place = None;
    }
}

/// For `--select-first` screenshot runs: select some creature once the world has any.
fn select_first_for_screenshot(
    args: Res<crate::Args>,
    snap: Res<Snap>,
    link: Res<SimLink>,
    mut cam: ResMut<crate::camera::OrbitCam>,
    mut done: Local<bool>,
) {
    if *done || !args.select_first {
        return;
    }
    if let Some(c) = snap.0.creatures.iter().max_by_key(|c| c.parts.len()) {
        link.cmd(Command::Select(Some(c.id)));
        cam.follow = true;
        cam.focus = Vec3::from_array(c.parts[0].pos);
        *done = true;
    }
}

/// `--pick-test`: press on whatever is under the screen center (the followed creature), drag it
/// sideways for a second and a half, and report how far the creature moved.
fn pick_test(
    args: Res<crate::Args>,
    frames: Res<bevy::diagnostic::FrameCount>,
    window: Single<&Window, With<PrimaryWindow>>,
    snap: Res<Snap>,
    mut fake: ResMut<FakePointer>,
    mut start: Local<Option<(u64, Vec3)>>,
) {
    if !args.pick_test {
        return;
    }
    let f = frames.0;
    let begin = args.frames.saturating_sub(150);
    let center = Vec2::new(window.width() * 0.5, window.height() * 0.5);
    fake.just_pressed = false;
    fake.just_released = false;
    let torso_of = |id: u64| {
        snap.0
            .creatures
            .iter()
            .find(|c| c.id == id)
            .map(|c| Vec3::from_array(c.parts[0].pos))
    };
    if f == begin {
        fake.pos = Some(center);
        fake.just_pressed = true;
        fake.pressed = true;
        if let Some(sel) = &snap.0.selected {
            *start = torso_of(sel.id).map(|p| (sel.id, p));
        }
    } else if f > begin && f < begin + 90 {
        fake.pos = Some(center + Vec2::new((f - begin) as f32 * 3.0, 0.0));
    } else if f == begin + 90 {
        if let Some((id, p0)) = *start {
            match torso_of(id) {
                Some(p1) => println!(
                    "pick-test: creature #{id} torso moved {:.2} m (from {p0:.2} to {p1:.2})",
                    (p1 - p0).length()
                ),
                None => println!("pick-test: creature #{id} is gone"),
            }
        }
        fake.just_released = true;
        fake.pressed = false;
    } else if f == begin + 91 {
        fake.pos = None;
    }
}
