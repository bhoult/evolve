//! Pictures for the MCP server (`evolve_sim::mcp`): when an assistant asks for a screenshot, the
//! panels are hidden and the view turned on the creature it names for a moment, captured, and
//! put back as it was.

use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender};

use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use evolve_sim::Command;
use evolve_sim::mcp::Shot;

use crate::camera::OrbitCam;
use crate::{Args, SimLink, Snap};

pub struct ObservatoryPlugin;

impl Plugin for ObservatoryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, serve_shots);
    }
}

#[derive(Resource)]
pub struct Shots {
    rx: Mutex<Receiver<Shot>>,
    pending: Option<Pending>,
}

impl Shots {
    pub fn new(rx: Receiver<Shot>) -> Self {
        Self { rx: Mutex::new(rx), pending: None }
    }
}

struct Pending {
    reply: Option<Sender<Result<Vec<u8>, String>>>,
    /// The creature to look at.
    creature: Option<u64>,
    /// Frames to wait: for the panels to go and the view to settle, then for the capture.
    wait: u32,
    /// How the view was before, to put back: focus, yaw, pitch, distance, following, panels hidden.
    restore: (Vec3, f32, f32, f32, bool, bool),
    /// The selection to put back, if the picture selected its creature.
    selected: Option<Option<u64>>,
}

/// Frames for the panels to disappear and the view to settle on the creature.
const SETTLE: u32 = 20;

fn serve_shots(
    mut commands: Commands,
    mut shots: ResMut<Shots>,
    mut cam: ResMut<OrbitCam>,
    mut args: ResMut<Args>,
    snap: Res<Snap>,
    link: Res<SimLink>,
) {
    if shots.pending.is_none() {
        let Some(shot) = shots.rx.lock().ok().and_then(|rx| rx.try_recv().ok()) else {
            return;
        };
        if let Some(id) = shot.creature
            && !snap.0.creatures.iter().any(|c| c.id == id)
        {
            let _ = shot.reply.send(Err(format!("no creature #{id} in the world on screen")));
            return;
        }
        let restore = (cam.focus, cam.yaw, cam.pitch, cam.distance, cam.follow, args.no_ui);
        // The picture is of the world, without the panels (the selection stays as it was unless
        // asked for).
        args.no_ui = true;
        let selected = shot.creature.filter(|_| shot.select).map(|id| {
            link.cmd(Command::Select(Some(id)));
            snap.0.selected.as_ref().map(|s| s.id)
        });
        shots.pending = Some(Pending { reply: Some(shot.reply), creature: shot.creature, wait: SETTLE, restore, selected });
    }
    let Some(p) = shots.pending.as_mut() else {
        return;
    };
    if let Some(c) = p.creature.and_then(|id| snap.0.creatures.iter().find(|c| c.id == id)) {
        let size = c.parts.iter().map(|p| p.shape.length()).fold(0.5f32, f32::max);
        cam.follow = false;
        cam.focus = crate::camera::center_of_mass(c);
        cam.distance = (size * 6.0).clamp(4.0, 25.0);
        cam.pitch = 0.4;
    }
    if p.wait > 0 {
        p.wait -= 1;
        return;
    }
    if let Some(reply) = p.reply.take() {
        commands.spawn(Screenshot::primary_window()).observe(move |mut cap: On<ScreenshotCaptured>| {
            let image = &mut cap.event_mut().image;
            let (w, h) = (image.width(), image.height());
            let bgra = matches!(image.texture_descriptor.format, TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb);
            let Some(data) = image.data.take() else {
                let _ = reply.send(Err("the capture came back empty".into()));
                return;
            };
            let _ = reply.send(Ok(half_size_png(w, h, &data, bgra)));
        });
        // Hold the view for the frames the capture takes.
        p.wait = 5;
        return;
    }
    let (focus, yaw, pitch, distance, follow, no_ui) = p.restore;
    cam.focus = focus;
    cam.yaw = yaw;
    cam.pitch = pitch;
    cam.distance = distance;
    cam.follow = follow;
    args.no_ui = no_ui;
    if let Some(was) = p.selected {
        link.cmd(Command::Select(was));
    }
    shots.pending = None;
}

/// The window's 4-byte pixels as a PNG at half the size (smaller for the assistant to read).
fn half_size_png(w: u32, h: u32, data: &[u8], bgra: bool) -> Vec<u8> {
    let (w2, h2) = (w / 2, h / 2);
    let mut rgb = Vec::with_capacity((w2 * h2 * 3) as usize);
    let px = |x: u32, y: u32, k: usize| data[((y * w + x) * 4) as usize + k] as u32;
    for y in 0..h2 {
        for x in 0..w2 {
            for c in 0..3 {
                let k = if bgra { 2 - c } else { c };
                let sum = px(2 * x, 2 * y, k) + px(2 * x + 1, 2 * y, k) + px(2 * x, 2 * y + 1, k) + px(2 * x + 1, 2 * y + 1, k);
                rgb.push((sum / 4) as u8);
            }
        }
    }
    evolve_sim::mcp::png_rgb(w2, h2, &rgb)
}
