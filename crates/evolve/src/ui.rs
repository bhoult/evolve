//! The egui interface: toolbar, parameter panel (generated from `SimParams`), statistics and
//! plots, creature inspector, save/load browser and help.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_egui::input::egui_wants_any_keyboard_input;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use evolve_sim::brain::{input_label, output_label};
use evolve_sim::params::{ParamInfo, ParamKind, ParamRef};
use evolve_sim::snapshot::{CreatureDetail, Snapshot};
use evolve_sim::stats::Sample;
use evolve_sim::wiring::SketchLimits;
use evolve_sim::zoo::ZooMeta;
use evolve_sim::{Command, Control, SimParams, ZooTarget, save};

use crate::audio::SoundSettings;
use crate::camera::OrbitCam;
use crate::director::Director;
use crate::env::{GraphicsSettings, ShadowQuality};
use crate::preview::{PREVIEW_SIZE, Sketch, ZooPreview, sketch};
use crate::sync::{ColorMode, ViewSettings};
use crate::tools::{Tool, ToolState};
use crate::{SimLink, Snap};

pub struct UiPlugin;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlotKind {
    Population,
    Progress,
    Traits,
    Generations,
}

#[derive(Resource)]
pub struct UiState {
    params: SimParams,
    defaults: HashMap<&'static str, f64>,
    /// After a local edit, don't overwrite the panel from the (one-frame-old) snapshot.
    hold_params_until: f64,
    params_filter: String,
    save_name: String,
    show_saves: bool,
    saves: Vec<(PathBuf, SystemTime, u64)>,
    saves_refreshed: f64,
    show_help: bool,
    help_topic: crate::help::Topic,
    show_genome: bool,
    plot: PlotKind,
    confirm_new_world: bool,
    show_zoo: bool,
    zoo_target: ZooTarget,
    zoo_count: u32,
    zoo_sort_newest: bool,
    zoo_selected: Option<u64>,
    /// Compacting the zoo: how many of the best to keep, and whether it's waiting to be confirmed.
    zoo_keep: u32,
    confirm_compact: bool,
    /// When Space was pressed once to pause (real seconds): a second press soon after pauses.
    pause_asked: Option<f64>,
    /// Brain view: shown on request, and on its own while following a creature (unless closed
    /// for that creature).
    show_brain: bool,
    brain_closed_for: Option<u64>,
    /// Which picture of the brain the brain view shows, and how much of the brain each shows.
    brain: BrainView,
    /// Body plans of zoo creatures, read as their rows come into view.
    sketches: Sketches,
    reports: crate::reports::Reports,
}

/// A finished sketch: (zoo id, its body plan, or None if the genome is unreadable).
type SketchDone = (u64, Option<Arc<Sketch>>);
/// Where to ask the loader thread for a sketch, and where it leaves them.
type SketchLoader = (Sender<(PathBuf, u64)>, Arc<Mutex<Vec<SketchDone>>>);

/// Zoo creatures' body plans, read on a thread of its own (unpacking a big brain takes a while,
/// and the zoo window shouldn't stall a frame for it), most recently asked for first.
#[derive(Default)]
struct Sketches {
    done: HashMap<u64, Option<Arc<Sketch>>>,
    asked: HashSet<u64>,
    loader: Option<SketchLoader>,
}

fn read_sketch(dir: &Path, id: u64) -> Option<Arc<Sketch>> {
    evolve_sim::zoo::load_genome(dir, id).ok().map(|g| Arc::new(sketch(&g)))
}

impl Sketches {
    /// Takes in what the loader has finished.
    fn collect(&mut self) {
        if let Some((_, out)) = &self.loader
            && let Ok(mut out) = out.lock()
        {
            self.done.extend(out.drain(..));
        }
    }

    /// A creature's body plan if it has been read, asking for it if not.
    fn get(&mut self, dir: &Path, id: u64) -> Option<Arc<Sketch>> {
        if let Some(sk) = self.done.get(&id) {
            return sk.clone();
        }
        if !self.asked.insert(id) {
            return None;
        }
        if self.loader.is_none() {
            self.loader = start_sketch_loader();
        }
        match &self.loader {
            Some((ask, _)) if ask.send((dir.to_path_buf(), id)).is_ok() => None,
            // No loader thread: read it here.
            _ => {
                let sk = read_sketch(dir, id);
                self.done.insert(id, sk.clone());
                sk
            }
        }
    }
}

fn start_sketch_loader() -> Option<SketchLoader> {
    let (ask, asks) = std::sync::mpsc::channel::<(PathBuf, u64)>();
    let done = Arc::new(Mutex::new(Vec::new()));
    let out = done.clone();
    std::thread::Builder::new()
        .name("evolve-zoo-sketches".into())
        .spawn(move || {
            let mut queue = Vec::new();
            loop {
                if queue.is_empty() {
                    match asks.recv() {
                        Ok(a) => queue.push(a),
                        Err(_) => return,
                    }
                }
                queue.extend(asks.try_iter());
                // Newest first: those are the rows on screen now.
                let Some((dir, id)) = queue.pop() else {
                    continue;
                };
                let sk = read_sketch(&dir, id);
                match out.lock() {
                    Ok(mut out) => out.push((id, sk)),
                    Err(_) => return,
                }
            }
        })
        .ok()?;
    Some((ask, done))
}

/// Zoo table column widths, fixed so the header lines up with the rows scrolling under it.
const ZOO_COLUMNS: [f32; 9] = [90.0, 150.0, 48.0, 56.0, 58.0, 44.0, 46.0, 50.0, 112.0];
/// Height of a zoo table row (the body sketch's).
const ZOO_ROW: f32 = 46.0;

/// One zoo table row: `cell` fills each column in turn.
fn zoo_row(ui: &mut egui::Ui, height: f32, stripe: bool, mut cell: impl FnMut(usize, &mut egui::Ui)) {
    let width: f32 = ZOO_COLUMNS.iter().sum();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    if stripe {
        ui.painter().rect_filled(rect, 2.0, ui.visuals().faint_bg_color);
    }
    let mut x = rect.left();
    for (k, w) in ZOO_COLUMNS.into_iter().enumerate() {
        let r = egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(w, height));
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .id_salt(k)
                .max_rect(r)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        child.set_clip_rect(r.intersect(ui.clip_rect()));
        cell(k, &mut child);
        x += w;
    }
}

impl Default for UiState {
    fn default() -> Self {
        let mut defaults = HashMap::new();
        SimParams::default().visit(&mut |info, r| {
            let v = match r {
                ParamRef::F32(x) => *x as f64,
                ParamRef::U32(x) => *x as f64,
                ParamRef::U64(x) => *x as f64,
                ParamRef::Bool(x) => *x as u8 as f64,
            };
            defaults.insert(info.name, v);
        });
        Self {
            params: SimParams::default(),
            defaults,
            hold_params_until: 0.0,
            params_filter: String::new(),
            save_name: "world".into(),
            show_saves: false,
            saves: Vec::new(),
            saves_refreshed: -100.0,
            show_help: false,
            help_topic: crate::help::Topic::default(),
            show_genome: false,
            plot: PlotKind::Population,
            confirm_new_world: false,
            show_zoo: false,
            zoo_target: ZooTarget::This,
            zoo_count: 1,
            zoo_sort_newest: false,
            zoo_selected: None,
            zoo_keep: 10,
            confirm_compact: false,
            pause_asked: None,
            show_brain: false,
            brain_closed_for: None,
            brain: BrainView::default(),
            sketches: Sketches::default(),
            reports: Default::default(),
        }
    }
}

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiState>()
            .add_systems(
                EguiPrimaryContextPass,
                (ui_system.run_if(|args: Res<crate::Args>| !args.no_ui), busy_banner),
            )
            .add_systems(Startup, |args: Res<crate::Args>, mut st: ResMut<UiState>| {
                st.show_zoo = args.zoo;
                if args.neural_view {
                    st.show_brain = true;
                    st.brain.tab = BrainTab::Neural;
                }
                if let Some(name) = &args.help_page {
                    st.show_help = true;
                    st.help_topic = crate::help::Topic::named(name).unwrap_or_default();
                }
                st.show_genome = args.genome_view;
            })
            .add_systems(Update, shortcuts.run_if(not(egui_wants_any_keyboard_input)));
    }
}

const SPEEDS: [(f32, &str); 7] = [
    (0.25, "¼×"),
    (0.5, "½×"),
    (1.0, "1×"),
    (2.0, "2×"),
    (4.0, "4×"),
    (8.0, "8×"),
    (0.0, "Max"),
];

fn quicksave_path() -> PathBuf {
    save::default_dir().join("quicksave.evo")
}

fn screenshot_dir() -> PathBuf {
    save::default_dir()
        .parent()
        .map(|p| p.join("screenshots"))
        .unwrap_or_else(|| PathBuf::from("screenshots"))
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn take_screenshot(commands: &mut Commands, link: &SimLink) {
    let dir = screenshot_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("evolve-{}.png", unix_time()));
    link.cmd(Command::Select(None));
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
}

#[allow(clippy::too_many_arguments)]
fn shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    snap: Res<Snap>,
    link: Res<SimLink>,
    mut st: ResMut<UiState>,
    mut director: ResMut<Director>,
    mut sound: ResMut<SoundSettings>,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::KeyC) {
        director.toggle();
    }
    if keys.just_pressed(KeyCode::KeyM) {
        sound.muted = !sound.muted;
    }
    // Space easily lands by mistake: pausing takes it twice (running again, once).
    let now = time.elapsed_secs_f64();
    if keys.just_pressed(KeyCode::Space) {
        if snap.0.paused {
            link.0.send(Control::Pause(false));
            st.pause_asked = None;
        } else if st.pause_asked.is_some_and(|t| now - t < PAUSE_CONFIRM) {
            link.0.send(Control::Pause(true));
            st.pause_asked = None;
        } else {
            st.pause_asked = Some(now);
        }
    }
    if st.pause_asked.is_some_and(|t| now - t >= PAUSE_CONFIRM) {
        st.pause_asked = None;
    }
    if keys.just_pressed(KeyCode::Period) {
        link.0.send(Control::StepOnce);
    }
    let i = SPEEDS.iter().position(|(s, _)| *s == snap.0.speed).unwrap_or(2);
    if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::NumpadAdd) {
        link.0.send(Control::Speed(SPEEDS[(i + 1).min(SPEEDS.len() - 1)].0));
    }
    if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::NumpadSubtract) {
        link.0.send(Control::Speed(SPEEDS[i.saturating_sub(1)].0));
    }
    if keys.just_pressed(KeyCode::F5) {
        link.0.send(Control::Save(quicksave_path()));
    }
    if keys.just_pressed(KeyCode::F9) {
        link.0.send(Control::Load(quicksave_path()));
    }
    if keys.just_pressed(KeyCode::F12) {
        take_screenshot(&mut commands, &link);
    }
    if keys.just_pressed(KeyCode::KeyH) {
        st.show_help = !st.show_help;
    }
}

#[allow(clippy::too_many_arguments)]
/// While the world is held up (saving, loading, or anything else keeping new pictures of it from
/// coming), says so across the top of the screen, so a still picture doesn't look like a broken
/// app.
/// Seconds after a first press of Space in which a second one pauses.
const PAUSE_CONFIRM: f64 = 2.0;

fn busy_banner(
    mut contexts: EguiContexts,
    link: Res<SimLink>,
    snap: Res<crate::Snap>,
    st: Res<UiState>,
    time: Res<Time<Real>>,
    mut seen: Local<(usize, f64)>,
) -> Result {
    let now = time.elapsed_secs_f64();
    let latest = std::sync::Arc::as_ptr(&snap.0) as usize;
    if latest != seen.0 {
        *seen = (latest, now);
    }
    let still = now - seen.1;
    let text = match link.0.busy() {
        Some((what, for_)) => format!("{what}… {:.0} s", for_.as_secs_f32()),
        None if st.pause_asked.is_some() && !snap.0.paused => "Press Space again to pause".to_string(),
        None if still > 1.5 && !snap.0.paused => format!("The simulation is busy… {still:.0} s"),
        None => return Ok(()),
    };
    let ctx = contexts.ctx_mut()?;
    egui::Area::new(egui::Id::new("busy banner"))
        .anchor(egui::Align2::CENTER_TOP, [0.0, 48.0])
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(egui::RichText::new(text).size(20.0));
            });
        });
    Ok(())
}

fn ui_system(
    mut contexts: EguiContexts,
    snap: Res<Snap>,
    link: Res<SimLink>,
    mut st: ResMut<UiState>,
    mut tools: ResMut<ToolState>,
    mut view: ResMut<ViewSettings>,
    mut cam: ResMut<OrbitCam>,
    mut preview: ResMut<ZooPreview>,
    mut director: ResMut<Director>,
    mut sound: ResMut<SoundSettings>,
    mut gfx: ResMut<GraphicsSettings>,
    diag: Res<DiagnosticsStore>,
    time: Res<Time>,
    mut rec: ResMut<crate::recording::Recorder>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let snap = &*snap.0;
    let st = &mut *st;
    let now = time.elapsed_secs_f64();
    if now > st.hold_params_until && snap.terrain.res >= 3 {
        st.params = snap.params.clone();
    }
    let fps = diag
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    // Below the toolbar, or at the very top when the director has hidden it.
    let top = if director.active && !director.panels { 10.0 } else { 56.0 };
    clock(ctx, snap, top);
    if snap.paused {
        paused_badge(ctx, top + 34.0);
    }
    if director.active {
        director_overlay(ctx, &mut director, &mut sound, &mut rec, now);
        if !director.panels {
            st.brain.ask(&link, None);
            return Ok(());
        }
    }
    toolbar(ctx, snap, &link, st, &mut tools, &mut view, &mut cam, &mut director, &mut sound, &mut rec, fps);
    params_panel(ctx, &link, st, &mut gfx, now);
    stats_panel(ctx, snap, st, fps, &link, &mut cam);
    events_panel(ctx, snap, rec.fresh_notice());
    // The runner sketches the brain only while the brain view is open, to what it shows.
    let mut brain_wants = None;
    if let Some(sel) = &snap.selected {
        inspector(ctx, sel, &link, &mut cam, &mut st.show_brain, &mut st.show_genome);
        if st.show_genome {
            crate::genome_view::genome_window(ctx, sel, &mut st.show_genome, &snap.params);
        }
        let following = cam.follow && st.brain_closed_for != Some(sel.id);
        if st.show_brain || following {
            let mut open = true;
            brain_window(ctx, sel, &mut open, &mut st.brain, now);
            if open {
                brain_wants = Some(st.brain.wants(sel));
            } else {
                st.show_brain = false;
                st.brain_closed_for = Some(sel.id);
            }
        }
    }
    st.brain.ask(&link, brain_wants);
    saves_window(ctx, &link, st, now);
    zoo_window(ctx, snap, &link, st, &mut tools, &mut preview);
    crate::help::help_window(ctx, &mut st.show_help, &mut st.help_topic, &st.params);
    crate::reports::reports_window(ctx, &mut st.reports, snap.world_uid, now);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn toolbar(
    ctx: &mut egui::Context,
    snap: &Snapshot,
    link: &SimLink,
    st: &mut UiState,
    tools: &mut ToolState,
    view: &mut ViewSettings,
    cam: &mut OrbitCam,
    director: &mut Director,
    sound: &mut SoundSettings,
    rec: &mut crate::recording::Recorder,
    fps: f64,
) {
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        if !link.0.alive() {
            ui.colored_label(
                egui::Color32::from_rgb(255, 110, 90),
                "The simulation stopped with an error (the reason is in the terminal). Restart evolve to continue.",
            );
        }
        ui.horizontal(|ui| {
            let label = if snap.paused { "▶ Run" } else { "⏸ Pause" };
            if ui.button(label).on_hover_text("Space").clicked() {
                link.0.send(Control::Pause(!snap.paused));
            }
            if ui.button("Step").on_hover_text("Advance one physics step (.)").clicked() {
                link.0.send(Control::StepOnce);
            }
            ui.separator();
            ui.label("Speed");
            for (s, name) in SPEEDS {
                if ui
                    .selectable_label(snap.speed == s, name)
                    .on_hover_text("+ / - keys")
                    .clicked()
                {
                    link.0.send(Control::Speed(s));
                }
            }
            ui.separator();
            ui.label(format!(
                "t = {}   {:.0} steps/s ({:.1}× real time)   {:.0} fps",
                fmt_time(snap.time),
                snap.steps_per_sec,
                snap.steps_per_sec / 60.0,
                fps
            ));
            if let Some(low) = snap.low_priority.as_ref().filter(|s| s.share < 1.0 || s.gpu_given_back) {
                ui.label(format!("low priority {:.0}%", low.share * 100.0)).on_hover_text(low.to_string());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Help").clicked() {
                    st.show_help = !st.show_help;
                }
                if ui
                    .selectable_label(st.reports.open, "Reports")
                    .on_hover_text("Graphs of how every world has done over time: travel, lifespans, births and sky spawns...")
                    .clicked()
                {
                    st.reports.open = !st.reports.open;
                }
                record_button(ui, rec);
                sound_controls(ui, sound);
                if ui
                    .selectable_label(director.active, "Director")
                    .on_hover_text("Watch hands-free: the camera finds fights, births, swimmers, sunsets and storms on its own (C)")
                    .clicked()
                {
                    director.toggle();
                }
                if st.confirm_new_world {
                    if ui.button("Cancel").clicked() {
                        st.confirm_new_world = false;
                    }
                    if ui.button("Yes, start over").clicked() {
                        link.0.send(Control::NewWorld(Box::new(st.params.clone())));
                        st.confirm_new_world = false;
                    }
                    ui.label("Discard this world?");
                } else if ui
                    .button("New world")
                    .on_hover_text("Start over with the current parameters")
                    .clicked()
                {
                    st.confirm_new_world = true;
                }
                if ui
                    .button(format!("Zoo ({})", snap.zoo.len()))
                    .on_hover_text("The best creatures from every world, kept across sessions")
                    .clicked()
                {
                    st.show_zoo = !st.show_zoo;
                }
                if ui.button("Save / Load").on_hover_text("F5 quick save, F9 quick load").clicked() {
                    st.show_saves = !st.show_saves;
                    st.saves_refreshed = -100.0;
                }
            });
        });
        ui.horizontal_wrapped(|ui| {
            for t in Tool::ALL.into_iter().filter(|t| !t.is_power()) {
                if ui
                    .selectable_label(tools.tool == t, t.label())
                    .on_hover_text(t.hint())
                    .clicked()
                {
                    tools.tool = t;
                }
            }
            ui.separator();
            // The acts of god, in a menu of their own: the one in hand shows on the button.
            let title = if tools.tool.is_power() { format!("Powers: {} ⏷", tools.tool.label()) } else { "Powers ⏷".to_string() };
            let button = egui::Button::new(title).selected(tools.tool.is_power());
            egui::containers::menu::MenuButton::from_button(button).ui(ui, |ui| {
                for t in Tool::ALL.into_iter().filter(|t| t.is_power()) {
                    if ui.selectable_label(tools.tool == t, t.label()).on_hover_text(t.hint()).clicked() {
                        tools.tool = t;
                        ui.close();
                    }
                }
                ui.separator();
                let storm = if snap.storm_left > 0.0 {
                    let s = snap.storm_left as u32;
                    format!("Calm the storm ({}:{:02} left)", s / 60, s % 60)
                } else {
                    "Call a storm".to_string()
                };
                if ui
                    .button(storm)
                    .on_hover_text("A storm over the whole island for four minutes: wind, rain and lightning")
                    .clicked()
                {
                    link.cmd(if snap.storm_left > 0.0 { Command::CalmStorm } else { Command::SummonStorm { seconds: 240.0 } });
                    ui.close();
                }
            });
            if snap.storm_left > 0.0 {
                let s = snap.storm_left as u32;
                ui.label(egui::RichText::new(format!("storm {}:{:02}", s / 60, s % 60)).weak());
            }
            ui.separator();
            match tools.tool {
                Tool::Plant | Tool::Raise | Tool::Lower => {
                    ui.add(egui::Slider::new(&mut tools.brush, 0.5..=15.0).text("brush m"));
                }
                Tool::Rock | Tool::Boulder => {
                    ui.add(egui::Slider::new(&mut tools.obstacle_size, 0.5..=10.0).text("size m"));
                }
                Tool::Totem => {
                    use evolve_sim::acts::TotemKind;
                    for (k, name) in [(TotemKind::Love, "Love"), (TotemKind::War, "War"), (TotemKind::Fear, "Fear")] {
                        ui.selectable_value(&mut tools.totem, k, name);
                    }
                }
                _ => {}
            }
            ui.separator();
            egui::ComboBox::from_label("color")
                .selected_text(view.color_mode.label())
                .show_ui(ui, |ui| {
                    for m in ColorMode::ALL {
                        ui.selectable_value(&mut view.color_mode, m, m.label());
                    }
                });
            ui.toggle_value(&mut cam.follow, "Follow selected")
                .on_hover_text("F");
            if let Some((_, name, count)) = &tools.zoo_place {
                ui.separator();
                ui.colored_label(
                    egui::Color32::from_rgb(240, 200, 90),
                    format!("Click the ground to drop {count} x {name} (Esc cancels)"),
                );
            }
        });
    });
}

fn fmt_time(t: f64) -> String {
    let s = t as u64;
    if s >= 3600 {
        format!("{}h{:02}m{:02}s", s / 3600, (s / 60) % 60, s % 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

/// A slider's label: the setting's name, said so that which way is faster can't be misread
/// (`wear_recovery` up means slower healing), with the unit of a duration.
fn param_label(info: &ParamInfo) -> String {
    let name = match info.name {
        "wear_recovery" => "wear heal time",
        "earth_recovery" => "earth heal time",
        "earth_rest" => "earth rest before healing",
        n => n,
    };
    let unit = match info.doc.split_whitespace().next() {
        Some("Minutes") => " (min)",
        Some("Seconds") => " (s)",
        _ => "",
    };
    format!("{}{unit}", name.replace('_', " "))
}

fn param_widget(ui: &mut egui::Ui, info: &ParamInfo, r: ParamRef<'_>, default: f64) -> bool {
    let mut label = param_label(info);
    if info.kind == ParamKind::NewWorld {
        label.push_str(" ⟳");
    }
    let resp = match r {
        ParamRef::F32(x) => {
            let log = info.min >= 0.0 && default > 0.0 && info.max / default >= 30.0;
            let mut s = egui::Slider::new(x, info.min as f32..=info.max as f32).text(label);
            if log {
                s = s.logarithmic(true).smallest_positive(default / 100.0);
            }
            ui.add(s)
        }
        ParamRef::U32(x) => {
            let mut s = egui::Slider::new(x, info.min as u32..=info.max as u32).text(label);
            if info.min >= 1.0 && info.max / info.min >= 100.0 {
                s = s.logarithmic(true);
            }
            ui.add(s)
        }
        ParamRef::U64(x) => ui.add(egui::DragValue::new(x).range(info.min as u64..=info.max as u64).prefix(format!("{label}: "))),
        ParamRef::Bool(x) => ui.checkbox(x, label),
    };
    let mut tip = info.doc.trim().to_string();
    if info.kind == ParamKind::NewWorld {
        tip.push_str("\n\n⟳ Takes effect when you press Regenerate terrain (or New world).");
    }
    tip.push_str(&format!("\nDefault: {default}  (setting: {})", info.name));
    resp.on_hover_text(tip).changed()
}

fn params_panel(ctx: &mut egui::Context, link: &SimLink, st: &mut UiState, gfx: &mut GraphicsSettings, now: f64) {
    egui::SidePanel::left("params")
        .resizable(true)
        .default_width(330.0)
        .show(ctx, |ui| {
            ui.heading("Parameters");
            ui.horizontal(|ui| {
                ui.label("Filter");
                ui.text_edit_singleline(&mut st.params_filter);
            });
            let mut changed = false;
            ui.horizontal(|ui| {
                if ui
                    .button("Regenerate terrain")
                    .on_hover_text("Rebuild the landscape from the World ⟳ settings, keeping all life")
                    .clicked()
                {
                    link.cmd(Command::RegenerateTerrain);
                }
                if ui.button("Defaults").on_hover_text("Reset every parameter").clicked() {
                    st.params = SimParams::default();
                    changed = true;
                }
            });
            egui::CollapsingHeader::new("Graphics").show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut gfx.clouds, "clouds");
                    ui.checkbox(&mut gfx.rain, "rain");
                    ui.checkbox(&mut gfx.lightning, "lightning");
                    ui.checkbox(&mut gfx.bloom, "bloom");
                    ui.checkbox(&mut gfx.sway, "swaying plants");
                });
                ui.horizontal(|ui| {
                    ui.label("shadows");
                    for (q, name) in [(ShadowQuality::Off, "off"), (ShadowQuality::Low, "low"), (ShadowQuality::High, "high")] {
                        ui.selectable_value(&mut gfx.shadows, q, name);
                    }
                    ui.checkbox(&mut gfx.soft_shadows, "soft (by weather)");
                });
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut gfx.flow, "flowing water");
                    ui.checkbox(&mut gfx.lod_tint, "tint terrain detail levels")
                        .on_hover_text("Ground drawn coarser far away is tinted green, yellow, orange and red as its detail halves");
                });
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.spacing_mut().slider_width = 150.0;
                let mut groups: Vec<&'static str> = Vec::new();
                st.params.visit(&mut |info, _| {
                    if groups.last() != Some(&info.group) {
                        groups.push(info.group);
                    }
                });
                let filter = st.params_filter.to_lowercase();
                let defaults = &st.defaults;
                // The three sources of creatures from the sky are weights; show the share each gets.
                let sky = {
                    let p = &st.params;
                    let w = [p.rain_from_world, p.rain_from_zoo, p.rain_from_random].map(|x| x.max(0.0));
                    let total: f32 = w.iter().sum();
                    if total > 0.0 { w.map(|x| x / total) } else { [0.0, 0.0, 1.0] }
                };
                for group in groups {
                    let open = !filter.is_empty() || matches!(group, "World" | "Population" | "Brain");
                    egui::CollapsingHeader::new(group)
                        .default_open(open)
                        .open(if filter.is_empty() { None } else { Some(true) })
                        .show(ui, |ui| {
                            st.params.visit(&mut |info, r| {
                                if info.group != group {
                                    return;
                                }
                                if !filter.is_empty()
                                    && !info.name.contains(&filter)
                                    && !info.doc.to_lowercase().contains(&filter)
                                {
                                    return;
                                }
                                let d = defaults.get(info.name).copied().unwrap_or(0.0);
                                changed |= param_widget(ui, info, r, d);
                                let share = match info.name {
                                    "rain_from_world" => Some(sky[0]),
                                    "rain_from_zoo" => Some(sky[1]),
                                    "rain_from_random" => Some(sky[2]),
                                    _ => None,
                                };
                                if let Some(x) = share {
                                    ui.label(format!("    = {:.0}% of creatures from the sky", 100.0 * x));
                                }
                            });
                        });
                }
            });
            if changed {
                st.hold_params_until = now + 0.5;
                link.cmd(Command::SetParams(Box::new(st.params.clone())));
            }
        });
}

fn stats_panel(
    ctx: &mut egui::Context,
    snap: &Snapshot,
    st: &mut UiState,
    fps: f64,
    link: &SimLink,
    cam: &mut OrbitCam,
) {
    egui::SidePanel::right("stats")
        .resizable(true)
        .default_width(330.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("World");
                let s = &snap.stats;
                let n = snap.creatures.len();
                let mean_gen = if n > 0 {
                    snap.creatures.iter().map(|c| c.generation as f32).sum::<f32>() / n as f32
                } else {
                    0.0
                };
                egui::Grid::new("stats_grid").num_columns(2).striped(true).show(ui, |ui| {
                    let mut row = |k: &str, v: String| {
                        ui.label(k);
                        ui.label(v);
                        ui.end_row();
                    };
                    row("Creatures", format!("{n}  (rain below {}, cap {})", snap.params.min_population, snap.params.max_population));
                    row("Plants", format!("{}", snap.plants.len()));
                    row("Meat", format!("{}", snap.meat.len()));
                    row("Obstacles", format!("{}", snap.obstacles.len()));
                    row("Born / rained", format!("{} / {}", s.births, s.rained));
                    row(
                        "Deaths",
                        format!(
                            "{} starved, {} killed, {} old, {} drowned, {} thirst, {} other",
                            s.deaths_starved,
                            s.deaths_killed,
                            s.deaths_old,
                            s.deaths_drowned,
                            s.deaths_thirst,
                            s.deaths_other
                        ),
                    );
                    row("Generation", format!("mean {mean_gen:.1}, max {}", s.max_generation));
                    row("Species (approx.)", format!("{}", snap.history.last().map(|h| h.species).unwrap_or(0)));
                    row("Gene pool", format!("{} genomes, best score {:.1}", snap.pool_size, snap.pool_best));
                    row(
                        "Recent lifetimes",
                        format!(
                            "{:.0} food, {:.0} m walked, {:.0} s, {:.2} children",
                            s.life_food, s.life_distance, s.life_age, s.life_children
                        ),
                    );
                });

                ui.add_space(6.0);
                species_list(ui, snap, link, cam);
                leaders_list(ui, snap, link, cam);
                if snap.worlds.len() > 1 {
                    ui.add_space(6.0);
                    worlds_list(ui, snap);
                }

                ui.add_space(6.0);
                ui.heading("History");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut st.plot, PlotKind::Population, "Population");
                    ui.selectable_value(&mut st.plot, PlotKind::Progress, "Progress");
                    ui.selectable_value(&mut st.plot, PlotKind::Traits, "Traits");
                    ui.selectable_value(&mut st.plot, PlotKind::Generations, "Generations");
                });
                history_plot(ui, &snap.history, st.plot);

                ui.add_space(6.0);
                ui.heading("Performance");
                let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
                let total_steps: f32 = snap.worlds.iter().map(|w| w.steps_per_sec).sum();
                egui::Grid::new("perf_grid").num_columns(2).show(ui, |ui| {
                    let p = &snap.perf;
                    let mut row = |k: &str, v: String| {
                        ui.label(k);
                        ui.label(v);
                        ui.end_row();
                    };
                    row("Render", format!("{fps:.0} fps"));
                    row(
                        "CPU",
                        format!(
                            "{:.0}% of one core ({:.0}% of all {cores})",
                            snap.cpu_percent,
                            snap.cpu_percent / cores as f32
                        ),
                    );
                    row(
                        "All worlds",
                        format!("{total_steps:.0} steps/s ({} worlds)", snap.worlds.len().max(1)),
                    );
                    row("This world", format!("{:.0} steps/s", snap.steps_per_sec));
                    // Every phase of a step, so the slow parts show.
                    for (name, ms) in evolve_sim::sim::PHASE_NAMES.iter().zip(&p.phases) {
                        row(&format!("  {name}"), format!("{ms:.2} ms/step"));
                    }
                    row(
                        "  physics worlds",
                        format!("slowest {:.2} ms, all {:.1} ms", p.shard_slowest_ms, p.shard_total_ms),
                    );
                    row(
                        "  for the screen",
                        format!("snapshot {:.2} ms, terrain copy {:.2} ms", p.publish_ms, p.terrain_copy_ms),
                    );
                    row("Brains run on", snap.brain_backend.clone());
                    if let Some(low) = &snap.low_priority {
                        row("Low priority", low.to_string());
                    }
                    if snap.brain_memory_bytes > 0 {
                        let all_busy: f32 = snap.worlds.iter().map(|w| w.gpu_busy).sum();
                        row(
                            "  GPU busy (brains)",
                            format!(
                                "{:.0}% all worlds ({:.0}% this one, {:.2} ms per brain tick)",
                                all_busy * 100.0,
                                p.gpu_busy * 100.0,
                                p.gpu_ms
                            ),
                        );
                        if snap.params.gpu_target > 0.0 {
                            row(
                                "  brain price",
                                format!(
                                    "×{:.2}, steering toward {:.0}% GPU at Max speed",
                                    snap.brain_price,
                                    snap.params.gpu_target * 100.0
                                ),
                            );
                        }
                        row("  waiting on GPU", format!("{:.2} ms per brain tick", p.gpu_wait_ms));
                        row("  brain memory", format!("{:.0} MB", snap.brain_memory_bytes as f64 / 1e6));
                    }
                });
            });
        });
}

fn history_plot(ui: &mut egui::Ui, h: &[Sample], kind: PlotKind) {
    let series = |f: &dyn Fn(&Sample) -> f64| -> PlotPoints<'static> {
        h.iter().map(|s| [s.time as f64 / 60.0, f(s)]).collect()
    };
    Plot::new(("history", kind as u8))
        .height(190.0)
        .legend(Legend::default())
        .allow_scroll(false)
        .x_axis_label("minutes")
        .show(ui, |p| match kind {
            PlotKind::Population => {
                let rgb = egui::Color32::from_rgb;
                p.line(Line::new("creatures", series(&|s| s.population as f64)).color(rgb(240, 170, 60)));
                p.line(Line::new("plants / 10", series(&|s| s.plants as f64 / 10.0)).color(rgb(90, 190, 80)));
                p.line(Line::new("meat", series(&|s| s.meat as f64)).color(rgb(200, 70, 60)));
                p.line(Line::new("species", series(&|s| s.species as f64)).color(rgb(170, 120, 230)));
            }
            PlotKind::Progress => {
                p.line(Line::new("food per life", series(&|s| s.life_food as f64)));
                p.line(Line::new("meters per life", series(&|s| s.life_distance as f64)));
            }
            PlotKind::Traits => {
                p.line(Line::new("neurons", series(&|s| s.mean_neurons as f64)));
                p.line(Line::new("limbs", series(&|s| s.mean_limbs as f64)));
                p.line(Line::new("diet x10 (meat)", series(&|s| s.mean_diet as f64 * 10.0)));
                p.line(Line::new("mass / 10 kg", series(&|s| s.mean_mass as f64 / 10.0)));
            }
            PlotKind::Generations => {
                p.line(Line::new("mean generation", series(&|s| s.mean_generation as f64)));
                p.line(Line::new("max generation", series(&|s| s.max_generation as f64)));
            }
        });
}

/// The world's latest events, and news about a recording.
fn events_panel(ctx: &mut egui::Context, snap: &Snapshot, notice: Option<&str>) {
    if snap.events.is_empty() && notice.is_none() {
        return;
    }
    egui::TopBottomPanel::bottom("events").show(ctx, |ui| {
        if let Some(n) = notice {
            ui.label(egui::RichText::new(n).small().strong());
        }
        for e in snap.events.iter().rev().take(if notice.is_some() { 1 } else { 2 }) {
            ui.label(egui::RichText::new(e).small());
        }
    });
}

fn value_color(v: f32) -> egui::Color32 {
    let v = v.clamp(-1.0, 1.0);
    let base = 70.0;
    if v >= 0.0 {
        egui::Color32::from_rgb((base + 185.0 * v) as u8, (base + 90.0 * v) as u8, base as u8)
    } else {
        let v = -v;
        egui::Color32::from_rgb(base as u8, (base + 80.0 * v) as u8, (base + 185.0 * v) as u8)
    }
}

fn activity(ui: &mut egui::Ui, values: &[f32], label: impl Fn(usize) -> String) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
        for (i, v) in values.iter().enumerate() {
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(11.0, 11.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, value_color(*v));
            resp.on_hover_text(format!("{}: {:+.2}", label(i), v));
        }
    });
}

fn bar(ui: &mut egui::Ui, label: &str, value: f32, max: f32) {
    let frac = if max > 0.0 { (value / max).clamp(0.0, 1.0) } else { 0.0 };
    ui.add(
        egui::ProgressBar::new(frac)
            .desired_height(14.0)
            .text(format!("{label} {value:.0} / {max:.0}")),
    );
}

fn inspector(
    ctx: &mut egui::Context,
    c: &CreatureDetail,
    link: &SimLink,
    cam: &mut OrbitCam,
    show_brain: &mut bool,
    show_genome: &mut bool,
) {
    egui::Window::new(format!("Creature #{}", c.id))
        .id(egui::Id::new("inspector"))
        .default_pos([360.0, 90.0])
        .default_width(320.0)
        .resizable(true)
        .show(ctx, |ui| {
            let origin = match c.parents {
                Some((a, b)) if a == b => format!("budded from #{a}"),
                Some((a, b)) => format!("child of #{a} and #{b}"),
                None if c.rained && c.generation > 0 => "fell from the sky, bred from the gene pool".to_string(),
                None if c.rained => "fell from the sky, random genes".to_string(),
                None => "placed by hand".to_string(),
            };
            ui.label(format!("Generation {}, {}, age {}", c.generation, origin, fmt_time(c.age as f64)));
            bar(ui, "energy", c.energy, c.max_energy);
            bar(ui, "health", c.health, c.max_health);
            bar(ui, "breath %", c.oxygen * 100.0, 100.0);
            bar(ui, "water %", c.hydration * 100.0, 100.0);
            bar(ui, "mating drive %", c.mate_drive * 100.0, 100.0);
            let calling = ((c.outputs.get(evolve_sim::brain::output::CALL).copied().unwrap_or(-1.0) - 0.2) / 0.8).max(0.0);
            bar(ui, "calling %", calling * 100.0, 100.0);
            ui.horizontal(|ui| {
                if ui.button("Follow").clicked() {
                    cam.follow = true;
                }
                if ui
                    .selectable_label(*show_brain, "Brain view")
                    .on_hover_text("A live picture of its neural network (opens by itself while following)")
                    .clicked()
                {
                    *show_brain = !*show_brain;
                }
                if ui
                    .selectable_label(*show_genome, "Genome")
                    .on_hover_text("Its genes, and what each one does")
                    .clicked()
                {
                    *show_genome = !*show_genome;
                }
                if ui.button("Feed").on_hover_text("Fill energy and health").clicked() {
                    link.cmd(Command::FeedCreature(c.id));
                }
                let blessed = c.heaven.is_some();
                let tip = match c.heaven {
                    Some(left) => format!("Glows and can't die; changes (remade from a mutated copy of its genome) in {}. Click to lift the blessing.", crate::help::dur(left.max(0.0))),
                    None => "It glows, can't die (but by Kill or Remove), and every hour of simulated time it's remade from a mutated copy of its genome (heaven_mutation_interval)".to_string(),
                };
                if ui.selectable_label(blessed, "Blessing of heaven").on_hover_text(tip).clicked() {
                    link.cmd(Command::Heaven { id: c.id, on: !blessed });
                }
                if ui
                    .button("Clone")
                    .on_hover_text("Drop a mutated copy from the sky")
                    .clicked()
                {
                    link.cmd(Command::SpawnCreature {
                        pos: None,
                        from: Some(c.id),
                        count: 1,
                    });
                }
                if ui.button("Kill").on_hover_text("Dies and leaves meat").clicked() {
                    link.cmd(Command::KillCreature(c.id));
                }
                if ui.button("Remove").clicked() {
                    link.cmd(Command::RemoveCreature(c.id));
                }
                if ui
                    .button("Add to zoo")
                    .on_hover_text("Keep this creature's genome in the zoo (pinned)")
                    .clicked()
                {
                    link.0.send(Control::ZooAdd(c.id));
                }
            });
            ui.separator();
            let t = &c.traits;
            egui::Grid::new("traits").num_columns(2).striped(true).show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                };
                row(
                    "Body",
                    format!(
                        "{:.0} kg, size {:.2}: {} body segment{}, {} limb segments",
                        c.mass,
                        t.size,
                        c.trunk,
                        if c.trunk == 1 { "" } else { "s" },
                        c.limbs
                    ),
                );
                row("Joints", format!("{} hinges, {} swivels", c.hinges, c.swivels));
                row("Diet", format!("{:.0}% meat", t.diet * 100.0));
                row("Leanings", format!("aggression {:+.2}, mating {:+.2}, appetite {:+.2}", t.aggression, t.libido, t.appetite));
                row("Armor / lungs", format!("{:.2} / {:.2}", t.armor, t.breath));
                row("Rhythm", format!("{:.2} Hz", t.cpg_freq));
                row(
                    "Brain",
                    format!(
                        "{} neurons, {} connections, costs {:.4} energy/s",
                        c.neurons,
                        if c.synapses >= 1_000_000 {
                            format!("{:.2}M", c.synapses as f64 / 1e6)
                        } else {
                            format!("{:.1}k", c.synapses as f64 / 1e3)
                        },
                        c.brain_cost
                    ),
                );
                row("Metabolism", format!("{:.3} energy/s at rest", c.basal_cost));
                row(
                    "Learning",
                    format!(
                        "rate {:.2}, explores {:.2}, remembers {:.1} s, seeks {:.2}",
                        t.plasticity, t.explore, t.trace_time, t.seek
                    ),
                );
                row(
                    "Learned",
                    format!(
                        "{:.0}% of its inherited wiring (offsets {:.3} vs weights {:.3})",
                        100.0 * c.learned / c.inherited.max(1e-6),
                        c.learned,
                        c.inherited
                    ),
                );
                row(
                    "Lifetime",
                    format!(
                        "ate {:.0} plant + {:.0} meat, walked {:.0} m",
                        c.life.plant_eaten, c.life.meat_eaten, c.life.distance
                    ),
                );
                row(
                    "Record",
                    format!("{} children, {} kills, {} bites", c.life.children, c.life.kills, c.life.attacks),
                );
                row("Exertion", format!("{:.2} W per kg of body", c.exertion));
                row("Holding", c.holding.clone().unwrap_or_else(|| "nothing".to_string()));
                let [eat, hunt, breed, travel] = c.score;
                row(
                    "Score",
                    format!(
                        "{:.1}: eating {eat:.1}, hunting {hunt:.1}, breeding {breed:.1}, travelling {travel:.1}",
                        eat + hunt + breed + travel
                    ),
                );
                let v = t.voice;
                row(
                    "Voice",
                    format!(
                        "{:.0} Hz, timbre {:.2}, rasp {:.2}, warble {:.2}",
                        110.0 * 10f32.powf(v[0].clamp(0.0, 1.0)),
                        v[1],
                        v[2],
                        v[3]
                    ),
                );
            });
            ui.separator();
            ui.label("Brain activity (hover a cell; orange positive, blue negative)");
            ui.label(egui::RichText::new("Senses").small());
            activity(ui, &c.inputs, input_label);
            const SHOWN: usize = 160;
            let more = c.hidden.len().saturating_sub(SHOWN);
            let title = if more > 0 {
                format!("Hidden (first {SHOWN} of {})", c.hidden.len())
            } else {
                format!("Hidden ({})", c.hidden.len())
            };
            ui.label(egui::RichText::new(title).small());
            activity(ui, &c.hidden[..c.hidden.len().min(SHOWN)], |i| format!("neuron {i}"));
            ui.label(egui::RichText::new("Actions").small());
            activity(ui, &c.outputs, output_label);
            if ui.button("Deselect").clicked() {
                link.cmd(Command::Select(None));
                cam.follow = false;
            }
        });
}

fn saves_window(ctx: &mut egui::Context, link: &SimLink, st: &mut UiState, now: f64) {
    if !st.show_saves {
        return;
    }
    let dir = save::default_dir();
    if now - st.saves_refreshed > 2.0 {
        st.saves = save::list(&dir);
        st.saves_refreshed = now;
    }
    let mut open = true;
    egui::Window::new("Save / Load")
        .open(&mut open)
        .default_pos([420.0, 140.0])
        .default_width(380.0)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new(dir.display().to_string()).small().weak());
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut st.save_name);
                if ui.button("Save").clicked() {
                    let name: String = st
                        .save_name
                        .chars()
                        .map(|ch| if ch.is_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' })
                        .collect();
                    let name = if name.is_empty() { "world".to_string() } else { name };
                    link.0.send(Control::Save(dir.join(format!("{name}.{}", save::EXTENSION))));
                    st.saves_refreshed = now - 1.5;
                }
            });
            ui.separator();
            if st.saves.is_empty() {
                ui.label("No saves yet.");
            }
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                for (path, modified, size) in &st.saves {
                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            link.0.send(Control::Load(path.clone()));
                        }
                        let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                        let age = SystemTime::now()
                            .duration_since(*modified)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let age = if age < 120 {
                            format!("{age}s ago")
                        } else if age < 7200 {
                            format!("{}m ago", age / 60)
                        } else {
                            format!("{}h ago", age / 3600)
                        };
                        ui.label(format!("{name}   {:.1} MB, {age}", *size as f64 / 1e6));
                    });
                }
            });
        });
    st.show_saves = open;
}

fn species_list(ui: &mut egui::Ui, snap: &Snapshot, link: &SimLink, cam: &mut OrbitCam) {
    egui::CollapsingHeader::new(format!("Species ({})", snap.species.len()))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Creatures that can interbreed. Click one to follow its best-scoring member.")
                    .small()
                    .weak(),
            );
            egui::Grid::new("species_grid").num_columns(6).striped(true).show(ui, |ui| {
                for h in ["", "alive", "gen", "meat", "limbs", "neurons"] {
                    ui.label(egui::RichText::new(h).small().strong());
                }
                ui.end_row();
                for s in snap.species.iter().take(12) {
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(28.0, 14.0), egui::Sense::click());
                    let [r, g, b] = s.color.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8);
                    ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgb(r, g, b));
                    let resp = resp.on_hover_text(format!(
                        "{:.0} kg on average, {} children among the living",
                        s.mean_mass, s.births
                    ));
                    let clicked = resp.clicked()
                        | ui.selectable_label(false, format!("{}", s.count)).clicked()
                        | ui.selectable_label(false, format!("{}", s.max_generation)).clicked()
                        | ui.selectable_label(false, format!("{:.0}%", s.mean_diet * 100.0)).clicked()
                        | ui.selectable_label(false, format!("{:.1}", s.mean_limbs)).clicked()
                        | ui.selectable_label(false, format!("{:.0}", s.mean_neurons)).clicked();
                    if clicked {
                        link.cmd(Command::Select(Some(s.representative)));
                        cam.follow = true;
                    }
                    ui.end_row();
                }
            });
            if snap.species.len() > 12 {
                ui.label(egui::RichText::new(format!("and {} smaller ones", snap.species.len() - 12)).small());
            }
        });
}

/// Who's ahead right now, in each way a creature can stand out; click to go and follow it.
fn leaders_list(ui: &mut egui::Ui, snap: &Snapshot, link: &SimLink, cam: &mut OrbitCam) {
    use evolve_sim::snapshot::CreatureView;
    type Measure = fn(&CreatureView) -> Option<f32>;
    // (title, how it's measured, the number shown)
    let records: [(&str, Measure, fn(f32) -> String); 11] = [
        ("Top score", |c| Some(c.score), |v| format!("{v:.1}")),
        ("Oldest", |c| Some(c.age), |v| fmt_time(v as f64)),
        ("Most evolved", |c| Some(c.generation as f32), |v| format!("generation {v:.0}")),
        ("Biggest brain", |c| Some(c.neurons as f32), |v| format!("{v:.0} neurons")),
        // Net progress over its life so far, once it has lived long enough to judge.
        ("Fastest", |c| (c.age > 30.0).then(|| c.distance / c.age), |v| format!("{v:.2} m/s")),
        ("Farthest travelled", |c| Some(c.distance), |v| format!("{v:.0} m")),
        ("Biggest eater", |c| Some(c.food), |v| format!("{v:.0} energy")),
        ("Most children", |c| Some(c.children as f32), |v| format!("{v:.0}")),
        ("Best hunter", |c| Some(c.kills as f32), |v| format!("{v:.0} kills")),
        ("Heaviest", |c| Some(c.mass), |v| format!("{v:.0} kg")),
        ("Best swimmer", |c| (c.in_water > 0.5 && !c.drowning).then_some(c.speed), |v| format!("{v:.2} m/s now")),
    ];
    egui::CollapsingHeader::new("Leaders").default_open(true).show(ui, |ui| {
        ui.label(egui::RichText::new("Who's ahead right now. Click one to go and follow it.").small().weak());
        egui::Grid::new("leaders_grid").num_columns(3).striped(true).show(ui, |ui| {
            for (title, measure, show) in records {
                let best = snap
                    .creatures
                    .iter()
                    .filter_map(|c| measure(c).map(|v| (c, v)))
                    .filter(|(_, v)| *v > 0.0)
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                ui.label(title);
                match best {
                    Some((c, v)) => {
                        let [r, g, b] = c.color.map(|x| (x.clamp(0.0, 1.0) * 255.0) as u8);
                        let text = egui::RichText::new(format!("#{}", c.id)).color(egui::Color32::from_rgb(r, g, b));
                        if ui
                            .selectable_label(snap.selected.as_ref().is_some_and(|s| s.id == c.id), text)
                            .on_hover_text("Go and follow it")
                            .clicked()
                        {
                            link.cmd(Command::Select(Some(c.id)));
                            cam.follow = true;
                        }
                        ui.label(show(v));
                    }
                    None => {
                        ui.label("–");
                        ui.label("");
                    }
                }
                ui.end_row();
            }
        });
    });
}

fn worlds_list(ui: &mut egui::Ui, snap: &Snapshot) {
    egui::CollapsingHeader::new(format!("Worlds ({})", snap.worlds.len()))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(
                    "Background worlds evolve on spare cores; their best genomes migrate into this one's gene pool. Named ones are an experiment's arms, on settings of their own, trading with nobody.",
                )
                .small()
                .weak(),
            );
            egui::Grid::new("worlds_grid").num_columns(6).striped(true).show(ui, |ui| {
                for h in ["", "time", "alive", "max gen", "neurons", "steps/s"] {
                    ui.label(egui::RichText::new(h).small().strong());
                }
                ui.end_row();
                for w in &snap.worlds {
                    ui.label(match (w.index, w.label.as_str()) {
                        (0, _) => "this".to_string(),
                        (k, "") => format!("#{k}"),
                        (k, arm) => format!("#{k} {arm}"),
                    });
                    if w.stopped {
                        ui.colored_label(egui::Color32::from_rgb(255, 110, 90), "stopped")
                            .on_hover_text("This world's thread stopped with an error; the reason is in the terminal.");
                        ui.end_row();
                        continue;
                    }
                    ui.label(fmt_time(w.time));
                    ui.label(format!("{}", w.population));
                    ui.label(format!("{}", w.max_generation));
                    ui.label(format!("{:.0}", w.mean_neurons));
                    ui.label(format!("{:.0}", w.steps_per_sec));
                    ui.end_row();
                }
            });
        });
}

fn zoo_window(
    ctx: &mut egui::Context,
    snap: &Snapshot,
    link: &SimLink,
    st: &mut UiState,
    tools: &mut ToolState,
    pv: &mut ZooPreview,
) {
    if !st.show_zoo {
        return;
    }
    let mut open = true;
    egui::Window::new("Zoo")
        .open(&mut open)
        .default_pos([300.0, 100.0])
        .default_width(1000.0)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Every world stores its best creature here every {:.0} simulated minutes; pinned ones are never pruned. {}",
                    snap.params.zoo_interval / 60.0,
                    snap.zoo_dir
                ))
                .small()
                .weak(),
            );
            ui.horizontal(|ui| {
                ui.label("Drop into");
                let label = |t: ZooTarget| match t {
                    ZooTarget::This => "this world".to_string(),
                    ZooTarget::World(k) => format!("background world {k}"),
                    ZooTarget::All => "every world".to_string(),
                };
                egui::ComboBox::from_id_salt("zoo_target")
                    .selected_text(label(st.zoo_target))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut st.zoo_target, ZooTarget::This, label(ZooTarget::This));
                        // Not into an experiment's arms: what they breed must be down to their settings.
                        for w in snap.worlds.iter().skip(1).filter(|w| w.label.is_empty()) {
                            ui.selectable_value(&mut st.zoo_target, ZooTarget::World(w.index), label(ZooTarget::World(w.index)));
                        }
                        ui.selectable_value(&mut st.zoo_target, ZooTarget::All, label(ZooTarget::All));
                    });
                ui.label("copies");
                for n in [1u32, 5, 10] {
                    ui.selectable_value(&mut st.zoo_count, n, format!("{n}"))
                        .on_hover_text("Each copy is a slightly mutated variant; they can breed with each other");
                }
                ui.separator();
                ui.selectable_value(&mut st.zoo_sort_newest, false, "best");
                ui.selectable_value(&mut st.zoo_sort_newest, true, "newest");
                ui.separator();
                ui.label("keep");
                ui.add(egui::DragValue::new(&mut st.zoo_keep).range(1..=500));
                if ui
                    .button("Compact")
                    .on_hover_text(
                        "Cut the zoo down to this many of the most different kinds, the best of each: \
                         the weaker of the two most alike goes, again and again (pinned ones always \
                         stay); asks first",
                    )
                    .clicked()
                {
                    st.confirm_compact = true;
                }
            });
            if st.confirm_compact {
                let unpinned = snap.zoo.iter().filter(|e| !e.pinned).count();
                let pinned = snap.zoo.len() - unpinned;
                let gone = unpinned.saturating_sub(st.zoo_keep as usize);
                ui.horizontal(|ui| {
                    if gone == 0 {
                        ui.label(format!("Nothing to delete: {unpinned} unpinned creatures, keeping {}.", st.zoo_keep));
                        if ui.button("OK").clicked() {
                            st.confirm_compact = false;
                        }
                        return;
                    }
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 110, 90),
                        format!(
                            "Delete {gone} of {} creatures, keeping {} of the most different kinds (the best of each) and the {pinned} pinned? Their genomes go to the zoo's trash folder.",
                            snap.zoo.len(),
                            st.zoo_keep
                        ),
                    );
                    if ui.button(format!("Delete {gone}")).clicked() {
                        link.0.send(Control::ZooKeepBest(st.zoo_keep as usize));
                        st.confirm_compact = false;
                    }
                    if ui.button("Cancel").clicked() {
                        st.confirm_compact = false;
                    }
                });
            }
            ui.separator();
            if snap.zoo.is_empty() {
                ui.label("The zoo is empty. Worlds add their best creature periodically, or use Add to zoo in a creature's inspector.");
                return;
            }
            let mut rows: Vec<&ZooMeta> = snap.zoo.iter().collect();
            if st.zoo_sort_newest {
                rows.sort_by(|a, b| b.saved_unix.cmp(&a.saved_unix));
            } else {
                // Scores worked out the old way don't compare with today's; they go last.
                rows.sort_by(|a, b| b.score_version.cmp(&a.score_version).then(b.score.total_cmp(&a.score)));
            }
            if !st.zoo_selected.is_some_and(|id| rows.iter().any(|e| e.id == id)) {
                st.zoo_selected = rows.first().map(|e| e.id);
            }
            let dir = PathBuf::from(&snap.zoo_dir);
            st.sketches.collect();
            let selected = rows.iter().find(|e| Some(e.id) == st.zoo_selected).copied();
            let sel_sketch = selected.and_then(|e| st.sketches.get(&dir, e.id));
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(ZOO_COLUMNS.iter().sum());
                    zoo_row(ui, 18.0, false, |k, ui| {
                        let h = ["body (side)", "name", "score", "neurons", "parts", "meat", "", "", ""][k];
                        ui.label(egui::RichText::new(h).small().strong());
                    });
                    // Only the rows in view are laid out, and only their genomes are read.
                    egui::ScrollArea::vertical().max_height(520.0).show_rows(ui, ZOO_ROW, rows.len(), |ui, range| {
                        for (i, e) in rows[range.clone()].iter().enumerate() {
                            let sk = st.sketches.get(&dir, e.id);
                            let picked = st.zoo_selected == Some(e.id);
                            let stripe = (range.start + i) % 2 == 1;
                            ui.push_id(e.id, |ui| {
                                zoo_row(ui, ZOO_ROW, stripe, |k, ui| match k {
                                    0 => {
                                        if draw_sketch(ui, sk.as_deref(), e.color, picked, egui::vec2(84.0, ZOO_ROW))
                                            .on_hover_text("Show this creature")
                                            .clicked()
                                        {
                                            st.zoo_selected = Some(e.id);
                                        }
                                    }
                                    1 => {
                                        let pin = if e.pinned { " (pinned)" } else { "" };
                                        if ui.selectable_label(picked, format!("{}{pin}", e.name)).clicked() {
                                            st.zoo_selected = Some(e.id);
                                        }
                                    }
                                    2 => {
                                        if e.score_version == evolve_sim::zoo::SCORE_VERSION {
                                            ui.label(format!("{:.1}", e.score));
                                        } else {
                                            ui.label(egui::RichText::new(format!("{:.1}*", e.score)).weak())
                                                .on_hover_text("Scored the old way (4 points per child, however little it was given), so it doesn't compare with newer scores");
                                        }
                                    }
                                    3 => {
                                        ui.label(format!("{}", e.neurons));
                                    }
                                    4 => {
                                        ui.label(match &sk {
                                            Some(sk) => format!("{} + {}", sk.trunk, sk.limbs),
                                            None => format!("{}", e.limbs + 1),
                                        })
                                        .on_hover_text("body segments + limb segments");
                                    }
                                    5 => {
                                        ui.label(format!("{:.0}%", e.diet * 100.0));
                                    }
                                    6 => {
                                        if ui.button("Drop").on_hover_text("Drop from the sky at a random spot").clicked() {
                                            link.0.send(Control::ZooDrop {
                                                id: e.id,
                                                target: st.zoo_target,
                                                pos: None,
                                                count: st.zoo_count,
                                            });
                                        }
                                    }
                                    7 => {
                                        if ui.button("Place").on_hover_text("Click the ground in this world to drop it there").clicked() {
                                            tools.zoo_place = Some((e.id, e.name.clone(), st.zoo_count));
                                            tools.tool = Tool::Creature;
                                        }
                                    }
                                    _ => {
                                        if ui
                                            .small_button(if e.pinned { "unpin" } else { "pin" })
                                            .on_hover_text("Pinned creatures are never pruned")
                                            .clicked()
                                        {
                                            link.0.send(Control::ZooPin(e.id, !e.pinned));
                                        }
                                        if ui.small_button("delete").on_hover_text("Its genome goes to the zoo's trash folder").clicked() {
                                            link.0.send(Control::ZooRemove(e.id));
                                        }
                                    }
                                });
                            });
                        }
                    });
                });
                ui.separator();
                ui.vertical(|ui| {
                    let Some(e) = selected else {
                        return;
                    };
                    ui.heading(&e.name);
                    ui.horizontal(|ui| {
                        if ui
                            .button(if e.pinned { "Unpin" } else { "Pin" })
                            .on_hover_text("Pinned creatures are never pruned")
                            .clicked()
                        {
                            link.0.send(Control::ZooPin(e.id, !e.pinned));
                        }
                        if ui
                            .button("Delete from the zoo")
                            .on_hover_text("Its genome goes to the zoo's trash folder")
                            .clicked()
                        {
                            link.0.send(Control::ZooRemove(e.id));
                        }
                    });
                    match &sel_sketch {
                        Some(sk) => {
                            pv.want = Some((e.id, sk.clone(), e.color));
                            pv.visible = true;
                            let size = egui::vec2(PREVIEW_SIZE[0] as f32, PREVIEW_SIZE[1] as f32) * 0.8;
                            match pv.texture {
                                Some(tex) => {
                                    let resp = ui
                                        .add(egui::Image::new((tex, size)).sense(egui::Sense::drag()))
                                        .on_hover_text("Drag to turn it");
                                    pv.dragging = resp.dragged();
                                    if resp.dragged() {
                                        pv.yaw += resp.drag_delta().x * 0.012;
                                    }
                                }
                                None => {
                                    ui.allocate_space(size);
                                }
                            }
                            ui.label(
                                egui::RichText::new("Yellow joints are hinges (one axis, cheap); pink ones are swivels (two axes, like a tentacle).")
                                    .small()
                                    .weak(),
                            );
                            egui::Grid::new("zoo_detail").num_columns(2).show(ui, |ui| {
                                let mut row = |k: &str, v: String| {
                                    ui.label(k);
                                    ui.label(v);
                                    ui.end_row();
                                };
                                row(
                                    "Body",
                                    format!(
                                        "{} body segment{}, {} limb segments, {:.0} kg",
                                        sk.trunk,
                                        if sk.trunk == 1 { "" } else { "s" },
                                        sk.limbs,
                                        e.mass
                                    ),
                                );
                                row("Joints", format!("{} hinges, {} swivels", sk.hinges, sk.swivels));
                                row("Brain", format!("{} neurons", e.neurons));
                                row("Diet", format!("{:.0}% meat", e.diet * 100.0));
                                row("Score", format!("{:.1}, generation {}", e.score, e.generation));
                                let age = now.saturating_sub(e.saved_unix);
                                let age = if age < 3600 {
                                    format!("{} min ago", age / 60)
                                } else if age < 86400 * 2 {
                                    format!("{} h ago", age / 3600)
                                } else {
                                    format!("{} days ago", age / 86400)
                                };
                                row("From", format!("{}, saved {age}", e.source));
                            });
                        }
                        None => {
                            ui.label("Reading its genome...");
                        }
                    }
                });
            });
        });
    st.show_zoo = open;
}

/// A side view of a body plan (forward to the right), fitted into a small box.
fn draw_sketch(ui: &mut egui::Ui, sk: Option<&Sketch>, color: [f32; 3], selected: bool, size: egui::Vec2) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter_at(rect);
    let bg = if selected {
        ui.visuals().selection.bg_fill
    } else {
        ui.visuals().extreme_bg_color
    };
    painter.rect_filled(rect, 3.0, bg);
    let Some(sk) = sk else {
        return resp;
    };
    enum Mark {
        Hull(Vec<Vec2>, bool),
        Stick(Vec2, Vec2, f32),
    }
    let mut marks = Vec::new();
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    let mut grow = |p: Vec2, r: f32| {
        lo = lo.min(p - Vec2::splat(r));
        hi = hi.max(p + Vec2::splat(r));
    };
    for p in &sk.parts {
        let rot = Quat::from_array(p.rot);
        let at = Vec3::from_array(p.pos);
        match p.shape {
            evolve_sim::body::PartShape::Box { half } => {
                let h = Vec3::from_array(half);
                let mut pts = Vec::with_capacity(8);
                for c in 0..8 {
                    let s = Vec3::new(
                        if c & 1 == 0 { -1.0 } else { 1.0 },
                        if c & 2 == 0 { -1.0 } else { 1.0 },
                        if c & 4 == 0 { -1.0 } else { 1.0 },
                    );
                    let w = at + rot * (h * s);
                    let q = Vec2::new(w.x, w.y);
                    grow(q, 0.0);
                    pts.push(q);
                }
                marks.push(Mark::Hull(hull(pts), p.trunk));
            }
            evolve_sim::body::PartShape::Capsule { half_height, radius } => {
                let a = at + rot * Vec3::new(0.0, -half_height, 0.0);
                let b = at + rot * Vec3::new(0.0, half_height, 0.0);
                let (a, b) = (Vec2::new(a.x, a.y), Vec2::new(b.x, b.y));
                grow(a, radius);
                grow(b, radius);
                marks.push(Mark::Stick(a, b, radius));
            }
        }
    }
    let span = (hi - lo).max(Vec2::splat(1e-3));
    let inner = rect.shrink(4.0);
    let scale = (inner.width() / span.x).min(inner.height() / span.y);
    let mid = (lo + hi) * 0.5;
    let to_screen = |p: Vec2| inner.center() + egui::vec2((p.x - mid.x) * scale, (mid.y - p.y) * scale);
    let shade = |k: f32| {
        let [r, g, b] = color.map(|v| (v * k).clamp(0.0, 1.0));
        egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    };
    // Limbs behind, body in front.
    for trunk_pass in [false, true] {
        for m in &marks {
            match m {
                Mark::Stick(a, b, r) if !trunk_pass => {
                    painter.line_segment([to_screen(*a), to_screen(*b)], egui::Stroke::new((r * 2.0 * scale).max(1.0), shade(0.75)));
                }
                Mark::Hull(pts, trunk) if *trunk == trunk_pass => {
                    let pts: Vec<egui::Pos2> = pts.iter().map(|p| to_screen(*p)).collect();
                    let fill = if *trunk { shade(1.0) } else { shade(0.75) };
                    painter.add(egui::Shape::convex_polygon(pts, fill, egui::Stroke::NONE));
                }
                _ => {}
            }
        }
    }
    for &(pos, swivel) in &sk.joints {
        let c = if swivel {
            egui::Color32::from_rgb(240, 90, 230)
        } else {
            egui::Color32::from_rgb(255, 215, 50)
        };
        painter.circle_filled(to_screen(Vec2::new(pos[0], pos[1])), 1.6, c);
    }
    resp
}

/// Convex hull, counterclockwise (monotone chain).
fn hull(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut out: Vec<Vec2> = Vec::with_capacity(pts.len() * 2);
    for pass in 0..2 {
        let start = out.len();
        let iter: Box<dyn Iterator<Item = &Vec2>> = if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &p in iter {
            while out.len() >= start + 2 && cross(out[out.len() - 2], out[out.len() - 1], p) <= 0.0 {
                out.pop();
            }
            out.push(p);
        }
        out.pop();
    }
    out
}

/// Starts or stops recording the window (with whatever panels are open) to a video file.
fn record_button(ui: &mut egui::Ui, rec: &mut crate::recording::Recorder) {
    let button = match rec.elapsed() {
        Some(t) => {
            let secs = t.as_secs();
            egui::Button::new(
                egui::RichText::new(format!("■ Stop {}:{:02}", secs / 60, secs % 60)).color(egui::Color32::WHITE),
            )
            .fill(egui::Color32::from_rgb(190, 30, 30))
        }
        None => egui::Button::new(egui::RichText::new("● Record").color(egui::Color32::from_rgb(230, 70, 60))),
    };
    if ui
        .add(button)
        .on_hover_text("Record the window to a video in ~/Videos, panels and all, as you see it (R or F10)")
        .clicked()
    {
        rec.toggle = true;
    }
}

fn sound_controls(ui: &mut egui::Ui, sound: &mut SoundSettings) {
    if let Some(e) = &sound.error {
        ui.label(egui::RichText::new("no sound").weak()).on_hover_text(e.as_str());
        return;
    }
    ui.add(egui::Slider::new(&mut sound.volume, 0.0..=1.0).show_value(false))
        .on_hover_text("Volume");
    let label = if sound.muted { "Unmute" } else { "Mute" };
    if ui.selectable_label(sound.muted, label).on_hover_text("M").clicked() {
        sound.muted = !sound.muted;
    }
}

/// While the director runs: what it's showing, and how to take over.
/// The world's clock, in every view: the day (counting from the first) and the time of day.
fn clock(ctx: &mut egui::Context, snap: &Snapshot, top: f32) {
    let t = snap.weather.time_of_day.rem_euclid(1.0) * 24.0;
    let (h, m) = (t as u32, (t.fract() * 60.0) as u32);
    egui::Area::new(egui::Id::new("clock"))
        .anchor(egui::Align2::CENTER_TOP, [0.0, top])
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(110))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(12, 4))
                .show(ui, |ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    ui.label(
                        egui::RichText::new(format!("Day {}   {h:02}:{m:02}", snap.day))
                            .size(17.0)
                            .color(egui::Color32::from_gray(235)),
                    );
                });
        });
}

/// A reminder at the top of the screen while the world is paused (the director hides the
/// toolbar, so nothing else would say why nothing moves).
fn paused_badge(ctx: &mut egui::Context, top: f32) {
    egui::Area::new(egui::Id::new("paused_badge"))
        .anchor(egui::Align2::CENTER_TOP, [0.0, top])
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(150))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(14, 6))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("⏸ Paused: Space to run").size(18.0).color(egui::Color32::WHITE));
                });
        });
}

fn director_overlay(
    ctx: &mut egui::Context,
    director: &mut Director,
    sound: &mut SoundSettings,
    rec: &mut crate::recording::Recorder,
    now: f64,
) {
    egui::Area::new(egui::Id::new("director_hint"))
        .anchor(egui::Align2::LEFT_TOP, [12.0, 10.0])
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(110))
                .corner_radius(6.0)
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Director").strong());
                        ui.label(
                            egui::RichText::new("Press C or move the camera to take over, R to record.")
                                .small()
                                .color(egui::Color32::from_gray(200)),
                        );
                        if ui.small_button("Stop").clicked() {
                            director.toggle();
                        }
                        ui.checkbox(&mut director.panels, "panels");
                        record_button(ui, rec);
                        sound_controls(ui, sound);
                    });
                });
        });
    let Some(c) = &director.caption else {
        return;
    };
    // Fade in, hold, fade out.
    let alpha = ((now - c.started) / 0.8).min((c.until - now) / 1.2).clamp(0.0, 1.0) as f32;
    if alpha <= 0.0 {
        return;
    }
    let a = (alpha * 255.0) as u8;
    egui::Area::new(egui::Id::new("director_caption"))
        .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -40.0])
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha((alpha * 120.0) as u8))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::symmetric(18, 10))
                .show(ui, |ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new(&c.title)
                                .size(28.0)
                                .strong()
                                .color(egui::Color32::from_white_alpha(a)),
                        );
                        ui.label(
                            egui::RichText::new(&c.detail)
                                .size(16.0)
                                .color(egui::Color32::from_white_alpha((a as f32 * 0.85) as u8)),
                        );
                    });
                });
        });
}

/// Diagram labels: joint channels by limb and segment, everything else by name.
pub(crate) fn short_input(k: usize) -> String {
    if k >= evolve_sim::brain::input::JOINTS {
        format!("{} angle", evolve_sim::brain::motor_short_label(k - evolve_sim::brain::input::JOINTS))
    } else {
        input_label(k)
    }
}

pub(crate) fn short_output(k: usize) -> String {
    if k < evolve_sim::brain::output::EAT {
        format!("move {}", evolve_sim::brain::motor_short_label(k))
    } else {
        output_label(k)
    }
}

/// Orange for positive, blue for negative, grey near zero.
fn signal_color(v: f32, alpha: u8) -> egui::Color32 {
    let v = v.clamp(-1.0, 1.0);
    let grey = [90.0, 90.0, 96.0];
    let to = if v >= 0.0 { [255.0, 150.0, 40.0] } else { [60.0, 140.0, 255.0] };
    let t = v.abs();
    let c = |k: usize| (grey[k] + (to[k] - grey[k]) * t) as u8;
    egui::Color32::from_rgba_unmultiplied(c(0), c(1), c(2), alpha)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BrainTab {
    /// The wiring as a diagram.
    Network,
    /// The same wiring as tissue: neurons, dendrites and synapses, with spikes.
    Neural,
    /// Every neuron in a brain you can turn, with the strongest connections.
    ThreeD,
}

impl BrainTab {
    /// The most neurons and connections this picture draws smoothly, for a brain of `neurons`.
    fn most(self, neurons: usize) -> (usize, usize) {
        match self {
            BrainTab::Network => (neurons, 10_000),
            // Tissue grows a dendrite tree for every neuron and an axon for every connection, and
            // grows them again whenever learning changes which connections are the strongest.
            BrainTab::Neural => (neurons.min(300), 2_000),
            BrainTab::ThreeD => (neurons, 100_000),
        }
    }
}

/// The brain view's state, kept while the app runs.
struct BrainView {
    tab: BrainTab,
    /// How much of the brain each picture shows (by tab), as its sliders set it; usize::MAX
    /// neurons is all of them, whatever the brain.
    limits: [SketchLimits; 3],
    neural: crate::neural_view::NeuralView,
    solid: crate::brain3d::Brain3d,
    /// What the runner was last asked to sketch (None: nothing, the view is closed).
    asked: Option<SketchLimits>,
}

impl Default for BrainView {
    fn default() -> Self {
        Self {
            tab: BrainTab::Network,
            limits: [
                SketchLimits::default(),
                SketchLimits::default(),
                SketchLimits { neurons: usize::MAX, connections: 20_000 },
            ],
            neural: Default::default(),
            solid: Default::default(),
            asked: None,
        }
    }
}

impl BrainView {
    /// What the picture on show wants sketched of `c`'s brain.
    fn wants(&self, c: &CreatureDetail) -> SketchLimits {
        let (neurons, connections) = self.tab.most(c.neurons);
        let l = self.limits[self.tab as usize];
        SketchLimits { neurons: l.neurons.min(neurons), connections: l.connections.min(connections) }
    }

    /// Asks the runner to sketch to `limits` (None: not at all), unless that's what it was last
    /// asked.
    fn ask(&mut self, link: &SimLink, limits: Option<SketchLimits>) {
        if limits != self.asked {
            link.0.send(Control::BrainView(limits));
            self.asked = limits;
        }
    }
}

/// The creature's brain, as a live diagram (senses on the left, the hidden neurons that matter
/// most in a ring, actions on the right, and the strongest connections between them), as living
/// tissue (see `neural_view`), or whole in 3D (see `brain3d`). Sliders set how many neurons and
/// connections each picture shows.
fn brain_window(ctx: &mut egui::Context, c: &CreatureDetail, open: &mut bool, view: &mut BrainView, now: f64) {
    use crate::brain3d::count;
    use evolve_sim::wiring::Node;
    egui::Window::new(format!("Brain of #{}", c.id))
        .id(egui::Id::new("brain_view"))
        .open(open)
        .default_pos([980.0, 60.0])
        .default_size([640.0, 720.0])
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut view.tab, BrainTab::Network, "Network")
                    .on_hover_text("The wiring as a diagram");
                ui.selectable_value(&mut view.tab, BrainTab::Neural, "Neural")
                    .on_hover_text("The same wiring as living tissue, firing");
                ui.selectable_value(&mut view.tab, BrainTab::ThreeD, "3D")
                    .on_hover_text("Every neuron it has, in a brain you can turn, with its strongest connections");
            });
            let tab = view.tab;
            let (most_neurons, most_links) = tab.most(c.neurons);
            let limits = &mut view.limits[tab as usize];
            ui.horizontal(|ui| {
                if tab != BrainTab::ThreeD {
                    let mut n = limits.neurons.clamp(1, most_neurons.max(1));
                    if ui
                        .add(egui::Slider::new(&mut n, 1..=most_neurons.max(1)).logarithmic(true).text("neurons"))
                        .on_hover_text("How many of its hidden neurons to show: those most strongly tied to its senses and actions first")
                        .changed()
                    {
                        // All the way up is all of them, in a bigger brain too.
                        limits.neurons = if n >= most_neurons { usize::MAX } else { n };
                    }
                }
                ui.add(egui::Slider::new(&mut limits.connections, 1..=most_links).logarithmic(true).text("connections"))
                    .on_hover_text("How many connections to show, the strongest first: shared among senses into neurons, neurons into neurons, and neurons into actions by how many of each there are among the neurons shown");
            });
            let w = &*c.wiring;
            if w.creature != c.id || w.inputs.is_empty() {
                ui.label("Reading its wiring...");
                return;
            }
            if tab == BrainTab::ThreeD {
                ui.label(
                    egui::RichText::new(format!(
                        "{} neurons, {} connections; showing {}. Every neuron it has, glowing as it works, with its senses on the disc behind and its actions on the disc in front, and its strongest connections: orange excites, blue inhibits, fainter is weaker. Drag to turn it, scroll to zoom.",
                        count(w.total_hidden),
                        count(c.synapses),
                        count(w.edges.len())
                    ))
                    .small()
                    .weak(),
                );
                let size = ui.available_size().max(egui::vec2(520.0, 480.0));
                let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                view.solid.show(ui, rect, &resp, c, now);
                return;
            }
            if tab == BrainTab::Neural {
                ui.label(
                    egui::RichText::new(format!(
                        "{} of its {} neurons and the strongest {} of {} connections, as tissue. Senses on the left, hidden neurons in the middle, motor neurons on the right. Each neuron fires as fast as it is active; spikes run down axons to synapses, warm ones exciting and cool ones inhibiting the next neuron. Hover a cell body for its name and value.",
                        count(w.hidden.len()),
                        count(w.total_hidden),
                        count(w.edges.len()),
                        count(c.synapses)
                    ))
                    .small()
                    .weak(),
                );
                let rows = w.inputs.len().max(w.outputs.len()) as f32;
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let width = ui.available_width().max(560.0);
                    let height = (rows * 13.0 + 28.0).max(ui.available_height()).max(480.0);
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
                    view.neural.show(ui, rect, &resp, c, now);
                });
                return;
            }
            ui.label(
                egui::RichText::new(format!(
                    "{} of its {} neurons and the strongest {} of {} connections. Orange excites, blue inhibits; brighter lines carry more signal right now. Hover for names and values.",
                    count(w.hidden.len()),
                    count(w.total_hidden),
                    count(w.edges.len()),
                    count(c.synapses)
                ))
                .small()
                .weak(),
            );
            let rows = w.inputs.len().max(w.outputs.len()) as f32;
            egui::ScrollArea::vertical().show(ui, |ui| {
                let width = ui.available_width().max(520.0);
                let height = (rows * 12.0 + 20.0).max(ui.available_height()).max(480.0);
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
                let painter = ui.painter_at(rect);
                let label_w = 124.0;
                let x_in = rect.left() + label_w;
                let x_out = rect.right() - label_w;
                let row_y = |i: usize, n: usize| rect.top() + 10.0 + (rect.height() - 20.0) * (i as f32 + 0.5) / n.max(1) as f32;
                let center = egui::pos2((x_in + x_out) * 0.5, rect.center().y);
                let radius = ((x_out - x_in) * 0.5 - 40.0).min(rect.height() * 0.42).max(40.0);
                let mut at: HashMap<Node, (egui::Pos2, f32)> = HashMap::new();
                for (i, &k) in w.inputs.iter().enumerate() {
                    at.insert(Node::Input(k), (egui::pos2(x_in, row_y(i, w.inputs.len())), c.inputs.get(k).copied().unwrap_or(0.0)));
                }
                for (i, &k) in w.outputs.iter().enumerate() {
                    at.insert(Node::Output(k), (egui::pos2(x_out, row_y(i, w.outputs.len())), c.outputs.get(k).copied().unwrap_or(0.0)));
                }
                let n = w.hidden.len().max(1) as f32;
                for (i, &k) in w.hidden.iter().enumerate() {
                    let a = -std::f32::consts::FRAC_PI_2 + i as f32 / n * std::f32::consts::TAU;
                    let p = center + egui::vec2(a.cos(), a.sin()) * radius;
                    at.insert(Node::Hidden(k), (p, c.hidden.get(k).copied().unwrap_or(0.0)));
                }
                // Smaller as the ring fills, down to dots for thousands.
                let hidden_r = (std::f32::consts::TAU * radius / n * 0.35).clamp(1.5, 6.0);
                // Weights are scaled per kind of link, so each layer's strongest reads as strong.
                let kind = |e: &evolve_sim::wiring::Edge| match (e.from, e.to) {
                    (Node::Input(_), _) => 0,
                    (_, Node::Output(_)) => 2,
                    _ => 1,
                };
                let mut max = [1e-6f32; 3];
                for e in &w.edges {
                    let k = kind(e);
                    max[k] = max[k].max(e.weight.abs());
                }
                // The lines go into one mesh: thousands of curves drawn one by one by egui would
                // take most of a frame. Fewer points to a curve when there are many.
                let mut lines = crate::brain3d::Lines::new(egui::Mesh::default(), ui.ctx().pixels_per_point(), true);
                let steps = if w.edges.len() > 2000 { 4 } else { 10 };
                let mut curve = Vec::with_capacity(steps + 1);
                for e in &w.edges {
                    let (Some(&(a, va)), Some(&(b, _))) = (at.get(&e.from), at.get(&e.to)) else {
                        continue;
                    };
                    let strength = e.weight.abs() / max[kind(e)];
                    let flow = va.abs().min(1.0);
                    let alpha = (25.0 + 210.0 * strength * (0.2 + 0.8 * flow)) as u8;
                    let color = signal_color(e.weight.signum(), alpha);
                    let width = 0.4 + 2.4 * strength;
                    if kind(e) == 1 {
                        if a == b {
                            // A neuron feeding itself: a small loop outside the ring.
                            let out = (a - center).normalized() * 9.0;
                            painter.circle_stroke(a + out, 6.0, egui::Stroke::new(width, color));
                        } else {
                            let ctrl = a + (b - a) * 0.5 + (center - (a + (b - a) * 0.5)) * 0.55;
                            curve.clear();
                            curve.extend((0..=steps).map(|i| {
                                let t = i as f32 / steps as f32;
                                let u = 1.0 - t;
                                let p = a.to_vec2() * (u * u) + ctrl.to_vec2() * (2.0 * u * t) + b.to_vec2() * (t * t);
                                (p.to_pos2(), color)
                            }));
                            lines.path(&curve, width);
                        }
                    } else {
                        lines.path(&[(a, color), (b, color)], width);
                    }
                }
                painter.add(egui::Shape::mesh(lines.mesh));
                let font = egui::FontId::proportional(10.5);
                let text = ui.visuals().text_color();
                for (&node, &(p, v)) in &at {
                    let r = if matches!(node, Node::Hidden(_)) { hidden_r } else { 4.0 };
                    let rim = if r >= 3.0 { egui::Stroke::new(0.8, egui::Color32::from_gray(30)) } else { egui::Stroke::NONE };
                    painter.circle(p, r, signal_color(v, 255), rim);
                    match node {
                        Node::Input(k) => {
                            painter.text(p - egui::vec2(8.0, 0.0), egui::Align2::RIGHT_CENTER, short_input(k), font.clone(), text);
                        }
                        Node::Output(k) => {
                            painter.text(p + egui::vec2(8.0, 0.0), egui::Align2::LEFT_CENTER, short_output(k), font.clone(), text);
                        }
                        Node::Hidden(_) => {}
                    }
                }
                if let Some(pointer) = resp.hover_pos()
                    && let Some((node, (p, v))) = at
                        .iter()
                        .map(|(n, x)| (*n, *x))
                        .min_by(|a, b| a.1.0.distance(pointer).total_cmp(&b.1.0.distance(pointer)))
                    && p.distance(pointer) < 10.0
                {
                    let name = match node {
                        Node::Input(k) => input_label(k),
                        Node::Output(k) => output_label(k),
                        Node::Hidden(k) => format!("neuron {k}"),
                    };
                    let links = w.edges.iter().filter(|e| e.from == node || e.to == node).count();
                    resp.on_hover_text_at_pointer(format!("{name}: {v:+.2}\n{links} connections shown"));
                }
            });
        });
}
