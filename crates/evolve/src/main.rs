//! evolve: a 3D evolution simulator. The world runs on its own thread (evolve-sim); this crate
//! draws it with Bevy on the GPU and lets the user poke at it.

mod audio;
mod camera;
mod director;
mod env;
mod genome_view;
mod help;
mod neural_view;
mod preview;
mod recording;
mod scene;
mod sky;
mod splash;
mod sync;
mod tools;
mod trees;
mod ui;
mod water;
mod weather_fx;

use std::path::PathBuf;
use std::sync::Arc;

use bevy::diagnostic::{FrameCount, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy::window::{PresentMode, WindowResolution};
use bevy_egui::EguiPlugin;
use evolve_sim::params::ParamRef;
use evolve_sim::{Control, Runner, Sim, SimParams, Snapshot, save};

/// The link to the simulation thread.
#[derive(Resource)]
pub struct SimLink(pub Runner);

impl SimLink {
    pub fn cmd(&self, c: evolve_sim::Command) {
        self.0.command(c);
    }
}

/// The latest world snapshot, refreshed at the start of every frame.
#[derive(Resource, Default)]
pub struct Snap(pub Arc<Snapshot>);

/// The snapshot before the latest one, so moving things can be drawn between the two. Snapshots
/// arrive at the simulation's pace, not the monitor's, so drawing them raw would stutter.
#[derive(Resource)]
pub struct Interp {
    pub prev: Arc<Snapshot>,
    pub prev_index: std::collections::HashMap<u64, usize>,
    pub arrived: f64,
    pub interval: f64,
}

impl Default for Interp {
    fn default() -> Self {
        Self {
            prev: Arc::default(),
            prev_index: Default::default(),
            arrived: 0.0,
            interval: 1.0 / 60.0,
        }
    }
}

impl Interp {
    /// How far to blend from the previous snapshot to the latest, 0..1.
    pub fn alpha(&self, now: f64) -> f32 {
        ((now - self.arrived) / self.interval).clamp(0.0, 1.0) as f32
    }
}

#[derive(Resource, Clone, Default)]
pub struct Args {
    /// Take a screenshot to this path after `frames` frames, then exit (for automated checks).
    pub screenshot: Option<PathBuf>,
    pub frames: u32,
    /// Camera preset for screenshots: focus x, z, distance, yaw, pitch.
    pub camera: Option<[f32; 5]>,
    pub select_first: bool,
    /// Performance probes: `--no-vsync`, `--no-shadows`, `--no-ui`, and an FPS report on exit.
    pub no_vsync: bool,
    pub no_shadows: bool,
    pub no_ui: bool,
    /// Debug: load this world from inside the running app after 60 frames, like the Save/Load
    /// window does.
    pub load_after: Option<PathBuf>,
    /// Open the Zoo window at startup.
    pub zoo: bool,
    /// Start with the director (hands-free camera) on.
    pub director: bool,
    /// Start with the sound off.
    pub mute: bool,
    /// Open the brain view on its neural picture (with --select-first, for screenshots).
    pub neural_view: bool,
    /// Open the help on this topic (the start of its title), for screenshots.
    pub help_page: Option<String>,
    /// Open the genome window (with --select-first, for screenshots).
    pub genome_view: bool,
    /// Start recording the window to this video file at launch.
    pub record: Option<PathBuf>,
    pub pick_test: bool,
}

fn usage() -> ! {
    eprintln!(
        "evolve [--load FILE] [--seed N] [--speed X] [--zoo] [--director] [--mute] [--set name=value]...\n       [--screenshot FILE --frames N] [--camera x,z,distance,yaw,pitch] [--select-first]\n       [--time-of-day F] [--weather clear|cloudy|rain|storm] [--lightning] [--record VIDEO.mp4]"
    );
    std::process::exit(2)
}

fn main() -> AppExit {
    let mut args = Args {
        frames: 240,
        ..default()
    };
    let mut load: Option<PathBuf> = None;
    let mut speed: Option<f32> = None;
    let mut params = SimParams::default();
    let mut visuals = env::VisualOverride::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--load" => load = Some(val().into()),
            "--seed" => params.seed = val().parse().unwrap_or_else(|_| usage()),
            "--speed" => speed = Some(val().parse().unwrap_or_else(|_| usage())),
            "--screenshot" => args.screenshot = Some(val().into()),
            "--frames" => args.frames = val().parse().unwrap_or_else(|_| usage()),
            "--select-first" => args.select_first = true,
            "--no-vsync" => args.no_vsync = true,
            "--no-shadows" => args.no_shadows = true,
            "--no-ui" => args.no_ui = true,
            "--load-after" => args.load_after = Some(val().into()),
            "--zoo" => args.zoo = true,
            "--director" => args.director = true,
            "--mute" => args.mute = true,
            "--neural-view" => args.neural_view = true,
            "--help-page" => args.help_page = Some(val()),
            "--genome-view" => args.genome_view = true,
            "--record" => args.record = Some(val().into()),
            "--time-of-day" => visuals.time_of_day = Some(val().parse().unwrap_or_else(|_| usage())),
            "--weather" => visuals.weather = Some(val().parse().unwrap_or_else(|_| usage())),
            "--lightning" => visuals.lightning = true,
            "--pick-test" => {
                args.select_first = true;
                args.pick_test = true;
            }
            "--camera" => {
                let v: Vec<f32> = val().split(',').filter_map(|x| x.parse().ok()).collect();
                if v.len() != 5 {
                    usage();
                }
                args.camera = Some([v[0], v[1], v[2], v[3], v[4]]);
            }
            "--set" => {
                let s = val();
                let (k, v) = s.split_once('=').unwrap_or_else(|| usage());
                let mut hit = false;
                params.visit(&mut |info, r| {
                    if info.name == k {
                        hit = true;
                        let ok = match r {
                            ParamRef::F32(x) => v.parse().map(|n| *x = n).is_ok(),
                            ParamRef::U32(x) => v.parse().map(|n| *x = n).is_ok(),
                            ParamRef::U64(x) => v.parse().map(|n| *x = n).is_ok(),
                            ParamRef::Bool(x) => v.parse().map(|n| *x = n).is_ok(),
                        };
                        if !ok {
                            usage();
                        }
                    }
                });
                if !hit {
                    eprintln!("evolve: no param named {k}");
                    usage();
                }
            }
            _ => usage(),
        }
    }

    evolve_sim::init_thread_pool();
    let sim = match &load {
        Some(p) => save::load(p).unwrap_or_else(|e| {
            eprintln!("evolve: {e}");
            std::process::exit(1)
        }),
        None => Sim::new(params),
    };
    let runner = Runner::start(sim);
    if let Some(s) = speed {
        runner.send(Control::Speed(s));
    }

    App::new()
        .insert_resource(ClearColor(scene::SKY))
        // Finer shadows, so small things don't look detached from their shadows.
        .insert_resource(bevy::light::DirectionalLightShadowMap { size: 4096 })
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.8, 0.88, 1.0),
            brightness: 350.0,
            ..default()
        })
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "evolve".into(),
                        resolution: WindowResolution::new(1600, 900),
                        present_mode: if args.no_vsync {
                            PresentMode::AutoNoVsync
                        } else {
                            PresentMode::AutoVsync
                        },
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    // Screenshot runs want complete first frames, not pop-in.
                    synchronous_pipeline_compilation: args.screenshot.is_some(),
                    ..default()
                }),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default())
        .insert_resource(SimLink(runner))
        .init_resource::<Snap>()
        .init_resource::<Interp>()
        .insert_resource(args)
        .insert_resource(visuals)
        .add_plugins((
            env::EnvPlugin,
            sky::SkyPlugin,
            water::WaterPlugin,
            splash::SplashPlugin,
            weather_fx::WeatherFxPlugin,
            trees::TreesPlugin,
            scene::ScenePlugin,
            sync::SyncPlugin,
            camera::CameraPlugin,
            tools::ToolsPlugin,
            ui::UiPlugin,
            preview::PreviewPlugin,
            director::DirectorPlugin,
            audio::AudioPlugin,
            recording::RecordingPlugin,
        ))
        .add_systems(PreUpdate, refresh_snapshot)
        .add_systems(Update, (auto_screenshot, load_after))
        .run()
}

fn refresh_snapshot(link: Res<SimLink>, time: Res<Time>, mut snap: ResMut<Snap>, mut interp: ResMut<Interp>) {
    let latest = link.0.snapshot();
    if Arc::ptr_eq(&latest, &snap.0) {
        return;
    }
    let now = time.elapsed_secs_f64();
    let gap = (now - interp.arrived).clamp(0.002, 0.25);
    interp.interval = interp.interval * 0.8 + gap * 0.2;
    interp.arrived = now;
    let prev = std::mem::replace(&mut snap.0, latest);
    // A new or loaded world reuses ids, so never blend a creature from one world into another.
    interp.prev = if prev.world_epoch == snap.0.world_epoch { prev } else { snap.0.clone() };
    interp.prev_index = interp
        .prev
        .creatures
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();
}

fn load_after(args: Res<Args>, frames: Res<FrameCount>, link: Res<SimLink>) {
    if frames.0 == 60
        && let Some(p) = &args.load_after
    {
        link.0.send(Control::Load(p.clone()));
    }
}

fn auto_screenshot(
    mut commands: Commands,
    frames: Res<FrameCount>,
    args: Res<Args>,
    diag: Res<bevy::diagnostic::DiagnosticsStore>,
    mut requested: Local<bool>,
) {
    let Some(path) = args.screenshot.clone() else {
        return;
    };
    if *requested || frames.0 < args.frames {
        return;
    }
    *requested = true;
    if let Some(ft) = diag.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
        println!(
            "frame time: mean {:.2} ms over the last {} frames ({:.0} fps)",
            ft.average().unwrap_or(0.0),
            ft.history_len(),
            1000.0 / ft.average().unwrap_or(1.0)
        );
    }
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path))
        .observe(|_c: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>| {
            exit.write(AppExit::Success);
        });
}
