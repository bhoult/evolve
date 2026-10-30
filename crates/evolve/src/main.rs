//! evolve: a 3D evolution simulator. The world runs on its own thread (evolve-sim); this crate
//! draws it with Bevy on the GPU and lets the user poke at it.

mod audio;
mod brain3d;
mod camera;
mod director;
mod env;
mod flow;
mod genome_view;
mod help;
mod miracles;
mod neural_view;
mod observatory;
mod preview;
#[cfg(feature = "raytracing")]
mod raytracing;
mod recording;
mod reports;
mod scene;
mod settings;
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
use evolve_sim::{Control, Runner, Sim, SimParams, Snapshot};

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
    /// `--miracle bless|smite|take:N`: work a miracle on the selected creature N frames before
    /// the screenshot.
    pub miracle: Option<(String, u32)>,
    /// Start recording the window to this video file at launch.
    pub record: Option<PathBuf>,
    pub pick_test: bool,
    /// Port of the MCP server (see `evolve_sim::mcp`); 0 for none.
    pub mcp_port: u16,
    /// Ray-traced lighting (see raytracing.rs), in builds with the `raytracing` feature.
    pub raytracing: bool,
}

fn usage() -> ! {
    eprintln!(
        "evolve [--load FILE] [--seed N] [--speed X] [--zoo] [--director] [--mute] [--set name=value]...\n       [--screenshot FILE --frames N] [--camera x,z,distance,yaw,pitch] [--select-first]\n       [--time-of-day F] [--weather clear|cloudy|rain|storm] [--lightning] [--record VIDEO.mp4]\n       [--mcp-port N (default 7311, 0 for none)] [--raytracing]"
    );
    std::process::exit(2)
}

fn main() -> AppExit {
    evolve_sim::tune_allocator();
    let mut args = Args {
        frames: 240,
        mcp_port: evolve_sim::mcp::DEFAULT_PORT,
        ..default()
    };
    let mut load: Option<PathBuf> = None;
    let mut speed: Option<f32> = None;
    let mut params = SimParams::default();
    // `--set` pairs, also applied on top of a loaded world's own settings.
    let mut sets: Vec<(String, String)> = Vec::new();
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
            "--raytracing" => {
                if !cfg!(feature = "raytracing") {
                    eprintln!("evolve: --raytracing needs a build with `--features raytracing`");
                    std::process::exit(2)
                }
                args.raytracing = true
            }
            "--neural-view" => args.neural_view = true,
            "--help-page" => args.help_page = Some(val()),
            "--genome-view" => args.genome_view = true,
            "--miracle" => {
                let v = val();
                let (kind, lead) = v.split_once(':').unwrap_or((&v, "30"));
                args.miracle = Some((kind.to_string(), lead.parse().unwrap_or_else(|_| usage())));
            }
            "--record" => args.record = Some(val().into()),
            "--mcp-port" => args.mcp_port = val().parse().unwrap_or_else(|_| usage()),
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
                sets.push((k.to_string(), v.to_string()));
            }
            _ => usage(),
        }
    }

    evolve_sim::init_thread_pool();
    // How every world does over time goes into the stats record, which the Reports window
    // graphs (screenshot runs are checks, and leave it alone).
    if args.screenshot.is_none() {
        let path = evolve_sim::record::default_path();
        match evolve_sim::record::start(&path) {
            Ok(()) => eprintln!("evolve: recording stats to {}", path.display()),
            Err(e) => eprintln!("evolve: not recording stats ({e})"),
        }
    }
    // A world to load is loaded by the runner once the window is up (a big one takes a minute,
    // which with no window looked like the app had died); it starts from a tiny empty island,
    // and the screen says what it's doing meanwhile. A loaded world keeps its own settings
    // except those given with --set.
    let sim = match &load {
        Some(p) => {
            if !p.exists() {
                eprintln!("evolve: no such file {}", p.display());
                std::process::exit(1)
            }
            Sim::new(SimParams {
                world_size: 60.0,
                terrain_resolution: 33,
                min_population: 0,
                stones: 0,
                plant_max_count: 0,
                plant_spawn_rate: 0.0,
                tree_max_count: 0,
                parallel_worlds: 0,
                ..params.clone()
            })
        }
        None => Sim::new(params),
    };
    let runner = Runner::start(sim);
    if let Some(p) = &load {
        runner.send(Control::LoadWith(p.clone(), sets.clone()));
    }
    if let Some(s) = speed {
        runner.send(Control::Speed(s));
    }
    // Screenshot runs are checks of their own, beside a real session that may hold the port.
    let (shot_tx, shot_rx) = std::sync::mpsc::channel();
    if args.mcp_port != 0 && args.screenshot.is_none() {
        match evolve_sim::mcp::serve(args.mcp_port, runner.sender(), Some(shot_tx)) {
            Ok(port) => eprintln!("evolve: MCP server at http://127.0.0.1:{port}/mcp"),
            Err(e) => eprintln!("evolve: no MCP server on port {} ({e})", args.mcp_port),
        }
    }

    let raytracing = args.raytracing;
    let mut app = App::new();
    app.insert_resource(ClearColor(scene::SKY))
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
                        // The app id (WM_CLASS on X11): the desktop finds evolve.desktop by it for
                        // the name and icon (tools/install-desktop.sh).
                        name: Some("evolve".into()),
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
        .insert_resource(observatory::Shots::new(shot_rx))
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
        .add_plugins((miracles::MiraclesPlugin, observatory::ObservatoryPlugin, flow::FlowPlugin, settings::SettingsPlugin))
        .add_systems(PreUpdate, refresh_snapshot)
        .add_systems(Update, (auto_screenshot, load_after))
        .add_systems(Last, log_slow_frames);
    // Only when asked: Solari draws every opaque material deferred from the moment it's added.
    #[cfg(feature = "raytracing")]
    if raytracing {
        app.add_plugins(raytracing::RaytracingPlugin);
    }
    let _ = raytracing;
    app.run()
}

fn refresh_snapshot(link: Res<SimLink>, time: Res<Time>, mut snap: ResMut<Snap>, mut interp: ResMut<Interp>) {
    let _cost = FrameCost::start("refresh_snapshot");
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

/// A frame this slow shows as a pause: logged, with how long since the last one, so pauses can
/// be told apart from the simulation's (the runner logs its own slow passes).
fn log_slow_frames(time: Res<Time<Real>>, frames: Res<FrameCount>, mut last: Local<f64>) {
    let dt = time.delta_secs_f64();
    let costs = std::mem::take(&mut *FRAME_COSTS.lock().unwrap_or_else(|e| e.into_inner()));
    if dt > 0.1 && frames.0 > 120 {
        let now = time.elapsed_secs_f64();
        let mut costs: Vec<(&str, f32)> = costs.into_iter().filter(|c| c.1 >= 1.0).collect();
        costs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let costs: Vec<String> = costs.iter().map(|(n, ms)| format!("{n} {ms:.0}")).collect();
        eprintln!(
            "evolve: slow frame {:.0} ms (frame {}, {:.1} s since the last slow one); updates over 1 ms: {}",
            dt * 1000.0,
            frames.0,
            now - *last,
            if costs.is_empty() { "none (so the time went to drawing)".to_string() } else { costs.join(", ") }
        );
        *last = now;
    }
}

/// Milliseconds the costly update systems took this frame, for `log_slow_frames`.
pub static FRAME_COSTS: std::sync::Mutex<Vec<(&'static str, f32)>> = std::sync::Mutex::new(Vec::new());

/// Times a system: records how long it ran in `FRAME_COSTS` when dropped.
pub struct FrameCost(&'static str, std::time::Instant);

impl FrameCost {
    pub fn start(name: &'static str) -> Self {
        Self(name, std::time::Instant::now())
    }
}

impl Drop for FrameCost {
    fn drop(&mut self) {
        let ms = self.1.elapsed().as_secs_f32() * 1000.0;
        if let Ok(mut c) = FRAME_COSTS.lock() {
            c.push((self.0, ms));
        }
    }
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
