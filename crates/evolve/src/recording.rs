//! Recording the window to a video file (F10, or the toolbar's Record button). Every frame the
//! window is captured as shown, the 3D view with whatever panels and overlays are open, and
//! streamed to an ffmpeg process that encodes it on the GPU (NVENC) if it can, else on the CPU.
//!
//! Frames are paced by the wall clock to a steady 30 per second, so the video plays back in real
//! time however the frame rate wanders: a slow frame is shown for longer, and frames that come
//! faster than that are skipped. If encoding falls behind, frames are dropped rather than
//! stalling the app. A window resized mid-recording starts a new file (`-2`, `-3`, ...).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy_egui::input::egui_wants_any_keyboard_input;

/// Frames per second of the video.
const FPS: u32 = 30;
/// Frames waiting for the encoder; past this, new frames are dropped rather than stall the app.
const QUEUE: usize = 8;
/// How long quitting waits for the last file to be finished.
const EXIT_WAIT: Duration = Duration::from_secs(10);

pub struct RecordingPlugin;

impl Plugin for RecordingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Recorder>()
            .add_systems(Startup, record_at_launch)
            .add_systems(Update, hotkey.run_if(not(egui_wants_any_keyboard_input)))
            .add_systems(Update, (control, capture).chain().after(hotkey))
            .add_systems(Last, stop_on_exit);
    }
}

#[derive(Resource, Default)]
pub struct Recorder {
    session: Option<Session>,
    /// Files being finished after a stop.
    finishing: Vec<JoinHandle<Result<Vec<PathBuf>, String>>>,
    /// Set by the button or the key: start if stopped, stop if recording.
    pub toggle: bool,
    /// Where the next recording goes, instead of a new name in the videos folder (`--record`).
    next_path: Option<PathBuf>,
    /// The latest news for the UI (a finished file, or why recording failed) and when it came.
    pub notice: Option<(String, Instant)>,
}

struct Session {
    started: Instant,
    tx: SyncSender<Frame>,
    worker: JoinHandle<Result<Vec<PathBuf>, String>>,
    dropped: Arc<AtomicU64>,
    /// The last video frame slot (1/30 s) a capture was asked for: faster frames aren't needed.
    slot: Option<u64>,
}

impl Recorder {
    /// How long the current recording has run, if one is.
    pub fn elapsed(&self) -> Option<Duration> {
        self.session.as_ref().map(|s| s.started.elapsed())
    }

    /// The news to show, while it's fresh.
    pub fn fresh_notice(&self) -> Option<&str> {
        self.notice
            .as_ref()
            .filter(|(_, at)| at.elapsed() < Duration::from_secs(20))
            .map(|(n, _)| n.as_str())
    }

    fn tell(&mut self, msg: String) {
        eprintln!("evolve: {msg}");
        self.notice = Some((msg, Instant::now()));
    }

    fn start(&mut self) {
        let path = self.next_path.take().unwrap_or_else(new_video_path);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let (tx, rx) = mpsc::sync_channel(QUEUE);
        let worker = std::thread::Builder::new()
            .name("evolve-recording".into())
            .spawn({
                let path = path.clone();
                move || encode(rx, path)
            });
        match worker {
            Ok(worker) => {
                self.session = Some(Session { started: Instant::now(), tx, worker, dropped: Arc::default(), slot: None });
                self.tell(format!("Recording to {}", path.display()));
            }
            Err(e) => self.tell(format!("Couldn't start recording: {e}")),
        }
    }

    /// Stops recording; the file is finished in the background.
    fn stop(&mut self) {
        if let Some(s) = self.session.take() {
            let dropped = s.dropped.load(Ordering::Relaxed);
            if dropped > 0 {
                eprintln!("evolve: recording dropped {dropped} frames the encoder couldn't keep up with");
            }
            drop(s.tx);
            self.finishing.push(s.worker);
        }
    }

    fn report(&mut self, result: std::thread::Result<Result<Vec<PathBuf>, String>>) {
        match result {
            Ok(Ok(files)) if !files.is_empty() => {
                let names: Vec<String> = files.iter().map(|f| f.display().to_string()).collect();
                self.tell(format!("Recording saved: {}", names.join(", ")));
            }
            Ok(Ok(_)) => self.tell("Recording stopped before any frame was captured".into()),
            Ok(Err(e)) => self.tell(format!("Recording failed: {e}")),
            Err(_) => self.tell("Recording failed: the encoder thread panicked".into()),
        }
    }
}

impl Drop for Recorder {
    /// Quitting while recording (or while a file is still being finished) waits a little for
    /// the file to be completed, so it's playable.
    fn drop(&mut self) {
        self.stop();
        let deadline = Instant::now() + EXIT_WAIT;
        for worker in self.finishing.drain(..) {
            while !worker.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            if worker.is_finished() {
                match worker.join() {
                    Ok(Ok(files)) => files.iter().for_each(|f| eprintln!("evolve: recording saved: {}", f.display())),
                    Ok(Err(e)) => eprintln!("evolve: recording failed: {e}"),
                    Err(_) => eprintln!("evolve: recording failed: the encoder thread panicked"),
                }
            } else {
                eprintln!("evolve: gave up waiting for the recording to finish");
            }
        }
    }
}

fn record_at_launch(args: Res<crate::Args>, mut rec: ResMut<Recorder>) {
    // Find out now (it takes a moment) whether the GPU encoder works, not when recording starts.
    let _ = std::thread::Builder::new().name("evolve-nvenc-probe".into()).spawn(nvenc_works);
    if let Some(p) = &args.record {
        rec.next_path = Some(p.clone());
        rec.toggle = true;
    }
}

fn hotkey(keys: Res<ButtonInput<KeyCode>>, mut rec: ResMut<Recorder>) {
    if keys.just_pressed(KeyCode::F10) {
        rec.toggle = true;
    }
}

fn control(mut rec: ResMut<Recorder>) {
    if std::mem::take(&mut rec.toggle) {
        if rec.session.is_some() {
            rec.stop();
        } else {
            rec.start();
        }
    }
    // The encoder gave up on its own (no ffmpeg, a full disk...).
    if rec.session.as_ref().is_some_and(|s| s.worker.is_finished()) {
        let s = rec.session.take().expect("checked");
        drop(s.tx);
        let result = s.worker.join();
        rec.report(result);
    }
    let mut i = 0;
    while i < rec.finishing.len() {
        if rec.finishing[i].is_finished() {
            let result = rec.finishing.swap_remove(i).join();
            rec.report(result);
        } else {
            i += 1;
        }
    }
}

/// Asks for this frame's image of the window (once per video frame). It arrives a frame or two
/// later.
fn capture(mut commands: Commands, mut rec: ResMut<Recorder>, mut exits: MessageReader<AppExit>) {
    let Some(s) = rec.session.as_mut() else {
        return;
    };
    // Quitting: the image would never arrive.
    if exits.read().next().is_some() {
        return;
    }
    let at = s.started.elapsed();
    let slot = (at.as_secs_f64() * FPS as f64) as u64;
    if s.slot.is_some_and(|last| slot <= last) {
        return;
    }
    s.slot = Some(slot);
    let (tx, dropped) = (s.tx.clone(), s.dropped.clone());
    commands.spawn(Screenshot::primary_window()).observe(move |mut cap: On<ScreenshotCaptured>| {
        let image = &mut cap.event_mut().image;
        let (width, height) = (image.width(), image.height());
        let Some(pixels) = Pixels::of(image.texture_descriptor.format) else {
            return;
        };
        let Some(data) = image.data.take() else {
            return;
        };
        if tx.try_send(Frame { at, width, height, pixels, data }).is_err() {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
    });
}

fn stop_on_exit(mut exits: MessageReader<AppExit>, mut rec: ResMut<Recorder>) {
    if exits.read().next().is_some() {
        rec.stop();
    }
}

/// `~/Videos/evolve-YYYY-MM-DD-HHMMSS.mp4` (or a `videos` folder next to the saves).
fn new_video_path() -> PathBuf {
    let home_videos = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Videos"));
    let dir = home_videos.filter(|d| std::fs::create_dir_all(d).is_ok()).unwrap_or_else(|| {
        evolve_sim::save::default_dir()
            .parent()
            .map(|p| p.join("videos"))
            .unwrap_or_else(|| PathBuf::from("videos"))
    });
    let stamp = Command::new("date")
        .arg("+%Y-%m-%d-%H%M%S")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            secs.to_string()
        });
    dir.join(format!("evolve-{stamp}.mp4"))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pixels {
    Bgra,
    Rgba,
}

impl Pixels {
    /// The window's own formats (8-bit; already sRGB-encoded either way).
    fn of(f: TextureFormat) -> Option<Self> {
        match f {
            TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => Some(Self::Bgra),
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => Some(Self::Rgba),
            _ => None,
        }
    }

    fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::Bgra => "bgra",
            Self::Rgba => "rgba",
        }
    }
}

struct Frame {
    /// When it was captured, since recording started.
    at: Duration,
    width: u32,
    height: u32,
    pixels: Pixels,
    data: Vec<u8>,
}

/// The encoder thread: writes frames to ffmpeg at a steady rate until the channel closes.
/// Returns the files written.
fn encode(rx: Receiver<Frame>, path: PathBuf) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut out: Option<Ffmpeg> = None;
    let mut last: Option<Vec<u8>> = None;
    for frame in rx {
        let fits = out
            .as_ref()
            .is_some_and(|o| (o.width, o.height, o.pixels) == (frame.width, frame.height, frame.pixels));
        if !fits {
            if let Some(o) = out.take() {
                files.push(o.finish()?);
            }
            let part = if files.is_empty() { path.clone() } else { numbered(&path, files.len() + 1) };
            out = Some(Ffmpeg::start(part, frame.width, frame.height, frame.pixels, frame.at)?);
            last = None;
        }
        let o = out.as_mut().expect("just started");
        // How many video frames should exist once this one is shown.
        let due = ((frame.at.saturating_sub(o.began)).as_secs_f64() * FPS as f64).floor() as u64 + 1;
        // A gap: hold the previous picture until this one's time.
        if let Some(prev) = &last {
            while o.written + 1 < due {
                o.write(prev)?;
            }
        }
        if o.written < due {
            o.write(&frame.data)?;
        }
        last = Some(frame.data);
    }
    if let Some(o) = out.take() {
        files.push(o.finish()?);
    }
    Ok(files)
}

/// `name-2.mp4` for `name.mp4`.
fn numbered(path: &Path, n: usize) -> PathBuf {
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "evolve".into());
    let ext = path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_else(|| "mp4".into());
    path.with_file_name(format!("{stem}-{n}.{ext}"))
}

struct Ffmpeg {
    child: Child,
    stdin: Option<ChildStdin>,
    path: PathBuf,
    log: PathBuf,
    width: u32,
    height: u32,
    pixels: Pixels,
    /// Capture time of the first frame: the video's time zero.
    began: Duration,
    written: u64,
}

/// Whether this machine's ffmpeg can encode on the GPU, found out once with a tiny test encode.
fn nvenc_works() -> bool {
    static WORKS: OnceLock<bool> = OnceLock::new();
    *WORKS.get_or_init(|| {
        Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.2"])
            .args(["-c:v", "h264_nvenc", "-f", "null", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

impl Ffmpeg {
    fn start(path: PathBuf, width: u32, height: u32, pixels: Pixels, began: Duration) -> Result<Self, String> {
        let codec: &[&str] = if nvenc_works() {
            &["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
        } else {
            &["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"]
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let log = std::env::temp_dir().join(format!("evolve-ffmpeg-{}-{n}.log", std::process::id()));
        let log_file = std::fs::File::create(&log).map_err(|e| format!("create {}: {e}", log.display()))?;
        let size = format!("{width}x{height}");
        let rate = FPS.to_string();
        let mut child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "rawvideo", "-pix_fmt", pixels.ffmpeg_name(), "-video_size", &size, "-framerate", &rate])
            .args(["-i", "-"])
            // yuv420p needs even sizes.
            .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            .args(codec)
            .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(log_file))
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    "ffmpeg isn't installed (sudo apt install ffmpeg)".to_string()
                } else {
                    format!("starting ffmpeg: {e}")
                }
            })?;
        let stdin = child.stdin.take();
        Ok(Self { child, stdin, path, log, width, height, pixels, began, written: 0 })
    }

    fn write(&mut self, data: &[u8]) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("ffmpeg's input is closed")?;
        if let Err(e) = stdin.write_all(data) {
            // ffmpeg has quit: its log says why.
            self.stdin = None;
            let _ = self.child.wait();
            let why = std::fs::read_to_string(&self.log).unwrap_or_default();
            return Err(format!("ffmpeg stopped ({e}): {}", why.trim()));
        }
        self.written += 1;
        Ok(())
    }

    /// Closes ffmpeg's input and waits for it to finish the file.
    fn finish(mut self) -> Result<PathBuf, String> {
        drop(self.stdin.take());
        let status = self.child.wait().map_err(|e| format!("waiting for ffmpeg: {e}"))?;
        let why = std::fs::read_to_string(&self.log).unwrap_or_default();
        let _ = std::fs::remove_file(&self.log);
        if status.success() {
            Ok(self.path)
        } else {
            Err(format!("ffmpeg failed ({status}): {}", why.trim()))
        }
    }
}
