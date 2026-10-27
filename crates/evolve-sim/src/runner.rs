//! Runs the simulation on its own thread. The UI sends `Control`s in and reads the latest
//! `Snapshot` out, so a slow frame never stalls the world and a fast world never stalls the UI.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::commands::Command;
use crate::math::Vec3;
use crate::zoo::{Offers, RainDraw, SharedZoo, Zoo, ZooMeta};
use crate::islands::{Board, Pacer, WorkerCtl, Worker, WorldStatus, main_threads, migrate, trial_params, world_pool};
use crate::params::SimParams;
use crate::mcp::Reply;
use crate::save;
use crate::sim::Sim;
use crate::snapshot::{Highlight, HighlightKind, Snapshot};
use crate::stats::{GenePool, Sample};
use crate::terrain::Terrain;
use crate::wiring::{SketchLimits, Wiring};

pub enum Control {
    Sim(Command),
    Pause(bool),
    /// Multiple of real time; 0 runs as fast as the CPU allows.
    Speed(f32),
    StepOnce,
    Save(PathBuf),
    Load(PathBuf),
    /// Load a world, then set these parameters on it (by name, as `--set` gives them).
    LoadWith(PathBuf, Vec<(String, String)>),
    NewWorld(Box<SimParams>),
    /// Put a living creature of this world in the zoo (pinned).
    ZooAdd(u64),
    /// Drop `count` copies of a zoo creature into a world, at `pos` (this world only) or a
    /// random spot.
    ZooDrop { id: u64, target: ZooTarget, pos: Option<[f32; 3]>, count: u32 },
    ZooRemove(u64),
    ZooPin(u64, bool),
    /// Keep only the best n zoo creatures (and the pinned ones).
    ZooKeepBest(usize),
    /// A question from the MCP server (see `mcp`) for world `world` (0 is this one).
    Ask { world: usize, ask: crate::mcp::Ask, reply: crate::mcp::Reply },
    /// How much of the selected creature's brain the brain view wants sketched; None while it's
    /// closed, so a big sketch costs nothing when nobody's looking.
    BrainView(Option<SketchLimits>),
    Quit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZooTarget {
    /// The world you're watching.
    This,
    /// Background world number k (1-based).
    World(usize),
    /// Every world.
    All,
}

/// Simulated seconds between the one-line summaries each world writes to stderr (the journal,
/// when the app runs as a service).
pub const LOG_EVERY: f64 = 600.0;

struct Shared {
    snapshot: Mutex<Arc<Snapshot>>,
    /// What the world is doing that holds it up (saving, loading), and since when, so the screen
    /// can say so rather than look frozen.
    busy: Mutex<Option<(String, Instant)>>,
}

impl Shared {
    fn busy(&self, what: Option<&str>) {
        if let Ok(mut b) = self.busy.lock() {
            *b = what.map(|w| (w.to_string(), Instant::now()));
        }
    }
}

pub struct Runner {
    tx: Sender<Control>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Runner {
    pub fn start(sim: Sim) -> Self {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared {
            snapshot: Mutex::new(Arc::new(Snapshot::default())),
            busy: Mutex::new(None),
        });
        let s2 = shared.clone();
        let thread = std::thread::Builder::new()
            .name("evolve-sim".into())
            .spawn(move || run(sim, rx, s2, Zoo::shared(Zoo::default_dir())))
            .expect("spawn sim thread");
        Self {
            tx,
            shared,
            thread: Some(thread),
        }
    }

    pub fn send(&self, c: Control) {
        let _ = self.tx.send(c);
    }

    pub fn command(&self, c: Command) {
        self.send(Control::Sim(c));
    }

    /// Another way in to the simulation thread, e.g. for the MCP server.
    pub fn sender(&self) -> Sender<Control> {
        self.tx.clone()
    }

    /// What holds the world up just now (saving, loading), and for how long so far.
    pub fn busy(&self) -> Option<(String, Duration)> {
        self.shared.busy.lock().ok()?.as_ref().map(|(w, t)| (w.clone(), t.elapsed()))
    }

    pub fn snapshot(&self) -> Arc<Snapshot> {
        match self.shared.snapshot.lock() {
            Ok(s) => s.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// False once the simulation thread has stopped (it panicked: the reason is on stderr).
    pub fn alive(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        let _ = self.tx.send(Control::Quit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Caches {
    terrain_version: u64,
    terrain: Arc<Terrain>,
    /// When the terrain was last copied out: creatures digging change it every tick, so the copy
    /// for the renderer is refreshed a few times a second at most.
    terrain_at: Instant,
    history_len: usize,
    history_time: f32,
    history: Arc<Vec<Sample>>,
}

impl Caches {
    fn new(sim: &Sim) -> Self {
        Self {
            terrain_version: sim.terrain.version,
            terrain: Arc::new(sim.terrain.clone()),
            terrain_at: Instant::now(),
            history_len: sim.history.len(),
            history_time: sim.history.last().map(|s| s.time).unwrap_or(-1.0),
            history: Arc::new(sim.history.clone()),
        }
    }

    /// Takes fresh copies of what changed; returns how long copying the terrain took, ms, if it
    /// was copied.
    fn refresh(&mut self, sim: &Sim) -> Option<f32> {
        // The screen's copy of the ground: four times a second, less often on a big terrain (a
        // second at 5000 a side). The copy shares its layers with the world's until the world
        // changes one, which then copies just that layer (`terrain::Layer`).
        let points = sim.terrain.heights.len() as f32;
        let every = Duration::from_secs_f32((points / 25.0e6).max(0.25));
        let mut copied = None;
        if sim.terrain.version != self.terrain_version && self.terrain_at.elapsed() >= every {
            let start = Instant::now();
            self.terrain_version = sim.terrain.version;
            self.terrain = Arc::new(sim.terrain.clone());
            self.terrain_at = Instant::now();
            copied = Some(start.elapsed().as_secs_f32() * 1000.0);
        }
        let last = sim.history.last().map(|s| s.time).unwrap_or(-1.0);
        if sim.history.len() != self.history_len || last != self.history_time {
            self.history_len = sim.history.len();
            self.history_time = last;
            self.history = Arc::new(sim.history.clone());
        }
        copied
    }
}

/// A pass of the runner's loop longer than this is logged with what it was spent on.
const SLOW_PASS_MS: u64 = 120;

/// How full GPU brain memory may get before it holds the brain price up like a busy GPU does (the
/// price rises while memory is fuller than this, and falls more slowly as memory nears it).
pub const MEMORY_TARGET: f32 = 0.9;

/// Turns the world into bytes for a save (it waits for that) and leaves them to be written on
/// other threads. A save still being written finishes first.
fn start_save(sim: &mut Sim, shared: &Shared, saving: &mut Option<(save::Saving, Option<Reply>)>, path: &std::path::Path, reply: Option<Reply>) {
    shared.busy(Some("Saving the world"));
    if let Some(s) = saving.take() {
        finish_save(sim, s);
    }
    match save::begin(sim, path) {
        Ok(s) => *saving = Some((s, reply)),
        Err(e) => {
            sim.log(format!("Save failed: {e}"));
            if let Some(r) = reply {
                let _ = r.send(Err(e));
            }
        }
    }
    shared.busy(None);
}

/// Waits for a save to be written, and says so in the log and to whoever asked for it.
fn finish_save(sim: &mut Sim, (saving, reply): (save::Saving, Option<Reply>)) {
    let (path, held, let_go) = (saving.path().display().to_string(), saving.held, saving.let_go);
    let r = saving.wait().map(|bytes| {
        format!(
            "{path} ({:.1} MB; the world waited {:.1} s, writing took {:.1} s more)",
            bytes as f64 / 1e6,
            held.as_secs_f32(),
            let_go.elapsed().as_secs_f32()
        )
    });
    match &r {
        Ok(m) => sim.log(format!("Saved {m}")),
        Err(e) => sim.log(format!("Save failed: {e}")),
    }
    if let Some(reply) = reply {
        let _ = reply.send(r.map(|m| format!("saved {m}")));
    }
}

/// Moves a brain price on by `error` (how far over its GPU target, negative when under) for
/// `window` seconds: a factor of e in two minutes at full error, faster when over, so an
/// overloaded GPU is relieved quickly.
fn steer_price(price: f32, error: f32, window: f32) -> f32 {
    let rate = if error > 0.0 { 1.0 / 30.0 } else { 1.0 / 120.0 };
    (price * (error * window * rate).exp()).clamp(0.05, 20.0)
}

fn run(mut sim: Sim, rx: Receiver<Control>, shared: Arc<Shared>, zoo: SharedZoo) {
    let mut paused = false;
    let mut speed = 1.0f32;
    let mut pacer = Pacer::steady();
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    // Sized to what the background worlds leave, and made again when there are more or fewer.
    let mut pool_for = sim.params.parallel_worlds;
    let mut pool = world_pool("evolve-sim", main_threads(cores, pool_for as usize));
    let mut last_publish = Instant::now() - Duration::from_secs(1);
    let mut rate_window = (Instant::now(), 0u64);
    let mut steps_per_sec = 0.0f32;
    // Time spent on the screen's copy of the terrain this pass.
    let mut copying = Duration::ZERO;
    let mut cpu_percent = 0.0f32;
    let mut cpu_mark = (Instant::now(), crate::process_cpu_seconds().unwrap_or(0.0));
    let mut events: Vec<String> = Vec::new();
    let mut highlights: Arc<Vec<Highlight>> = Arc::default();
    // The brain sketch: what it was made to, when, and how long it took.
    let mut wiring: Option<(Arc<Wiring>, SketchLimits, Instant, Duration)> = None;
    let mut brain_view: Option<SketchLimits> = None;
    let mut caches = Caches::new(&sim);
    let mut step_once = false;
    let mut board: Board = Arc::new(Mutex::new(GenePool::default()));
    let mut workers: Vec<Worker> = Vec::new();
    // An experiment's arms (see `Worker::spawn_trial`), and a line on each: its world, name and
    // settings.
    let mut trials: Vec<Worker> = Vec::new();
    let mut experiment: Vec<String> = Vec::new();
    // The arms' own brain price (see the steering below).
    let mut trial_price = 1.0f32;
    let mut next_migration = sim.time;
    let mut world_epoch = 1u64;
    let mut next_zoo = sim.time + sim.params.zoo_interval as f64;
    let mut next_log = sim.time + LOG_EVERY;
    let mut zoo_cache: (u64, Arc<Vec<ZooMeta>>) = (0, Arc::new(Vec::new()));
    let zoo_dir = zoo.lock().map(|z| z.dir().display().to_string()).unwrap_or_default();
    // The zoo's disk work is done on other threads so the world doesn't stop for it: storing the
    // world's best, and reading a new rain sample after.
    let mut zoo_offer: Option<JoinHandle<()>> = None;
    let mut rain_draw: Option<JoinHandle<(u64, Vec<(crate::genome::Genome, u32)>)>> = None;
    // A save being compressed and written on other threads, and who asked for it (over MCP).
    let mut saving: Option<(save::Saving, Option<Reply>)> = None;
    // A world that hasn't started yet (not a loaded one) begins with zoo creatures.
    if sim.tick == 0 && sim.creatures.is_empty() {
        crate::zoo::seed_world(&zoo, &mut sim);
    }

    // Background worlds whose thread stopped, already reported.
    let mut reported_stopped: Vec<usize> = Vec::new();
    loop {
        let pass = Instant::now();
        if saving.as_ref().is_some_and(|(s, _)| s.is_finished()) {
            finish_save(&mut sim, saving.take().expect("just looked"));
        }
        // Background worlds must exist before commands that target them are handled.
        sync_workers(&mut workers, &sim, speed, paused, &board, &zoo);
        let mut reset_islands = false;
        // A new world asked for (by the New world button or the MCP server), started once the
        // messages are read.
        let mut new_world: Option<Box<SimParams>> = None;
        // Zoo creatures bound for background worlds go out after a new or loaded world has
        // restarted them, so a drop sent alongside a new world isn't lost with the old workers.
        let mut worker_drops: Vec<(ZooTarget, Box<crate::genome::Genome>, u32, u32)> = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(Control::Quit) | Err(TryRecvError::Disconnected) => {
                    // Not quitting halfway through putting genomes in the zoo, or writing a save.
                    if let Some(h) = zoo_offer.take() {
                        let _ = h.join();
                    }
                    if let Some(s) = saving.take() {
                        finish_save(&mut sim, s);
                    }
                    crate::record::flush();
                    return;
                }
                Ok(Control::Sim(c)) => {
                    if let Command::SetParams(p) = &c {
                        for w in &workers {
                            w.send(WorkerCtl::Params(p.clone()));
                        }
                    }
                    sim.apply(c);
                }
                Ok(Control::Pause(p)) => {
                    paused = p;
                    // (An experiment's arms keep their own pace, but pause with the rest.)
                    workers.iter().chain(&trials).for_each(|w| w.send(WorkerCtl::Pause(p)));
                }
                Ok(Control::Speed(s)) => {
                    speed = s.max(0.0);
                    workers.iter().for_each(|w| w.send(WorkerCtl::Speed(speed)));
                }
                Ok(Control::StepOnce) => step_once = true,
                Ok(Control::BrainView(limits)) => brain_view = limits,
                Ok(Control::Save(path)) => start_save(&mut sim, &shared, &mut saving, &path, None),
                Ok(Control::Load(path)) => {
                    shared.busy(Some("Loading the world"));
                    // A save still being written finishes first (it may be the one asked for).
                    if let Some(s) = saving.take() {
                        finish_save(&mut sim, s);
                    }
                    match save::load(&path) {
                        Ok(loaded) => {
                            sim = loaded;
                            caches = Caches::new(&sim);
                            reset_islands = true;
                            sim.log(format!("Loaded {}", path.display()));
                            sim.record_event(format!("loaded from {}", path.display()));
                        }
                        Err(e) => sim.log(format!("Load failed: {e}")),
                    }
                    shared.busy(None);
                }
                Ok(Control::LoadWith(path, sets)) => {
                    shared.busy(Some("Loading the world"));
                    if let Some(s) = saving.take() {
                        finish_save(&mut sim, s);
                    }
                    match save::load(&path) {
                        Ok(mut loaded) => {
                            // Settings given with it override the loaded world's own.
                            let mut p = loaded.params.clone();
                            for (k, v) in &sets {
                                if let Err(e) = p.set_by_name(k, v) {
                                    loaded.log(format!("{k}: {e}"));
                                }
                            }
                            loaded.set_params(p);
                            sim = loaded;
                            caches = Caches::new(&sim);
                            reset_islands = true;
                            sim.log(format!("Loaded {}", path.display()));
                            sim.record_event(format!("loaded from {}", path.display()));
                        }
                        Err(e) => sim.log(format!("Load failed: {e}")),
                    }
                    shared.busy(None);
                }
                Ok(Control::NewWorld(p)) => new_world = Some(p),
                Ok(Control::ZooAdd(cid)) => {
                    if let Some(c) = sim.creature(cid) {
                        let score = sim.life_score(c);
                        let source = format!("picked by hand, seed {}, {:.0} min", sim.params.seed, sim.time / 60.0);
                        let max = sim.params.zoo_max_entries as usize;
                        let added = zoo
                            .lock()
                            .map_err(|_| "zoo is unavailable".to_string())
                            .and_then(|mut z| z.add(&c.genome, score, c.generation, &source, true, max));
                        match added {
                            Ok(Some(_)) => sim.log(format!("Creature #{cid} is in the zoo")),
                            Ok(None) => sim.log(format!("Creature #{cid} was already in the zoo (now pinned)")),
                            Err(e) => sim.log(format!("Zoo: {e}")),
                        }
                    }
                }
                Ok(Control::ZooDrop { id, target, pos, count }) => {
                    let found = zoo.lock().map_err(|_| "zoo is unavailable".to_string()).and_then(|z| {
                        let meta = z.get(id).cloned().ok_or("no such zoo creature")?;
                        Ok((meta, z.genome(id)?))
                    });
                    match found {
                        Ok((meta, genome)) => {
                            let here = matches!(target, ZooTarget::This | ZooTarget::All);
                            if here {
                                sim.drop_genome(&genome, meta.generation, pos.map(Vec3::from), count);
                            }
                            let worlds = sim.params.parallel_worlds as usize;
                            match target {
                                ZooTarget::World(k) if k == 0 || k > worlds => {
                                    sim.log(format!("Zoo: there is no background world {k}"));
                                    continue;
                                }
                                ZooTarget::This => {}
                                _ => worker_drops.push((target, Box::new(genome), meta.generation, count)),
                            }
                            let whom = match target {
                                ZooTarget::This => "this world".to_string(),
                                ZooTarget::World(k) => format!("background world {k}"),
                                ZooTarget::All => "every world".to_string(),
                            };
                            sim.log(format!("Dropped {count} x {} into {whom}", meta.name));
                        }
                        Err(e) => sim.log(format!("Zoo: {e}")),
                    }
                }
                Ok(Control::ZooRemove(id)) => {
                    if let Ok(mut z) = zoo.lock() {
                        let name = z.get(id).map(|e| e.name.clone()).unwrap_or_default();
                        match z.remove(id) {
                            Ok(()) => sim.log(format!("Zoo: {name} deleted (its genome is in the zoo's trash folder)")),
                            Err(e) => sim.log(format!("Zoo: {e}")),
                        }
                    }
                }
                Ok(Control::ZooKeepBest(n)) => {
                    if let Ok(mut z) = zoo.lock() {
                        match z.keep_best(n) {
                            Ok(gone) => sim.log(format!(
                                "Zoo: kept the best {n} and the pinned, {gone} deleted (their genomes are in the zoo's trash folder)"
                            )),
                            Err(e) => sim.log(format!("Zoo: {e}")),
                        }
                    }
                }
                Ok(Control::Ask { world, ask, reply }) => {
                    use crate::mcp::Ask;
                    match ask {
                        Ask::Worlds => {
                            let _ = reply.send(Ok((workers.len() + trials.len()).to_string()));
                        }
                        Ask::StartExperiment { arms, replicates, speed: pace } => {
                            let arms: Vec<(String, Vec<(String, String)>)> =
                                std::iter::once(("control".to_string(), Vec::new())).chain(arms).collect();
                            let n = arms.len() * replicates as usize;
                            let planned: Result<Vec<_>, String> = (0..replicates)
                                .flat_map(|r| arms.iter().map(move |arm| (r, arm)))
                                .map(|(r, (label, set))| {
                                    trial_params(&sim.params, r, set, n).map(|p| (r, label, set, p)).map_err(|e| format!("{label}: {e}"))
                                })
                                .collect();
                            match planned {
                                Err(e) => {
                                    let _ = reply.send(Err(format!("{e}; no experiment started")));
                                }
                                Ok(planned) => {
                                    trials.clear();
                                    experiment.clear();
                                    trial_price = sim.brain_price;
                                    let first = 1 + workers.len();
                                    for (k, (r, label, set, p)) in planned.into_iter().enumerate() {
                                        let index = first + k;
                                        let name = if replicates > 1 { format!("{label} #{}", r + 1) } else { label.clone() };
                                        let settings = if set.is_empty() {
                                            "the world on screen's settings".to_string()
                                        } else {
                                            set.iter().map(|(k, v)| format!("{k} = {v}")).collect::<Vec<_>>().join(", ")
                                        };
                                        experiment.push(format!("world {index}: {name}: {settings}"));
                                        trials.push(Worker::spawn_trial(index, p, name, sim.pool.clone(), pace, paused, zoo.clone()));
                                    }
                                    sim.log(format!("Experiment started: {} arms on {replicates} islands", arms.len()));
                                    sim.record_event(format!("experiment started: {}", experiment.join("; ")));
                                    let pace = if pace > 0.0 { format!("{pace}x real time") } else { "Max".to_string() };
                                    let _ = reply.send(Ok(format!(
                                        "{}\nEach runs at {pace}; experiment_report compares them.",
                                        experiment.join("\n")
                                    )));
                                }
                            }
                        }
                        Ask::StopExperiment => {
                            let had = trials.len();
                            trials.clear();
                            experiment.clear();
                            if had > 0 {
                                sim.log("Experiment stopped");
                                sim.record_event("experiment stopped");
                            }
                            let _ = reply.send(Ok(format!("stopped {had} experiment worlds")));
                        }
                        Ask::Experiment => {
                            let _ = reply.send(if experiment.is_empty() {
                                Err("no experiment is running (start_experiment starts one)".into())
                            } else {
                                Ok(experiment.join("\n"))
                            });
                        }
                        Ask::SetParam { name, value } => {
                            let mut p = sim.params.clone();
                            let before = sim.params_text(&name);
                            match p.set_by_name(&name, &value) {
                                Ok(()) => {
                                    for w in &workers {
                                        w.send(WorkerCtl::Params(Box::new(p.clone())));
                                    }
                                    sim.apply(Command::SetParams(Box::new(p)));
                                    sim.log(format!("{name} set to {value} (by the MCP server)"));
                                    let _ = reply.send(Ok(format!("was: {before}now: {}", sim.params_text(&name))));
                                }
                                Err(e) => {
                                    let _ = reply.send(Err(e));
                                }
                            }
                        }
                        Ask::SetParams(pairs) => {
                            let mut p = sim.params.clone();
                            match pairs.iter().try_for_each(|(k, v)| p.set_by_name(k, v)) {
                                Ok(()) => {
                                    for w in &workers {
                                        w.send(WorkerCtl::Params(Box::new(p.clone())));
                                    }
                                    sim.apply(Command::SetParams(Box::new(p)));
                                    let names: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
                                    sim.log(format!("{} set (by the MCP server)", names.join(", ")));
                                    let now: String = names.iter().map(|k| sim.params_text(k)).collect();
                                    let _ = reply.send(Ok(format!("now:\n{now}")));
                                }
                                Err(e) => {
                                    let _ = reply.send(Err(format!("{e}; nothing was changed")));
                                }
                            }
                        }
                        Ask::RegenerateTerrain => {
                            let started = Instant::now();
                            sim.apply(Command::RegenerateTerrain);
                            let t = &sim.terrain;
                            let _ = reply.send(Ok(format!(
                                "the world on screen has new terrain: {} x {} samples over {:.0} m ({:.2} m apart), made in {:.1} s",
                                t.res,
                                t.res,
                                t.size,
                                t.size / (t.res - 1) as f32,
                                started.elapsed().as_secs_f32()
                            )));
                        }
                        Ask::NewWorld => {
                            new_world = Some(Box::new(sim.params.clone()));
                            let _ = reply.send(Ok("starting a new world with the current settings".into()));
                        }
                        // Answered once the file is written.
                        Ask::Save(path) => start_save(&mut sim, &shared, &mut saving, &path, Some(reply)),
                        // The world on screen leads with how fast it runs against real time.
                        Ask::Summary if world == 0 => {
                            let rate = steps_per_sec / crate::sim::TICKS_PER_SECOND as f32;
                            let how = if paused { " (paused)" } else if speed == 0.0 { " at Max" } else { "" };
                            let _ = reply.send(Ok(format!("{rate:.2}x real time{how}; {}", sim.summary_line(600.0))));
                        }
                        Ask::Line { minutes } if world == 0 => {
                            let rate = steps_per_sec / crate::sim::TICKS_PER_SECOND as f32;
                            let _ = reply.send(Ok(format!("{rate:.2}x real time; {}", sim.summary_line(minutes * 60.0))));
                        }
                        ask if world == 0 => sim.answer(ask, reply),
                        ask => match trials.iter().find(|t| t.index == world).or_else(|| workers.get(world - 1)) {
                            Some(w) if w.alive() => w.send(WorkerCtl::Ask(ask, reply)),
                            _ => {
                                let _ = reply.send(Err(format!("there is no background world {world} running")));
                            }
                        },
                    }
                }
                Ok(Control::ZooPin(id, pinned)) => {
                    if let Ok(mut z) = zoo.lock() {
                        z.set_pinned(id, pinned);
                    }
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(p) = new_world {
            sim = Sim::new(*p);
            crate::zoo::seed_world(&zoo, &mut sim);
            caches = Caches::new(&sim);
            reset_islands = true;
            sim.log("New world");
            sim.record_event("new world");
        }

        // Background worlds: restart them for a new or loaded world, and match the param.
        if reset_islands {
            world_epoch += 1;
            highlights = Arc::default();
            wiring = None;
            workers.clear();
            // An experiment was on the old world's settings and gene pool.
            if !trials.is_empty() {
                sim.log("Experiment stopped with the old world");
            }
            trials.clear();
            experiment.clear();
            board = Arc::new(Mutex::new(GenePool::default()));
            next_migration = sim.time;
        }
        sync_workers(&mut workers, &sim, speed, paused, &board, &zoo);
        if reset_islands {
            reported_stopped.clear();
        }
        for (target, genome, generation, count) in worker_drops {
            for (k, w) in workers.iter().enumerate() {
                if target == ZooTarget::All || target == ZooTarget::World(k + 1) {
                    w.send(WorkerCtl::Spawn { genome: genome.clone(), generation, count });
                }
            }
        }
        for (k, w) in workers.iter().enumerate() {
            if !w.alive() && !reported_stopped.contains(&k) {
                reported_stopped.push(k);
                sim.log(format!("Background world {} stopped with an error (see the terminal)", k + 1));
            }
        }

        if step_once {
            sim.step();
            rate_window.1 += 1;
            step_once = false;
        }
        let (before_steps, before_tick) = (Instant::now(), sim.tick);
        if sim.params.parallel_worlds != pool_for {
            pool_for = sim.params.parallel_worlds;
            pool = world_pool("evolve-sim", main_threads(cores, pool_for as usize));
        }
        rate_window.1 += pacer.run(&mut sim, speed, paused, Duration::from_millis(30), Some(&pool));
        let stepping = before_steps.elapsed();
        let stepped = before_tick..sim.tick;
        if reset_islands {
            next_zoo = sim.time + sim.params.zoo_interval as f64;
        }
        if sim.time >= next_zoo {
            if zoo_offer.as_ref().is_none_or(|h| h.is_finished()) {
                let (offers, zoo) = (Offers::of(&sim, 0), zoo.clone());
                zoo_offer = std::thread::Builder::new().name("evolve-zoo".into()).spawn(move || offers.give(&zoo)).ok();
            }
            next_zoo = sim.time + sim.params.zoo_interval as f64;
        }
        if sim.time >= next_log {
            eprintln!("evolve world 0: {}", sim.summary_line(LOG_EVERY));
            next_log = sim.time + LOG_EVERY;
        }
        if sim.time >= next_migration {
            let interval = sim.params.migration_interval;
            migrate(&mut sim.pool, &board, interval, false);
            if rain_draw.is_none()
                && let Some(draw) = RainDraw::due(&zoo, &sim)
            {
                rain_draw = std::thread::Builder::new().name("evolve-rain-sample".into()).spawn(move || draw.read()).ok();
            }
            next_migration = sim.time + interval as f64;
        }
        if rain_draw.as_ref().is_some_and(|h| h.is_finished())
            && let Some(Ok(drawn)) = rain_draw.take().map(|h| h.join())
        {
            (sim.zoo_sample_version, sim.zoo_sample) = drawn;
        }

        let window = rate_window.0.elapsed().as_secs_f32();
        if window >= 0.5 {
            // Steer the price of brains toward the GPU target, judged at Max speed only: at a
            // set speed the GPU idles by design.
            let steering = speed == 0.0 && !paused && sim.params.gpu_target > 0.0 && sim.brains.is_some();
            let busy_of = |ws: &[Worker]| -> f32 {
                ws.iter().filter(|w| w.alive()).filter_map(|w| w.status.lock().ok().map(|s| s.gpu_busy)).sum()
            };
            // Brain memory full counts as busy too: brains that don't fit are trimmed, so their
            // price mustn't fall for want of GPU time while there's no room for them (it sat at
            // the floor with memory 105% full, and brains grew into halving).
            let fill_of = |ws: &[Worker]| -> f32 {
                ws.iter().filter(|w| w.alive()).filter_map(|w| w.status.lock().ok().map(|s| s.memory_fill)).fold(0.0, f32::max)
            };
            let busy = sim.perf.gpu_busy + busy_of(&workers);
            if steering {
                let fill = sim.brain_memory_fill().max(fill_of(&workers));
                let error = (busy - sim.params.gpu_target).max(fill - MEMORY_TARGET);
                sim.brain_price = steer_price(sim.brain_price, error, window);
                sim.gpu_headroom = -error;
            } else {
                sim.gpu_headroom = 0.0;
                if sim.params.gpu_target <= 0.0 {
                    sim.brain_price = 1.0;
                }
            }
            for w in &workers {
                w.send(WorkerCtl::BrainPrice(sim.brain_price, sim.gpu_headroom));
            }
            // An experiment's arms have a price of their own, steered into the GPU time this world
            // and its islands leave (a twentieth at least). Counted in with this world's, their
            // load made brains here 15 times dearer, and this world's brains shrank to a quarter.
            if !trials.is_empty() && !paused && sim.params.gpu_target > 0.0 {
                let room = (sim.params.gpu_target - busy).max(0.05);
                let error = (busy_of(&trials) - room).max(fill_of(&trials) - MEMORY_TARGET);
                trial_price = steer_price(trial_price, error, window);
                for w in &trials {
                    w.send(WorkerCtl::BrainPrice(trial_price, -error));
                }
            }
            // Smoothed over a second or two, so the speed shown doesn't flicker with each
            // half-second's luck.
            let rate = rate_window.1 as f32 / window;
            steps_per_sec = if steps_per_sec <= 0.0 { rate } else { steps_per_sec + (rate - steps_per_sec) * 0.3 };
            rate_window = (Instant::now(), 0);
            if let Some(now) = crate::process_cpu_seconds() {
                let (at, before) = cpu_mark;
                cpu_percent = ((now - before) / at.elapsed().as_secs_f64() * 100.0) as f32;
                cpu_mark = (Instant::now(), now);
            }
        }

        if last_publish.elapsed() >= Duration::from_millis(15) {
            last_publish = Instant::now();
            events.append(&mut sim.events);
            if events.len() > 8 {
                events.drain(..events.len() - 8);
            }
            if !sim.highlights.is_empty() {
                // Impacts are many and only matter for sound; keep them from crowding out births
                // and kills by keeping the latest of each separately.
                let all: Vec<Highlight> = highlights.iter().copied().chain(sim.highlights.drain(..)).collect();
                let frequent = |h: &Highlight| matches!(h.kind, HighlightKind::Impact | HighlightKind::Call);
                let latest = |impacts: bool, n: usize| {
                    let v: Vec<Highlight> = all.iter().filter(|h| frequent(h) == impacts).copied().collect();
                    v[v.len().saturating_sub(n)..].to_vec()
                };
                let mut h = latest(false, 64);
                h.extend(latest(true, 48));
                h.sort_by_key(|h| h.seq);
                highlights = Arc::new(h);
            }
            let publish = Instant::now();
            if let Some(ms) = caches.refresh(&sim) {
                sim.perf.terrain_copy_ms += (ms - sim.perf.terrain_copy_ms) * 0.2;
            }
            copying = publish.elapsed();
            let mut snap = sim.snapshot(caches.terrain.clone(), caches.history.clone());
            let ms = publish.elapsed().as_secs_f32() * 1000.0;
            sim.perf.publish_ms += (ms - sim.perf.publish_ms) * 0.05;
            // The brain sketch is redone once a second while the brain view is open (learning
            // moves the weights), and at once for another creature. Every neuron and 100,000
            // connections of a big brain hold the world up a while, so a sketch is never
            // redone sooner than ten times as long as it took (and new limits from a slider
            // being dragged wait a moment).
            if let Some(limits) = brain_view
                && let Some(sel) = snap.selected.as_mut()
            {
                let stale = wiring.as_ref().is_none_or(|(w, made_to, at, took)| {
                    let every = if *made_to == limits { Duration::from_secs(1) } else { Duration::from_millis(200) };
                    w.creature != sel.id || at.elapsed() > every.max(*took * 10)
                });
                if stale && let Some(c) = sim.creature(sel.id) {
                    let start = Instant::now();
                    let w = pool.install(|| crate::wiring::sketch(c, limits));
                    wiring = Some((Arc::new(w), limits, Instant::now(), start.elapsed()));
                }
                if let Some((w, ..)) = &wiring {
                    sel.wiring = w.clone();
                }
            } else {
                wiring = None;
            }
            snap.paused = paused;
            snap.speed = speed;
            snap.steps_per_sec = steps_per_sec;
            snap.cpu_percent = cpu_percent;
            snap.world_epoch = world_epoch;
            // Not waiting while the zoo is being written to: the list on screen catches up later.
            if let Ok(z) = zoo.try_lock()
                && z.version != zoo_cache.0
            {
                zoo_cache = (z.version, Arc::new(z.entries.clone()));
            }
            snap.zoo = zoo_cache.1.clone();
            snap.zoo_dir = zoo_dir.clone();
            snap.events = events.clone();
            snap.highlights = highlights.clone();
            snap.worlds = std::iter::once(WorldStatus::of(0, &sim, steps_per_sec))
                .chain(workers.iter().chain(&trials).filter_map(|w| {
                    let mut s = w.status.lock().ok()?.clone();
                    s.stopped = !w.alive();
                    Some(s)
                }))
                .collect();
            if let Ok(mut s) = shared.snapshot.lock() {
                *s = Arc::new(snap);
            }
        }
        // A pass this slow shows as a pause on screen: say what it was spent on.
        let took = pass.elapsed();
        if took > Duration::from_millis(SLOW_PASS_MS) {
            let ms = |d: Duration| d.as_secs_f32() * 1000.0;
            let hydro = stepped.clone().any(|t| t % (crate::sim::TICKS_PER_SECOND * 2) == 0);
            // The costliest phases of a step lately (see `PHASE_NAMES`).
            let mut phases: Vec<(f32, &str)> = sim.perf.phases.iter().copied().zip(crate::sim::PHASE_NAMES).collect();
            phases.sort_by(|a, b| b.0.total_cmp(&a.0));
            let top: Vec<String> = phases.iter().take(4).map(|(t, n)| format!("{n} {t:.1}")).collect();
            eprintln!(
                "evolve: slow pass {:.0} ms: stepping {:.0} ms ({} steps{}), copying the terrain {:.0} ms; per step lately {}",
                ms(took),
                ms(stepping),
                stepped.end - stepped.start,
                if hydro { ", with water" } else { "" },
                ms(copying),
                top.join(", ")
            );
        }
        copying = Duration::ZERO;

        if paused || speed > 0.0 {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

/// Starts or stops background worlds to match the `parallel_worlds` param.
fn sync_workers(workers: &mut Vec<Worker>, sim: &Sim, speed: f32, paused: bool, board: &Board, zoo: &SharedZoo) {
    let want = sim.params.parallel_worlds as usize;
    while workers.len() > want {
        workers.pop();
    }
    while workers.len() < want {
        let index = workers.len() + 1;
        workers.push(Worker::spawn(index, sim.params.clone(), speed, paused, board.clone(), zoo.clone()));
    }
}
