//! Background worlds (an "island model"): extra copies of the simulation evolve on spare cores
//! and swap their best genomes through a shared board, so the world you watch keeps receiving
//! the best brains and bodies evolved anywhere.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::params::SimParams;
use crate::sim::{Sim, TICKS_PER_SECOND};
use crate::stats::GenePool;
use crate::zoo::{SharedZoo, store_best};

/// How many genomes travel each way per migration.
const MIGRANTS: usize = 8;
/// Threads for the world you watch, which steps its physics worlds (`physics_shards`) on them,
/// at least (see `main_threads`).
pub const MAIN_THREADS: usize = 12;
/// Threads for each background world.
pub const BACKGROUND_THREADS: usize = 4;

/// Threads for the world you watch: `MAIN_THREADS`, or more when fewer background worlds leave
/// cores spare, keeping a tenth of the machine free. With the background worlds off, a fixed 12
/// left half of 24 cores idle, and raising `physics_shards` past 12 gained nothing: the extra
/// physics worlds waited for a thread.
pub fn main_threads(cores: usize, background_worlds: usize) -> usize {
    let spare = (cores * 9 / 10).saturating_sub(background_worlds * BACKGROUND_THREADS);
    spare.max(MAIN_THREADS).min(cores)
}

/// A thread pool of its own for one world, so worlds don't contend in one shared pool.
pub fn world_pool(name: &str, threads: usize) -> rayon::ThreadPool {
    pool(name, threads, false)
}

/// A thread pool; `background` ones run at a lower priority, so they use spare cores without
/// slowing the world you watch.
pub fn pool(name: &str, threads: usize, background: bool) -> rayon::ThreadPool {
    let name = name.to_string();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .thread_name(move |i| format!("{name}-{i}"))
        .start_handler(move |_| {
            if background {
                lower_priority();
            }
        })
        .build()
        .expect("thread pool")
}

/// Lowers the calling thread's scheduling priority (nice 10).
pub fn lower_priority() {
    #[cfg(target_os = "linux")]
    // Safe: only changes this thread's own priority.
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t, 10);
    }
}

/// Shared pool of the best genomes from every world.
pub type Board = Arc<Mutex<GenePool>>;

/// Sends this world's best genomes to the board and takes the board's best into its own pool.
/// Only the world you watch passes an `age_by` (its migration interval), so the board ages at
/// one rate however many worlds there are.
pub fn migrate(pool: &mut GenePool, board: &Board, age_by: f32) {
    let Ok(mut b) = board.lock() else {
        return;
    };
    if age_by > 0.0 {
        b.age(0.998f32.powf(age_by));
    }
    for e in pool.entries.iter().take(MIGRANTS) {
        b.consider(e.score, e.generation, &e.genome);
    }
    for e in b.entries.iter().take(MIGRANTS) {
        pool.consider(e.score, e.generation, &e.genome);
    }
}

#[derive(Clone, Debug, Default)]
pub struct WorldStatus {
    pub index: usize,
    pub time: f64,
    pub population: usize,
    pub max_generation: u32,
    pub mean_neurons: f32,
    pub births: u64,
    pub steps_per_sec: f32,
    pub gpu_busy: f32,
    /// The world's thread has stopped (it panicked).
    pub stopped: bool,
}

impl WorldStatus {
    pub fn of(index: usize, sim: &Sim, steps_per_sec: f32) -> Self {
        let n = sim.creatures.len().max(1) as f32;
        Self {
            index,
            time: sim.time,
            population: sim.creatures.len(),
            max_generation: sim.stats.max_generation,
            mean_neurons: sim.creatures.iter().map(|c| c.genome.brain.h as f32).sum::<f32>() / n,
            births: sim.stats.births,
            steps_per_sec,
            gpu_busy: sim.perf.gpu_busy,
            stopped: false,
        }
    }
}

pub enum WorkerCtl {
    Spawn { genome: Box<crate::genome::Genome>, generation: u32, count: u32 },
    Speed(f32),
    Pause(bool),
    Params(Box<SimParams>),
    /// The going price of brain energy and the GPU headroom, set by the main world's runner.
    BrainPrice(f32, f32),
    /// A question from the MCP server.
    Ask(crate::mcp::Ask, crate::mcp::Reply),
    Quit,
}

pub struct Worker {
    tx: Sender<WorkerCtl>,
    pub status: Arc<Mutex<WorldStatus>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(
        index: usize,
        params: SimParams,
        speed: f32,
        paused: bool,
        board: Board,
        zoo: SharedZoo,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        let status = Arc::new(Mutex::new(WorldStatus {
            index,
            ..Default::default()
        }));
        let s2 = status.clone();
        let handle = std::thread::Builder::new()
            .name(format!("evolve-world-{index}"))
            .spawn(move || worker_loop(index, params, speed, paused, rx, s2, board, zoo))
            .expect("spawn background world");
        Self {
            tx,
            status,
            handle: Some(handle),
        }
    }

    pub fn send(&self, c: WorkerCtl) {
        let _ = self.tx.send(c);
    }

    /// False once the world's thread has stopped (it panicked: the reason is on stderr).
    pub fn alive(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerCtl::Quit);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Each background world gets its own landscape so the islands don't all face the same problem.
fn island_params(index: usize, p: &SimParams) -> SimParams {
    let mut p = p.clone();
    p.seed = p.seed.wrapping_add(7919 * index as u64);
    // As many physics worlds as it has threads.
    p.physics_shards = p.physics_shards.min(BACKGROUND_THREADS as u32);
    let share = p.background_population.clamp(0.05, 1.0);
    p.min_population = ((p.min_population as f32 * share).round() as u32).max(p.min_population.min(4));
    p.max_population = ((p.max_population as f32 * share).round() as u32).max(p.min_population);
    // Nobody looks at a background world's ground: no finer than the water is worked out at (a
    // 5000-a-side terrain each, 25 million points, nearly ran the app out of memory).
    let finest = (p.world_size / crate::hydro::HYDRO_CELL).ceil() as u32 + 1;
    p.terrain_resolution = p.terrain_resolution.min(finest);
    p
}

/// Steps a world at a multiple of real time (or flat out), for `budget` at most per call.
pub struct Pacer {
    pending: f64,
    last: Instant,
}

impl Pacer {
    pub fn new() -> Self {
        Self {
            pending: 0.0,
            last: Instant::now(),
        }
    }

    /// Runs the steps that are due and returns how many ran. With a `pool`, all of the step's
    /// parallel work (physics, senses, brains) stays inside that pool.
    pub fn run(
        &mut self,
        sim: &mut Sim,
        speed: f32,
        paused: bool,
        budget: Duration,
        pool: Option<&rayon::ThreadPool>,
    ) -> u64 {
        match pool {
            Some(p) => p.install(|| self.run_here(sim, speed, paused, budget)),
            None => self.run_here(sim, speed, paused, budget),
        }
    }

    fn run_here(&mut self, sim: &mut Sim, speed: f32, paused: bool, budget: Duration) -> u64 {
        let now = Instant::now();
        let elapsed = (now - self.last).as_secs_f64();
        self.last = now;
        if paused {
            self.pending = 0.0;
            return 0;
        }
        let start = Instant::now();
        let mut steps = 0;
        if speed <= 0.0 {
            while start.elapsed() < budget {
                sim.step();
                steps += 1;
            }
        } else {
            let per_sec = TICKS_PER_SECOND as f64 * speed as f64;
            // Don't build up a backlog if the CPU can't keep up with the requested speed.
            self.pending = (self.pending + elapsed * per_sec).min(per_sec * 0.25 + 1.0);
            while self.pending >= 1.0 {
                sim.step();
                steps += 1;
                self.pending -= 1.0;
                if start.elapsed() > budget {
                    self.pending = 0.0;
                    break;
                }
            }
        }
        steps
    }
}

impl Default for Pacer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_loop(
    index: usize,
    params: SimParams,
    mut speed: f32,
    mut paused: bool,
    rx: Receiver<WorkerCtl>,
    status: Arc<Mutex<WorldStatus>>,
    board: Board,
    zoo: SharedZoo,
) {
    lower_priority();
    let pool = self::pool(&format!("evolve-world-{index}"), BACKGROUND_THREADS, true);
    let mut sim = pool.install(|| Sim::new_background(island_params(index, &params)));
    pool.install(|| crate::zoo::seed_world(&zoo, &mut sim));
    crate::zoo::refresh_rain_sample(&zoo, &mut sim);
    // Start from the best genomes known so far rather than from nothing.
    migrate(&mut sim.pool, &board, 0.0);
    let mut pacer = Pacer::new();
    let mut next_migration = sim.time + sim.params.migration_interval as f64;
    let mut next_zoo = sim.time + sim.params.zoo_interval as f64;
    let mut next_log = sim.time + crate::runner::LOG_EVERY;
    let mut window = (Instant::now(), 0u64);
    let mut rate = 0.0f32;
    loop {
        loop {
            match rx.try_recv() {
                Ok(WorkerCtl::Quit) | Err(TryRecvError::Disconnected) => return,
                Ok(WorkerCtl::Spawn { genome, generation, count }) => {
                    pool.install(|| sim.drop_genome(&genome, generation, None, count));
                }
                Ok(WorkerCtl::Speed(s)) => speed = s,
                Ok(WorkerCtl::Pause(p)) => paused = p,
                Ok(WorkerCtl::BrainPrice(x, room)) => {
                    sim.brain_price = x;
                    sim.gpu_headroom = room;
                }
                // Like the world you watch, a new island shape waits for a new world.
                Ok(WorkerCtl::Params(p)) => sim.set_params(island_params(index, &p)),
                Ok(WorkerCtl::Ask(ask, reply)) => sim.answer(ask, reply),
                Err(TryRecvError::Empty) => break,
            }
        }
        window.1 += pacer.run(&mut sim, speed, paused, Duration::from_millis(30), Some(&pool));
        sim.events.clear();
        sim.highlights.clear();
        if sim.time >= next_migration {
            migrate(&mut sim.pool, &board, 0.0);
            crate::zoo::refresh_rain_sample(&zoo, &mut sim);
            next_migration = sim.time + sim.params.migration_interval as f64;
        }
        if sim.time >= next_log {
            eprintln!("evolve world {index}: {}", sim.summary_line(crate::runner::LOG_EVERY));
            next_log = sim.time + crate::runner::LOG_EVERY;
        }
        if sim.time >= next_zoo {
            store_best(&zoo, &sim, index);
            next_zoo = sim.time + sim.params.zoo_interval as f64;
        }
        let secs = window.0.elapsed().as_secs_f32();
        if secs >= 0.5 {
            rate = window.1 as f32 / secs;
            window = (Instant::now(), 0);
            if let Ok(mut s) = status.lock() {
                *s = WorldStatus::of(index, &sim, rate);
            }
        }
        if paused || speed > 0.0 {
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = rate;
    }
}
