//! evolve-sim: the headless core of the evolution simulator. No rendering in here, so it builds
//! fast and can run (and be tested) without a window.

pub mod acts;
pub mod body;
pub mod carry;
pub mod brain;
pub mod commands;
pub mod forest;
pub mod genome;
pub mod grid;
pub mod islands;
pub mod knock;
pub mod gpu;
pub mod hydro;
mod life;
pub mod math;
pub mod mcp;
pub mod noise;
pub mod observe;
pub mod params;
pub mod rng;
pub mod runner;
pub mod save;
pub mod sim;
pub mod snapshot;
pub mod stats;
pub mod terrain;
pub mod weather;
pub mod wiring;
pub mod zoo;

pub use commands::Command;
pub use params::SimParams;
pub use runner::{Control, Runner, ZooTarget};
pub use sim::Sim;
pub use snapshot::Snapshot;

/// CPU time used by this process so far (user + system), in seconds. Linux only.
pub fn process_cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Fields after the parenthesized command name; utime and stime are fields 14 and 15.
    let rest = stat.rsplit_once(')')?.1;
    let f: Vec<&str> = rest.split_whitespace().collect();
    let utime: f64 = f.get(11)?.parse().ok()?;
    let stime: f64 = f.get(12)?.parse().ok()?;
    Some((utime + stime) / 100.0)
}

/// Sizes the shared rayon pool that physics and brains of every world run on, unless
/// `RAYON_NUM_THREADS` is set. One world's physics stops getting faster past about 8 threads, so
/// the rest of the cores go to the background worlds, which share this pool.
pub fn init_thread_pool() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("evolve-worker-{i}"))
        .build_global();
}
