//! Run a world without a window, as fast as the CPU allows, printing progress.
//!
//!     evolve-headless --minutes 10 --report 30 --save run.evo
//!     evolve-headless --load run.evo --minutes 5
//!     evolve-headless --set min_population=60 --set mutation_rate=0.1

use std::path::PathBuf;
use std::time::Instant;

use evolve_sim::params::ParamRef;
use evolve_sim::sim::TICKS_PER_SECOND;
use evolve_sim::{Sim, SimParams, save};

fn main() {
    evolve_sim::init_thread_pool();
    // Experiments shouldn't fill the real zoo; `EVOLVE_ZOO_DIR=...` picks one on purpose.
    if std::env::var_os("EVOLVE_ZOO_DIR").is_none() {
        let dir = std::env::temp_dir().join(format!("evolve-headless-zoo-{}", std::process::id()));
        // Safe: nothing else is running yet.
        unsafe { std::env::set_var("EVOLVE_ZOO_DIR", &dir) };
    }
    let mut minutes = 5.0f64;
    let mut report = 30.0f64;
    let mut save_to: Option<PathBuf> = None;
    let mut load_from: Option<PathBuf> = None;
    let mut sets: Vec<(String, String)> = Vec::new();
    let mut islands: Option<u32> = None;
    let mut mcp: Option<u16> = None;
    let mut wall: f64 = 60.0;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| die(&format!("{a} needs a value")));
        match a.as_str() {
            "--minutes" => minutes = val().parse().unwrap_or_else(|_| die("bad --minutes")),
            "--report" => report = val().parse().unwrap_or_else(|_| die("bad --report")),
            "--save" => save_to = Some(val().into()),
            "--islands" => islands = Some(val().parse().unwrap_or_else(|_| die("bad --islands"))),
            "--mcp" => mcp = Some(val().parse().unwrap_or_else(|_| die("bad --mcp"))),
            "--wall-seconds" => wall = val().parse().unwrap_or_else(|_| die("bad --wall-seconds")),
            "--load" => load_from = Some(val().into()),
            "--set" => {
                let v = val();
                let (k, x) = v.split_once('=').unwrap_or_else(|| die("--set wants name=value"));
                sets.push((k.to_string(), x.to_string()));
            }
            "-h" | "--help" => {
                println!(
                    "evolve-headless [--minutes M] [--report SECONDS] [--save FILE] [--load FILE] [--set name=value]...\n                [--islands N --wall-seconds S [--mcp PORT]]"
                );
                return;
            }
            _ => die(&format!("unknown argument {a}")),
        }
    }

    // The settings go in before a new world is made, so those that shape it (the terrain, how
    // many stones) count too.
    let with_sets = |mut params: SimParams| {
        for (k, v) in &sets {
            let mut hit = false;
            params.visit(&mut |info, r| {
                if info.name != k {
                    return;
                }
                hit = true;
                match r {
                    ParamRef::F32(x) => *x = v.parse().unwrap_or_else(|_| die(&format!("bad {k}"))),
                    ParamRef::U32(x) => *x = v.parse().unwrap_or_else(|_| die(&format!("bad {k}"))),
                    ParamRef::U64(x) => *x = v.parse().unwrap_or_else(|_| die(&format!("bad {k}"))),
                    ParamRef::Bool(x) => *x = v.parse().unwrap_or_else(|_| die(&format!("bad {k}"))),
                }
            });
            if !hit {
                die(&format!("no param named {k}"));
            }
        }
        params
    };
    let mut sim = match &load_from {
        Some(p) => {
            let mut sim = save::load(p).unwrap_or_else(|e| die(&e));
            let params = with_sets(sim.params.clone());
            sim.set_params(params);
            sim
        }
        None => Sim::new(with_sets(SimParams::default())),
    };

    if let Some(n) = islands {
        run_islands(sim, n, wall, report, save_to, mcp);
        return;
    }
    println!(
        "{:>7} {:>4} {:>5} {:>4} {:>6} {:>6} {:>6} {:>5} {:>5} {:>6} {:>6} {:>5} {:>6} {:>4} {:>6} {:>6} {:>6} {:>7} {:>6} {:>5} {:>6} {:>5}",
        "sim_s", "pop", "plant", "meat", "births", "rained", "deaths", "maxG", "meanG", "food", "dist", "age", "neur", "spec", "think", "phys", "other", "steps/s", "gpu_ms", "gpu%", "wait", "cpu%"
    );
    println!("# brains: {}", sim.brain_backend);
    let mut cpu_last = (Instant::now(), evolve_sim::process_cpu_seconds().unwrap_or(0.0));
    let total_ticks = (minutes * 60.0 * TICKS_PER_SECOND as f64) as u64;
    let report_ticks = ((report * TICKS_PER_SECOND as f64) as u64).max(1);
    let start = Instant::now();
    let mut last = (Instant::now(), sim.tick);
    for _ in 0..total_ticks {
        sim.step();
        for e in sim.events.drain(..) {
            println!("# {e}");
        }
        if sim.tick % report_ticks == 0 {
            let s = &sim.stats;
            let n = sim.creatures.len().max(1) as f32;
            let mean_gen = sim.creatures.iter().map(|c| c.generation as f32).sum::<f32>() / n;
            let neurons = sim.creatures.iter().map(|c| c.genome.brain.h as f32).sum::<f32>() / n;
            let rate = (sim.tick - last.1) as f64 / last.0.elapsed().as_secs_f64();
            last = (Instant::now(), sim.tick);
            let cpu_now = evolve_sim::process_cpu_seconds().unwrap_or(0.0);
            let cpu_pct = (cpu_now - cpu_last.1) / cpu_last.0.elapsed().as_secs_f64() * 100.0;
            cpu_last = (Instant::now(), cpu_now);
            println!(
                "{:>7.0} {:>4} {:>5} {:>4} {:>6} {:>6} {:>6} {:>5} {:>5.1} {:>6.1} {:>6.1} {:>5.0} {:>6.1} {:>4} {:>6.2} {:>6.2} {:>6.2} {:>7.0} {:>6.2} {:>5.0} {:>6.2} {:>5.0}",
                sim.time,
                sim.creatures.len(),
                sim.plants.len(),
                sim.meat.len(),
                s.births,
                s.rained,
                format!("{}", s.deaths()),
                s.max_generation,
                mean_gen,
                s.life_food,
                s.life_distance,
                s.life_age,
                neurons,
                sim.count_species(),
                sim.perf.think_ms,
                sim.perf.physics_ms,
                sim.perf.other_ms,
                rate,
                sim.perf.gpu_ms,
                sim.perf.gpu_busy * 100.0,
                sim.perf.gpu_wait_ms,
                cpu_pct
            );
        }
    }
    let s = &sim.stats;
    println!("# step: {}", sim.perf.phases_line());
    println!(
        "# physics: forces {:.2} ms, slowest world {:.2} ms, all worlds {:.2} ms",
        sim.perf.forces_ms, sim.perf.shard_slowest_ms, sim.perf.shard_total_ms
    );
    // Loose stones and boulders: how many are still moving (or never settled), and how many
    // ended up under the ground.
    let (mut awake, mut buried) = (0, 0);
    for o in sim.obstacles.iter().filter(|o| o.dynamic) {
        if let Some(b) = sim.shards[o.shard].world.bodies.get(o.body) {
            awake += usize::from(!b.is_sleeping());
            let t = b.translation();
            buried += usize::from(t.y < sim.terrain.height_at(t.x, t.z) - 0.05);
        }
    }
    println!("# obstacles: {} loose, {awake} awake, {buried} under the ground", sim.obstacles.iter().filter(|o| o.dynamic).count());
    println!(
        "done: {:.0} sim-seconds in {:.1}s. deaths: starved {} killed {} old {} drowned {} thirst {} other {}",
        sim.time,
        start.elapsed().as_secs_f64(),
        s.deaths_starved,
        s.deaths_killed,
        s.deaths_old,
        s.deaths_drowned,
        s.deaths_thirst,
        s.deaths_other
    );
    if let Some(p) = save_to {
        match save::save(&mut sim, &p) {
            Ok(bytes) => println!("saved {} ({} bytes)", p.display(), bytes),
            Err(e) => die(&e),
        }
    }
}

/// Runs the world through the real `Runner` at maximum speed with `n` background worlds, for
/// `wall` seconds of real time, reporting every `report` seconds: how hard the hardware works.
fn run_islands(mut sim: Sim, n: u32, wall: f64, report: f64, save_to: Option<PathBuf>, mcp: Option<u16>) {
    let mut p = sim.params.clone();
    p.parallel_worlds = n;
    sim.set_params(p);
    println!("# brains: {}; {} background worlds", sim.brain_backend, n);
    let runner = evolve_sim::Runner::start(sim);
    runner.send(evolve_sim::Control::Speed(0.0));
    if let Some(port) = mcp {
        match evolve_sim::mcp::serve(port, runner.sender(), None) {
            Ok(port) => println!("# MCP server at http://127.0.0.1:{port}/mcp"),
            Err(e) => println!("# no MCP server on port {port} ({e})"),
        }
    }
    let start = Instant::now();
    let mut cpu_last = (Instant::now(), evolve_sim::process_cpu_seconds().unwrap_or(0.0));
    while start.elapsed().as_secs_f64() < wall {
        std::thread::sleep(std::time::Duration::from_secs_f64(report.min(wall)));
        let snap = runner.snapshot();
        let cpu_now = evolve_sim::process_cpu_seconds().unwrap_or(0.0);
        let cpu = (cpu_now - cpu_last.1) / cpu_last.0.elapsed().as_secs_f64() * 100.0;
        cpu_last = (Instant::now(), cpu_now);
        let gpu = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        println!(
            "t={:>4.0}s  process cpu {:>5.0}%  nvidia-smi gpu [{}]  brain price x{:.3}",
            start.elapsed().as_secs_f64(),
            cpu,
            gpu,
            snap.brain_price
        );
        for w in &snap.worlds {
            println!(
                "   world {:>2}: sim {:>6.0}s  pop {:>3}  maxG {:>3}  neurons {:>7.1}  {:>5.0} steps/s  brain gpu busy {:>3.0}%",
                w.index, w.time, w.population, w.max_generation, w.mean_neurons, w.steps_per_sec, w.gpu_busy * 100.0
            );
        }
    }
    if let Some(p) = save_to {
        // Saving happens on the runner's thread, ahead of the quit that dropping it sends.
        let modified = |p: &PathBuf| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        let before = modified(&p);
        runner.send(evolve_sim::Control::Save(p.clone()));
        drop(runner);
        match std::fs::metadata(&p) {
            Ok(m) if modified(&p) != before => println!("saved {} ({} bytes)", p.display(), m.len()),
            _ => die(&format!("couldn't save {}", p.display())),
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("evolve-headless: {msg}");
    std::process::exit(2)
}
