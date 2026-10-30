//! What the machine is doing, this process and the rest: a sampler thread looks at the CPU and
//! RAM (from /proc) and the GPU's time and memory (from nvidia-smi) every second or two, keeps the
//! last couple of hours for the app's Load window, and steers low priority mode
//! (low_priority.rs) with each look.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant, SystemTime};

/// Looks kept: at one every second or two, about two hours.
const KEEP: usize = 4000;

static SAMPLER: Once = Once::new();
static HISTORY: Mutex<VecDeque<Sample>> = Mutex::new(VecDeque::new());
/// Counts looks taken, so a reader knows when to copy the history again.
static TAKEN: AtomicU64 = AtomicU64::new(0);
/// The world on screen's speed, × real time, as f32 bits.
static SPEED: AtomicU32 = AtomicU32::new(0);

/// Starts the sampler thread, once (later calls do nothing).
pub fn start() {
    SAMPLER.call_once(|| {
        let _ = std::thread::Builder::new().name("evolve-machine".into()).spawn(sampler);
    });
}

/// The runner says how fast the world on screen runs, × real time.
pub fn report_speed(x_real_time: f32) {
    SPEED.store(x_real_time.to_bits(), Ordering::Relaxed);
}

/// How many looks have been taken (it changes when there's a new one).
pub fn looks() -> u64 {
    TAKEN.load(Ordering::Relaxed)
}

/// The looks kept, oldest first.
pub fn history() -> Vec<Sample> {
    HISTORY.lock().map(|h| h.iter().copied().collect()).unwrap_or_default()
}

/// One look at the machine: shares of all its CPU and of the GPU's time (0..1), busy in all and
/// busy with this process, and RAM and GPU memory in MB. The GPU's figures are `None` without
/// nvidia-smi.
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    pub cpu_busy: f32,
    pub cpu_ours: f32,
    pub gpu_busy: Option<f32>,
    pub gpu_ours: Option<f32>,
    pub ram_total_mb: f32,
    pub ram_used_mb: f32,
    pub ram_ours_mb: f32,
    pub vram_total_mb: Option<f32>,
    pub vram_used_mb: Option<f32>,
    pub vram_ours_mb: Option<f32>,
}

/// A look as the Load window draws it: this process's part and the rest's.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    /// When, in seconds since 1970.
    pub at: f64,
    /// Shares of all the CPU and of the GPU's time (0..1).
    pub cpu_ours: f32,
    pub cpu_others: f32,
    pub gpu_ours: Option<f32>,
    pub gpu_others: Option<f32>,
    /// GB.
    pub ram_ours: f32,
    pub ram_others: f32,
    pub ram_total: f32,
    pub vram_ours: Option<f32>,
    pub vram_others: Option<f32>,
    pub vram_total: Option<f32>,
    /// The world on screen, × real time, and low priority's share of its full pace (1 when off).
    pub speed: f32,
    pub pace: f32,
}

impl Sample {
    fn of(l: &Load, at: f64) -> Self {
        let gb = |mb: f32| mb / 1000.0;
        let vram_ours = l.vram_used_mb.map(|_| l.vram_ours_mb.unwrap_or(0.0));
        Self {
            at,
            cpu_ours: l.cpu_ours,
            cpu_others: (l.cpu_busy - l.cpu_ours).max(0.0),
            gpu_ours: l.gpu_busy.and(l.gpu_ours.map(|o| o.min(l.gpu_busy.unwrap_or(1.0)))),
            gpu_others: l.gpu_busy.zip(l.gpu_ours).map(|(b, o)| (b - o).max(0.0)),
            ram_ours: gb(l.ram_ours_mb),
            ram_others: gb((l.ram_used_mb - l.ram_ours_mb).max(0.0)),
            ram_total: gb(l.ram_total_mb),
            vram_ours: vram_ours.map(gb),
            vram_others: l.vram_used_mb.zip(vram_ours).map(|(u, o)| gb((u - o).max(0.0))),
            vram_total: l.vram_total_mb.map(gb),
            speed: f32::from_bits(SPEED.load(Ordering::Relaxed)),
            pace: crate::low_priority::share(),
        }
    }
}

fn sampler() {
    let mut cpu = CpuMeter::default();
    let mut last = Instant::now();
    loop {
        // nvidia-smi's pmon itself takes about a second, so a look every second or two.
        std::thread::sleep(Duration::from_millis(1000));
        let Some((cpu_busy, cpu_ours)) = cpu.sample() else {
            continue;
        };
        let mut load = Load { cpu_busy, cpu_ours, ..Load::default() };
        ram_sample(&mut load);
        nvidia_sample(&mut load);
        let dt = last.elapsed().as_secs_f32().min(5.0);
        last = Instant::now();
        crate::low_priority::on_sample(&load, dt);
        let at = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
        if let Ok(mut h) = HISTORY.lock() {
            if h.len() >= KEEP {
                h.pop_front();
            }
            h.push_back(Sample::of(&load, at));
        }
        TAKEN.fetch_add(1, Ordering::Relaxed);
    }
}

/// The machine's and this process's CPU time between samples, from /proc (both in clock ticks).
#[derive(Default)]
struct CpuMeter {
    last: Option<(u64, u64, u64)>,
}

impl CpuMeter {
    /// (busy, ours) as shares of all the machine's CPU since the last call; `None` the first time.
    fn sample(&mut self) -> Option<(f32, f32)> {
        let (busy, total) = parse_proc_stat(&std::fs::read_to_string("/proc/stat").ok()?)?;
        let ours = parse_self_ticks(&std::fs::read_to_string("/proc/self/stat").ok()?)?;
        let now = (busy, total, ours);
        let before = self.last.replace(now)?;
        let dt = now.1.saturating_sub(before.1).max(1) as f32;
        let b = now.0.saturating_sub(before.0) as f32 / dt;
        let o = now.2.saturating_sub(before.2) as f32 / dt;
        Some((b.min(1.0), o.min(b).min(1.0)))
    }
}

/// RAM in use (all but what's available) and this process's resident set, from /proc.
fn ram_sample(l: &mut Load) {
    if let Some((total, available)) = std::fs::read_to_string("/proc/meminfo").ok().and_then(|t| parse_meminfo(&t)) {
        l.ram_total_mb = total;
        l.ram_used_mb = total - available;
    }
    if let Some(rss) = std::fs::read_to_string("/proc/self/status").ok().and_then(|t| parse_rss_mb(&t)) {
        l.ram_ours_mb = rss;
    }
}

/// (busy, total) clock ticks from /proc/stat's first line: busy is all but idle and iowait.
pub fn parse_proc_stat(text: &str) -> Option<(u64, u64)> {
    let line = text.lines().next()?.strip_prefix("cpu ")?;
    let f: Vec<u64> = line.split_whitespace().take(8).map(|v| v.parse().ok()).collect::<Option<_>>()?;
    let total: u64 = f.iter().sum();
    Some((total - f.get(3)? - f.get(4)?, total))
}

/// This process's user + system clock ticks, from /proc/self/stat.
pub fn parse_self_ticks(text: &str) -> Option<u64> {
    let f: Vec<&str> = text.rsplit_once(')')?.1.split_whitespace().collect();
    Some(f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?)
}

/// (total, available) MB from /proc/meminfo.
pub fn parse_meminfo(text: &str) -> Option<(f32, f32)> {
    let field = |name: &str| -> Option<f32> {
        let kb: f32 = text.lines().find_map(|l| l.strip_prefix(name))?.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb / 1000.0)
    };
    Some((field("MemTotal:")?, field("MemAvailable:")?))
}

/// This process's resident set, MB, from /proc/self/status.
pub fn parse_rss_mb(text: &str) -> Option<f32> {
    let kb: f32 = text.lines().find_map(|l| l.strip_prefix("VmRSS:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kb / 1000.0)
}

/// Fills in the GPU's figures from nvidia-smi (the first GPU): its busy time and memory, this
/// process's memory, and its share of the GPU's time from pmon. Anything that fails stays `None`.
fn nvidia_sample(l: &mut Load) {
    let run = |args: &[&str]| -> Option<String> {
        let out = std::process::Command::new("nvidia-smi").args(args).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let Some(gpu) = run(&["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]) else {
        return;
    };
    let f: Vec<f32> = gpu.lines().next().unwrap_or("").split(',').filter_map(|v| v.trim().parse().ok()).collect();
    if let [busy, used, total] = f[..] {
        l.gpu_busy = Some(busy / 100.0);
        l.vram_used_mb = Some(used);
        l.vram_total_mb = Some(total);
    }
    let pid = std::process::id();
    if let Some(apps) = run(&["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]) {
        l.vram_ours_mb = Some(parse_apps_mb(&apps, pid));
    }
    if let Some(pmon) = run(&["pmon", "-c", "1", "-s", "u"]) {
        l.gpu_ours = parse_pmon_sm(&pmon, pid);
    }
}

/// This process's GPU memory, MB, from `--query-compute-apps=pid,used_memory` (0 if not listed).
pub fn parse_apps_mb(text: &str, pid: u32) -> f32 {
    text.lines()
        .filter_map(|l| l.split_once(','))
        .filter(|(p, _)| p.trim().parse() == Ok(pid))
        .filter_map(|(_, mb)| mb.trim().parse::<f32>().ok())
        // (A float sum of nothing is -0.)
        .fold(0.0, |a, b| a + b)
}

/// This process's share of the GPU's time (0..1) from `pmon -s u`; "-" (no sample) reads as 0.
/// `None` if pmon printed no process lines at all.
pub fn parse_pmon_sm(text: &str, pid: u32) -> Option<f32> {
    let rows: Vec<Vec<&str>> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().collect())
        .collect();
    if rows.is_empty() {
        return None;
    }
    Some(
        rows.iter()
            .filter(|r| r.get(1).and_then(|p| p.parse().ok()) == Some(pid))
            .filter_map(|r| r.get(3)?.parse::<f32>().ok())
            .fold(0.0, |a, b| a + b)
            / 100.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proc_and_nvidia_smi() {
        let stat = "cpu  100 50 30 800 20 0 5 0 0 0\ncpu0 1 2 3 4 5 6 7 8 9 10\n";
        assert_eq!(parse_proc_stat(stat), Some((185, 1005)));
        let me = "42 (evolve sim) S 1 2 3 4 5 6 7 8 9 10 250 70 0 0 20 0";
        assert_eq!(parse_self_ticks(me), Some(320));
        let meminfo = "MemTotal:       64000000 kB\nMemFree:         8000000 kB\nMemAvailable:   24000000 kB\n";
        assert_eq!(parse_meminfo(meminfo), Some((64000.0, 24000.0)));
        assert_eq!(parse_rss_mb("Name:\tevolve\nVmRSS:\t10209528 kB\n"), Some(10209.528));
        assert_eq!(parse_apps_mb("221995, 189\n4149183, 6712\n", 4149183), 6712.0);
        let pmon = "# gpu         pid   type     sm    mem    enc    dec\n\
                    # Idx           #    C/G      %      %      %      %\n    \
                    0      53745     G      2      2      -      -\n    \
                    0    4149183   C+G     11     10      -      -\n    \
                    0     220919   C+G      -      -      -      -\n";
        assert_eq!(parse_pmon_sm(pmon, 4149183), Some(0.11));
        assert!(parse_pmon_sm(pmon, 220919).is_some_and(|x| x == 0.0 && x.is_sign_positive()), "no sample reads as 0, not -0");
        assert_eq!(parse_pmon_sm("# gpu pid\n# Idx #\n", 1), None);
    }

    #[test]
    fn a_sample_splits_ours_from_the_rest() {
        let l = Load {
            cpu_busy: 0.6,
            cpu_ours: 0.2,
            gpu_busy: Some(0.9),
            gpu_ours: Some(0.95),
            ram_total_mb: 64000.0,
            ram_used_mb: 30000.0,
            ram_ours_mb: 10000.0,
            vram_total_mb: Some(16000.0),
            vram_used_mb: Some(9000.0),
            vram_ours_mb: Some(7000.0),
        };
        let s = Sample::of(&l, 0.0);
        assert!((s.cpu_others - 0.4).abs() < 1e-6);
        // pmon's sample can read past the GPU's own figure: ours never shows more than all.
        assert_eq!((s.gpu_ours, s.gpu_others), (Some(0.9), Some(0.0)));
        assert_eq!((s.ram_ours, s.ram_others, s.ram_total), (10.0, 20.0, 64.0));
        assert_eq!((s.vram_ours, s.vram_others, s.vram_total), (Some(7.0), Some(2.0), Some(16.0)));
        // Without nvidia-smi there's no GPU to show.
        let s = Sample::of(&Load { gpu_busy: None, gpu_ours: None, vram_total_mb: None, vram_used_mb: None, vram_ours_mb: None, ..l }, 0.0);
        assert_eq!((s.gpu_ours, s.vram_ours, s.vram_others), (None, None, None));
    }
}
