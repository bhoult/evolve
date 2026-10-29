//! Low priority mode (the `low_priority` param): evolve gives way to other programs. A sampler
//! thread watches what the rest of the machine uses and sets the share of its own pace every
//! world steps at, down to `FLOOR` when other programs want the CPU or the GPU, and back up when
//! they let go. When other programs run short of GPU memory, every world's brains move to the
//! CPU and the GPU memory is handed back, until there is room again.
//!
//! Only the pace changes, never the world: the brain price holds still while the share is below
//! 1 (as it does at a set speed), and brains moved to the CPU think the same as on the GPU.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

/// The least share of its pace evolve keeps however busy the machine is.
pub const FLOOR: f32 = 0.05;
/// Free CPU or GPU the machine keeps: below it, with other programs using some, evolve slows.
const RESERVE: f32 = 0.10;
/// Free CPU and GPU it takes before speeding up again (between the two it holds its pace, so it
/// doesn't hunt around the line).
const ROOMY: f32 = 0.25;
/// What other programs must use before evolve counts the machine as theirs: below it, a busy
/// machine is busy with evolve itself.
const OTHERS_MIN: f32 = 0.05;
/// Each second squeezed, the share shrinks to this much of itself: from all to the floor in
/// about 6 s.
const SHRINK: f32 = 0.6;
/// Each second with room, the share grows by this much: from the floor to all in about 20 s.
const GROW: f32 = 0.05;
/// GPU memory left free below which (with other programs holding some) brains move to the CPU.
const VRAM_LOW_MB: f32 = 1024.0;
/// Free GPU memory, besides what the brains held, it waits for before moving them back...
const VRAM_ROOMY_MB: f32 = 2048.0;
/// ...for this long, so a program that frees and takes memory in turns doesn't bounce them.
const VRAM_ROOMY_SECS: f32 = 30.0;

static ENABLED: AtomicBool = AtomicBool::new(false);
/// The share, as f32 bits (1.0 when off).
static SHARE: AtomicU32 = AtomicU32::new(0x3f80_0000);
static GPU_GIVEN_BACK: AtomicBool = AtomicBool::new(false);
static STATUS: Mutex<Option<Status>> = Mutex::new(None);
static SAMPLER: Once = Once::new();

/// The share of their pace the worlds step at: 1 normally, down to `FLOOR` in low priority mode.
pub fn share() -> f32 {
    f32::from_bits(SHARE.load(Ordering::Relaxed))
}

/// Should the worlds' brains be off the GPU, its memory handed back to other programs?
pub fn gpu_given_back() -> bool {
    GPU_GIVEN_BACK.load(Ordering::Relaxed)
}

/// What the sampler saw last, for the screen; `None` with the mode off.
pub fn status() -> Option<Status> {
    if !ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    STATUS.lock().ok().and_then(|s| s.clone())
}

/// Turns the mode on or off (cheap to call every pass). The sampler thread starts the first time
/// it's turned on; turned off, the worlds go back to full pace and the GPU at once.
pub fn set_enabled(on: bool) {
    if ENABLED.swap(on, Ordering::Relaxed) == on {
        return;
    }
    if on {
        SAMPLER.call_once(|| {
            let _ = std::thread::Builder::new().name("evolve-low-priority".into()).spawn(sampler);
        });
    } else {
        SHARE.store(1.0f32.to_bits(), Ordering::Relaxed);
        GPU_GIVEN_BACK.store(false, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug, Default)]
pub struct Status {
    pub share: f32,
    /// Shares of the whole machine's CPU and of the GPU's time used by other programs.
    pub cpu_others: f32,
    pub gpu_others: Option<f32>,
    /// GPU memory, MB: free, and held by other programs.
    pub vram_free_mb: Option<f32>,
    pub vram_others_mb: Option<f32>,
    pub gpu_given_back: bool,
}

/// "low priority: 12% of full pace (other programs: CPU 85%, GPU 97%), brains on the CPU".
impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "low priority: {:.0}% of full pace (other programs: CPU {:.0}%", self.share * 100.0, self.cpu_others * 100.0)?;
        if let Some(g) = self.gpu_others {
            write!(f, ", GPU {:.0}%", g * 100.0)?;
        }
        if let Some(mb) = self.vram_free_mb {
            write!(f, ", {:.1} GB GPU memory free", mb / 1000.0)?;
        }
        write!(f, ")")?;
        if self.gpu_given_back {
            write!(f, ", brains on the CPU to leave the GPU's memory free")?;
        }
        Ok(())
    }
}

/// One look at the machine: shares of all its CPU and of the GPU's time (0..1), busy in all and
/// busy with this process, and GPU memory in MB. The GPU's figures are `None` without
/// nvidia-smi.
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    pub cpu_busy: f32,
    pub cpu_ours: f32,
    pub gpu_busy: Option<f32>,
    pub gpu_ours: Option<f32>,
    pub vram_total_mb: Option<f32>,
    pub vram_used_mb: Option<f32>,
    pub vram_ours_mb: Option<f32>,
}

/// Turns looks at the machine into the share and whether to hand the GPU's memory back.
#[derive(Clone, Debug)]
pub struct Controller {
    pub share: f32,
    pub gpu_given_back: bool,
    /// GPU memory this process held when it handed the brains' back, MB.
    held_mb: f32,
    roomy_secs: f32,
}

impl Default for Controller {
    fn default() -> Self {
        Self { share: 1.0, gpu_given_back: false, held_mb: 0.0, roomy_secs: 0.0 }
    }
}

impl Controller {
    /// Folds in a look at the machine taken `dt` seconds after the last.
    pub fn update(&mut self, l: &Load, dt: f32) {
        let cpu_free = 1.0 - l.cpu_busy;
        let cpu_others = (l.cpu_busy - l.cpu_ours).max(0.0);
        // The GPU counts only when this process's own part of its time is known; otherwise a GPU
        // kept busy by evolve alone would read as other programs wanting it.
        let gpu = l.gpu_busy.zip(l.gpu_ours).map(|(busy, ours)| (1.0 - busy, (busy - ours).max(0.0)));
        let squeezed = (cpu_free < RESERVE && cpu_others >= OTHERS_MIN)
            || gpu.is_some_and(|(free, others)| free < RESERVE && others >= OTHERS_MIN);
        let roomy = cpu_free > ROOMY && gpu.is_none_or(|(free, _)| free > ROOMY);
        if squeezed {
            self.share = (self.share * SHRINK.powf(dt)).max(FLOOR);
        } else if roomy {
            self.share = (self.share + GROW * dt).min(1.0);
        }

        let (Some(total), Some(used)) = (l.vram_total_mb, l.vram_used_mb) else {
            return;
        };
        let ours = l.vram_ours_mb.unwrap_or(0.0);
        let others = (used - ours).max(0.0);
        if !self.gpu_given_back {
            let low = VRAM_LOW_MB.max(total * 0.06);
            if total - used < low && others >= VRAM_LOW_MB {
                self.gpu_given_back = true;
                self.held_mb = ours;
                self.roomy_secs = 0.0;
            }
        } else {
            // Room for what the brains held, and some to spare, for a while.
            let room = total - others - self.held_mb;
            self.roomy_secs = if room > VRAM_ROOMY_MB { self.roomy_secs + dt } else { 0.0 };
            if self.roomy_secs >= VRAM_ROOMY_SECS {
                self.gpu_given_back = false;
            }
        }
    }
}

fn sampler() {
    let mut c = Controller::default();
    let mut cpu = CpuMeter::default();
    let mut last = Instant::now();
    loop {
        // nvidia-smi's pmon itself takes about a second, so a look every second or two.
        std::thread::sleep(Duration::from_millis(1000));
        if !ENABLED.load(Ordering::Relaxed) {
            c = Controller::default();
            continue;
        }
        let Some((cpu_busy, cpu_ours)) = cpu.sample() else {
            continue;
        };
        let mut load = Load { cpu_busy, cpu_ours, ..Load::default() };
        nvidia_sample(&mut load);
        let dt = last.elapsed().as_secs_f32().min(5.0);
        last = Instant::now();
        c.update(&load, dt);
        if !ENABLED.load(Ordering::Relaxed) {
            continue;
        }
        SHARE.store(c.share.to_bits(), Ordering::Relaxed);
        GPU_GIVEN_BACK.store(c.gpu_given_back, Ordering::Relaxed);
        if let Ok(mut s) = STATUS.lock() {
            let vram_others = load.vram_used_mb.map(|u| (u - load.vram_ours_mb.unwrap_or(0.0)).max(0.0));
            *s = Some(Status {
                share: c.share,
                cpu_others: (cpu_busy - cpu_ours).max(0.0),
                gpu_others: load.gpu_busy.zip(load.gpu_ours).map(|(b, o)| (b - o).max(0.0)),
                vram_free_mb: load.vram_total_mb.zip(load.vram_used_mb).map(|(t, u)| t - u),
                vram_others_mb: vram_others,
                gpu_given_back: c.gpu_given_back,
            });
        }
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
        .sum()
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
            .sum::<f32>()
            / 100.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(cpu_busy: f32, cpu_ours: f32) -> Load {
        Load { cpu_busy, cpu_ours, ..Load::default() }
    }

    #[test]
    fn shrinks_to_the_floor_when_others_want_the_cpu_and_grows_back() {
        let mut c = Controller::default();
        for _ in 0..10 {
            c.update(&load(0.97, 0.10), 1.0);
        }
        assert_eq!(c.share, FLOOR);
        // Busy with evolve alone is not a reason to slow down.
        let mut alone = Controller::default();
        alone.update(&load(0.95, 0.93), 1.0);
        assert_eq!(alone.share, 1.0);
        // In between the lines it holds; with room it climbs back in about 20 s.
        c.update(&load(0.85, 0.05), 1.0);
        assert_eq!(c.share, FLOOR);
        for _ in 0..19 {
            c.update(&load(0.40, 0.05), 1.0);
        }
        assert!(c.share > 0.95, "{}", c.share);
    }

    #[test]
    fn gpu_counts_only_when_our_part_is_known() {
        let mut c = Controller::default();
        let busy_gpu = Load { gpu_busy: Some(0.98), gpu_ours: None, ..load(0.3, 0.1) };
        c.update(&busy_gpu, 1.0);
        assert_eq!(c.share, 1.0);
        c.update(&Load { gpu_ours: Some(0.05), ..busy_gpu }, 1.0);
        assert!(c.share < 1.0);
        // Evolve keeping the GPU busy itself.
        let mut alone = Controller::default();
        alone.update(&Load { gpu_busy: Some(0.95), gpu_ours: Some(0.93), ..load(0.3, 0.1) }, 1.0);
        assert_eq!(alone.share, 1.0);
    }

    #[test]
    fn hands_gpu_memory_back_and_takes_it_again_after_a_while() {
        let mut c = Controller::default();
        let vram = |used: f32, ours: f32| Load {
            vram_total_mb: Some(16000.0),
            vram_used_mb: Some(used),
            vram_ours_mb: Some(ours),
            ..load(0.3, 0.1)
        };
        // Full, but of evolve's own brains: keep them.
        c.update(&vram(15500.0, 15000.0), 1.0);
        assert!(!c.gpu_given_back);
        // Another program takes most of what's left.
        c.update(&vram(15500.0, 7000.0), 1.0);
        assert!(c.gpu_given_back);
        // It still holds 8.5 GB: 7 GB back would leave too little.
        for _ in 0..60 {
            c.update(&vram(9000.0, 500.0), 1.0);
        }
        assert!(c.gpu_given_back);
        // It lets go: back after 30 s of room, not before.
        for _ in 0..29 {
            c.update(&vram(2000.0, 500.0), 1.0);
        }
        assert!(c.gpu_given_back);
        c.update(&vram(2000.0, 500.0), 1.0);
        assert!(!c.gpu_given_back);
    }

    #[test]
    fn parses_proc_and_nvidia_smi() {
        let stat = "cpu  100 50 30 800 20 0 5 0 0 0\ncpu0 1 2 3 4 5 6 7 8 9 10\n";
        assert_eq!(parse_proc_stat(stat), Some((185, 1005)));
        let me = "42 (evolve sim) S 1 2 3 4 5 6 7 8 9 10 250 70 0 0 20 0";
        assert_eq!(parse_self_ticks(me), Some(320));
        assert_eq!(parse_apps_mb("221995, 189\n4149183, 6712\n", 4149183), 6712.0);
        let pmon = "# gpu         pid   type     sm    mem    enc    dec\n\
                    # Idx           #    C/G      %      %      %      %\n    \
                    0      53745     G      2      2      -      -\n    \
                    0    4149183   C+G     11     10      -      -\n    \
                    0     220919   C+G      -      -      -      -\n";
        assert_eq!(parse_pmon_sm(pmon, 4149183), Some(0.11));
        assert_eq!(parse_pmon_sm(pmon, 220919), Some(0.0));
        assert_eq!(parse_pmon_sm("# gpu pid\n# Idx #\n", 1), None);
    }
}
