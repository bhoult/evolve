//! Low priority mode: evolve gives way to other programs. The app turns it on (always, or while
//! its window is minimized: its Settings window). A sampler thread watches what the rest of the
//! machine uses and sets the share of its own pace every world steps at: down to the slowest pace
//! allowed when other programs want the CPU or the GPU, and back up when they let go. When other
//! programs run short of GPU memory, every world's brains move to the CPU and the GPU memory is
//! handed back, until there is room again. The thresholds are a `Config`.
//!
//! Only the pace changes, never the world: the brain price holds still while the share is below
//! 1 (as it does at a set speed), and brains moved to the CPU think the same as on the GPU.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::machine::Load;

/// The least share of its pace any setting lets evolve fall to (the screen is fed less often in
/// proportion, so this bounds that too).
pub const FLOOR: f32 = 0.01;
/// How much more free CPU and GPU than `Config::give_way_below` it takes to speed up again
/// (between the two it holds its pace, so it doesn't hunt around the line).
const ROOMY_ABOVE: f32 = 0.15;
/// What other programs must use before evolve counts the machine as theirs: below it, a busy
/// machine is busy with evolve itself.
const OTHERS_MIN: f32 = 0.05;
/// Each second squeezed, the share shrinks to this much of itself: from all to 5% in about 6 s.
const SHRINK: f32 = 0.6;
/// Each second with room, the share grows by this much: from 5% to all in about 20 s.
const GROW: f32 = 0.05;
/// GPU memory other programs must hold before running short of it counts as theirs, MB.
const VRAM_OTHERS_MIN_MB: f32 = 1024.0;
/// Free GPU memory, besides what the brains held, it waits for before moving them back...
const VRAM_ROOMY_MB: f32 = 2048.0;
/// ...for this long, so a program that frees and takes memory in turns doesn't bounce them.
const VRAM_ROOMY_SECS: f32 = 30.0;

/// Where the mode gives way, and how far.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Free CPU or GPU (a share of the whole, 0..1) below which, with other programs using some,
    /// evolve slows down.
    pub give_way_below: f32,
    /// The slowest it goes, as a share of its full pace (at least `FLOOR`).
    pub slowest: f32,
    /// Free GPU memory, MB, below which (with other programs holding some) brains move to the CPU.
    pub vram_free_mb: f32,
}

pub const DEFAULT_CONFIG: Config = Config { give_way_below: 0.10, slowest: 0.05, vram_free_mb: 1024.0 };

impl Default for Config {
    fn default() -> Self {
        DEFAULT_CONFIG
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static WINDOW_MINIMIZED: AtomicBool = AtomicBool::new(false);
static CONFIG: Mutex<Config> = Mutex::new(DEFAULT_CONFIG);
/// The share, as f32 bits (1.0 when off).
static SHARE: AtomicU32 = AtomicU32::new(0x3f80_0000);
static GPU_GIVEN_BACK: AtomicBool = AtomicBool::new(false);
static STATUS: Mutex<Option<Status>> = Mutex::new(None);
static CONTROLLER: Mutex<Controller> = Mutex::new(Controller::new());

/// The share of their pace the worlds step at: 1 normally, less in low priority mode.
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

/// Turns the mode on or off (cheap to call every frame); `minimized` says it's on only because
/// the window is minimized (for the status). The sampler thread starts the first time it's turned
/// on (the machine sampler, if it isn't running yet); turned off, the worlds go back to full pace and the GPU at once.
pub fn set_enabled(on: bool, minimized: bool) {
    WINDOW_MINIMIZED.store(on && minimized, Ordering::Relaxed);
    if ENABLED.swap(on, Ordering::Relaxed) == on {
        return;
    }
    if on {
        crate::machine::start();
    } else {
        SHARE.store(1.0f32.to_bits(), Ordering::Relaxed);
        GPU_GIVEN_BACK.store(false, Ordering::Relaxed);
    }
}

pub fn set_config(c: Config) {
    if let Ok(mut g) = CONFIG.lock() {
        *g = c;
    }
}

fn config() -> Config {
    CONFIG.lock().map(|c| *c).unwrap_or_default()
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
    /// On because the window is minimized.
    pub window_minimized: bool,
}

/// "low priority: 12% of full pace (other programs: CPU 85%, GPU 97%), brains on the CPU".
impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let why = if self.window_minimized { " (window minimized)" } else { "" };
        write!(f, "low priority{why}: {:.0}% of full pace (other programs: CPU {:.0}%", self.share * 100.0, self.cpu_others * 100.0)?;
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
        Self::new()
    }
}

impl Controller {
    pub const fn new() -> Self {
        Self { share: 1.0, gpu_given_back: false, held_mb: 0.0, roomy_secs: 0.0 }
    }

    /// Folds in a look at the machine taken `dt` seconds after the last.
    pub fn update(&mut self, l: &Load, dt: f32, cfg: &Config) {
        let slowest = cfg.slowest.clamp(FLOOR, 1.0);
        let roomy_above = (cfg.give_way_below + ROOMY_ABOVE).min(0.95);
        let cpu_free = 1.0 - l.cpu_busy;
        let cpu_others = (l.cpu_busy - l.cpu_ours).max(0.0);
        // The GPU counts only when this process's own part of its time is known; otherwise a GPU
        // kept busy by evolve alone would read as other programs wanting it.
        let gpu = l.gpu_busy.zip(l.gpu_ours).map(|(busy, ours)| (1.0 - busy, (busy - ours).max(0.0)));
        let squeezed = (cpu_free < cfg.give_way_below && cpu_others >= OTHERS_MIN)
            || gpu.is_some_and(|(free, others)| free < cfg.give_way_below && others >= OTHERS_MIN);
        let roomy = cpu_free > roomy_above && gpu.is_none_or(|(free, _)| free > roomy_above);
        if squeezed {
            self.share *= SHRINK.powf(dt);
        } else if roomy {
            self.share = (self.share + GROW * dt).min(1.0);
        }
        // (Also lifts it at once when the slowest allowed is raised.)
        self.share = self.share.max(slowest);

        let (Some(total), Some(used)) = (l.vram_total_mb, l.vram_used_mb) else {
            return;
        };
        let ours = l.vram_ours_mb.unwrap_or(0.0);
        let others = (used - ours).max(0.0);
        if !self.gpu_given_back {
            if total - used < cfg.vram_free_mb && others >= VRAM_OTHERS_MIN_MB {
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

/// Each look at the machine (machine.rs, `dt` seconds after the last) steers the mode while it's
/// on.
pub(crate) fn on_sample(load: &Load, dt: f32) {
    let Ok(mut c) = CONTROLLER.lock() else {
        return;
    };
    if !ENABLED.load(Ordering::Relaxed) {
        *c = Controller::new();
        return;
    }
    c.update(load, dt, &config());
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    SHARE.store(c.share.to_bits(), Ordering::Relaxed);
    GPU_GIVEN_BACK.store(c.gpu_given_back, Ordering::Relaxed);
    if let Ok(mut s) = STATUS.lock() {
        *s = Some(Status {
            share: c.share,
            cpu_others: (load.cpu_busy - load.cpu_ours).max(0.0),
            gpu_others: load.gpu_busy.zip(load.gpu_ours).map(|(b, o)| (b - o).max(0.0)),
            vram_free_mb: load.vram_total_mb.zip(load.vram_used_mb).map(|(t, u)| t - u),
            vram_others_mb: load.vram_used_mb.map(|u| (u - load.vram_ours_mb.unwrap_or(0.0)).max(0.0)),
            gpu_given_back: c.gpu_given_back,
            window_minimized: WINDOW_MINIMIZED.load(Ordering::Relaxed),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Controller {
        fn update_default(&mut self, l: &Load, dt: f32) {
            self.update(l, dt, &Config::default());
        }
    }

    fn load(cpu_busy: f32, cpu_ours: f32) -> Load {
        Load { cpu_busy, cpu_ours, ..Load::default() }
    }

    #[test]
    fn shrinks_to_the_floor_when_others_want_the_cpu_and_grows_back() {
        let mut c = Controller::default();
        for _ in 0..10 {
            c.update_default(&load(0.97, 0.10), 1.0);
        }
        assert_eq!(c.share, DEFAULT_CONFIG.slowest);
        // Busy with evolve alone is not a reason to slow down.
        let mut alone = Controller::default();
        alone.update_default(&load(0.95, 0.93), 1.0);
        assert_eq!(alone.share, 1.0);
        // In between the lines it holds; with room it climbs back in about 20 s.
        c.update_default(&load(0.85, 0.05), 1.0);
        assert_eq!(c.share, DEFAULT_CONFIG.slowest);
        for _ in 0..19 {
            c.update_default(&load(0.40, 0.05), 1.0);
        }
        assert!(c.share > 0.95, "{}", c.share);
    }

    #[test]
    fn the_settings_move_the_line_and_the_floor() {
        let cfg = Config { give_way_below: 0.30, slowest: 0.20, vram_free_mb: 1024.0 };
        // 25% free: fine by default, squeezed with the line at 30%.
        let mut c = Controller::default();
        c.update_default(&load(0.75, 0.10), 1.0);
        assert_eq!(c.share, 1.0);
        for _ in 0..10 {
            c.update(&load(0.75, 0.10), 1.0, &cfg);
        }
        assert_eq!(c.share, 0.20);
        // Raising the slowest pace lifts it at once; it can't go under 1%.
        c.update(&load(0.75, 0.10), 1.0, &Config { slowest: 0.5, ..cfg });
        assert_eq!(c.share, 0.5);
        c.update(&load(0.99, 0.10), 30.0, &Config { slowest: 0.0, ..cfg });
        assert_eq!(c.share, FLOOR);
        // It speeds up only with 15% more than the line free (45% here).
        let share = c.share;
        c.update(&load(0.60, 0.10), 1.0, &cfg);
        assert_eq!(c.share, share.max(cfg.slowest));
        c.update(&load(0.50, 0.10), 1.0, &cfg);
        assert!(c.share > cfg.slowest);
    }

    #[test]
    fn gpu_counts_only_when_our_part_is_known() {
        let mut c = Controller::default();
        let busy_gpu = Load { gpu_busy: Some(0.98), gpu_ours: None, ..load(0.3, 0.1) };
        c.update_default(&busy_gpu, 1.0);
        assert_eq!(c.share, 1.0);
        c.update_default(&Load { gpu_ours: Some(0.05), ..busy_gpu }, 1.0);
        assert!(c.share < 1.0);
        // Evolve keeping the GPU busy itself.
        let mut alone = Controller::default();
        alone.update_default(&Load { gpu_busy: Some(0.95), gpu_ours: Some(0.93), ..load(0.3, 0.1) }, 1.0);
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
        c.update_default(&vram(15500.0, 15000.0), 1.0);
        assert!(!c.gpu_given_back);
        // Another program takes most of what's left.
        c.update_default(&vram(15500.0, 7000.0), 1.0);
        assert!(c.gpu_given_back);
        // It still holds 8.5 GB: 7 GB back would leave too little.
        for _ in 0..60 {
            c.update_default(&vram(9000.0, 500.0), 1.0);
        }
        assert!(c.gpu_given_back);
        // It lets go: back after 30 s of room, not before.
        for _ in 0..29 {
            c.update_default(&vram(2000.0, 500.0), 1.0);
        }
        assert!(c.gpu_given_back);
        c.update_default(&vram(2000.0, 500.0), 1.0);
        assert!(!c.gpu_given_back);
    }
}
