//! What belongs to this machine rather than to a world: low priority mode and the graphics
//! toggles. They're set in the Settings window and kept in `~/.config/evolve/settings.json`
//! (worlds keep their own parameters, in their saves).

use std::path::PathBuf;
use std::time::Duration;

use bevy::ecs::system::NonSendMarker;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::winit::{UpdateMode, WINIT_WINDOWS, WinitSettings};
use evolve_sim::low_priority;
use serde::{Deserialize, Serialize};

use crate::Snap;
use crate::env::GraphicsSettings;

pub struct SettingsPlugin;

impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        let file = load();
        app.insert_resource(file.app)
            .insert_resource(file.graphics)
            .init_resource::<Minimized>()
            .add_systems(Update, (apply_low_priority, pace_frames, save_settings).chain());
    }
}

/// When the worlds give way to other programs (see `evolve_sim::low_priority`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LowPriority {
    Never,
    #[default]
    WhenMinimized,
    Always,
}

#[derive(Resource, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub low_priority: LowPriority,
    /// Slow down when less than this share of the CPU or the GPU is free (0..1).
    pub give_way_below: f32,
    /// The slowest the worlds go, as a share of their full pace.
    pub slowest: f32,
    /// Move brains off the GPU when less than this much of its memory is free, GB.
    pub vram_free_gb: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        let c = low_priority::DEFAULT_CONFIG;
        Self {
            low_priority: LowPriority::default(),
            give_way_below: c.give_way_below,
            slowest: c.slowest,
            vram_free_gb: c.vram_free_mb / 1000.0,
        }
    }
}

impl AppSettings {
    fn config(&self) -> low_priority::Config {
        low_priority::Config {
            give_way_below: self.give_way_below,
            slowest: self.slowest,
            vram_free_mb: self.vram_free_gb * 1000.0,
        }
    }
}

/// The settings file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct File {
    #[serde(flatten)]
    app: AppSettings,
    graphics: GraphicsSettings,
}

/// Whether the window is minimized, as of the last look.
#[derive(Resource, Default)]
pub struct Minimized(pub bool);

pub fn path() -> PathBuf {
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    config.join("evolve").join("settings.json")
}

/// The settings file, or the defaults when there's none yet (or it can't be read, which is said).
fn load() -> File {
    let p = path();
    match std::fs::read_to_string(&p) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            eprintln!("evolve: {} can't be read ({e}); using the default settings", p.display());
            File::default()
        }),
        Err(_) => File::default(),
    }
}

/// Turns low priority mode on or off to match the setting and the window, and hands it the
/// settings' thresholds. Minimized is asked of the window system (X11, Windows and macOS can
/// tell; Wayland can't, so there it never counts as minimized) twice a second, or every frame
/// while minimized (frames are a second apart then).
fn apply_low_priority(
    _main_thread: NonSendMarker,
    settings: Res<AppSettings>,
    window: Query<Entity, With<PrimaryWindow>>,
    time: Res<Time>,
    mut minimized: ResMut<Minimized>,
    mut next_look: Local<f64>,
) {
    let now = time.elapsed_secs_f64();
    if (now >= *next_look || minimized.0)
        && let Ok(entity) = window.single()
    {
        *next_look = now + 0.5;
        let m = WINIT_WINDOWS.with_borrow(|w| w.get_window(entity).and_then(|win| win.is_minimized())).unwrap_or(false);
        if m != minimized.0 {
            minimized.0 = m;
        }
    }
    let on = match settings.low_priority {
        LowPriority::Never => false,
        LowPriority::WhenMinimized => minimized.0,
        LowPriority::Always => true,
    };
    low_priority::set_enabled(on, settings.low_priority == LowPriority::WhenMinimized);
    if settings.is_changed() {
        low_priority::set_config(settings.config());
    }
}

/// Low priority mode draws less often as the worlds slow down (a few frames a second at 5% of
/// their pace), and about once a second while the window is minimized. Input still wakes it at
/// once. Otherwise, Bevy's usual pace for games.
fn pace_frames(snap: Res<Snap>, minimized: Res<Minimized>, mut winit: ResMut<WinitSettings>) {
    let want = match &snap.0.low_priority {
        Some(_) if minimized.0 => wait(Duration::from_secs(1)),
        Some(s) if s.share < 1.0 => wait(Duration::from_secs_f32(1.0 / (60.0 * s.share))),
        _ => WinitSettings::game(),
    };
    if winit.focused_mode != want.focused_mode || winit.unfocused_mode != want.unfocused_mode {
        *winit = want;
    }
}

fn wait(d: Duration) -> WinitSettings {
    let mode = UpdateMode::reactive_low_power(d);
    WinitSettings { focused_mode: mode, unfocused_mode: mode }
}

/// Writes the settings file when something in it changes. What they were at the first look
/// (after the command line's say, like --no-shadows) isn't written: only changes made here are.
fn save_settings(settings: Res<AppSettings>, graphics: Res<GraphicsSettings>, mut saved: Local<Option<File>>) {
    let now = File { app: settings.clone(), graphics: graphics.clone() };
    let Some(before) = saved.as_ref() else {
        *saved = Some(now);
        return;
    };
    if *before == now {
        return;
    }
    let p = path();
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(p.parent().unwrap_or(&p))?;
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&now).map_err(std::io::Error::other)?)?;
        std::fs::rename(&tmp, &p)
    };
    if let Err(e) = write() {
        eprintln!("evolve: can't save the settings to {}: {e}", p.display());
    }
    *saved = Some(now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_file_round_trips_and_fills_gaps_with_defaults() {
        let f = File {
            app: AppSettings { low_priority: LowPriority::Always, slowest: 0.2, ..AppSettings::default() },
            graphics: GraphicsSettings { bloom: false, ..GraphicsSettings::default() },
        };
        let text = serde_json::to_string_pretty(&f).unwrap();
        assert_eq!(serde_json::from_str::<File>(&text).unwrap(), f);
        // A file from before a setting existed (or edited by hand) keeps the rest at their defaults.
        let partial: File = serde_json::from_str(r#"{"low_priority": "Never", "graphics": {"rain": false}}"#).unwrap();
        assert_eq!(partial.app, AppSettings { low_priority: LowPriority::Never, ..AppSettings::default() });
        assert_eq!(partial.graphics, GraphicsSettings { rain: false, ..GraphicsSettings::default() });
    }
}
