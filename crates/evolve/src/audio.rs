//! Sound, all synthesized (no sound files): wind, rain, surf and thunder, set by the weather and
//! where the camera is; the creatures' own calls, each in its genetic voice; and bites, kills,
//! births, new generations, splashes, bubbles and bodies hitting the ground. Every sound comes
//! from where it happens (left or right) and fades with distance from the camera.
//!
//! The synth runs on the audio device's thread (cpal). Bevy hands it ambience levels through
//! atomics and one-shot sounds through a channel, so the two never wait on each other.

use std::f32::consts::{PI, TAU};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use evolve_sim::snapshot::HighlightKind;

use crate::camera::MainCamera;
use crate::weather_fx::LightningStrike;
use crate::{Args, Snap};

pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SoundSettings>()
            .add_systems(Startup, start_audio)
            .add_systems(Update, feed_audio);
    }
}

/// What the user controls.
#[derive(Resource)]
pub struct SoundSettings {
    pub muted: bool,
    pub volume: f32,
    /// Why there's no sound, if there isn't.
    pub error: Option<String>,
}

impl Default for SoundSettings {
    fn default() -> Self {
        Self {
            muted: false,
            volume: 0.7,
            error: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sfx {
    Bite,
    Kill,
    Birth,
    Record,
    Splash,
    Bubbles,
    /// `near` thunder cracks before it rumbles.
    Thunder { near: bool },
    /// A body hitting the ground: a thud, deeper for heavier parts (see `SoundEvent::pitch`).
    Impact,
    /// A creature calling, in its voice (`SoundEvent::voice`) at a pitch its brain picked.
    Call,
    /// A miracle: a choir's chord swelling in, with bells (feeding by the hand of god).
    Blessing,
    /// A shimmer rising away (a creature taken out of the world).
    Ascend,
}

impl Sfx {
    /// Loudness trim per sound so they sit together in the mix, checked by `the_mix_is_balanced`
    /// (targets in LUFS at gain 1, close up).
    fn trim(self) -> f32 {
        match self {
            Sfx::Bite => 2.265,
            Sfx::Kill => 1.738,
            Sfx::Birth => 0.631,
            Sfx::Record => 0.513,
            Sfx::Splash => 0.832,
            Sfx::Bubbles => 0.335,
            Sfx::Thunder { near: true } => 1.047,
            Sfx::Thunder { near: false } => 0.759,
            Sfx::Impact => 2.985,
            Sfx::Call => 0.359,
            Sfx::Blessing => 0.822,
            Sfx::Ascend => 0.776,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoundEvent {
    pub sfx: Sfx,
    pub gain: f32,
    /// -1 left .. 1 right.
    pub pan: f32,
    /// Pitch multiplier (1 normal).
    pub pitch: f32,
    /// A caller's voice genes (see `Traits::voice`); unused by other sounds.
    pub voice: [f32; 4],
}

/// Loudness trims for the ambience, at full intensity (see `Sfx::trim`).
const WIND_TRIM: f32 = 0.359;
const RAIN_TRIM: f32 = 0.285;
const SURF_TRIM: f32 = 0.305;

/// Ambience levels, 0..1, written by Bevy and read by the synth.
#[derive(Default)]
pub struct Levels {
    master: AtomicU32,
    wind: AtomicU32,
    rain: AtomicU32,
    surf: AtomicU32,
}

fn store(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

fn load(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

// ── The synth ──────────────────────────────────────────────────────────

struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn unit(&mut self) -> f32 {
        self.next() * 0.5 + 0.5
    }
}

/// One-pole low-pass.
#[derive(Default, Clone, Copy)]
struct Lp(f32);

impl Lp {
    fn run(&mut self, x: f32, cutoff: f32, rate: f32) -> f32 {
        let a = 1.0 - (-TAU * cutoff / rate).exp();
        self.0 += a * (x - self.0);
        self.0
    }
}

/// A one-shot sound in progress.
struct Voice {
    sfx: Sfx,
    t: f32,
    gain: f32,
    pan: f32,
    phase: f32,
    lp: Lp,
    lp2: Lp,
    lp3: Lp,
    /// Per-voice randomness (pitch, timing).
    seed: f32,
    pitch: f32,
    phase2: f32,
    voice: [f32; 4],
}

impl Voice {
    fn length(&self) -> f32 {
        match self.sfx {
            Sfx::Bite => 0.18,
            Sfx::Kill => 0.5,
            Sfx::Birth => 1.4,
            Sfx::Record => 1.3,
            Sfx::Splash => 0.6,
            Sfx::Bubbles => 0.9,
            Sfx::Thunder { .. } => 6.0,
            Sfx::Impact => 0.35,
            // Steady voices hold a note; warbling ones chirp.
            Sfx::Call => 0.55 - 0.3 * self.voice[3],
            Sfx::Blessing => 3.6,
            Sfx::Ascend => 2.2,
        }
    }

    fn sample(&mut self, n: &mut Noise, rate: f32) -> f32 {
        let t = self.t;
        let dt = 1.0 / rate;
        self.t += dt;
        let tone = |phase: &mut f32, f: f32| {
            *phase = (*phase + f * dt).fract();
            (*phase * TAU).sin()
        };
        match self.sfx {
            Sfx::Bite => {
                // A snap of mid noise over a quick low knock.
                let snap = self.lp.run(n.next(), 1800.0, rate) - self.lp2.run(n.next(), 500.0, rate);
                let knock = tone(&mut self.phase, 140.0 - 300.0 * t) * (-t * 40.0).exp();
                snap * 2.5 * (-t * 45.0).exp() + knock * 0.6
            }
            Sfx::Kill => {
                let crunch = self.lp.run(n.next(), 900.0, rate) * (-t * 12.0).exp() * (1.0 + (t * 90.0).sin());
                let thud = tone(&mut self.phase, 75.0 - 40.0 * t) * (-t * 9.0).exp();
                crunch * 1.6 + thud * 0.8
            }
            Sfx::Birth => {
                // Two soft bell notes, a fifth apart.
                let base = 660.0 * (1.0 + 0.06 * self.seed);
                let mut s = 0.0;
                for (k, (f, at)) in [(base, 0.0), (base * 1.5, 0.14)].into_iter().enumerate() {
                    if t >= at {
                        let u = t - at;
                        let w = (TAU * f * u).sin() + 0.3 * (TAU * 2.0 * f * u).sin();
                        s += w * (-u * 3.5).exp() * (1.0 - (-u * 200.0).exp()) * if k == 0 { 0.5 } else { 0.4 };
                    }
                }
                s
            }
            Sfx::Record => {
                // A rising arpeggio.
                let mut s = 0.0;
                for (k, f) in [523.0, 659.0, 784.0, 1047.0].into_iter().enumerate() {
                    let at = k as f32 * 0.09;
                    if t >= at {
                        let u = t - at;
                        s += ((TAU * f * u).sin() + 0.25 * (TAU * 3.0 * f * u).sin()) * (-u * 4.0).exp() * 0.3;
                    }
                }
                s
            }
            Sfx::Splash => {
                let cutoff = 300.0 + 3500.0 * (-t * 6.0).exp();
                self.lp.run(n.next(), cutoff, rate) * 2.0 * (-t * 7.0).exp() * (1.0 - (-t * 300.0).exp())
            }
            Sfx::Bubbles => {
                // Little blips rising in pitch, at uneven moments.
                let slot = (t / 0.11).floor();
                let u = t - slot * 0.11 - 0.02 * ((slot * 7.3 + self.seed * 13.0).sin() + 1.0);
                if !(0.0..=0.05).contains(&u) {
                    return 0.0;
                }
                let f = 350.0 + 500.0 * ((slot * 3.1 + self.seed * 5.0).sin() * 0.5 + 0.5) + 6000.0 * u;
                tone(&mut self.phase, f) * (u / 0.05 * PI).sin() * 0.5
            }
            Sfx::Thunder { near } => {
                // A crack and a tearing sound (if close), then a long rolling rumble with enough
                // midrange to carry on small speakers, not just sub-bass.
                let (crack, tear) = if near {
                    let hiss = n.next() - self.lp2.run(n.next(), 2500.0, rate);
                    (hiss * 1.2 * (-t * 25.0).exp(), self.lp3.run(n.next(), 1400.0, rate) * 2.5 * (-t * 4.0).exp())
                } else {
                    (0.0, 0.0)
                };
                let roll = 0.55 + 0.45 * (t * (1.7 + self.seed)).sin() * (t * 0.6 + 3.0 * self.seed).sin();
                let attack = 1.0 - (-t * if near { 40.0 } else { 2.5 }).exp();
                let body = self.lp.run(n.next(), 380.0 + 120.0 * self.seed, rate);
                crack + tear + body * 5.0 * attack * (-t * 0.6).exp() * roll
            }
            Sfx::Call => {
                // An FM tone: odd timbre genes give inharmonic, metallic or bubbling sounds. It
                // wavers by the warble gene, has noise in it by the rasp gene, and glides up for a
                // high call and down for a low one.
                let [vp, timbre, rasp, warble] = self.voice;
                let len = self.length();
                let u = (t / len).min(1.0);
                let base = 110.0 * 10f32.powf(vp.clamp(0.0, 1.0)) * self.pitch;
                let glide = 1.0 + 0.5 * self.pitch.log2() * u;
                let waver = 1.0 + (0.02 + 0.12 * warble) * (TAU * (2.0 + 26.0 * warble) * t).sin();
                let f = base * glide * waver;
                self.phase2 = (self.phase2 + f * (0.5 + 3.2 * timbre) * dt).fract();
                let depth = (0.5 + 5.0 * timbre) * (1.0 - 0.6 * u);
                let fm = (self.phase2 * TAU).sin() * depth;
                self.phase = (self.phase + f * dt).fract();
                let tone = (self.phase * TAU + fm).sin();
                let grit = self.lp.run(n.next(), base * 3.0, rate) - self.lp2.run(n.next(), base * 0.5, rate);
                let env = (t / 0.02).min(1.0) * (1.0 - u).powf(0.7);
                // High voices and their wider noise band sound louder; even them out.
                let even = (330.0 / base).powf(0.35);
                let grit = grit * (330.0 / base).sqrt().min(1.5);
                (tone * (1.0 - 0.7 * rasp) + grit * 3.0 * rasp) * env * even
            }
            Sfx::Blessing => {
                // A soft "ah" on a major chord, each note two voices a little out of tune with
                // each other and gently wavering, swelling in and dying away, with high bells.
                let len = self.length();
                let env = (t / 0.7).min(1.0).powf(1.5) * ((len - t) / 1.8).clamp(0.0, 1.0);
                let mut choir = 0.0;
                for (k, f) in [220.0f32, 277.2, 329.6, 440.0, 554.4].into_iter().enumerate() {
                    for d in [-1.0f32, 1.0] {
                        let rate = 5.0 + 0.4 * k as f32 + 0.3 * d;
                        let ph = TAU * f * (1.0 + d * 0.003) * t + 0.002 * f / rate * (TAU * rate * t + k as f32).sin();
                        choir += ph.sin() + 0.45 * (2.0 * ph).sin() + 0.25 * (3.0 * ph).sin() + 0.1 * (4.0 * ph).sin();
                    }
                }
                let mut bells = 0.0;
                for (k, f) in [1760.0f32, 2217.5, 2637.0, 3520.0].into_iter().enumerate() {
                    let at = 0.15 + k as f32 * 0.22;
                    if t >= at {
                        let u = t - at;
                        bells += (TAU * f * u).sin() * (-u * 2.5).exp() * (1.0 - (-u * 300.0).exp());
                    }
                }
                choir * 0.06 * env + bells * 0.12
            }
            Sfx::Ascend => {
                // Bell partials gliding up an octave as they fade, over a breath of rising air.
                let len = self.length();
                let u = (t / len).min(1.0);
                let env = (t / 0.04).min(1.0) * (1.0 - u).powf(1.5);
                let glide = len / std::f32::consts::LN_2 * (2f32.powf(t / len) - 1.0);
                let mut s = 0.0;
                for (k, f0) in [660.0f32, 990.0, 1320.0, 1980.0].into_iter().enumerate() {
                    s += (TAU * f0 * glide + k as f32).sin() * 0.5 / (k as f32 + 1.0);
                }
                let air = self.lp.run(n.next(), 2500.0 + 3000.0 * u, rate);
                (s * 0.5 + air * 0.6 * (1.0 - u)) * env
            }
            Sfx::Impact => {
                // A short low knock with a little grit on top; `pitch` drops for heavy parts.
                let f = 140.0 * self.pitch * (1.0 - 0.5 * (t / 0.08).min(1.0));
                let knock = tone(&mut self.phase, f) * (-t * 22.0).exp();
                let grit = self.lp.run(n.next(), 900.0 * self.pitch, rate) * (-t * 45.0).exp();
                knock + grit * 1.5
            }
        }
    }
}

/// Most one-shot sounds playing at once, and most of those that are calls.
const MAX_VOICES: usize = 24;
const MAX_CALLS: usize = 8;

pub struct Synth {
    pub rate: f32,
    n: Noise,
    t: f32,
    levels: Arc<Levels>,
    events: Receiver<SoundEvent>,
    voices: Vec<Voice>,
    // Smoothed copies of the levels, so nothing clicks when they change.
    smooth: [f32; 4],
    wind_l: [Lp; 4],
    rain_lp: [Lp; 2],
    surf_lp: [Lp; 2],
    gust: Lp,
    drop: (f32, f32, Lp),
}

impl Synth {
    pub fn new(rate: f32, levels: Arc<Levels>, events: Receiver<SoundEvent>) -> Self {
        Self {
            rate,
            n: Noise(0x1234_5678),
            t: 0.0,
            levels,
            events,
            voices: Vec::new(),
            smooth: [0.0; 4],
            wind_l: [Lp::default(); 4],
            rain_lp: [Lp::default(); 2],
            surf_lp: [Lp::default(); 2],
            gust: Lp::default(),
            drop: (0.0, 0.0, Lp::default()),
        }
    }

    /// Fills interleaved frames of `channels` channels (stereo in the first two).
    pub fn render(&mut self, out: &mut [f32], channels: usize) {
        while let Ok(e) = self.events.try_recv() {
            // A crowd calling at once could otherwise take every voice.
            let calls = self.voices.iter().filter(|v| v.sfx == Sfx::Call).count();
            if self.voices.len() < MAX_VOICES && (e.sfx != Sfx::Call || calls < MAX_CALLS) {
                let seed = self.n.unit();
                self.voices.push(Voice {
                    sfx: e.sfx,
                    t: 0.0,
                    gain: e.gain,
                    pan: e.pan.clamp(-1.0, 1.0),
                    phase: 0.0,
                    lp: Lp::default(),
                    lp2: Lp::default(),
                    lp3: Lp::default(),
                    seed,
                    pitch: e.pitch,
                    phase2: 0.0,
                    voice: e.voice,
                });
            }
        }
        let l = &self.levels;
        let target = [
            load(&l.master),
            load(&l.wind),
            load(&l.rain),
            load(&l.surf),
        ];
        let rate = self.rate;
        let dt = 1.0 / rate;
        let channels = channels.max(1);
        for frame in out.chunks_mut(channels) {
            for (s, t) in self.smooth.iter_mut().zip(target) {
                *s += (t - *s) * 0.0004;
                // Land on the target rather than creeping into denormals, which are slow.
                if (t - *s).abs() < 1e-6 {
                    *s = t;
                }
            }
            let [master, wind, rain, surf] = self.smooth;
            self.t += dt;
            let (mut left, mut right) = (0.0, 0.0);
            if master > 1e-4 {
                let n = &mut self.n;
                // Wind: band-limited noise whose brightness and loudness follow slow gusts.
                if wind > 1e-3 {
                    let gust = self.gust.run(n.next(), 0.3, rate) * 6.0;
                    let bright = (250.0 + 500.0 * (wind + gust).clamp(0.0, 1.5)).max(80.0);
                    let a = self.wind_l[0].run(n.next(), bright, rate) - self.wind_l[1].run(n.next(), 60.0, rate);
                    let b = self.wind_l[2].run(n.next(), bright * 1.1, rate) - self.wind_l[3].run(n.next(), 60.0, rate);
                    let g = wind * wind * 1.8 * (0.6 + gust.abs().min(0.8)) * WIND_TRIM;
                    left += a * g;
                    right += b * g;
                }
                // Rain: a high hiss with the odd louder drop.
                if rain > 1e-3 {
                    let h0 = n.next() - self.rain_lp[0].run(n.next(), 1500.0, rate);
                    let h1 = n.next() - self.rain_lp[1].run(n.next(), 1500.0, rate);
                    left += h0 * rain * 0.12 * RAIN_TRIM;
                    right += h1 * rain * 0.12 * RAIN_TRIM;
                    if n.unit() < rain * 0.0015 {
                        self.drop = (1.0, n.next(), Lp::default());
                    }
                    let (env, pan, lp) = &mut self.drop;
                    if *env > 1e-3 {
                        let d = lp.run(n.next(), 2500.0, rate) * *env * rain * 0.6 * RAIN_TRIM;
                        *env *= 0.996;
                        left += d * (1.0 - *pan) * 0.5;
                        right += d * (1.0 + *pan) * 0.5;
                    }
                }
                // Surf: low noise swelling and ebbing like waves on a beach.
                if surf > 1e-3 {
                    let swell = ((self.t * TAU / 7.3 + (self.t * 0.43).sin()).sin() * 0.5 + 0.5).powf(2.0);
                    let cut = 250.0 + 900.0 * swell;
                    let a = self.surf_lp[0].run(n.next(), cut, rate);
                    let b = self.surf_lp[1].run(n.next(), cut, rate);
                    let g = surf * (0.25 + 1.4 * swell) * SURF_TRIM;
                    left += a * g;
                    right += b * g;
                }
                for v in &mut self.voices {
                    let s = v.sample(n, rate) * v.gain * v.sfx.trim();
                    let a = (v.pan + 1.0) * 0.25 * PI;
                    left += s * a.cos();
                    right += s * a.sin();
                }
            } else {
                for v in &mut self.voices {
                    v.t += dt;
                }
            }
            let (l, r) = ((left * master).tanh(), (right * master).tanh());
            frame[0] = if channels == 1 { (l + r) * 0.5 } else { l };
            if channels > 1 {
                frame[1] = r;
            }
            for x in frame.iter_mut().skip(2) {
                *x = 0.0;
            }
        }
        self.voices.retain(|v| v.t < v.length());
    }
}

// ── The device ─────────────────────────────────────────────────────────

/// Keeps the link to the audio thread.
#[derive(Resource)]
struct Sound {
    levels: Arc<Levels>,
    /// Bounded, so sounds don't pile up while no device is draining them.
    events: SyncSender<SoundEvent>,
    /// The device playing, or why there's none (the thread keeps trying).
    status: Arc<Mutex<Result<String, String>>>,
}

/// Output devices to try, best first: through PipeWire or PulseAudio, which share the speakers
/// and follow the desktop's choice of them, then ALSA's default, then anything else.
fn output_devices(host: &cpal::Host) -> Vec<cpal::Device> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let rank = |d: &cpal::Device| match d.name().unwrap_or_default().as_str() {
        "pipewire" => 0,
        "pulse" => 1,
        "default" => 2,
        _ => 3,
    };
    let mut all: Vec<cpal::Device> = host.output_devices().map(|d| d.collect()).unwrap_or_default();
    if let Some(d) = host.default_output_device()
        && !all.iter().any(|x| x.name().ok() == d.name().ok())
    {
        all.push(d);
    }
    all.sort_by_key(rank);
    all
}

/// How a stream is doing, as seen from its callbacks.
struct Health {
    started: Instant,
    /// Set by any stream error.
    broken: AtomicBool,
    /// When audio was last asked for, in milliseconds since `started` (0: not yet).
    last_callback_ms: AtomicU64,
    /// When an error was last logged, likewise (errors can repeat thousands of times a second).
    last_log_ms: AtomicU64,
}

/// A stream that hasn't asked for audio in this long has stalled.
const STALL: Duration = Duration::from_secs(4);

impl Health {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            broken: AtomicBool::new(false),
            last_callback_ms: AtomicU64::new(0),
            last_log_ms: AtomicU64::new(0),
        }
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// Not broken, and not stalled. A stream can take several seconds to ask for its first
    /// audio, and tearing it down then only starts the wait over, so only a stream that has
    /// played and then stopped counts as stalled.
    fn ok(&self) -> bool {
        let last = self.last_callback_ms.load(Ordering::Relaxed);
        let stalled = last > 0 && self.now_ms().saturating_sub(last) >= STALL.as_millis() as u64;
        !self.broken.load(Ordering::Relaxed) && !stalled
    }
}

/// Starts playing on the first device that works. Returns the stream, its device's name, and how
/// it's doing.
fn open_any(synth: &Arc<Mutex<Synth>>) -> Result<(cpal::Stream, String, Arc<Health>), String> {
    use cpal::traits::{DeviceTrait, StreamTrait};
    let host = cpal::default_host();
    let mut why = String::from("no audio output device");
    for device in output_devices(&host) {
        let name = device.name().unwrap_or_else(|_| "audio".into());
        let tried = (|| -> Result<(cpal::Stream, Arc<Health>), String> {
            let supported = device.default_output_config().map_err(|e| e.to_string())?;
            let format = supported.sample_format();
            let config: cpal::StreamConfig = supported.into();
            if let Ok(mut s) = synth.lock() {
                s.rate = config.sample_rate.0 as f32;
            }
            let health = Arc::new(Health::new());
            let stream = match format {
                cpal::SampleFormat::F32 => build::<f32>(&device, &config, synth.clone(), health.clone()),
                cpal::SampleFormat::I16 => build::<i16>(&device, &config, synth.clone(), health.clone()),
                cpal::SampleFormat::U16 => build::<u16>(&device, &config, synth.clone(), health.clone()),
                cpal::SampleFormat::I32 => build::<i32>(&device, &config, synth.clone(), health.clone()),
                other => return Err(format!("unsupported sample format {other}")),
            }
            .map_err(|e| e.to_string())?;
            stream.play().map_err(|e| e.to_string())?;
            Ok((stream, health))
        })();
        match tried {
            Ok((stream, health)) => return Ok((stream, name, health)),
            Err(e) => why = format!("{name}: {e}"),
        }
    }
    Err(why)
}

/// Runs the audio device on a thread of its own (a stream must stay on the thread that made it),
/// reopening it whenever it fails, stalls or there was none: unplugged headsets, a call app
/// taking the speakers, and so on.
fn run_output(synth: Arc<Mutex<Synth>>, status: Arc<Mutex<Result<String, String>>>) {
    let set = |s: Result<String, String>| {
        if let Ok(mut st) = status.lock() {
            *st = s;
        }
    };
    loop {
        match open_any(&synth) {
            Ok((stream, name, health)) => {
                set(Ok(name));
                while health.ok() {
                    std::thread::sleep(Duration::from_millis(500));
                }
                set(Err("audio device was lost; reconnecting".into()));
                // Dropping a dead stream can block for good (cpal waits for its worker, which may
                // be stuck retrying the device), so reconnect from a fresh thread and leave this
                // one to drop it.
                let (synth, status) = (synth.clone(), status.clone());
                let respawned = std::thread::Builder::new().name("evolve-audio".into()).spawn(move || {
                    std::thread::sleep(Duration::from_secs(1));
                    run_output(synth, status);
                });
                if let Err(e) = respawned {
                    set(Err(format!("audio thread: {e}")));
                }
                drop(stream);
                return;
            }
            Err(e) => {
                set(Err(e));
                std::thread::sleep(Duration::from_secs(5));
            }
        }
    }
}

fn build<T: cpal::SizedSample + cpal::FromSample<f32>>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    synth: Arc<Mutex<Synth>>,
    health: Arc<Health>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    use cpal::traits::DeviceTrait;
    let channels = config.channels as usize;
    let mut buf: Vec<f32> = Vec::new();
    let errors = health.clone();
    device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            health.last_callback_ms.store(health.now_ms().max(1), Ordering::Relaxed);
            buf.resize(data.len(), 0.0);
            match synth.try_lock() {
                Ok(mut s) => s.render(&mut buf, channels),
                Err(_) => buf.fill(0.0),
            }
            for (o, s) in data.iter_mut().zip(&buf) {
                *o = T::from_sample(*s);
            }
        },
        move |e| {
            // Any error means a reconnect. A dead device can report errors in a tight loop, so
            // log at most every 10 s, and slow the loop down once the stream is given up on.
            let was_broken = errors.broken.swap(true, Ordering::Relaxed);
            let now = errors.now_ms();
            let last = errors.last_log_ms.load(Ordering::Relaxed);
            if !was_broken || now.saturating_sub(last) >= 10_000 {
                errors.last_log_ms.store(now, Ordering::Relaxed);
                eprintln!("evolve: audio: {e}");
            }
            if was_broken {
                std::thread::sleep(Duration::from_millis(100));
            }
        },
        None,
    )
}

fn start_audio(mut commands: Commands, args: Res<Args>, mut settings: ResMut<SoundSettings>) {
    // Automated screenshot runs stay quiet.
    settings.muted = args.mute || args.screenshot.is_some();
    let levels = Arc::new(Levels::default());
    let (tx, rx) = mpsc::sync_channel(256);
    let synth = Arc::new(Mutex::new(Synth::new(48_000.0, levels.clone(), rx)));
    let opening = "opening the audio device".to_string();
    settings.error = Some(opening.clone());
    let status = Arc::new(Mutex::new(Err(opening)));
    let st = status.clone();
    if let Err(e) = std::thread::Builder::new()
        .name("evolve-audio".into())
        .spawn(move || run_output(synth, st))
    {
        settings.error = Some(e.to_string());
    }
    commands.insert_resource(Sound { levels, events: tx, status });
}

/// Within this many meters of the camera a sound plays at full strength.
const NEAR: f32 = 6.0;

/// How loud a sound is at a distance from the camera: full within `NEAR`, then falling as 1/d
/// (halving with each doubling of distance, as sound does in open air).
fn distance_gain(d: f32) -> f32 {
    (NEAR / d.max(NEAR)).min(1.0)
}

fn rand(s: &mut u32) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s as f32 / u32::MAX as f32
}

#[derive(Default)]
struct FeedState {
    epoch: u64,
    last_seq: u64,
    /// When each kind of sound last played, to keep a busy world from turning into noise.
    last: Vec<(Sfx, f64)>,
    next_thunder: f64,
    /// Thunder on its way: (when it arrives, near, gain, pan).
    thunder: Vec<(f64, bool, f32, f32)>,
    rng: u32,
}

fn feed_audio(
    sound: Option<Res<Sound>>,
    mut settings: ResMut<SoundSettings>,
    snap: Res<Snap>,
    time: Res<Time>,
    camera: Query<&GlobalTransform, With<MainCamera>>,
    mut strikes: MessageReader<LightningStrike>,
    mut st: Local<FeedState>,
) {
    let Some(sound) = sound else {
        return;
    };
    if let Ok(st) = sound.status.try_lock() {
        let error = st.as_ref().err().cloned();
        if error != settings.error {
            match &*st {
                Ok(name) => info!("sound on {name}"),
                Err(e) => warn!("no sound yet: {e}"),
            }
            settings.error = error;
        }
    }
    let lv = &sound.levels;
    store(&lv.master, if settings.muted { 0.0 } else { settings.volume });
    let Ok(cam) = camera.single() else {
        return;
    };
    let snap = &*snap.0;
    let t = &snap.terrain;
    if t.res < 3 {
        return;
    }
    let now = time.elapsed_secs_f64();
    if st.rng == 0 {
        st.rng = 0x2545_F491;
    }
    let w = &snap.weather;
    let wl = snap.params.water_level;
    let pos = cam.translation();
    let ground = t.height_at(pos.x, pos.z).max(wl);
    // How much of what's below the camera is water, sampled around the spot it looks at.
    let fwd = cam.forward().as_vec3();
    let look = if fwd.y < -0.05 { pos + fwd * ((pos.y - ground) / -fwd.y).min(400.0) } else { pos };
    let mut water = 0.0;
    for k in 0..12 {
        let a = k as f32 * TAU / 12.0;
        for r in [8.0, 20.0] {
            water += (t.height_at(look.x + a.cos() * r, look.z + a.sin() * r) < wl) as u8 as f32;
        }
    }
    let water = water / 24.0;
    let near = 1.0 / (1.0 + ((pos.y - ground).max(0.0) / 60.0).powi(2));
    let wind = ((w.wind[0].powi(2) + w.wind[1].powi(2)).sqrt() / 18.0).min(1.2);
    store(&lv.wind, wind * (0.5 + 0.5 * (1.0 - near)));
    store(&lv.rain, w.rain * (0.4 + 0.6 * near));
    // Loudest at the shoreline, where land and water meet.
    store(&lv.surf, (4.0 * water * (1.0 - water) * 0.8 + water * 0.2) * near * (0.6 + 0.6 * wind.min(1.0)));

    // Nothing plays without a device, so don't queue anything for it.
    let playing = settings.error.is_none();
    let send = |sfx: Sfx, gain: f32, pan: f32, pitch: f32, voice: [f32; 4]| {
        if playing {
            let _ = sound.events.try_send(SoundEvent { sfx, gain, pan, pitch, voice });
        }
    };
    // Thunder follows each lightning flash after the time sound takes to get here.
    let right = cam.right().as_vec3();
    for s in strikes.read() {
        let d = s.at.distance(pos);
        let pan = (s.at - pos).normalize_or_zero().dot(right);
        let gain = (1.4 / (1.0 + d / 300.0)).min(1.0);
        st.thunder.push((now + (d / 343.0) as f64, d < 260.0, gain, pan));
    }
    st.thunder.retain(|&(at, near, gain, pan)| {
        if now >= at {
            send(Sfx::Thunder { near }, gain, pan, 1.0, [0.0; 4]);
        }
        now < at
    });
    // And storms rumble now and then from lightning out of sight.
    if w.storm > 0.15 && now >= st.next_thunder {
        if st.next_thunder > 0.0 {
            send(Sfx::Thunder { near: false }, 0.3 + 0.3 * w.storm, rand(&mut st.rng) * 2.0 - 1.0, 1.0, [0.0; 4]);
        }
        st.next_thunder = now + (8.0 + 20.0 * (1.0 - w.storm) * rand(&mut st.rng) + 8.0 * rand(&mut st.rng)) as f64;
    } else if w.storm <= 0.15 {
        st.next_thunder = 0.0;
    }

    // One-shot sounds for what happens near the camera.
    if snap.world_epoch != st.epoch {
        st.epoch = snap.world_epoch;
        st.last_seq = snap.highlights.last().map(|h| h.seq).unwrap_or(0);
        return;
    }
    for h in snap.highlights.iter() {
        if h.seq <= st.last_seq {
            continue;
        }
        st.last_seq = h.seq;
        // (sound, loudness at the source, least seconds between two of them, pitch)
        let (sfx, loud, gap, pitch) = match h.kind {
            HighlightKind::Attack => (Sfx::Bite, 1.0, 0.12, 1.0),
            HighlightKind::Kill => (Sfx::Kill, 1.0, 0.25, 1.0),
            HighlightKind::Birth => (Sfx::Birth, 1.0, 0.4, 1.0),
            HighlightKind::Record => (Sfx::Record, 1.0, 1.0, 1.0),
            HighlightKind::Splash => (Sfx::Splash, 1.0, 0.2, 1.0),
            HighlightKind::Drowning => (Sfx::Bubbles, 1.0, 0.5, 1.0),
            // Harder hits are louder and, being mostly heavy parts, deeper.
            HighlightKind::Impact => (Sfx::Impact, h.strength.min(1.2), 0.04, (1.3 - 0.5 * h.strength).clamp(0.6, 1.3)),
            // Calls are paced by the sim already (a few a second per caller).
            HighlightKind::Call => (Sfx::Call, h.strength, 0.0, 1.0),
            HighlightKind::Blessing => (Sfx::Blessing, 1.0, 0.5, 1.0),
            HighlightKind::Ascension => (Sfx::Ascend, 1.0, 0.5, 1.0),
            // A smite's sound is its thunder, which the lightning sends.
            HighlightKind::Smite => continue,
        };
        let caller = (sfx == Sfx::Call).then(|| snap.creatures.iter().find(|c| c.id == h.subject)).flatten();
        let (pitch, voice) = match caller {
            Some(c) => (2f32.powf(0.6 * c.call_pitch), c.voice),
            None if sfx == Sfx::Call => continue,
            None => (pitch, [0.0; 4]),
        };
        let at = Vec3::from_array(h.pos);
        let mut gain = loud * distance_gain(at.distance(pos));
        if sfx == Sfx::Record {
            // News worth hearing from anywhere, quietly.
            gain = gain.max(0.1);
        }
        if matches!(sfx, Sfx::Blessing | Sfx::Ascend) {
            // Acts of god are heard wherever the camera is.
            gain = gain.max(0.5);
        }
        if gain < 0.015 || st.last.iter().any(|(s, t)| *s == sfx && now - t < gap) {
            continue;
        }
        st.last.retain(|(s, _)| *s != sfx);
        st.last.push((sfx, now));
        let pan = (at - pos).normalize_or_zero().dot(right);
        send(sfx, gain, pan, pitch, voice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(levels: [f32; 4], events: &[Sfx], seconds: f32) -> Vec<f32> {
        let lv = Arc::new(Levels::default());
        for (a, v) in [&lv.master, &lv.wind, &lv.rain, &lv.surf].into_iter().zip(levels) {
            store(a, v);
        }
        let (tx, rx) = mpsc::channel();
        for &sfx in events {
            tx.send(SoundEvent { sfx, gain: 1.0, pan: 0.3, pitch: 1.0, voice: [0.5; 4] }).unwrap();
        }
        let mut synth = Synth::new(48_000.0, lv, rx);
        let mut out = vec![0.0; (48_000.0 * seconds) as usize * 2];
        for chunk in out.chunks_mut(1024) {
            synth.render(chunk, 2);
        }
        out
    }

    #[test]
    fn everything_at_once_stays_finite_and_in_range() {
        let all = [
            Sfx::Bite,
            Sfx::Kill,
            Sfx::Birth,
            Sfx::Record,
            Sfx::Splash,
            Sfx::Bubbles,
            Sfx::Thunder { near: true },
            Sfx::Thunder { near: false },
            Sfx::Impact,
            Sfx::Call,
            Sfx::Blessing,
            Sfx::Ascend,
        ];
        let out = render([1.0; 4], &all, 4.0);
        assert!(out.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.02, "should be audible: rms {rms}");
    }

    #[test]
    fn each_sound_makes_a_sound_and_muting_silences() {
        for sfx in [Sfx::Bite, Sfx::Kill, Sfx::Birth, Sfx::Record, Sfx::Splash, Sfx::Bubbles, Sfx::Thunder { near: true }, Sfx::Impact, Sfx::Call, Sfx::Blessing, Sfx::Ascend] {
            let out = render([1.0, 0.0, 0.0, 0.0], &[sfx], 1.5);
            let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(peak > 0.05, "{sfx:?} is too quiet: peak {peak}");
        }
        for (k, name) in ["wind", "rain", "surf"].into_iter().enumerate() {
            let mut lv = [0.0; 4];
            lv[0] = 1.0;
            lv[k + 1] = 1.0;
            let out = render(lv, &[], 12.0);
            let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(peak > 0.01, "{name} is silent: peak {peak}");
        }
        let muted = render([0.0, 1.0, 1.0, 1.0], &[Sfx::Kill], 2.0);
        assert!(muted.iter().all(|x| x.abs() < 1e-6));
    }
}

#[cfg(test)]
mod mix {
    use super::*;

    /// Loudness in LUFS (ITU-R BS.1770 K-weighting, 48 kHz coefficients), over the whole buffer.
    pub fn lufs(out: &[f32]) -> f32 {
        let mut ms = 0.0f64;
        for ch in 0..2 {
            let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0, 0.0, 0.0);
            let (mut u1, mut u2, mut z1, mut z2) = (0.0f64, 0.0, 0.0, 0.0);
            let mut sum = 0.0;
            let mut n = 0;
            for f in out.chunks(2) {
                let x = f[ch] as f64;
                let y = 1.53512485958697 * x - 2.69169618940638 * x1 + 1.19839281085285 * x2
                    + 1.69065929318241 * y1
                    - 0.73248077421585 * y2;
                x2 = x1;
                x1 = x;
                y2 = y1;
                y1 = y;
                let z = y - 2.0 * u1 + u2 + 1.99004745483398 * z1 - 0.99007225036621 * z2;
                u2 = u1;
                u1 = y;
                z2 = z1;
                z1 = z;
                sum += z * z;
                n += 1;
            }
            ms += sum / n.max(1) as f64;
        }
        (-0.691 + 10.0 * ms.max(1e-12).log10()) as f32
    }

    pub fn render(levels: [f32; 4], events: &[SoundEvent], seconds: f32) -> Vec<f32> {
        let lv = Arc::new(Levels::default());
        for (a, v) in [&lv.master, &lv.wind, &lv.rain, &lv.surf].into_iter().zip(levels) {
            store(a, v);
        }
        let (tx, rx) = mpsc::channel();
        for &e in events {
            tx.send(e).unwrap();
        }
        let mut synth = Synth::new(48_000.0, lv, rx);
        // Skip the first second while ambience fades in from silence.
        let mut warm = vec![0.0; 48_000 * 2];
        if events.is_empty() {
            for c in warm.chunks_mut(1024) {
                synth.render(c, 2);
            }
        }
        let mut out = vec![0.0; (48_000.0 * seconds) as usize * 2];
        for chunk in out.chunks_mut(1024) {
            synth.render(chunk, 2);
        }
        out
    }

    /// (what, target LUFS, measured LUFS)
    pub fn measure() -> Vec<(String, f32, f32)> {
        let mut out = Vec::new();
        let ambience = [("wind", -22.0), ("rain", -27.0), ("surf", -24.0)];
        for (k, (name, target)) in ambience.into_iter().enumerate() {
            let mut lv = [0.0; 4];
            lv[0] = 1.0;
            lv[k + 1] = 1.0;
            out.push((name.to_string(), target, lufs(&render(lv, &[], 12.0))));
        }
        let events = [
            (Sfx::Thunder { near: true }, -12.0),
            (Sfx::Thunder { near: false }, -18.0),
            (Sfx::Impact, -16.0),
            (Sfx::Kill, -16.0),
            (Sfx::Bite, -20.0),
            (Sfx::Splash, -20.0),
            (Sfx::Birth, -22.0),
            (Sfx::Record, -24.0),
            (Sfx::Bubbles, -26.0),
            (Sfx::Call, -20.0),
            (Sfx::Blessing, -18.0),
            (Sfx::Ascend, -20.0),
        ];
        for (sfx, target) in events {
            let e = SoundEvent { sfx, gain: 1.0, pan: 0.0, pitch: 1.0, voice: [0.5; 4] };
            let v = Voice {
                sfx,
                t: 0.0,
                gain: 1.0,
                pan: 0.0,
                phase: 0.0,
                lp: Lp::default(),
                lp2: Lp::default(),
                lp3: Lp::default(),
                seed: 0.5,
                pitch: 1.0,
                phase2: 0.0,
                voice: [0.5; 4],
            };
            // Over the sound itself (the first two seconds of thunder).
            let buf = render([1.0, 0.0, 0.0, 0.0], &[e], v.length().min(2.0));
            out.push((format!("{sfx:?}"), target, lufs(&buf)));
        }
        out
    }

    #[test]
    fn voices_vary_without_any_being_much_louder() {
        let mut levels = Vec::new();
        for voice in [[0.0; 4], [1.0; 4], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.5, 0.2, 0.8, 0.1], [0.2, 0.9, 0.1, 0.6]] {
            for pitch in [0.66, 1.0, 1.5] {
                let e = SoundEvent { sfx: Sfx::Call, gain: 1.0, pan: 0.0, pitch, voice };
                let l = lufs(&render([1.0, 0.0, 0.0, 0.0], &[e], 0.4));
                println!("voice {voice:?} pitch {pitch}: {l:.1} LUFS");
                levels.push(l);
            }
        }
        let (lo, hi) = levels.iter().fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
        assert!(hi - lo < 12.0 && hi < -12.0, "calls range {lo:.1}..{hi:.1} LUFS");
    }

    #[test]
    fn the_mix_is_balanced() {
        let mut worst = 0.0f32;
        for (name, target, got) in measure() {
            println!("{name:>24}: {got:6.1} LUFS (target {target:5.1}, off by {:+5.1} dB)", got - target);
            worst = worst.max((got - target).abs());
        }
        assert!(worst < 1.5, "a sound is {worst:.1} dB off its target loudness");
    }
}
