//! The director: a hands-off camera for just watching. It looks for something worth seeing (a
//! kill, a birth, a fight, a swimmer, the fastest walker, a sunset, a storm), frames it, slowly
//! circles it, and glides on to the next thing. Moving the camera yourself takes over.

use std::collections::{HashMap, VecDeque};

use bevy::input::InputSystems;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;
use evolve_sim::snapshot::{CreatureView, HighlightKind, Snapshot};

use crate::camera::{FOLLOW_RATE, OrbitCam, center_of_mass, spring};
use crate::env::{Env, VisualOverride};
use crate::tools::ToolState;
use crate::{Args, Snap};

/// The sky as shown on screen (which command-line overrides can differ from the world's).
#[derive(Clone, Copy)]
struct Sky {
    sun: Vec3,
    rain: f32,
    storm: f32,
    time_of_day: f32,
}

pub struct DirectorPlugin;

impl Plugin for DirectorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Director>()
            .add_systems(Startup, |args: Res<Args>, mut d: ResMut<Director>| d.active = args.director)
            .add_systems(PreUpdate, direct.after(crate::refresh_snapshot).after(InputSystems));
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Kill,
    Fight,
    Birth,
    Record,
    Drowning,
    Splash,
    Swimmer,
    Sprinter,
    Call,
    Star,
    Brain,
    Oddball,
    Vista,
}

impl Kind {
    /// How long a shot of this kind typically lasts, seconds (each shot varies around it).
    fn typical_time(self) -> f64 {
        match self {
            Kind::Splash => 9.0,
            Kind::Drowning | Kind::Brain => 13.0,
            Kind::Kill | Kind::Fight | Kind::Birth | Kind::Record | Kind::Star | Kind::Oddball | Kind::Call => 15.0,
            Kind::Vista => 18.0,
            Kind::Swimmer | Kind::Sprinter => 20.0,
        }
    }

    /// Big moments may cut into a shot early.
    fn urgent(self) -> bool {
        matches!(self, Kind::Kill | Kind::Birth | Kind::Record)
    }
}

/// How long the next shot lasts: around its kind's typical time, mostly a little shorter or
/// longer, now and then lingering two or three times as long, so the pace doesn't tick like a
/// metronome.
fn shot_length(d: &mut Director, kind: Kind) -> f64 {
    let r = d.random() as f64;
    let mut f = 0.65 + 0.9 * r * r;
    if d.random() < 0.15 {
        f *= 2.0 + d.random() as f64;
    }
    kind.typical_time() * f
}

struct Candidate {
    kind: Kind,
    subject: Option<u64>,
    other: Option<u64>,
    /// Where to look for a vista.
    place: Vec3,
    score: f32,
    title: String,
    detail: String,
}

struct Shot {
    kind: Kind,
    subject: Option<u64>,
    other: Option<u64>,
    place: Vec3,
    started: f64,
    yaw: f32,
    yaw_rate: f32,
    pitch: f32,
    distance: f32,
    /// Camera (focus, distance, yaw, pitch) when the shot began, to glide from.
    from: (Vec3, f32, f32, f32),
    glide: f64,
    /// How long it holds the shot, seconds.
    length: f64,
    /// Extra pull-back halfway through a long glide, so it flies over rather than through.
    arc: f32,
    focus: Vec3,
    /// How fast `focus` is moving, for the spring that follows the subject.
    focus_vel: Vec3,
    /// The camera distance the shot is easing toward (two subjects drifting apart widen it).
    spread: f32,
    /// When the subject disappeared (died, or left the world).
    lost: Option<f64>,
}

pub struct Caption {
    pub title: String,
    pub detail: String,
    pub started: f64,
    pub until: f64,
}

#[derive(Resource, Default)]
pub struct Director {
    pub active: bool,
    /// Show the usual panels while directing (off: a clean screen).
    pub panels: bool,
    pub caption: Option<Caption>,
    shot: Option<Shot>,
    recent: VecDeque<(Kind, Option<u64>)>,
    /// Highlights not yet used, with the wall time they arrived.
    pending: Vec<(f64, evolve_sim::snapshot::Highlight)>,
    /// Calls are many and minor, so only the latest few are kept, apart from `pending` so they
    /// never crowd out a kill or a birth.
    calls: VecDeque<(f64, evolve_sim::snapshot::Highlight)>,
    last_seq: u64,
    epoch: u64,
    /// Smoothed ground speed of each creature, to find steady walkers rather than tumblers.
    speeds: HashMap<u64, f32>,
    since_vista: u32,
    rng: u64,
}

impl Director {
    pub fn toggle(&mut self) {
        self.active = !self.active;
        self.shot = None;
        self.caption = None;
    }

    fn random(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Less interest in what was just shown.
    fn novelty(&self, kind: Kind, subject: Option<u64>) -> f32 {
        let mut f = 1.0;
        for (i, (k, s)) in self.recent.iter().rev().enumerate() {
            if *k == kind && i < 2 {
                f *= 0.35;
            }
            if subject.is_some() && *s == subject && i < 5 {
                f *= 0.25;
            }
        }
        f
    }
}

fn find(snap: &Snapshot, id: Option<u64>) -> Option<&CreatureView> {
    let id = id?;
    snap.creatures.iter().find(|c| c.id == id)
}

fn torso(c: &CreatureView) -> Vec3 {
    Vec3::from_array(c.parts[0].pos)
}

/// Rough radius of the whole body.
fn size(c: &CreatureView) -> f32 {
    let t = torso(c);
    c.parts.iter().map(|p| Vec3::from_array(p.pos).distance(t)).fold(0.0, f32::max) + 0.4
}

fn diet(c: &CreatureView) -> &'static str {
    match c.diet {
        d if d < 0.3 => "plant-eater",
        d if d > 0.7 => "meat-eater",
        _ => "omnivore",
    }
}

fn who(c: &CreatureView) -> String {
    format!("#{} ({}, generation {}, {} parts)", c.id, diet(c), c.generation, c.parts.len())
}

/// How long a highlight stays worth cutting to, seconds.
const FRESH: f64 = 6.0;

fn candidates(d: &mut Director, snap: &Snapshot, sky: Sky, now: f64) -> Vec<Candidate> {
    let mut out = Vec::new();
    let mut push = |d: &mut Director, kind, subject, other, place, score: f32, title: &str, detail: String| {
        let jitter = 0.8 + 0.4 * d.random();
        out.push(Candidate {
            kind,
            subject,
            other,
            place,
            score: score * d.novelty(kind, subject) * jitter,
            title: title.into(),
            detail,
        });
    };
    let pending = std::mem::take(&mut d.pending);
    let calls = std::mem::take(&mut d.calls);
    for &(_, h) in pending.iter().chain(&calls) {
        let Some(s) = find(snap, Some(h.subject)) else {
            continue;
        };
        let o = find(snap, h.other);
        let at = Vec3::from_array(h.pos);
        match h.kind {
            HighlightKind::Kill => {
                let detail = format!("{} killed #{}", who(s), h.other.unwrap_or(0));
                push(d, Kind::Kill, Some(s.id), None, at, 10.0, "A kill", detail);
            }
            HighlightKind::Attack if o.is_some() => {
                let detail = format!("{} bites #{}", who(s), h.other.unwrap_or(0));
                push(d, Kind::Fight, Some(s.id), h.other, at, 6.0, "A fight", detail);
            }
            HighlightKind::Birth => {
                let detail = format!("{} born to #{}", who(s), h.other.unwrap_or(0));
                push(d, Kind::Birth, Some(s.id), h.other.filter(|_| o.is_some()), at, 7.0, "A birth", detail);
            }
            HighlightKind::Record => {
                let detail = format!("{} is the first of generation {}", who(s), s.generation);
                push(d, Kind::Record, Some(s.id), None, at, 8.0, "A new generation", detail);
            }
            HighlightKind::Drowning => {
                let detail = format!("{} is out of breath", who(s));
                push(d, Kind::Drowning, Some(s.id), None, at, 3.0, "Trouble in the water", detail);
            }
            HighlightKind::Splash => {
                let detail = format!("{} hits the water", who(s));
                push(d, Kind::Splash, Some(s.id), None, at, 3.0, "Splash", detail);
            }
            HighlightKind::Call => {
                let detail = format!("{} calls out", who(s));
                push(d, Kind::Call, Some(s.id), None, at, 3.5, "A call", detail);
            }
            // Acts of god happen where the person at the controls is already looking.
            HighlightKind::Attack | HighlightKind::Impact | HighlightKind::Blessing | HighlightKind::Smite | HighlightKind::Ascension | HighlightKind::Meteor | HighlightKind::Landfall | HighlightKind::Manna => {}
        }
    }
    // Keep recent highlights around for a few seconds in case nothing better comes along.
    d.pending = pending.into_iter().filter(|(t, _)| now - t < FRESH).collect();
    d.calls = calls.into_iter().filter(|(t, _)| now - t < FRESH).collect();

    let best = |f: &dyn Fn(&CreatureView) -> Option<f32>| {
        snap.creatures
            .iter()
            .filter_map(|c| f(c).map(|v| (c, v)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    };
    let speeds = std::mem::take(&mut d.speeds);
    if let Some((c, v)) = best(&|c| (c.in_water > 0.5 && !c.drowning && c.speed > 0.35).then_some(c.speed)) {
        let detail = format!("{} swims at {v:.1} m/s", who(c));
        push(d, Kind::Swimmer, Some(c.id), None, torso(c), 4.0 + v, "A swimmer", detail);
    }
    if let Some((c, v)) = best(&|c| {
        let v = speeds.get(&c.id).copied().unwrap_or(0.0);
        (c.in_water < 0.2 && c.age > 5.0 && v > 0.6).then_some(v)
    }) {
        let detail = format!("{} keeps up {v:.1} m/s", who(c));
        push(d, Kind::Sprinter, Some(c.id), None, torso(c), 3.0 + v, "The fastest walker", detail);
    }
    d.speeds = speeds;
    if let Some((c, score)) = best(&|c| Some(c.score)) {
        let detail = format!("{}: scored {score:.1} so far eating, hunting, breeding and travelling", who(c));
        push(d, Kind::Star, Some(c.id), None, torso(c), 2.5 + score.min(10.0) * 0.2, "The island's best", detail);
    }
    if let Some((c, n)) = best(&|c| Some(c.neurons as f32)) {
        let detail = format!("{}: {n:.0} neurons", who(c));
        push(d, Kind::Brain, Some(c.id), None, torso(c), 2.0, "The biggest brain", detail);
    }
    if let Some((c, _)) = best(&|c| Some(c.parts.len() as f32 + c.id as f32 * 1e-9)) {
        let detail = who(c);
        push(d, Kind::Oddball, Some(c.id), None, torso(c), 2.5, "The strangest body", detail);
    }

    // The island itself, best at dawn and dusk and in bad weather.
    let sun = sky.sun.y;
    let (mut score, mut title) = (2.5 + 0.8 * d.since_vista as f32, "The island");
    if (-0.08..0.2).contains(&sun) {
        score += 5.0;
        title = if sky.time_of_day < 0.5 { "Sunrise" } else { "Sunset" };
    } else if sun < -0.08 {
        score += 1.0;
        title = "Night on the island";
    }
    if sky.storm > 0.4 {
        score += 5.0 * sky.storm;
        title = "A storm";
    } else if sky.rain > 0.5 {
        score += 2.0;
        title = "Rain";
    }
    let t = &snap.terrain;
    let center = Vec3::new(0.0, t.height_at(0.0, 0.0).max(snap.params.water_level), 0.0);
    let detail = format!(
        "day {}, {:02}:{:02}",
        snap.day,
        (sky.time_of_day * 24.0) as u32,
        ((sky.time_of_day * 24.0).fract() * 60.0) as u32
    );
    push(d, Kind::Vista, None, None, center, score, title, detail);
    out
}

fn start_shot(d: &mut Director, cam: &OrbitCam, snap: &Snapshot, sky: Sky, c: Candidate, now: f64) {
    let from = (cam.focus, cam.distance, cam.yaw, cam.pitch);
    let r1 = d.random();
    let r2 = d.random();
    let (focus, distance, yaw, pitch, yaw_rate) = match find(snap, c.subject) {
        Some(s) => {
            let mut focus = center_of_mass(s);
            let mut distance = (size(s) * 5.0 + 3.0).clamp(4.5, 22.0);
            if let Some(o) = find(snap, c.other) {
                distance = distance.max(focus.distance(center_of_mass(o)) * 1.6 + 4.0);
                focus = (focus + center_of_mass(o)) * 0.5;
            }
            // Side on to where it's heading, from either side.
            let rot = Quat::from_array(s.parts[0].rot);
            let fwd = rot * Vec3::X;
            let heading = fwd.z.atan2(fwd.x);
            let side = if r1 < 0.5 { 1.0 } else { -1.0 };
            let yaw = -heading + side * std::f32::consts::FRAC_PI_2 + (r2 - 0.5) * 0.8;
            (focus, distance, yaw, 0.28 + 0.2 * r2, (r1 - 0.5) * 0.15)
        }
        None => {
            let reach = snap.terrain.size * 0.5 * snap.params.island_size;
            // Face the sun when it's low, so the sky glows behind the island.
            let yaw = if sky.sun.y < 0.25 {
                (-sky.sun.x).atan2(-sky.sun.z)
            } else {
                r1 * std::f32::consts::TAU
            };
            let pitch = if sky.storm > 0.4 { 0.38 } else { 0.22 + 0.15 * r2 };
            (c.place, (reach * 1.9).clamp(60.0, 400.0), yaw, pitch, 0.025)
        }
    };
    // Turn the short way round, worked out once: unwrapping every frame instead would flip by a
    // full turn mid-glide whenever the slowly circling yaw crossed the half-turn mark.
    let yaw = from.2 + wrap(yaw - from.2);
    let jump = from.0.distance(focus);
    d.recent.push_back((c.kind, c.subject));
    if d.recent.len() > 8 {
        d.recent.pop_front();
    }
    d.since_vista = if c.kind == Kind::Vista { 0 } else { d.since_vista + 1 };
    let length = shot_length(d, c.kind);
    // A long shot keeps its caption only for a while, then has a clean screen.
    d.caption = Some(Caption {
        title: c.title,
        detail: c.detail,
        started: now,
        until: now + length.min(10.0),
    });
    d.shot = Some(Shot {
        kind: c.kind,
        subject: c.subject,
        other: c.other,
        place: c.place,
        started: now,
        yaw,
        yaw_rate,
        pitch,
        distance,
        from,
        glide: (1.6 + jump as f64 / 80.0).min(4.0),
        length,
        arc: (jump * 0.35).min(120.0),
        focus,
        focus_vel: Vec3::ZERO,
        spread: distance,
        lost: None,
    });
}

fn smoothstep(x: f64) -> f32 {
    let t = x.clamp(0.0, 1.0) as f32;
    t * t * (3.0 - 2.0 * t)
}

fn wrap(a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    (a + std::f32::consts::PI).rem_euclid(tau) - std::f32::consts::PI
}

#[allow(clippy::too_many_arguments)]
fn direct(
    mut d: ResMut<Director>,
    snap: Res<Snap>,
    time: Res<Time>,
    mut cam: ResMut<OrbitCam>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    egui: Res<EguiWantsInput>,
    env: Res<Env>,
    over: Res<VisualOverride>,
    tools: Res<ToolState>,
) {
    let d = &mut *d;
    let snap = &*snap.0;
    let now = time.elapsed_secs_f64();
    let dt = time.delta_secs();
    if d.rng == 0 {
        d.rng = 0x9E37_79B9_7F4A_7C15;
    }
    // Watch the world even when off, so it's ready the moment it's switched on.
    if snap.world_epoch != d.epoch {
        d.epoch = snap.world_epoch;
        d.last_seq = 0;
        d.pending.clear();
        d.calls.clear();
        d.speeds.clear();
        d.recent.clear();
        d.shot = None;
        d.caption = None;
    }
    for h in snap.highlights.iter() {
        if h.seq <= d.last_seq {
            continue;
        }
        d.last_seq = h.seq;
        match h.kind {
            // Bodies thud against the ground all the time; never worth a shot.
            HighlightKind::Impact => {}
            HighlightKind::Call => {
                d.calls.push_back((now, *h));
                if d.calls.len() > 8 {
                    d.calls.pop_front();
                }
            }
            _ => d.pending.push((now, *h)),
        }
    }
    // Rebuilt each frame from the creatures alive, which also forgets the dead.
    let k = 1.0 - (-dt / 3.0).exp();
    let old = std::mem::take(&mut d.speeds);
    d.speeds = snap
        .creatures
        .iter()
        .map(|c| {
            let v = old.get(&c.id).copied().unwrap_or(0.0);
            (c.id, v + (c.speed - v) * k)
        })
        .collect();
    if d.pending.len() > 64 {
        let n = d.pending.len() - 64;
        d.pending.drain(..n);
    }
    if !d.active {
        return;
    }

    // Any camera control hands it back, and so does carrying a creature.
    let pointer_free = !egui.wants_any_pointer_input();
    let dragging = (buttons.pressed(MouseButton::Right) || buttons.pressed(MouseButton::Middle))
        && motion.delta != Vec2::ZERO;
    let moving = !egui.wants_any_keyboard_input()
        && [KeyCode::KeyW, KeyCode::KeyA, KeyCode::KeyS, KeyCode::KeyD, KeyCode::KeyQ, KeyCode::KeyE, KeyCode::KeyF]
            .iter()
            .any(|k| keys.pressed(*k));
    if (pointer_free && (dragging || scroll.delta.y != 0.0)) || moving || tools.drag.is_some() {
        d.toggle();
        return;
    }

    // Is it time for something new? When the shot runs its course or its subject is gone, or
    // early if something big just happened somewhere else.
    let switch = match &mut d.shot {
        None => true,
        Some(shot) => {
            let held = now - shot.started;
            if shot.subject.is_some() && find(snap, shot.subject).is_none() {
                shot.lost.get_or_insert(now);
            }
            let subject = shot.subject;
            // A shot gets at least half its length (and 8 s) before news elsewhere can cut in.
            let patience = (0.5 * shot.length).max(if shot.kind.urgent() { 10.0 } else { 8.0 });
            let started = shot.started;
            // Only news since this shot began: what it already passed over stays passed over.
            // Births are too common in a big population to cut in for; they wait their turn.
            let big_news = held >= patience
                && d.pending.iter().any(|&(t, h)| {
                    t > started
                        && now - t < FRESH
                        && matches!(h.kind, HighlightKind::Kill | HighlightKind::Record)
                        && Some(h.subject) != subject
                        && find(snap, Some(h.subject)).is_some()
                });
            held >= shot.length || shot.lost.is_some_and(|t| now - t > 1.5) || big_news
        }
    };
    if switch {
        let sky = Sky {
            sun: env.sun_dir,
            rain: env.rain,
            storm: env.storm,
            time_of_day: over.time_of_day.unwrap_or(snap.weather.time_of_day),
        };
        let mut cands = candidates(d, snap, sky, now);
        if let Some(shot) = &d.shot {
            cands.retain(|c| !(c.kind == shot.kind && c.subject == shot.subject));
        }
        if let Some(best) = cands.into_iter().max_by(|a, b| a.score.total_cmp(&b.score)) {
            start_shot(d, &cam, snap, sky, best, now);
        }
    }

    let Some(shot) = &mut d.shot else {
        return;
    };
    let held = now - shot.started;
    // Where the shot wants the camera now.
    let mut target = shot.place;
    let mut distance = shot.distance;
    if let Some(s) = find(snap, shot.subject) {
        // The body's center of mass, not its torso, and followed by a spring: a creature
        // thrashing about would otherwise shake the camera with it.
        target = center_of_mass(s);
        if let Some(o) = find(snap, shot.other) {
            let other = center_of_mass(o);
            distance = distance.max(target.distance(other) * 1.6 + 4.0);
            target = (target + other) * 0.5;
        }
    } else if shot.subject.is_some() {
        target = shot.focus;
    }
    spring(&mut shot.focus, &mut shot.focus_vel, target, FOLLOW_RATE, dt);
    shot.spread += (distance - shot.spread) * (1.0 - (-1.5 * dt).exp());
    let distance = shot.spread;
    let yaw = shot.yaw + shot.yaw_rate * held as f32;
    let s = smoothstep(held / shot.glide);
    let (f0, d0, y0, p0) = shot.from;
    cam.follow = false;
    cam.focus = f0.lerp(shot.focus, s);
    cam.distance = d0 + (distance - d0) * s + shot.arc * (std::f32::consts::PI * s).sin();
    cam.yaw = y0 + (yaw - y0) * s;
    cam.pitch = p0 + (shot.pitch - p0) * s;
}
