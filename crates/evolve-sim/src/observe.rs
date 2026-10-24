//! The observatory: a running record of recent lives and births, and a plain-text report of
//! how the world is doing (who dies of what, where their energy goes, how far they travel,
//! whether lineages carry on by breeding or the sky keeps refilling the world).
//!
//! The record is not saved: it covers the last `KEEP` simulated seconds of whatever is running.

use std::collections::VecDeque;
use std::fmt::Write;
use std::sync::mpsc::Sender;

use crate::math::Vec3;
use crate::sim::{Creature, DeathCause, LifeStats, Origin, Sim, Spending};

/// Seconds of deaths and births the census remembers.
pub const KEEP: f64 = 30.0 * 60.0;

/// One finished life.
#[derive(Clone, Debug)]
pub struct Obituary {
    pub time: f64,
    pub rained: bool,
    pub origin: Origin,
    /// Births in a row behind it, back to an ancestor that fell from the sky (0 for one that fell).
    pub lineage: u32,
    pub cause: DeathCause,
    pub age: f32,
    pub mass: f32,
    pub parts: u32,
    pub neurons: u32,
    pub life: LifeStats,
    pub energy_left: f32,
    /// How far the nearest plant was when it died, m (infinite if none within `NEAR_PLANT`).
    pub nearest_plant: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct BirthNote {
    pub time: f64,
    pub lineage: u32,
    pub mass: f32,
}

/// Distance out to which the census looks for the plant nearest a dead creature, m.
pub const NEAR_PLANT: f32 = 25.0;

#[derive(Clone, Debug, Default)]
pub struct Census {
    pub deaths: VecDeque<Obituary>,
    pub births: VecDeque<BirthNote>,
    /// Creatures that fell from the sky: (time, energy they started with).
    pub sky: VecDeque<(f64, f32)>,
}

impl Census {
    pub fn forget_before(&mut self, t: f64) {
        while self.deaths.front().is_some_and(|d| d.time < t) {
            self.deaths.pop_front();
        }
        while self.births.front().is_some_and(|b| b.time < t) {
            self.births.pop_front();
        }
        while self.sky.front().is_some_and(|s| s.0 < t) {
            self.sky.pop_front();
        }
    }
}

impl Sim {
    /// Notes a death in the census (called as the body is removed).
    pub(crate) fn record_death(&mut self, c: &Creature, at: Vec3) {
        let r2 = NEAR_PLANT * NEAR_PLANT;
        let nearest_plant = self
            .plants
            .iter()
            .map(|f| (f.pos.x - at.x).powi(2) + (f.pos.z - at.z).powi(2))
            .filter(|&d2| d2 < r2)
            .fold(f32::INFINITY, f32::min)
            .sqrt();
        let o = Obituary {
            time: self.time,
            rained: c.rained,
            origin: c.origin,
            lineage: c.lineage,
            cause: c.dead.unwrap_or(DeathCause::Removed),
            age: c.age,
            mass: c.mass,
            parts: c.plan.len() as u32,
            neurons: c.genome.brain.h as u32,
            life: c.life.clone(),
            energy_left: c.energy,
            nearest_plant,
        };
        self.census.deaths.push_back(o);
        self.census.forget_before(self.time - KEEP);
    }

    pub(crate) fn record_birth(&mut self, lineage: u32, mass: f32) {
        self.census.births.push_back(BirthNote { time: self.time, lineage, mass });
        self.census.forget_before(self.time - KEEP);
    }

    pub(crate) fn record_sky(&mut self, energy: f32) {
        self.census.sky.push_back((self.time, energy));
        self.census.forget_before(self.time - KEEP);
    }

    /// One line on how the last `window` seconds went, for logs.
    pub fn summary_line(&self, window: f64) -> String {
        let from = self.time - window;
        let deaths: Vec<&Obituary> = self.census.deaths.iter().filter(|d| d.time >= from).collect();
        let births = self.census.births.iter().filter(|b| b.time >= from).count();
        let sky = self.census.sky.iter().filter(|s| s.0 >= from).count();
        let born_alive = self.creatures.iter().filter(|c| !c.rained).count();
        let mins = (window / 60.0).max(1e-6);
        let ages: Vec<f32> = deaths.iter().map(|d| d.age).collect();
        let ate: f32 = deaths.iter().map(|d| d.life.food()).sum();
        let lived: f32 = ages.iter().sum::<f32>().max(1e-6) / 60.0;
        let travelled: f32 = deaths.iter().map(|d| d.life.distance).sum();
        let deepest = self.creatures.iter().map(|c| c.lineage).max().unwrap_or(0);
        let mut masses: Vec<f32> = self.creatures.iter().map(|c| c.mass).collect();
        let mut neurons: Vec<f32> = self.creatures.iter().map(|c| c.genome.brain.h as f32).collect();
        format!(
            "t {:.0} min: {} alive ({} born here), births {:.1}/min, from the sky {:.1}/min, deaths {:.1}/min, \
             median age at death {:.0} s, eaten {:.1}/creature-min, travel {:.1} m/creature-min, \
             longest line of births {}, mass p50 {:.0} kg, neurons p50 {:.0}, plants {}",
            self.time / 60.0,
            self.creatures.len(),
            born_alive,
            births as f64 / mins,
            sky as f64 / mins,
            deaths.len() as f64 / mins,
            median(&mut ages.clone()),
            ate / lived,
            travelled / lived,
            deepest,
            median(&mut masses),
            median(&mut neurons),
            self.plants.len(),
        )
    }

    /// A report on the last `window` seconds and the world as it stands.
    pub fn report(&self, window: f64) -> String {
        let p = &self.params;
        let from = self.time - window;
        let mins = window / 60.0;
        let mut out = String::new();
        let o = &mut out;
        let born_alive = self.creatures.iter().filter(|c| !c.rained).count();
        let _ = writeln!(
            o,
            "World seed {} at {:.1} sim-minutes. {} alive (rain below {}, no births above {}): {} born here, {} from the sky.",
            p.seed,
            self.time / 60.0,
            self.creatures.len(),
            p.min_population,
            p.max_population,
            born_alive,
            self.creatures.len() - born_alive
        );
        let plant_energy: f32 = self.plants.iter().map(|f| f.energy).sum();
        let meat_energy: f32 = self.meat.iter().map(|f| f.energy).sum();
        let full = self.plants.iter().filter(|f| f.energy >= 0.95 * p.plant_max_energy).count();
        let _ = writeln!(
            o,
            "Food standing: {} plants (most {}), {} of them full grown, holding {:.0} energy; {} pieces of meat holding {:.0}.",
            self.plants.len(),
            p.plant_max_count,
            full,
            plant_energy,
            self.meat.len(),
            meat_energy
        );

        let deaths: Vec<&Obituary> = self.census.deaths.iter().filter(|d| d.time >= from).collect();
        let births: Vec<&BirthNote> = self.census.births.iter().filter(|b| b.time >= from).collect();
        let sky: Vec<&(f64, f32)> = self.census.sky.iter().filter(|s| s.0 >= from).collect();
        let covered = (self.time - self.census.deaths.front().map(|d| d.time).unwrap_or(self.time)).min(window);
        let _ = writeln!(
            o,
            "\nLast {:.0} sim-minutes{}: {} births ({:.1}/min), {} fell from the sky ({:.1}/min, bringing {:.0} energy/s), {} deaths.",
            mins,
            if covered + 1.0 < window { format!(" (record covers {:.1})", covered / 60.0) } else { String::new() },
            births.len(),
            births.len() as f64 / mins,
            sky.len(),
            sky.len() as f64 / mins,
            sky.iter().map(|s| s.1 as f64).sum::<f64>() / window.max(1e-6),
            deaths.len()
        );

        let groups = [
            ("From the sky, made at random", Origin::Random),
            ("From the sky, a mutated copy of a gene-pool genome", Origin::PoolClone),
            ("From the sky, a cross of two gene-pool genomes", Origin::PoolCross),
            ("Dropped in from the zoo or by hand", Origin::Dropped),
            ("Born in the world", Origin::Born),
            ("Of unknown origin (from an older save)", Origin::Unknown),
        ];
        for (label, origin) in groups {
            let d: Vec<&&Obituary> = deaths.iter().filter(|d| d.origin == origin).collect();
            if d.is_empty() && !matches!(origin, Origin::Born) {
                continue;
            }
            if d.is_empty() {
                let _ = writeln!(o, "\n{label}: no deaths in the window.");
                continue;
            }
            let n = d.len() as f32;
            let count = |c: DeathCause| d.iter().filter(|x| x.cause == c).count() as f32 / n * 100.0;
            let mut ages: Vec<f32> = d.iter().map(|x| x.age).collect();
            let lived = ages.iter().sum::<f32>().max(1e-6) / 60.0;
            let sum = |f: &dyn Fn(&LifeStats) -> f32| d.iter().map(|x| f(&x.life)).sum::<f32>();
            let spent = |f: &dyn Fn(&Spending) -> f32| sum(&|l: &LifeStats| f(&l.spent)) / lived;
            let mass = d.iter().map(|x| x.mass).sum::<f32>() / n;
            let _ = writeln!(
                o,
                "\n{label}: {} deaths, mean body {:.0} kg, median age at death {:.0} s (p90 {:.0} s).",
                d.len(),
                mass,
                median(&mut ages),
                quantile(&mut ages, 0.9)
            );
            let _ = writeln!(
                o,
                "  Causes: starved {:.0}%, killed {:.0}%, old age {:.0}%, drowned {:.0}%, thirst {:.0}%, other {:.0}%.",
                count(DeathCause::Starved),
                count(DeathCause::Killed),
                count(DeathCause::OldAge),
                count(DeathCause::Drowned),
                count(DeathCause::Dehydrated),
                count(DeathCause::Removed) + count(DeathCause::Glitch)
            );
            let _ = writeln!(
                o,
                "  Energy per creature-minute: ate {:.1} (plants {:.1}, meat {:.1}); spent basal {:.1}, brain {:.1}, muscle {:.1}, healing {:.1}, digging {:.1}, calls+attacks {:.1}; gave children {:.1}.",
                sum(&|l| l.food()) / lived,
                sum(&|l| l.plant_eaten) / lived,
                sum(&|l| l.meat_eaten) / lived,
                spent(&|s| s.basal),
                spent(&|s| s.brain),
                spent(&|s| s.muscle),
                spent(&|s| s.healing),
                spent(&|s| s.digging),
                spent(&|s| s.other),
                sum(&|l| l.invested) / lived
            );
            let dist = sum(&|l| l.distance);
            let muscle = sum(&|l| l.spent.muscle);
            let kg_m: f32 = d.iter().map(|x| x.mass * x.life.distance).sum::<f32>().max(1e-6);
            let _ = writeln!(
                o,
                "  Per life: {:.1} children, {:.2} kills, {:.0} m of net travel ({:.1} m/min); muscle energy per 100 kg per 100 m travelled: {:.2}.",
                sum(&|l| l.children as f32) / n,
                sum(&|l| l.kills as f32) / n,
                dist / n,
                dist / lived,
                muscle / kg_m * 1e4
            );
            let starved: Vec<&&&Obituary> = d.iter().filter(|x| x.cause == DeathCause::Starved).collect();
            if !starved.is_empty() {
                let ns = starved.len() as f32;
                let within = |r: f32| starved.iter().filter(|x| x.nearest_plant <= r).count() as f32 / ns * 100.0;
                let _ = writeln!(
                    o,
                    "  The starved: a plant within 2 m of {:.0}%, within 5 m {:.0}%, within 10 m {:.0}%, none within {:.0} m {:.0}%; {:.0}% never ate at all.",
                    within(2.0),
                    within(5.0),
                    within(10.0),
                    NEAR_PLANT,
                    100.0 - within(NEAR_PLANT),
                    starved.iter().filter(|x| x.life.food() < 0.5).count() as f32 / ns * 100.0
                );
            }
        }

        // The living.
        let alive = &self.creatures;
        if !alive.is_empty() {
            let v = |f: &dyn Fn(&Creature) -> f32| {
                let mut x: Vec<f32> = alive.iter().map(f).collect();
                format!("{:.0} / {:.0} / {:.0}", quantile(&mut x, 0.1), median(&mut x), quantile(&mut x, 0.9))
            };
            let _ = writeln!(o, "\nThe living (p10 / median / p90):");
            let _ = writeln!(o, "  body mass {} kg, parts {}, limbs {}", v(&|c| c.mass), v(&|c| c.plan.len() as f32), v(&|c| c.genome.limb_count() as f32));
            let _ = writeln!(o, "  age {} s, energy {} % of full", v(&|c| c.age), v(&|c| 100.0 * c.energy / c.max_energy.max(1e-6)));
            let _ = writeln!(o, "  brain neurons {}, generation {}", v(&|c| c.genome.brain.h as f32), v(&|c| c.generation as f32));
            let mut lines: Vec<u32> = alive.iter().map(|c| c.lineage).collect();
            lines.sort_unstable();
            let _ = writeln!(
                o,
                "  births in a row since the sky: median {}, longest {}; {} of the living have parents born here.",
                lines[lines.len() / 2],
                lines.last().copied().unwrap_or(0),
                alive.iter().filter(|c| c.lineage >= 2).count()
            );
        }
        let memory = match &self.brains {
            Some(g) => format!(
                "; brain memory {:.2} of this world's {:.2} GB (a brain that doesn't fit is trimmed)",
                g.memory_bytes() as f64 / 1e9,
                g.budget_bytes as f64 / 1e9
            ),
            None => String::new(),
        };
        let _ = writeln!(
            o,
            "\nBrains: price {:.2}, this world's brains keep the GPU {:.0}% busy (target for all worlds together {:.0}%){memory}; {}.",
            self.brain_price,
            self.perf.gpu_busy * 100.0,
            p.gpu_target * 100.0,
            self.brain_backend
        );

        // The best of the living.
        let mut best: Vec<&Creature> = alive.iter().collect();
        best.sort_by(|a, b| self.life_score(b).total_cmp(&self.life_score(a)));
        if !best.is_empty() {
            let _ = writeln!(o, "\nBest of the living by life score:");
            for c in best.iter().take(5) {
                let _ = writeln!(o, "  {}", self.describe(c));
            }
        }
        out
    }

    /// One line about a creature.
    pub fn describe(&self, c: &Creature) -> String {
        format!(
            "#{} score {:.1}: {} kg, {} parts, {} neurons, gen {}, {} births since the sky, age {:.0} s, energy {:.0}/{:.0}, ate {:.0} (meat {:.0}), {} children, {} kills, travelled {:.0} m",
            c.id,
            self.life_score(c),
            c.mass.round(),
            c.plan.len(),
            c.genome.brain.h,
            c.generation,
            c.lineage,
            c.age,
            c.energy,
            c.max_energy,
            c.life.food(),
            c.life.meat_eaten,
            c.life.children,
            c.life.kills,
            c.life.distance
        )
    }
}

fn median(v: &mut [f32]) -> f32 {
    quantile(v, 0.5)
}

fn quantile(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * q).round() as usize]
}

/// A creature followed for a few seconds, sampled ten times a second, to see how it moves.
pub struct Watch {
    pub id: u64,
    pub start: f64,
    pub until: f64,
    pub before: (f32, LifeStats),
    pub samples: Vec<WatchSample>,
    pub reply: Sender<Result<String, String>>,
}

#[derive(Clone, Debug)]
pub struct WatchSample {
    pub t: f32,
    pub pos: Vec3,
    pub fwd: Vec3,
    /// How upright the torso is: 1 level, 0 on its side, -1 upside down.
    pub up: f32,
    /// Torso center above the ground, m.
    pub height: f32,
    /// Lowest point of each limb tip above the ground, m.
    pub feet: Vec<f32>,
    /// Lowest point of the whole body above the ground, m.
    pub lowest: f32,
    pub nearest_plant: f32,
    pub eating: bool,
}

/// Ground clearance under which a part counts as touching, m.
const TOUCH: f32 = 0.05;

impl Sim {
    /// Follows creature `id` for `seconds` of simulated time; the description goes to `reply`
    /// when it's done (or the creature dies).
    pub fn watch(&mut self, id: u64, seconds: f32, reply: Sender<Result<String, String>>) {
        let Some(c) = self.creature(id) else {
            let _ = reply.send(Err(format!("no creature #{id} in this world")));
            return;
        };
        let before = (c.energy, c.life.clone());
        let start = self.time;
        self.watches.push(Watch {
            id,
            start,
            until: start + seconds.clamp(0.5, 120.0) as f64,
            before,
            samples: Vec::new(),
            reply,
        });
    }

    /// Takes a sample of every watched creature, and sends off the finished watches.
    pub(crate) fn sample_watches(&mut self) {
        let mut done = Vec::new();
        for w in 0..self.watches.len() {
            let id = self.watches[w].id;
            let sample = self.creature(id).and_then(|c| self.watch_sample(c, (self.time - self.watches[w].start) as f32));
            let alive = sample.is_some();
            if let Some(s) = sample {
                self.watches[w].samples.push(s);
            }
            if !alive || self.time >= self.watches[w].until {
                done.push(w);
            }
        }
        for w in done.into_iter().rev() {
            let watch = self.watches.swap_remove(w);
            let text = self.describe_watch(&watch);
            let _ = watch.reply.send(Ok(text));
        }
    }

    fn watch_sample(&self, c: &Creature, t: f32) -> Option<WatchSample> {
        let world = &self.shards[c.shard].world;
        let torso = world.bodies.get(c.parts[0].body)?;
        let pos = torso.translation();
        let rot = *torso.rotation();
        let ground = |p: Vec3| p.y - self.terrain.height_at(p.x, p.z);
        let mut feet = Vec::new();
        let mut lowest = f32::INFINITY;
        for (k, part) in c.parts.iter().enumerate() {
            let Some(b) = world.bodies.get(part.body) else {
                continue;
            };
            let (bp, br) = (b.translation(), *b.rotation());
            let low = match c.plan[k].shape {
                crate::body::PartShape::Capsule { half_height, radius } => {
                    let a = bp + br * Vec3::Y * half_height;
                    let z = bp - br * Vec3::Y * half_height;
                    ground(a).min(ground(z)) - radius
                }
                crate::body::PartShape::Box { half } => {
                    let mut m = f32::INFINITY;
                    for sx in [-1.0, 1.0] {
                        for sy in [-1.0, 1.0] {
                            for sz in [-1.0, 1.0] {
                                m = m.min(ground(bp + br * Vec3::new(sx * half[0], sy * half[1], sz * half[2])));
                            }
                        }
                    }
                    m
                }
            };
            lowest = lowest.min(low);
            let leaf = !c.plan[k].trunk && !c.plan.iter().any(|s| s.parent == Some(k));
            if leaf {
                feet.push(low);
            }
        }
        let nearest_plant = self
            .plants
            .iter()
            .map(|f| (f.pos.x - pos.x).powi(2) + (f.pos.z - pos.z).powi(2))
            .fold(f32::INFINITY, f32::min)
            .sqrt();
        Some(WatchSample {
            t,
            pos,
            fwd: crate::math::heading_frame(rot).0,
            up: (rot * Vec3::Y).y,
            height: ground(pos),
            feet,
            lowest,
            nearest_plant,
            eating: c.output(crate::brain::output::EAT) > 0.0,
        })
    }

    fn describe_watch(&self, w: &Watch) -> String {
        let mut out = String::new();
        let o = &mut out;
        let s = &w.samples;
        let now = self.creature(w.id);
        let span = s.last().map(|x| x.t).unwrap_or(0.0);
        let _ = writeln!(
            o,
            "Watched #{} for {:.1} s{}.",
            w.id,
            span,
            if now.is_none() { " (it died or left the world before the end)" } else { "" }
        );
        if let Some(c) = now {
            let _ = writeln!(o, "{}", self.describe(c));
            let _ = writeln!(o, "Body: {}", body_summary(c));
        }
        if s.len() < 2 {
            return out;
        }
        let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
        let net = flat(s[s.len() - 1].pos - s[0].pos).length();
        let path: f32 = s.windows(2).map(|p| flat(p[1].pos - p[0].pos).length()).sum();
        let turned: f32 = s
            .windows(2)
            .map(|p| p[0].fwd.dot(p[1].fwd).clamp(-1.0, 1.0).acos())
            .sum::<f32>()
            .to_degrees();
        // Which way it moves relative to where it faces.
        let along: f32 = s.windows(2).map(|p| flat(p[1].pos - p[0].pos).dot(p[0].fwd)).sum();
        let _ = writeln!(
            o,
            "Moved: net {:.1} m, path {:.1} m (straightness {:.2}), {:.2} m/s along its path; {:.0}% of the way was forward (toward its head); turned {:.0} degrees in all.",
            net,
            path,
            net / path.max(1e-6),
            path / span.max(1e-6),
            100.0 * along / path.max(1e-6),
            turned
        );
        let n = s.len() as f32;
        let mean = |f: &dyn Fn(&WatchSample) -> f32| s.iter().map(f).sum::<f32>() / n;
        let _ = writeln!(
            o,
            "Torso: upright (up > 0.7) {:.0}% of the time, mean up {:.2}; center {:.2} m above the ground on average (min {:.2}, max {:.2}); whole body off the ground {:.0}% of the time.",
            100.0 * mean(&|x| (x.up > 0.7) as u8 as f32),
            mean(&|x| x.up),
            mean(&|x| x.height),
            s.iter().map(|x| x.height).fold(f32::INFINITY, f32::min),
            s.iter().map(|x| x.height).fold(f32::NEG_INFINITY, f32::max),
            100.0 * mean(&|x| (x.lowest > TOUCH) as u8 as f32)
        );
        let nf = s.iter().map(|x| x.feet.len()).min().unwrap_or(0);
        if nf > 0 {
            let touch: Vec<String> = (0..nf)
                .map(|f| format!("{:.0}%", 100.0 * mean(&|x| (x.feet[f] <= TOUCH) as u8 as f32)))
                .collect();
            // How often the feet change between touching and lifted: a gait shows as steady
            // steps, flailing as none or a blur.
            let steps: Vec<f32> = (0..nf)
                .map(|f| s.windows(2).filter(|p| (p[0].feet[f] <= TOUCH) != (p[1].feet[f] <= TOUCH)).count() as f32 / 2.0 / span.max(1e-6))
                .collect();
            let _ = writeln!(
                o,
                "Limb tips ({nf}): on the ground {} of the time; steps per second {}.",
                touch.join(", "),
                steps.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(", ")
            );
        } else {
            let _ = writeln!(o, "No limb tips: it moves, if at all, by its trunk alone.");
        }
        let closer = s.windows(2).filter(|p| p[1].nearest_plant < p[0].nearest_plant - 0.01).count() as f32;
        let _ = writeln!(
            o,
            "Food: nearest plant {:.1} m at the start, {:.1} m at the end; getting closer to the nearest plant in {:.0}% of samples; mouth open to eat {:.0}% of the time.",
            s[0].nearest_plant,
            s[s.len() - 1].nearest_plant,
            100.0 * closer / (n - 1.0),
            100.0 * mean(&|x| x.eating as u8 as f32)
        );
        if let Some(c) = now {
            let (e0, l0) = &w.before;
            let d = |a: f32, b: f32| a - b;
            let sp = &c.life.spent;
            let _ = writeln!(
                o,
                "Energy: {:.1} -> {:.1}; ate {:.1}; spent basal {:.1}, brain {:.1}, muscle {:.1}, healing {:.1}, other {:.1}.",
                e0,
                c.energy,
                d(c.life.food(), l0.food()),
                d(sp.basal, l0.spent.basal),
                d(sp.brain, l0.spent.brain),
                d(sp.muscle, l0.spent.muscle),
                d(sp.healing, l0.spent.healing),
                d(sp.other + sp.digging, l0.spent.other + l0.spent.digging)
            );
        }
        let _ = write!(o, "Track, each second (x, z; heading; up; height; nearest plant):");
        let mut next = 0.0;
        for x in s {
            if x.t + 1e-3 >= next {
                let _ = write!(
                    o,
                    "\n  {:4.1} s: ({:6.1}, {:6.1}) heading {:4.0} up {:5.2} h {:4.2} plant {:4.1} m{}",
                    x.t,
                    x.pos.x,
                    x.pos.z,
                    x.fwd.z.atan2(x.fwd.x).to_degrees(),
                    x.up,
                    x.height,
                    x.nearest_plant,
                    if x.eating { " eating" } else { "" }
                );
                next += 1.0;
            }
        }
        out.push('\n');
        out
    }
}

/// What a body is made of, in a line.
pub fn body_summary(c: &Creature) -> String {
    let trunk = c.plan.iter().filter(|s| s.trunk).count();
    let limbs = c.plan.iter().filter(|s| !s.trunk && s.parent.is_some_and(|p| c.plan[p].trunk)).count();
    let segments = c.plan.iter().filter(|s| !s.trunk).count();
    let torso = c.torso_half() * 2.0;
    let swivels = c
        .plan
        .iter()
        .filter_map(|s| s.joint.as_ref())
        .filter(|j| matches!(j.kind, crate::body::JointKind::Swivel))
        .count();
    format!(
        "torso {:.2} x {:.2} x {:.2} m, {} trunk segments, {} limbs of {} segments in all, {} joints ({} two-axis), CPG {:.2} Hz",
        torso.x,
        torso.y,
        torso.z,
        trunk,
        limbs,
        segments,
        c.joints.len(),
        swivels,
        c.genome.traits.cpg_freq
    )
}
