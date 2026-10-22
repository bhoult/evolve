//! The genetic code: body plan, inherited traits and brain wiring.
//!
//! The body is a torso (the head end, with eyes and mouth) followed by a spine of up to
//! `MAX_SPINE` more segments, so it can be one block or a snake. Limb genes are a fixed array of
//! slots (each can be switched off), so crossover lines up slot by slot instead of splicing trees.
//! A limb grows from a body segment or from the tip of an earlier limb, has up to
//! `MAX_LIMB_SEGMENTS` segments (thigh, shin, foot), and can be mirrored, which gives a symmetric
//! pair driven by separate brain outputs.
//!
//! Joints are hinges (one axis, cheap) or swivels (two axes, bending any way like a tentacle,
//! but costing more energy per unit of work). Every joint's range of motion is genetic.
//!
//! The brain is a recurrent network (see `brain.rs`) whose neuron count is itself genetic:
//! creatures made from scratch start at `brain_min_neurons`, and mutation can grow brains up to
//! `brain_max_neurons`. Every connection costs energy to run.

use serde::{Deserialize, Serialize};

use crate::brain::BrainGenes;
use crate::params::SimParams;
use crate::rng::Rng;

pub const MAX_LIMB_GENES: usize = 6;
/// Body segments behind the torso.
pub const MAX_SPINE: usize = 5;
pub const MAX_LIMB_SEGMENTS: usize = 3;
/// Body parts beyond this many aren't grown, whatever the genes say.
pub const MAX_PARTS: usize = 24;

#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub lo: f32,
    pub hi: f32,
}

const fn r(lo: f32, hi: f32) -> Range {
    Range { lo, hi }
}

pub const TORSO_HALF: [Range; 3] = [r(0.12, 0.45), r(0.06, 0.2), r(0.08, 0.3)];
pub const LIMB_LENGTH: Range = r(0.15, 0.9);
pub const LIMB_RADIUS: Range = r(0.04, 0.14);
pub const LIMB_LIMIT: Range = r(0.2, 1.5);
pub const SPINE_LIMIT: Range = r(0.1, 0.9);
pub const TAPER: Range = r(0.5, 1.15);
pub const SPINE_LENGTH: Range = r(0.4, 1.3);
pub const LIMB_STRENGTH: Range = r(0.2, 2.0);
pub const UNIT: Range = r(-1.0, 1.0);
pub const ANGLE: Range = r(0.0, std::f32::consts::PI);

pub const SIZE: Range = r(0.6, 1.6);
pub const ZERO_ONE: Range = r(0.0, 1.0);
pub const CPG_FREQ: Range = r(0.2, 3.0);
pub const TRACE_TIME: Range = r(0.1, 4.0);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TorsoGene {
    /// Half extents: x is forward, y up, z sideways.
    pub half: [f32; 3],
}

/// Segments behind the torso, each joined to the one in front.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpineGene {
    /// Extra segments (0: the body is a single block).
    pub segments: u8,
    /// Each segment's thickness relative to the one in front.
    pub taper: f32,
    /// Each segment's length relative to the torso's.
    pub length: f32,
    /// Which way the hinges bend: 0 side to side like a snake, pi/2 up and down like a caterpillar.
    pub axis_angle: f32,
    /// Two-axis joints that bend any way (more flexible, more costly).
    pub swivel: bool,
    pub limit: f32,
    pub strength: f32,
}

impl Default for SpineGene {
    fn default() -> Self {
        Self {
            segments: 0,
            taper: 0.85,
            length: 0.8,
            axis_angle: 0.0,
            swivel: false,
            limit: 0.5,
            strength: 1.0,
        }
    }
}

fn one() -> u8 {
    1
}

fn default_taper() -> f32 {
    0.8
}

fn default_knee_limit() -> f32 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimbGene {
    pub present: bool,
    /// 0 attaches to the torso; k attaches to the tip of limb gene k-1 (must be an earlier gene).
    pub parent: u8,
    /// For torso limbs: which point of the torso surface it grows from.
    pub attach: [f32; 3],
    /// Tilts the limb away from straight out (torso) or straight on (chained limb).
    pub bend: [f32; 3],
    pub length: f32,
    pub radius: f32,
    /// Which way the hinge swings, as a rotation around the limb's own axis.
    pub axis_angle: f32,
    /// Hinge range, plus or minus this many radians.
    pub limit: f32,
    /// Motor strength multiplier.
    pub strength: f32,
    /// Grow a mirror-image copy on the other side of the body.
    pub mirror: bool,
    /// Segments in the limb: 1 a stick, 2 a leg with a knee, 3 with a foot too.
    #[serde(default = "one")]
    pub segments: u8,
    /// Each segment's length and thickness relative to the one before.
    #[serde(default = "default_taper")]
    pub taper: f32,
    /// Resting bend at each knee, -1..1 (times 0.8 rad). Knees bend mostly toward this side.
    #[serde(default)]
    pub curl: f32,
    /// Range of the knees, radians.
    #[serde(default = "default_knee_limit")]
    pub knee_limit: f32,
    /// Two-axis joint where the limb meets the body (a hip or shoulder rather than a hinge).
    #[serde(default)]
    pub swivel_root: bool,
    /// Two-axis knees too, which makes a tentacle.
    #[serde(default)]
    pub swivel_knees: bool,
    /// A flat paddle or foot at the tip, 0..1 (below 0.2 the tip stays round).
    #[serde(default)]
    pub paddle: f32,
    /// Which body segment a limb on the body grows from (0 is the torso).
    #[serde(default)]
    pub segment: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Traits {
    /// Overall body scale.
    pub size: f32,
    /// Body color hue; drifts slowly so lineages stay visibly related.
    pub hue: f32,
    /// 0 eats only plants, 1 eats only meat.
    pub diet: f32,
    /// Built-in leaning toward attacking (added to the brain's attack output).
    pub aggression: f32,
    /// Built-in leaning toward mating.
    pub libido: f32,
    /// Built-in leaning toward eating.
    pub appetite: f32,
    /// Frequency of the internal rhythm fed to the brain, Hz.
    pub cpg_freq: f32,
    /// Lung capacity; longer breath-holds cost a little metabolism.
    pub breath: f32,
    /// Damage reduction; costs metabolism.
    pub armor: f32,
    /// How fast the brain learns during its lifetime.
    pub plasticity: f32,
    /// How much it varies its movements to find better ones (0 repeats itself exactly). Learning
    /// needs some; too much spoils a good gait.
    #[serde(default = "default_explore")]
    pub explore: f32,
    /// How long, in seconds, a movement stays in mind to be credited when a reward comes (the
    /// half-life of its eligibility traces).
    #[serde(default = "default_trace_time")]
    pub trace_time: f32,
    /// How rewarding it feels to close in on food when hungry and on water when thirsty.
    #[serde(default = "default_seek")]
    pub seek: f32,
    /// The voice, each 0..1: natural pitch, timbre (how inharmonic and bright), rasp (how much
    /// noise) and warble (how fast it wavers). Kin sound alike. Negative means not yet given
    /// (a genome from before voices); `Genome::upgrade` fills it in.
    #[serde(default = "no_voice")]
    pub voice: [f32; 4],
}

fn no_voice() -> [f32; 4] {
    [-1.0; 4]
}

/// Creatures from before lifetime exploration explore mildly, so they behave almost as before.
fn default_explore() -> f32 {
    0.25
}

fn default_trace_time() -> f32 {
    1.0
}

fn default_seek() -> f32 {
    0.5
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub torso: TorsoGene,
    #[serde(default)]
    pub spine: SpineGene,
    pub limbs: Vec<LimbGene>,
    pub traits: Traits,
    pub brain: BrainGenes,
}

fn rand_in(rng: &mut Rng, rg: Range) -> f32 {
    rng.range(rg.lo, rg.hi)
}

fn mutate_f(rng: &mut Rng, v: &mut f32, rg: Range, rate: f32, strength: f32) {
    if rng.chance(rate) {
        *v = (*v + rng.normal() * strength * (rg.hi - rg.lo)).clamp(rg.lo, rg.hi);
    }
}

fn norm_diff(a: f32, b: f32, rg: Range) -> f32 {
    ((a - b) / (rg.hi - rg.lo)).abs()
}

impl LimbGene {
    fn random(rng: &mut Rng, index: usize) -> Self {
        let present = if index == 0 { rng.chance(0.95) } else { rng.chance(0.5) };
        let parent = if index == 0 || rng.chance(0.55) {
            0
        } else {
            rng.range_u32(1, index as u32) as u8
        };
        Self {
            present,
            parent,
            attach: [rng.signed(), rng.signed() * 0.6, rng.signed()],
            bend: [rng.signed() * 0.5, rng.signed() * 0.5 - 0.3, rng.signed() * 0.5],
            length: rand_in(rng, LIMB_LENGTH),
            radius: rand_in(rng, LIMB_RADIUS),
            axis_angle: rand_in(rng, ANGLE),
            limit: rand_in(rng, LIMB_LIMIT),
            strength: rng.range(0.6, 1.6),
            mirror: rng.chance(0.6),
            segments: [1, 1, 1, 2, 2, 3][rng.below(6)],
            taper: rng.range(0.6, 1.0),
            curl: rng.signed() * 0.8,
            knee_limit: rand_in(rng, LIMB_LIMIT),
            swivel_root: rng.chance(0.3),
            swivel_knees: rng.chance(0.15),
            paddle: if rng.chance(0.15) { rng.range(0.2, 1.0) } else { 0.0 },
            segment: if rng.chance(0.7) { 0 } else { rng.range_u32(0, MAX_SPINE as u32 + 1) as u8 },
        }
    }

    fn mutate(&mut self, rng: &mut Rng, index: usize, rate: f32, s: f32, q: f32) {
        if rng.chance(q) {
            self.present = !self.present;
        }
        if rng.chance(q) {
            self.mirror = !self.mirror;
        }
        if index > 0 && rng.chance(q * 0.5) {
            self.parent = rng.range_u32(0, index as u32) as u8;
        }
        if rng.chance(q) {
            self.segments = if rng.coin() { self.segments + 1 } else { self.segments.saturating_sub(1) }
                .clamp(1, MAX_LIMB_SEGMENTS as u8);
        }
        if rng.chance(q * 0.5) {
            self.swivel_root = !self.swivel_root;
        }
        if rng.chance(q * 0.5) {
            self.swivel_knees = !self.swivel_knees;
        }
        if rng.chance(q * 0.5) {
            self.segment = rng.range_u32(0, MAX_SPINE as u32 + 1) as u8;
        }
        for k in 0..3 {
            mutate_f(rng, &mut self.attach[k], UNIT, rate, s);
            mutate_f(rng, &mut self.bend[k], UNIT, rate, s);
        }
        mutate_f(rng, &mut self.length, LIMB_LENGTH, rate, s);
        mutate_f(rng, &mut self.radius, LIMB_RADIUS, rate, s);
        mutate_f(rng, &mut self.limit, LIMB_LIMIT, rate, s);
        mutate_f(rng, &mut self.knee_limit, LIMB_LIMIT, rate, s);
        mutate_f(rng, &mut self.strength, LIMB_STRENGTH, rate, s);
        mutate_f(rng, &mut self.taper, TAPER, rate, s);
        mutate_f(rng, &mut self.curl, UNIT, rate, s);
        mutate_f(rng, &mut self.paddle, ZERO_ONE, rate, s);
        if rng.chance(rate) {
            self.axis_angle = (self.axis_angle + rng.normal() * s * ANGLE.hi).rem_euclid(std::f32::consts::PI);
        }
    }

    /// 0 for identical limbs, up to 1.
    fn distance(&self, b: &LimbGene) -> f32 {
        let a = self;
        if a.present != b.present {
            return 1.0;
        }
        if !a.present {
            return 0.0;
        }
        let mut ld = norm_diff(a.length, b.length, LIMB_LENGTH)
            + norm_diff(a.radius, b.radius, LIMB_RADIUS)
            + norm_diff(a.limit, b.limit, LIMB_LIMIT)
            + norm_diff(a.strength, b.strength, LIMB_STRENGTH);
        for k in 0..3 {
            ld += norm_diff(a.attach[k], b.attach[k], UNIT);
        }
        ld /= 7.0;
        if a.parent != b.parent {
            ld += 0.5;
        }
        if a.mirror != b.mirror {
            ld += 0.3;
        }
        if a.segments != b.segments {
            ld += 0.3;
        }
        if a.swivel_root != b.swivel_root || a.swivel_knees != b.swivel_knees {
            ld += 0.2;
        }
        if (a.paddle >= 0.2) != (b.paddle >= 0.2) {
            ld += 0.2;
        }
        ld.min(1.0)
    }
}

impl SpineGene {
    fn random(rng: &mut Rng) -> Self {
        let segments = match rng.f32() {
            x if x < 0.5 => 0,
            x if x < 0.85 => rng.range_u32(1, 3) as u8,
            _ => rng.range_u32(3, MAX_SPINE as u32 + 1) as u8,
        };
        Self {
            segments,
            taper: rng.range(0.7, 1.05),
            length: rand_in(rng, SPINE_LENGTH),
            axis_angle: if rng.chance(0.6) { 0.0 } else { rand_in(rng, ANGLE) },
            swivel: rng.chance(0.25),
            limit: rand_in(rng, SPINE_LIMIT),
            strength: rng.range(0.6, 1.6),
        }
    }
}

impl Genome {
    pub fn random(rng: &mut Rng, p: &SimParams) -> Self {
        // Made from scratch: start near the smallest brain allowed and let evolution grow it.
        let (lo, hi) = p.neuron_range();
        let n_hidden = rng.range_u32(lo, (lo + lo / 4).min(hi)) as usize;
        Self {
            torso: TorsoGene {
                half: [
                    rand_in(rng, TORSO_HALF[0]),
                    rand_in(rng, TORSO_HALF[1]),
                    rand_in(rng, TORSO_HALF[2]),
                ],
            },
            spine: SpineGene::random(rng),
            limbs: (0..MAX_LIMB_GENES).map(|i| LimbGene::random(rng, i)).collect(),
            traits: Traits {
                size: rng.range(0.7, 1.3),
                hue: rng.f32(),
                diet: rng.f32().powi(3),
                aggression: rng.range(-1.0, 0.2),
                libido: rng.range(-0.2, 1.0),
                appetite: rng.range(0.0, 1.0),
                cpg_freq: rand_in(rng, CPG_FREQ),
                breath: rng.f32(),
                armor: rng.f32() * 0.5,
                plasticity: rng.f32(),
                explore: rng.f32(),
                trace_time: rand_in(rng, TRACE_TIME),
                seek: rng.f32(),
                voice: std::array::from_fn(|_| rng.f32()),
            },
            brain: {
                // Born idle with a spade: digging is costly, so evolution has to find it worth it.
                let mut b = BrainGenes::random(rng, n_hidden);
                b.b_out[crate::brain::output::DIG] = -1.0;
                b.b_out[crate::brain::output::PILE] = -1.0;
                b
            },
        }
    }

    pub fn crossover(a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
        let pick = |rng: &mut Rng, x: f32, y: f32| if rng.coin() { x } else { y };
        let torso = TorsoGene {
            half: [
                pick(rng, a.torso.half[0], b.torso.half[0]),
                pick(rng, a.torso.half[1], b.torso.half[1]),
                pick(rng, a.torso.half[2], b.torso.half[2]),
            ],
        };
        let limbs = (0..MAX_LIMB_GENES)
            .map(|i| if rng.coin() { a.limbs[i].clone() } else { b.limbs[i].clone() })
            .collect();
        let blend = |rng: &mut Rng, x: f32, y: f32| {
            let t = rng.f32();
            x + (y - x) * t
        };
        let (ta, tb) = (&a.traits, &b.traits);
        // Hue: take one parent's so lineage colors stay crisp instead of averaging to mud.
        let traits = Traits {
            size: blend(rng, ta.size, tb.size),
            hue: pick(rng, ta.hue, tb.hue),
            diet: blend(rng, ta.diet, tb.diet),
            aggression: blend(rng, ta.aggression, tb.aggression),
            libido: blend(rng, ta.libido, tb.libido),
            appetite: blend(rng, ta.appetite, tb.appetite),
            cpg_freq: pick(rng, ta.cpg_freq, tb.cpg_freq),
            breath: blend(rng, ta.breath, tb.breath),
            armor: blend(rng, ta.armor, tb.armor),
            plasticity: blend(rng, ta.plasticity, tb.plasticity),
            explore: blend(rng, ta.explore, tb.explore),
            trace_time: blend(rng, ta.trace_time, tb.trace_time),
            seek: blend(rng, ta.seek, tb.seek),
            voice: std::array::from_fn(|k| blend(rng, ta.voice[k], tb.voice[k])),
        };
        let brain = BrainGenes::crossover(&a.brain, &b.brain, rng);
        Genome {
            torso,
            spine: if rng.coin() { a.spine.clone() } else { b.spine.clone() },
            limbs,
            traits,
            brain,
        }
    }

    pub fn mutate(&mut self, rng: &mut Rng, p: &SimParams) {
        let rate = p.mutation_rate;
        let s = p.mutation_strength;
        let q = p.structural_mutation;
        for (k, rg) in TORSO_HALF.iter().enumerate() {
            mutate_f(rng, &mut self.torso.half[k], *rg, rate, s);
        }
        for (i, l) in self.limbs.iter_mut().enumerate() {
            l.mutate(rng, i, rate, s, q);
        }
        let sp = &mut self.spine;
        if rng.chance(q) {
            sp.segments = if rng.coin() { sp.segments + 1 } else { sp.segments.saturating_sub(1) }
                .min(MAX_SPINE as u8);
        }
        if rng.chance(q * 0.5) {
            sp.swivel = !sp.swivel;
        }
        mutate_f(rng, &mut sp.taper, TAPER, rate, s);
        mutate_f(rng, &mut sp.length, SPINE_LENGTH, rate, s);
        mutate_f(rng, &mut sp.limit, SPINE_LIMIT, rate, s);
        mutate_f(rng, &mut sp.strength, LIMB_STRENGTH, rate, s);
        if rng.chance(rate) {
            sp.axis_angle = (sp.axis_angle + rng.normal() * s * ANGLE.hi).rem_euclid(std::f32::consts::PI);
        }
        let t = &mut self.traits;
        mutate_f(rng, &mut t.size, SIZE, rate, s);
        t.hue = (t.hue + rng.normal() * 0.01).rem_euclid(1.0);
        mutate_f(rng, &mut t.diet, ZERO_ONE, rate, s);
        mutate_f(rng, &mut t.aggression, UNIT, rate, s);
        mutate_f(rng, &mut t.libido, UNIT, rate, s);
        mutate_f(rng, &mut t.appetite, UNIT, rate, s);
        mutate_f(rng, &mut t.cpg_freq, CPG_FREQ, rate, s);
        mutate_f(rng, &mut t.breath, ZERO_ONE, rate, s);
        mutate_f(rng, &mut t.armor, ZERO_ONE, rate, s);
        mutate_f(rng, &mut t.plasticity, ZERO_ONE, rate, s);
        mutate_f(rng, &mut t.explore, ZERO_ONE, rate, s);
        mutate_f(rng, &mut t.trace_time, TRACE_TIME, rate, s);
        mutate_f(rng, &mut t.seek, ZERO_ONE, rate, s);
        for v in &mut t.voice {
            mutate_f(rng, v, ZERO_ONE, rate, s);
        }

        self.brain.mutate(rng, p);
    }

    /// Rough genetic distance from body plan and traits (not brain weights), 0 for identical.
    pub fn distance(&self, other: &Genome) -> f32 {
        let mut d = 0.0;
        let mut n = 0.0;
        for k in 0..3 {
            d += norm_diff(self.torso.half[k], other.torso.half[k], TORSO_HALF[k]);
            n += 1.0;
        }
        for (a, b) in self.limbs.iter().zip(&other.limbs) {
            n += 1.0;
            d += a.distance(b);
        }
        let (sa, sb) = (&self.spine, &other.spine);
        n += 1.0;
        d += ((sa.segments as f32 - sb.segments as f32).abs() * 0.3
            + norm_diff(sa.length, sb.length, SPINE_LENGTH)
            + if sa.swivel != sb.swivel { 0.2 } else { 0.0 })
        .min(1.0);
        let (ta, tb) = (&self.traits, &other.traits);
        let hue = (ta.hue - tb.hue).abs();
        let hue = hue.min(1.0 - hue) * 2.0;
        d += norm_diff(ta.size, tb.size, SIZE)
            + norm_diff(ta.diet, tb.diet, ZERO_ONE)
            + norm_diff(ta.aggression, tb.aggression, UNIT)
            + norm_diff(ta.libido, tb.libido, UNIT)
            + norm_diff(ta.cpg_freq, tb.cpg_freq, CPG_FREQ)
            + norm_diff(ta.armor, tb.armor, ZERO_ONE)
            + hue;
        n += 7.0;
        d / n
    }

    /// A hash identifying this exact genome (cheap: samples a few hundred values).
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for v in self.torso.half {
            v.to_bits().hash(&mut h);
        }
        for l in &self.limbs {
            (l.present, l.parent, l.length.to_bits(), l.radius.to_bits(), l.axis_angle.to_bits()).hash(&mut h);
            (l.segments, l.swivel_root, l.swivel_knees, l.curl.to_bits(), l.paddle.to_bits(), l.segment).hash(&mut h);
        }
        let sp = &self.spine;
        (sp.segments, sp.swivel, sp.length.to_bits(), sp.axis_angle.to_bits()).hash(&mut h);
        let t = &self.traits;
        for v in [t.size, t.hue, t.diet, t.aggression, t.libido, t.appetite, t.cpg_freq] {
            v.to_bits().hash(&mut h);
        }
        let b = &self.brain;
        b.h.hash(&mut h);
        for v in b.b_out.iter().chain(b.bias.iter().take(64)) {
            v.to_bits().hash(&mut h);
        }
        let step = (b.w_rec.0.len() / 128).max(1);
        for w in b.w_rec.0.iter().step_by(step).chain(b.w_in.0.iter().take(64)) {
            w.to_bits().hash(&mut h);
        }
        // Output weights too: they carry what parents learned, which can be all that differs.
        let step = (b.w_out.0.len() / 256).max(1);
        for w in b.w_out.0.iter().step_by(step) {
            w.to_bits().hash(&mut h);
        }
        for v in [t.explore, t.trace_time, t.seek].into_iter().chain(t.voice) {
            v.to_bits().hash(&mut h);
        }
        h.finish()
    }

    pub fn limb_count(&self) -> usize {
        crate::body::body_plan(self).len() - 1
    }

    /// Brings a genome saved by an older version up to date. Returns true if anything changed.
    pub fn upgrade(&mut self) -> bool {
        let mut rng = Rng::new(self.fingerprint());
        let mut changed = self.brain.upgrade_io(&mut rng);
        if self.traits.voice.iter().any(|v| *v < 0.0) {
            // A voice of its own, the same every time this genome is upgraded.
            self.traits.voice = std::array::from_fn(|_| rng.f32());
            changed = true;
        }
        changed
    }

    pub fn color(&self) -> [f32; 3] {
        hsv_to_rgb(self.traits.hue, 0.65, 0.85)
    }
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = (h.rem_euclid(1.0)) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}
