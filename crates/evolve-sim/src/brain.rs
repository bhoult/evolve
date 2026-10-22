//! Each creature's brain is a recurrent network: a hidden layer of genetic size whose neurons feed
//! back into each other every tick (so the brain keeps a state between ticks), plus a few memory
//! outputs fed back as inputs.
//!
//! Output weights also carry a lifetime-learned offset, learned by reward-modulated node
//! perturbation:
//! - The creature tries small random variations on its joint movements (exploration noise).
//! - Each output connection keeps an eligibility trace: a fading record of which variations it
//!   made, and with what hidden activity.
//! - When things go better than usual (a reward above the creature's running average), the
//!   variations in the trace are written into the learned offsets; when worse, the opposite.
//!
//! That credits what the creature did in the last few seconds, not just at the instant of reward.
//! A fraction of what parents learned is folded into their children's genes (see `Sim::breed`).
//!
//! Weights are stored as f16. The recurrent matrix grows with the square of the neuron count, so
//! a big brain is a lot of arithmetic: that work runs on the GPU when one is available
//! (`gpu.rs`), and this file holds the CPU reference implementation and the genetics.

use half::f16;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::genome::{MAX_LIMB_GENES, MAX_LIMB_SEGMENTS, MAX_SPINE};
use crate::params::SimParams;
use crate::rng::Rng;

/// Motor channels. Every joint axis has a fixed channel tied to its gene, so a mutation elsewhere
/// in the body never rewires it: two per spine joint, and two per segment of each limb gene and
/// of its mirror image. A hinge uses the first of its pair; a two-axis swivel uses both.
pub const SPINE_MOTORS: usize = MAX_SPINE * 2;
pub const N_MOTORS: usize = SPINE_MOTORS + MAX_LIMB_GENES * MAX_LIMB_SEGMENTS * 2 * 2;
pub const N_MEMORY: usize = 3;

pub fn spine_motor(joint: usize) -> usize {
    joint * 2
}

pub fn limb_motor(gene: usize, segment: usize, mirror: bool) -> usize {
    SPINE_MOTORS + ((gene * MAX_LIMB_SEGMENTS + segment) * 2 + mirror as usize) * 2
}

/// What a motor channel drives, e.g. "limb 2 seg 1 (mirror) axis b".
pub fn motor_label(k: usize) -> String {
    let axis = if k % 2 == 0 { "a" } else { "b" };
    if k < SPINE_MOTORS {
        return format!("spine {} axis {axis}", k / 2 + 1);
    }
    let m = (k - SPINE_MOTORS) / 2;
    let mirror = if m % 2 == 1 { " (mirror)" } else { "" };
    let seg = (m / 2) % MAX_LIMB_SEGMENTS;
    let gene = m / 2 / MAX_LIMB_SEGMENTS;
    format!("limb {} seg {}{mirror} axis {axis}", gene + 1, seg + 1)
}

/// A compact name for a motor channel, e.g. "limb 2.1 (m)" or "spine 3 b".
pub fn motor_short_label(k: usize) -> String {
    let axis = if k % 2 == 0 { "" } else { " b" };
    if k < SPINE_MOTORS {
        return format!("spine {}{axis}", k / 2 + 1);
    }
    let m = (k - SPINE_MOTORS) / 2;
    let mirror = if m % 2 == 1 { " (m)" } else { "" };
    let seg = (m / 2) % MAX_LIMB_SEGMENTS;
    let gene = m / 2 / MAX_LIMB_SEGMENTS;
    format!("limb {}.{}{mirror}{axis}", gene + 1, seg + 1)
}

pub mod input {
    use super::{N_MEMORY, N_MOTORS};
    pub const BIAS: usize = 0;
    pub const CPG_SIN: usize = 1;
    pub const CPG_COS: usize = 2;
    pub const ENERGY: usize = 3;
    pub const HEALTH: usize = 4;
    pub const OXYGEN: usize = 5;
    pub const AGE: usize = 6;
    pub const UPRIGHT: usize = 7;
    pub const PITCH: usize = 8;
    pub const SPEED_FWD: usize = 9;
    pub const SPEED_SIDE: usize = 10;
    pub const YAW_RATE: usize = 11;
    /// Forward, sideways, closeness.
    pub const PLANT: usize = 12;
    pub const MEAT: usize = 15;
    pub const OTHER: usize = 18;
    pub const OTHER_KIN: usize = 21;
    pub const OTHER_SIZE: usize = 22;
    pub const OTHER_AGGRO: usize = 23;
    /// Left, center, right closeness.
    pub const RAYS: usize = 24;
    pub const PAIN: usize = 27;
    pub const IN_WATER: usize = 28;
    /// Nearest compatible, mature creature: forward, sideways, closeness.
    pub const MATE: usize = 29;
    pub const HYDRATION: usize = 32;
    /// Nearest open water: forward, sideways, closeness.
    pub const WATER: usize = 33;
    /// The urge to mate, 0..1: builds while ready to breed, resets after mating.
    pub const MATE_DRIVE: usize = 36;
    /// The loudest call heard: forward, sideways, loudness.
    pub const HEAR: usize = 37;
    /// Its pitch, -1..1.
    pub const HEAR_PITCH: usize = 40;
    /// How much the caller's voice is like its own, 0..1 (kin sound alike).
    pub const HEAR_LIKE: usize = 41;
    /// How full its load of carried earth is, 0..1.
    pub const SOIL: usize = 42;
    pub const MEMORY: usize = 43;
    pub const JOINTS: usize = MEMORY + N_MEMORY;
    pub const COUNT: usize = JOINTS + N_MOTORS;
}

pub mod output {
    use super::{N_MEMORY, N_MOTORS};
    pub const JOINTS: usize = 0;
    pub const EAT: usize = N_MOTORS;
    pub const ATTACK: usize = N_MOTORS + 1;
    pub const MATE: usize = N_MOTORS + 2;
    /// Calling: silent below 0.2, louder above.
    pub const CALL: usize = N_MOTORS + 3;
    /// The call's pitch, -1..1 around the voice's natural pitch.
    pub const CALL_PITCH: usize = N_MOTORS + 4;
    /// Digging out the ground at its mouth (above 0.2), and piling carried earth there.
    pub const DIG: usize = N_MOTORS + 5;
    pub const PILE: usize = N_MOTORS + 6;
    pub const MEMORY: usize = N_MOTORS + 7;
    pub const COUNT: usize = MEMORY + N_MEMORY;
}

pub const N_IN: usize = input::COUNT;
pub const N_OUT: usize = output::COUNT;

pub fn input_label(i: usize) -> String {
    use input::*;
    let fixed = [
        "bias", "rhythm sin", "rhythm cos", "energy", "health", "oxygen", "age", "upright",
        "pitch", "speed fwd", "speed side", "turn rate", "plant fwd", "plant side", "plant near",
        "meat fwd", "meat side", "meat near", "other fwd", "other side", "other near", "other kin",
        "other size", "other aggro", "ray left", "ray mid", "ray right", "pain", "in water",
        "mate fwd", "mate side", "mate near", "hydration", "water fwd", "water side", "water near",
        "mating drive", "hear fwd", "hear side", "hear loud", "hear pitch", "hear kin", "soil carried",
    ];
    if i < fixed.len() {
        fixed[i].to_string()
    } else if i < JOINTS {
        format!("memory {}", i - MEMORY)
    } else {
        format!("{} angle", motor_label(i - JOINTS))
    }
}

pub fn output_label(i: usize) -> String {
    use output::*;
    match i {
        EAT => "eat".into(),
        ATTACK => "attack".into(),
        MATE => "mate".into(),
        CALL => "call".into(),
        CALL_PITCH => "call pitch".into(),
        DIG => "dig".into(),
        PILE => "pile".into(),
        _ if i >= MEMORY => format!("memory {}", i - MEMORY),
        _ => format!("{} target", motor_label(i)),
    }
}


/// How the brain's outputs are shifted by the built-in leanings in the genome.
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputBias {
    pub eat: f32,
    pub attack: f32,
    pub mate: f32,
}

impl OutputBias {
    pub fn apply(&self, k: usize) -> f32 {
        match k {
            output::EAT => self.eat,
            output::ATTACK => self.attack,
            output::MATE => self.mate,
            _ => 0.0,
        }
    }
}

/// Base learning rate (times the `learning_rate` param and the plasticity gene).
pub const LEARNING_RATE: f32 = 0.15;
/// Seconds over which the running-average reward (what counts as "usual") adapts.
pub const BASELINE_TIME: f32 = 10.0;
/// Recurrent weights start at this gain over sqrt(neurons): just under the edge of chaos.
const RECURRENT_GAIN: f32 = 0.9;
/// Input weights are scaled for about this many active inputs (most motor channels of a given
/// body are unused and read zero).
const TYPICAL_INPUTS: f32 = 48.0;

/// f16 weights, serialized as raw little-endian bytes (half the size of a list of numbers).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HalfVec(pub Vec<f16>);

impl Serialize for HalfVec {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(bytemuck::cast_slice(&self.0))
    }
}

impl<'de> Deserialize<'de> for HalfVec {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = HalfVec;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("f16 weights as bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, b: &[u8]) -> Result<HalfVec, E> {
                if b.len() % 2 != 0 {
                    return Err(E::custom("odd number of bytes for f16 weights"));
                }
                Ok(HalfVec(
                    b.chunks_exact(2)
                        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                        .collect(),
                ))
            }
            fn visit_byte_buf<E: serde::de::Error>(self, b: Vec<u8>) -> Result<HalfVec, E> {
                self.visit_bytes(&b)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<HalfVec, A::Error> {
                let mut bytes = Vec::new();
                while let Some(b) = seq.next_element::<u8>()? {
                    bytes.push(b);
                }
                self.visit_bytes(&bytes)
            }
        }
        d.deserialize_bytes(V)
    }
}

/// The inherited wiring of a brain with `h` hidden neurons. Matrices are row-major:
/// `w_in` is h x N_IN, `w_rec` is h x h (row i holds neuron i's inputs from every neuron), and
/// `w_out` is N_OUT x h.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainGenes {
    pub h: usize,
    pub w_in: HalfVec,
    pub w_rec: HalfVec,
    pub bias: Vec<f32>,
    pub w_out: HalfVec,
    pub b_out: Vec<f32>,
}

fn half_normal(rng: &mut Rng, sigma: f32) -> f16 {
    f16::from_f32(rng.normal() * sigma)
}

/// Draws how many items to skip before the next one hit with probability `p`.
fn geometric(rng: &mut Rng, p: f32) -> usize {
    if p >= 1.0 {
        return 0;
    }
    let u = rng.f32().max(1e-12);
    (u.ln() / (1.0 - p).ln()).floor().min(1e12) as usize
}

fn mutate_halves(rng: &mut Rng, v: &mut [f16], rate: f32, sigma: f32) {
    if rate <= 0.0 || v.is_empty() {
        return;
    }
    let mut i = geometric(rng, rate);
    while i < v.len() {
        v[i] = f16::from_f32(v[i].to_f32() + rng.normal() * sigma);
        i += 1 + geometric(rng, rate);
    }
}

impl BrainGenes {
    pub fn synapses(&self) -> usize {
        synapses(self.h)
    }

    pub fn random(rng: &mut Rng, h: usize) -> Self {
        let h = h.max(1);
        let sigma_in = 1.5 / TYPICAL_INPUTS.sqrt();
        let mut w_in = Vec::with_capacity(h * N_IN);
        for _ in 0..h {
            for k in 0..N_IN {
                // Strong rhythm inputs so even random brains wiggle their limbs.
                let s = if k == input::CPG_SIN || k == input::CPG_COS { 1.0 } else { sigma_in };
                w_in.push(half_normal(rng, s));
            }
        }
        let sigma_rec = RECURRENT_GAIN / (h as f32).sqrt();
        let w_rec = (0..h * h).map(|_| half_normal(rng, sigma_rec)).collect();
        let sigma_out = 1.5 / (h as f32).sqrt();
        Self {
            h,
            w_in: HalfVec(w_in),
            w_rec: HalfVec(w_rec),
            bias: (0..h).map(|_| rng.normal() * 0.2).collect(),
            w_out: HalfVec((0..N_OUT * h).map(|_| half_normal(rng, sigma_out)).collect()),
            b_out: (0..N_OUT).map(|_| rng.normal() * 0.3).collect(),
        }
    }

    /// A new brain made of `keep` (old neuron indices, in their new order) followed by `added`
    /// fresh neurons. Fresh neurons start nearly silent so they don't wreck a working brain.
    fn remap(&self, keep: &[usize], added: usize, rng: &mut Rng) -> Self {
        let nh = keep.len() + added;
        let mut w_in = Vec::with_capacity(nh * N_IN);
        let mut bias = Vec::with_capacity(nh);
        let sigma_in = 1.5 / TYPICAL_INPUTS.sqrt();
        for &o in keep {
            w_in.extend_from_slice(&self.w_in.0[o * N_IN..(o + 1) * N_IN]);
            bias.push(self.bias[o]);
        }
        for _ in 0..added {
            w_in.extend((0..N_IN).map(|_| half_normal(rng, sigma_in)));
            bias.push(rng.normal() * 0.2);
        }
        let sigma_new = 0.3 * RECURRENT_GAIN / (nh as f32).sqrt();
        let mut w_rec = Vec::with_capacity(nh * nh);
        for i in 0..nh {
            for j in 0..nh {
                let w = match (keep.get(i), keep.get(j)) {
                    (Some(&oi), Some(&oj)) => self.w_rec.0[oi * self.h + oj],
                    _ => half_normal(rng, sigma_new),
                };
                w_rec.push(w);
            }
        }
        let mut w_out = Vec::with_capacity(N_OUT * nh);
        for k in 0..N_OUT {
            for i in 0..nh {
                w_out.push(match keep.get(i) {
                    Some(&o) => self.w_out.0[k * self.h + o],
                    None => half_normal(rng, 0.05),
                });
            }
        }
        Self {
            h: nh,
            w_in: HalfVec(w_in),
            w_rec: HalfVec(w_rec),
            bias,
            w_out: HalfVec(w_out),
            b_out: self.b_out.clone(),
        }
    }

    /// Grow or shrink to `nh` neurons, keeping as much of the existing wiring as possible.
    pub fn resize(&self, nh: usize, rng: &mut Rng) -> Self {
        let nh = nh.max(1);
        if nh == self.h {
            return self.clone();
        }
        if nh > self.h {
            let keep: Vec<usize> = (0..self.h).collect();
            return self.remap(&keep, nh - self.h, rng);
        }
        // Drop random neurons.
        let mut keep: Vec<usize> = (0..self.h).collect();
        while keep.len() > nh {
            let i = rng.below(keep.len());
            keep.remove(i);
        }
        self.remap(&keep, 0, rng)
    }

    /// Moves a brain saved by an older version onto today's layout, so the creature behaves
    /// exactly as it did: new senses start unconnected, and new actions (calling) start silent.
    /// Returns false if the brain already has today's layout. Past layouts:
    /// - 47 inputs, 18 outputs: before multi-jointed bodies. Channels for joints it doesn't have
    ///   get small random weights for evolution to work with.
    /// - 121 inputs, 88 outputs: before the mating drive.
    /// - 122 inputs, 88 outputs: before calls and hearing.
    /// - 127 inputs, 90 outputs: before digging.
    pub fn upgrade_io(&mut self, rng: &mut Rng) -> bool {
        let h = self.h;
        let (n_in, n_out) = (self.w_in.0.len() / h.max(1), self.b_out.len());
        if (n_in, n_out) == (N_IN, N_OUT) {
            return false;
        }
        // Each old input and output index, mapped to today's.
        let genes = MAX_LIMB_GENES;
        let motor = |slot: usize| limb_motor(slot % genes, 0, slot >= genes);
        let (map_in, map_out, fresh_motors): (Box<dyn Fn(usize) -> usize>, Box<dyn Fn(usize) -> usize>, bool) =
            match (n_in, n_out) {
                (47, 18) => (
                    Box::new(move |i| match i {
                        0..32 => i,
                        32..35 => input::MEMORY + i - 32,
                        _ => input::JOINTS + motor(i - 35),
                    }),
                    Box::new(move |k| match k {
                        0..12 => output::JOINTS + motor(k),
                        12 => output::EAT,
                        13 => output::ATTACK,
                        14 => output::MATE,
                        _ => output::MEMORY + k - 15,
                    }),
                    true,
                ),
                (121 | 122, 88) => {
                    // The 122-input layout added the mating drive at 36.
                    let drive = n_in == 122;
                    let mem = if drive { 37 } else { 36 };
                    (
                        Box::new(move |i| match i {
                            0..36 => i,
                            36 if drive => input::MATE_DRIVE,
                            _ if i < mem + N_MEMORY => input::MEMORY + i - mem,
                            _ => input::JOINTS + i - mem - N_MEMORY,
                        }),
                        Box::new(|k| match k {
                            0..85 => k,
                            _ => output::MEMORY + k - 85,
                        }),
                        false,
                    )
                }
                (127, 90) => (
                    Box::new(|i| match i {
                        0..42 => i,
                        _ if i < 42 + N_MEMORY => input::MEMORY + i - 42,
                        _ => input::JOINTS + i - 42 - N_MEMORY,
                    }),
                    Box::new(|k| match k {
                        0..87 => k,
                        _ => output::MEMORY + k - 87,
                    }),
                    false,
                ),
                _ => return false,
            };
        let sigma_in = 1.5 / TYPICAL_INPUTS.sqrt();
        let sigma_out = 1.5 / (h as f32).sqrt();
        let mut w_in = vec![f16::ZERO; h * N_IN];
        let mut w_out = vec![f16::ZERO; N_OUT * h];
        let mut b_out = vec![0.0f32; N_OUT];
        if fresh_motors {
            for r in 0..h {
                for k in input::JOINTS..N_IN {
                    w_in[r * N_IN + k] = half_normal(rng, sigma_in);
                }
            }
            for k in 0..N_OUT {
                for i in 0..h {
                    w_out[k * h + i] = half_normal(rng, sigma_out);
                }
                b_out[k] = rng.normal() * 0.3;
            }
        }
        for r in 0..h {
            for i in 0..n_in {
                w_in[r * N_IN + map_in(i)] = self.w_in.0[r * n_in + i];
            }
        }
        for k in 0..n_out {
            let nk = map_out(k);
            w_out[nk * h..(nk + 1) * h].copy_from_slice(&self.w_out.0[k * h..(k + 1) * h]);
            b_out[nk] = self.b_out[k];
        }
        // Silent and idle until evolution finds a use for a voice or a spade.
        for k in [output::CALL, output::CALL_PITCH, output::DIG, output::PILE] {
            if (n_in, n_out) == (127, 90) && k < output::DIG {
                continue;
            }
            w_out[k * h..(k + 1) * h].fill(f16::ZERO);
            b_out[k] = if k == output::CALL_PITCH { 0.0 } else { -1.0 };
        }
        self.w_in = HalfVec(w_in);
        self.w_out = HalfVec(w_out);
        self.b_out = b_out;
        true
    }

    /// Writes a fraction of learned output-weight offsets (N_OUT x h) into the inherited weights:
    /// how a parent passes on part of what it learned.
    pub fn absorb(&mut self, plastic: &[f32], fraction: f32) {
        if fraction <= 0.0 || plastic.len() != self.w_out.0.len() {
            return;
        }
        for (w, p) in self.w_out.0.iter_mut().zip(plastic) {
            // Bounded, so many generations of lessons can't run away.
            *w = f16::from_f32((w.to_f32() + fraction * p).clamp(-4.0, 4.0));
        }
    }

    /// Neuron-by-neuron crossover: each hidden neuron keeps its input row, bias and output column
    /// from one parent.
    pub fn crossover(a: &BrainGenes, b: &BrainGenes, rng: &mut Rng) -> BrainGenes {
        let h = if rng.coin() { a.h } else { b.h };
        let src: Vec<&BrainGenes> = (0..h)
            .map(|i| match (i < a.h, i < b.h) {
                (true, true) => if rng.coin() { a } else { b },
                (true, false) => a,
                _ => b,
            })
            .collect();
        let mut w_in = Vec::with_capacity(h * N_IN);
        let mut bias = Vec::with_capacity(h);
        let mut w_rec = Vec::with_capacity(h * h);
        for (i, s) in src.iter().enumerate() {
            w_in.extend_from_slice(&s.w_in.0[i * N_IN..(i + 1) * N_IN]);
            bias.push(s.bias[i]);
            for j in 0..h {
                w_rec.push(if j < s.h { s.w_rec.0[i * s.h + j] } else { f16::ZERO });
            }
        }
        let mut w_out = Vec::with_capacity(N_OUT * h);
        for k in 0..N_OUT {
            for (i, s) in src.iter().enumerate() {
                w_out.push(s.w_out.0[k * s.h + i]);
            }
        }
        let b_out = (0..N_OUT)
            .map(|k| if rng.coin() { a.b_out[k] } else { b.b_out[k] })
            .collect();
        BrainGenes {
            h,
            w_in: HalfVec(w_in),
            w_rec: HalfVec(w_rec),
            bias,
            w_out: HalfVec(w_out),
            b_out,
        }
    }

    pub fn mutate(&mut self, rng: &mut Rng, p: &SimParams) {
        let s = p.mutation_strength;
        // Per-weight rate shrinks for big brains, so a birth changes roughly the same amount of
        // wiring rather than scrambling millions of weights.
        let rate = p.mutation_rate * (2000.0 / self.synapses().max(2000) as f32).sqrt();
        mutate_halves(rng, &mut self.w_in.0, rate, s);
        mutate_halves(rng, &mut self.w_rec.0, rate, s / (self.h as f32).sqrt().max(1.0));
        mutate_halves(rng, &mut self.w_out.0, rate, s);
        for b in self.bias.iter_mut().chain(self.b_out.iter_mut()) {
            if rng.chance(p.mutation_rate) {
                *b += rng.normal() * s;
            }
        }
        let (lo, hi) = p.neuron_range();
        if rng.chance(p.brain_resize_rate) {
            // Multiplicative steps, so small brains can grow and big ones can change in big jumps.
            let factor = (rng.normal() * 0.2 + p.brain_growth).exp();
            let mut nh = (self.h as f32 * factor).round() as usize;
            if nh == self.h {
                nh = if factor >= 1.0 { self.h + 1 } else { self.h.saturating_sub(1) };
            }
            let nh = nh.clamp(lo as usize, hi as usize);
            *self = self.resize(nh, rng);
        }
        let nh = self.h.clamp(lo as usize, hi as usize);
        if nh != self.h {
            *self = self.resize(nh, rng);
        }
    }
}

pub fn synapses(h: usize) -> usize {
    h * (N_IN + h + N_OUT)
}

/// What a brain remembers from tick to tick. In GPU mode the GPU holds the live copy and this one
/// is refreshed only when needed (saving, the inspector).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainState {
    pub memory: [f32; N_MEMORY],
    pub inputs: Vec<f32>,
    /// Hidden-neuron activations after the last tick.
    pub hidden: Vec<f32>,
    /// Outputs of the last tick.
    pub outputs: Vec<f32>,
    /// Lifetime-learned offsets on the output weights, N_OUT x h.
    pub plastic: Vec<f32>,
    /// Reward collected since the last brain tick (food eaten minus damage taken, normalized).
    pub reward: f32,
    /// Eligibility traces, N_OUT x h: recent exploration (the noise added to each output times
    /// the hidden activity behind it), fading, waiting for a reward to credit it.
    #[serde(default)]
    pub traces: Vec<f32>,
    /// Exploration noise on each output right now: the creature acts on outputs + noise.
    #[serde(default)]
    pub noise: Vec<f32>,
    /// Running average reward per tick: the "usual" that rewards are compared with.
    #[serde(default)]
    pub baseline: f32,
    /// How close the nearest food, water and mate were last tick, for the approach reward.
    #[serde(default)]
    pub closeness: [f32; 3],
}

impl BrainState {
    pub fn new(genes: &BrainGenes) -> Self {
        Self {
            memory: [0.0; N_MEMORY],
            inputs: vec![0.0; N_IN],
            hidden: vec![0.0; genes.h],
            outputs: vec![0.0; N_OUT],
            plastic: vec![0.0; N_OUT * genes.h],
            reward: 0.0,
            traces: vec![0.0; N_OUT * genes.h],
            noise: vec![0.0; N_OUT],
            baseline: 0.0,
            closeness: [0.0; 3],
        }
    }

    /// Fills in fields missing from states saved by older versions.
    pub fn fit(&mut self, h: usize) {
        if self.traces.len() != N_OUT * h {
            self.traces = vec![0.0; N_OUT * h];
        }
        if self.plastic.len() != N_OUT * h {
            self.plastic = vec![0.0; N_OUT * h];
        }
        if self.noise.len() != N_OUT {
            self.noise = vec![0.0; N_OUT];
        }
    }
}

/// One tick's learning settings.
#[derive(Clone, Copy, Debug)]
pub struct Learning {
    /// Learning rate times how much better (or worse) than usual things just went.
    pub lr: f32,
    /// Per-tick fade of the eligibility traces.
    pub trace_decay: f32,
    /// Per-tick fade of the learned offsets back toward the inherited weights.
    pub plastic_decay: f32,
}

impl Learning {
    pub const OFF: Learning = Learning {
        lr: 0.0,
        trace_decay: 0.0,
        plastic_decay: 1.0,
    };
}

/// The surprise in this tick's reward, and an update of the running average it's measured
/// against. `rate` is the creature's learning rate; `dt` the seconds per brain tick.
pub fn learning_signal(state: &mut BrainState, rate: f32, dt: f32) -> f32 {
    let r = state.reward.clamp(-1.0, 1.0);
    let surprise = r - state.baseline;
    state.baseline += surprise * (dt / BASELINE_TIME).min(1.0);
    rate * surprise.clamp(-1.0, 1.0)
}

/// Per-tick fade for something that should halve every `half_life` seconds.
pub fn per_tick_decay(half_life: f32, dt: f32) -> f32 {
    if half_life <= 0.0 { 0.0 } else { 0.5f32.powf(dt / half_life) }
}

/// One brain tick on the CPU. The GPU kernels in `gpu.rs` compute exactly this.
///
/// 1. Learning, from last tick: each trace fades and takes in the exploration noise that was
///    on that output times the hidden activity behind it; then the learned offsets move along
///    the traces by `lr` (rate times reward surprise) and fade toward the inherited weights.
/// 2. h' = tanh(W_in x + W_rec h + b)
/// 3. o = tanh((W_out + P) h' + b_out + leanings)
pub fn step_cpu(g: &BrainGenes, bias: OutputBias, st: &mut BrainState, x: &[f32; N_IN], learn: Learning) {
    let h = g.h;
    st.fit(h);
    for k in 0..N_OUT {
        let n = st.noise[k];
        let traces = &mut st.traces[k * h..(k + 1) * h];
        let row = &mut st.plastic[k * h..(k + 1) * h];
        for i in 0..h {
            let e = traces[i] * learn.trace_decay + n * st.hidden[i];
            traces[i] = e;
            row[i] = ((row[i] + learn.lr * e) * learn.plastic_decay).clamp(-2.0, 2.0);
        }
    }
    let mut next = vec![0.0f32; h];
    for (i, n) in next.iter_mut().enumerate() {
        let mut s = g.bias[i];
        for (w, xv) in g.w_in.0[i * N_IN..(i + 1) * N_IN].iter().zip(x) {
            s += w.to_f32() * xv;
        }
        for (w, hv) in g.w_rec.0[i * h..(i + 1) * h].iter().zip(&st.hidden) {
            s += w.to_f32() * hv;
        }
        *n = s.tanh();
    }
    for k in 0..N_OUT {
        let mut s = g.b_out[k] + bias.apply(k);
        let w = &g.w_out.0[k * h..(k + 1) * h];
        let p = &st.plastic[k * h..(k + 1) * h];
        for i in 0..h {
            s += (w[i].to_f32() + p[i]) * next[i];
        }
        st.outputs[k] = s.tanh();
    }
    st.hidden = next;
    st.inputs.copy_from_slice(x);
    st.reward = 0.0;
}
