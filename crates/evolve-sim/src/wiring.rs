//! A readable sketch of one creature's brain for the brain view: the senses and actions its body
//! actually uses, the hidden neurons that matter most to them, and the strongest connections
//! between all of these. A big brain has millions of connections; the brain view asks for as many
//! neurons and connections as it wants to draw (`SketchLimits`), from a few hundred connections
//! for a diagram to every neuron and a hundred thousand connections for the 3D picture.

use half::f16;
use rayon::prelude::*;

use crate::brain::{BrainGenes, N_IN, N_MEMORY, input, output};
use crate::sim::Creature;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    Input(usize),
    Hidden(usize),
    Output(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: Node,
    pub to: Node,
    pub weight: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Wiring {
    pub creature: u64,
    /// Input, hidden and output indices shown, in order.
    pub inputs: Vec<usize>,
    pub hidden: Vec<usize>,
    pub outputs: Vec<usize>,
    /// Senses into neurons, then neurons into neurons, then neurons into actions; each run
    /// strongest first.
    pub edges: Vec<Edge>,
    /// Neurons in the whole brain.
    pub total_hidden: usize,
}

/// How much of a brain to sketch: up to `neurons` of the hidden neurons that matter most, and
/// up to `connections` of the strongest connections among them and the senses and actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SketchLimits {
    pub neurons: usize,
    pub connections: usize,
}

impl Default for SketchLimits {
    /// A diagram that stays readable.
    fn default() -> Self {
        Self { neurons: 40, connections: 320 }
    }
}

/// The senses and actions a creature's body uses: the general senses and its own joints'
/// readings in, its joints' motors and the general actions out.
fn senses_and_actions(c: &Creature) -> (Vec<usize>, Vec<usize>) {
    let mut inputs: Vec<usize> = (0..input::JOINTS).collect();
    let mut outputs = Vec::new();
    for j in &c.joints {
        for d in 0..j.kind.axes() {
            inputs.push(input::JOINTS + j.slot + d);
            outputs.push(output::JOINTS + j.slot + d);
        }
    }
    inputs.sort_unstable();
    outputs.sort_unstable();
    outputs.extend([output::EAT, output::ATTACK, output::MATE, output::CALL, output::CALL_PITCH, output::DIG, output::PILE]);
    outputs.extend(output::MEMORY..output::MEMORY + N_MEMORY);
    (inputs, outputs)
}

pub fn sketch(c: &Creature, limits: SketchLimits) -> Wiring {
    let (inputs, outputs) = senses_and_actions(c);
    // What it acts on: inherited output weights plus what it has learned.
    let learned = (c.brain.plastic.len() == c.genome.brain.w_out.0.len()).then_some(&c.brain.plastic[..]);
    Wiring { creature: c.id, ..sketch_brain(&c.genome.brain, learned, inputs, outputs, limits) }
}

/// Size as an f16 bit pattern, which orders the same as the size does. Weights are stored as
/// f16, so for them this is exact; it lets the strongest of 17 million be found by counting
/// rather than sorting.
fn size_key(w: f16) -> u16 {
    w.to_bits() & 0x7FFF
}

/// Keys up to infinity; a NaN (never a real weight) falls outside and is never picked.
const KEYS: usize = 0x7C01;

/// Which of a `rows` x `cols` grid of candidates have the `k` largest keys (see `size_key`), as
/// (row, column) in grid order. Zero weights are never picked, and ties at the cut go to the
/// earliest, so the answer depends only on the weights. Two passes on every core: one counts how
/// many candidates have each key, which fixes the smallest key that makes the cut, and the
/// other collects those that make it.
fn top_k(rows: usize, cols: usize, k: usize, key: impl Fn(usize, usize) -> u16 + Sync) -> Vec<(usize, usize)> {
    if k == 0 || rows == 0 || cols == 0 {
        return Vec::new();
    }
    let counts = (0..rows)
        .into_par_iter()
        .fold(
            || vec![0u32; KEYS],
            |mut n, r| {
                for c in 0..cols {
                    if let Some(x) = n.get_mut(key(r, c) as usize) {
                        *x += 1;
                    }
                }
                n
            },
        )
        .reduce(
            || vec![0u32; KEYS],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    // The cut: the smallest key with at most k candidates above it and k or more at or above.
    let mut above = 0;
    let mut cut = 1;
    for b in (1..KEYS).rev() {
        if above + counts[b] as usize >= k {
            cut = b;
            break;
        }
        above += counts[b] as usize;
    }
    let mut at_cut = k - above.min(k);
    let candidates: Vec<(usize, usize, usize)> = (0..rows)
        .into_par_iter()
        .flat_map_iter(|r| {
            let key = &key;
            (0..cols).filter_map(move |c| {
                let b = key(r, c) as usize;
                (cut..KEYS).contains(&b).then_some((r, c, b))
            })
        })
        .collect();
    let mut picked = Vec::with_capacity(k);
    for (r, c, b) in candidates {
        if b > cut {
            picked.push((r, c));
        } else if at_cut > 0 {
            at_cut -= 1;
            picked.push((r, c));
        }
    }
    picked
}

/// Strongest first; equals keep their order.
fn strongest_first(edges: &mut [Edge]) {
    edges.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
}

/// The sketch of a brain given the senses and actions to show and what it has learned (output
/// weight offsets, if any), within `limits`.
///
/// The hidden neurons shown are those most strongly tied to the senses and actions. The
/// connection budget is shared between the three kinds of connection in proportion to how many
/// of each there are among what's shown, and each kind gets its strongest: sizes can't be
/// compared across kinds, since a big brain's neuron-to-neuron weights are an order of magnitude
/// smaller than its sense weights (they're scaled by how many there are), and the strongest
/// overall would be nothing but senses.
pub fn sketch_brain(g: &BrainGenes, learned: Option<&[f32]>, inputs: Vec<usize>, outputs: Vec<usize>, limits: SketchLimits) -> Wiring {
    let h = g.h;
    let learned = learned.filter(|l| l.len() == g.w_out.0.len());
    let w_out = |k: usize, i: usize| g.w_out.0[k * h + i].to_f32() + learned.map_or(0.0, |l| l[k * h + i]);

    let n = limits.neurons.min(h);
    let hidden: Vec<usize> = if n == h {
        (0..h).collect()
    } else {
        // The hidden neurons most strongly tied to what the body senses and does.
        let mut ranked: Vec<(f32, usize)> = (0..h)
            .into_par_iter()
            .map(|i| {
                let out: f32 = outputs.iter().map(|&k| w_out(k, i).abs()).sum();
                let inp: f32 = inputs.iter().map(|&k| g.w_in.0[i * N_IN + k].to_f32().abs()).sum();
                (out + 0.5 * inp, i)
            })
            .collect();
        let order = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        if n > 0 {
            ranked.select_nth_unstable_by(n - 1, order);
        }
        let mut hidden: Vec<usize> = ranked[..n].iter().map(|&(_, i)| i).collect();
        hidden.sort_unstable();
        hidden
    };

    let n_in = hidden.len() * inputs.len();
    let n_rec = hidden.len() * hidden.len();
    let n_out = outputs.len() * hidden.len();
    let total = (n_in + n_rec + n_out).max(1) as u128;
    let k = limits.connections.min(n_in + n_rec + n_out);
    let mut k_in = (k as u128 * n_in as u128 / total) as usize;
    let mut k_out = (k as u128 * n_out as u128 / total) as usize;
    // Rounding the others down can leave neuron-to-neuron a connection or two more than there
    // are; those go to the others.
    let spare = (k - k_in - k_out).saturating_sub(n_rec);
    let to_in = spare.min(n_in - k_in);
    k_in += to_in;
    k_out += spare - to_in;
    let k_rec = k - k_in - k_out;

    let mut edges = Vec::with_capacity(k);
    let picked = top_k(hidden.len(), inputs.len(), k_in, |r, c| size_key(g.w_in.0[hidden[r] * N_IN + inputs[c]]));
    let start = edges.len();
    for (r, c) in picked {
        let (i, s) = (hidden[r], inputs[c]);
        edges.push(Edge { from: Node::Input(s), to: Node::Hidden(i), weight: g.w_in.0[i * N_IN + s].to_f32() });
    }
    strongest_first(&mut edges[start..]);
    // Row i of the recurrent weights is what neuron i hears from every neuron.
    let picked = top_k(hidden.len(), hidden.len(), k_rec, |r, c| size_key(g.w_rec.0[hidden[r] * h + hidden[c]]));
    let start = edges.len();
    for (r, c) in picked {
        let (to, from) = (hidden[r], hidden[c]);
        edges.push(Edge { from: Node::Hidden(from), to: Node::Hidden(to), weight: g.w_rec.0[to * h + from].to_f32() });
    }
    strongest_first(&mut edges[start..]);
    // Learned offsets make these f32; their keys round them to f16, so near-equals at the cut
    // may come in either order.
    let picked = top_k(outputs.len(), hidden.len(), k_out, |r, c| size_key(f16::from_f32(w_out(outputs[r], hidden[c]))));
    let start = edges.len();
    for (r, c) in picked {
        let (o, i) = (outputs[r], hidden[c]);
        edges.push(Edge { from: Node::Hidden(i), to: Node::Output(o), weight: w_out(o, i) });
    }
    strongest_first(&mut edges[start..]);

    Wiring {
        creature: 0,
        inputs,
        hidden,
        outputs,
        edges,
        total_hidden: h,
    }
}
