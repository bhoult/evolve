//! A readable sketch of one creature's brain for the brain view: the senses and actions its body
//! actually uses, the hidden neurons that matter most to them, and the strongest connections
//! between all of these. A big brain has millions of connections; this keeps a few hundred.

use crate::brain::{N_IN, N_MEMORY, input, output};
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
    pub edges: Vec<Edge>,
    /// Neurons in the whole brain.
    pub total_hidden: usize,
}

pub const MAX_HIDDEN_SHOWN: usize = 40;
/// Strongest connections kept into each hidden neuron and into each output.
const FAN_IN: usize = 3;

fn strongest<T: Copy>(mut v: Vec<(f32, T)>, n: usize) -> Vec<(f32, T)> {
    v.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    v.truncate(n);
    v
}

pub fn sketch(c: &Creature) -> Wiring {
    let g = &c.genome.brain;
    let h = g.h;
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

    let w_in = |i: usize, k: usize| g.w_in.0[i * N_IN + k].to_f32();
    // What it acts on: inherited output weights plus what it has learned.
    let learned = c.brain.plastic.len() == g.w_out.0.len();
    let w_out = |k: usize, i: usize| {
        g.w_out.0[k * h + i].to_f32() + if learned { c.brain.plastic[k * h + i] } else { 0.0 }
    };
    let w_rec = |i: usize, j: usize| g.w_rec.0[i * h + j].to_f32();

    // The hidden neurons most strongly tied to what the body senses and does.
    let ranked: Vec<(f32, usize)> = (0..h)
        .map(|i| {
            let out: f32 = outputs.iter().map(|&k| w_out(k, i).abs()).sum();
            let inp: f32 = inputs.iter().map(|&k| w_in(i, k).abs()).sum();
            (out + 0.5 * inp, i)
        })
        .collect();
    let mut hidden: Vec<usize> = strongest(ranked, MAX_HIDDEN_SHOWN).into_iter().map(|(_, i)| i).collect();
    hidden.sort_unstable();

    let mut edges = Vec::new();
    for &i in &hidden {
        let into = inputs.iter().map(|&k| (w_in(i, k), k)).collect();
        for (w, k) in strongest(into, FAN_IN) {
            edges.push(Edge { from: Node::Input(k), to: Node::Hidden(i), weight: w });
        }
    }
    let mut rec = Vec::with_capacity(hidden.len() * hidden.len());
    for &i in &hidden {
        for &j in &hidden {
            rec.push((w_rec(i, j), (j, i)));
        }
    }
    for (w, (from, to)) in strongest(rec, hidden.len() * 2) {
        edges.push(Edge { from: Node::Hidden(from), to: Node::Hidden(to), weight: w });
    }
    for &k in &outputs {
        let into = hidden.iter().map(|&i| (w_out(k, i), i)).collect();
        for (w, i) in strongest(into, FAN_IN) {
            edges.push(Edge { from: Node::Hidden(i), to: Node::Output(k), weight: w });
        }
    }
    Wiring {
        creature: c.id,
        inputs,
        hidden,
        outputs,
        edges,
        total_hidden: h,
    }
}
