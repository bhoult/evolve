//! The neural view: a creature's brain drawn as living tissue rather than a diagram. Sensory
//! neurons line the left edge, each with a receptor ending; motor neurons line the right, each
//! sending its axon to a motor end plate; the hidden neurons that matter most sit between, with
//! branching dendrites. Axons run to synapses on the dendrites of the neurons they feed.
//!
//! It shows the same sketch as the network view (`Wiring`) and the same live activity. Each
//! neuron fires at a rate set by its activity; an action potential runs down its axon to every
//! synapse, which lights up warm if it excites and cool if it inhibits, and the postsynaptic
//! potential runs up the dendrite to the next soma. A motor neuron's spike ends at its end plate.
//!
//! The tissue (dendrites, axons, synapses, endings) is laid out deterministically from the
//! creature and its wiring, and tessellated once into a mesh; each frame only the somas, the
//! moving pulses and the flashes are drawn on top.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bevy_egui::egui::{self, Color32, Pos2, Shape, Stroke, Vec2, pos2, vec2};
use evolve_sim::snapshot::CreatureDetail;
use evolve_sim::wiring::{Node, Wiring};

/// Room for the sense and action names at either side.
const LABEL_W: f32 = 108.0;
/// Most action potentials in flight at once; past it, new spikes are skipped.
const MAX_PULSES: usize = 400;
/// Pulse speeds, points per second.
const AXON_SPEED: f32 = 320.0;
const DENDRITE_SPEED: f32 = 150.0;
const RECEPTOR_SPEED: f32 = 180.0;

const BACKGROUND: Color32 = Color32::from_rgb(9, 10, 18);

fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

/// Warm for excitation, cool for inhibition.
fn sign_color(sign: f32, alpha: u8) -> Color32 {
    if sign >= 0.0 { rgba(255, 160, 60, alpha) } else { rgba(80, 150, 255, alpha) }
}

/// splitmix64: cheap, deterministic randomness keyed by the creature and neuron.
struct Rng(u64);

impl Rng {
    fn new(a: u64, b: u64) -> Self {
        let mut r = Rng(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03);
        r.next();
        r
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn node_key(n: Node) -> u64 {
    match n {
        Node::Input(k) => k as u64,
        Node::Hidden(k) => 1_000_000 + k as u64,
        Node::Output(k) => 2_000_000 + k as u64,
    }
}

fn dir(a: f32) -> Vec2 {
    vec2(a.cos(), a.sin())
}

fn perp(v: Vec2) -> Vec2 {
    vec2(-v.y, v.x)
}

/// A cubic Bézier sampled into `n` + 1 points, bent sideways by a gentle wave so no two axons
/// look ruled.
fn curve(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, n: usize, wave: f32, phase: f32) -> Vec<Pos2> {
    let side = perp((p3 - p0).normalized());
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let u = 1.0 - t;
            let p = p0.to_vec2() * (u * u * u)
                + p1.to_vec2() * (3.0 * u * u * t)
                + p2.to_vec2() * (3.0 * u * t * t)
                + p3.to_vec2() * (t * t * t);
            let bend = wave * (t * std::f32::consts::PI).sin() * (t * 7.0 + phase).sin();
            (p + side * bend).to_pos2()
        })
        .collect()
}

/// A polyline a pulse can travel, with the distance to each point.
struct Path {
    pts: Vec<Pos2>,
    cum: Vec<f32>,
}

impl Path {
    fn new(pts: Vec<Pos2>) -> Self {
        let mut cum = Vec::with_capacity(pts.len());
        let mut d = 0.0;
        for (i, p) in pts.iter().enumerate() {
            if i > 0 {
                d += p.distance(pts[i - 1]);
            }
            cum.push(d);
        }
        Self { pts, cum }
    }

    fn len(&self) -> f32 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    fn at(&self, s: f32) -> Pos2 {
        let s = s.clamp(0.0, self.len());
        let i = self.cum.partition_point(|&c| c < s).clamp(1, self.pts.len().max(2) - 1);
        let (c0, c1) = (self.cum[i - 1], self.cum[i]);
        let t = if c1 > c0 { (s - c0) / (c1 - c0) } else { 0.0 };
        self.pts[i - 1] + (self.pts[i] - self.pts[i - 1]) * t
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Sensory,
    Hidden,
    Motor,
}

/// A dendrite branch: its points from the soma out to its tip.
struct Branch {
    from_soma: Vec<Pos2>,
    /// Where this branch's own points start in `from_soma`.
    own: usize,
    width: f32,
}

struct Neuron {
    node: Node,
    role: Role,
    at: Pos2,
    radius: f32,
    /// Outline of the cell body, relative to `at`.
    outline: Vec<Vec2>,
    nucleus: Vec2,
    branches: Vec<Branch>,
    /// Synapses this neuron's axon makes.
    out: Vec<usize>,
    /// For a sensory neuron, the path from its receptor ending in to the soma.
    receptor: Option<usize>,
    /// For a motor neuron, the path down its axon to its end plate, and where that is.
    end_plate: Option<(usize, Pos2)>,
    /// Where its name goes (sensory and motor neurons).
    label: Option<Pos2>,
}

struct Synapse {
    at: Pos2,
    /// Weight relative to the strongest of its kind, with its sign.
    weight: f32,
    source: usize,
    target: usize,
    axon: usize,
    dendrite: usize,
}

/// Where a connection lands: target neuron, the point on its dendrite, the dendrite's outward
/// direction there, relative weight, and the branch and point (branch usize::MAX: on the soma).
type Landing = (usize, Pos2, Vec2, f32, usize, usize);

/// Everything about the tissue that only changes with the wiring or the room to draw it in.
struct Layout {
    topology: u64,
    style: u64,
    size: Vec2,
    neurons: Vec<Neuron>,
    synapses: Vec<Synapse>,
    paths: Vec<Path>,
    /// Axon trunks shared by a neuron's synapses: (path, how many of its points are trunk).
    trunks: Vec<(usize, usize, f32)>,
    /// The tissue, tessellated, and the screen position it was made for.
    mesh: Arc<egui::Mesh>,
    origin: Pos2,
    ppp: f32,
}

#[derive(Clone, Copy)]
enum Arrive {
    Fire(usize),
    Synapse(usize),
    Soma(usize, f32),
    EndPlate(usize),
}

#[derive(Clone, Copy, PartialEq)]
enum PulseKind {
    Receptor,
    Axon,
    Dendrite(f32),
}

struct Pulse {
    path: usize,
    born: f64,
    speed: f32,
    kind: PulseKind,
    arrive: Arrive,
}

/// The neural view's state: its layout and what is firing.
#[derive(Default)]
pub struct NeuralView {
    layout: Option<Layout>,
    pulses: Vec<Pulse>,
    phase: Vec<f32>,
    fired: Vec<f64>,
    psp: Vec<(f64, f32)>,
    synapse_lit: Vec<f64>,
    plate_lit: Vec<f64>,
    last: f64,
}

/// How strongly connections of this kind can be: weights are scaled per kind, as in the network
/// view, so each layer's strongest reads as strong.
fn kind(e: &evolve_sim::wiring::Edge) -> usize {
    match (e.from, e.to) {
        (Node::Input(_), _) => 0,
        (_, Node::Output(_)) => 2,
        _ => 1,
    }
}

fn keys(w: &Wiring, id: u64, size: Vec2) -> (u64, u64) {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    (size.x.round() as i32, size.y.round() as i32).hash(&mut h);
    w.inputs.hash(&mut h);
    w.hidden.hash(&mut h);
    w.outputs.hash(&mut h);
    for e in &w.edges {
        e.from.hash(&mut h);
        e.to.hash(&mut h);
    }
    let topology = h.finish();
    let mut max = [1e-6f32; 3];
    for e in &w.edges {
        max[kind(e)] = max[kind(e)].max(e.weight.abs());
    }
    for e in &w.edges {
        ((e.weight / max[kind(e)] * 12.0).round() as i32).hash(&mut h);
    }
    (topology, h.finish())
}

/// Grows a dendrite tree from a soma: a few primary branches, each forking once or twice,
/// wandering a little and tapering. `aim` biases them toward one side (motor neurons reach
/// back toward the hidden layer).
fn grow_dendrites(rng: &mut Rng, at: Pos2, radius: f32, reach: f32, aim: Option<f32>) -> Vec<Branch> {
    let mut out = Vec::new();
    let n = if aim.is_some() { 2 + (rng.next() % 2) as usize } else { 3 + (rng.next() % 3) as usize };
    let start = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..n {
        let a = match aim {
            Some(a) => a + (i as f32 + 0.5 - n as f32 * 0.5) / n as f32 * 2.4 + rng.range(-0.2, 0.2),
            None => start + i as f32 / n as f32 * std::f32::consts::TAU + rng.range(-0.35, 0.35),
        };
        let base = at + dir(a) * radius * 0.9;
        let len = reach * rng.range(0.75, 1.2);
        grow_branch(rng, &mut out, vec![at], base, a, len, 2.4, 0);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn grow_branch(rng: &mut Rng, out: &mut Vec<Branch>, before: Vec<Pos2>, from: Pos2, angle: f32, len: f32, width: f32, depth: u32) {
    let steps = 4;
    let mut a = angle;
    let mut p = from;
    let mut from_soma = before;
    let own = from_soma.len();
    from_soma.push(p);
    for _ in 0..steps {
        a += rng.range(-0.3, 0.3);
        p += dir(a) * (len / steps as f32);
        from_soma.push(p);
    }
    let tip = from_soma.clone();
    out.push(Branch { from_soma, own, width });
    if depth < 2 {
        let forks = if depth == 0 { 2 } else { 1 + (rng.next() % 2) as usize };
        for k in 0..forks {
            let side = if k == 0 { 1.0 } else { -1.0 };
            let fa = a + side * rng.range(0.3, 0.75);
            let child = len * rng.range(0.5, 0.7);
            grow_branch(rng, out, tip.clone(), p, fa, child, width * 0.6, depth + 1);
        }
    }
}

/// A slightly irregular cell body, smooth enough to stay convex.
fn soma_outline(rng: &mut Rng, radius: f32) -> Vec<Vec2> {
    let (p1, p2) = (rng.range(0.0, 6.3), rng.range(0.0, 6.3));
    let (a1, a2) = (rng.range(0.06, 0.13), rng.range(0.03, 0.07));
    let stretch = rng.range(0.85, 1.15);
    let turn = rng.range(0.0, std::f32::consts::PI);
    (0..16)
        .map(|i| {
            let t = i as f32 / 16.0 * std::f32::consts::TAU;
            let r = radius * (1.0 + a1 * (2.0 * t + p1).sin() + a2 * (3.0 * t + p2).sin());
            let v = vec2(t.cos() * r * stretch, t.sin() * r / stretch);
            vec2(v.x * turn.cos() - v.y * turn.sin(), v.x * turn.sin() + v.y * turn.cos())
        })
        .collect()
}

fn build(w: &Wiring, id: u64, size: Vec2, topology: u64, style: u64) -> Layout {
    let (width, height) = (size.x, size.y);
    let row_y = |i: usize, n: usize| 14.0 + (height - 28.0) * (i as f32 + 0.5) / n.max(1) as f32;
    let x_sense = LABEL_W + 34.0;
    let x_motor = width - LABEL_W - 34.0;
    let mut neurons: Vec<Neuron> = Vec::new();
    let mut index: HashMap<Node, usize> = HashMap::new();
    let new_neuron = |node: Node, role: Role, at: Pos2, radius: f32, rng: &mut Rng| Neuron {
        node,
        role,
        at,
        radius,
        outline: soma_outline(rng, radius),
        nucleus: vec2(rng.range(-0.25, 0.25), rng.range(-0.25, 0.25)) * radius,
        branches: Vec::new(),
        out: Vec::new(),
        receptor: None,
        end_plate: None,
        label: None,
    };

    // Senses down the left, actions down the right.
    let in_gap = (height - 28.0) / w.inputs.len().max(1) as f32;
    for (i, &k) in w.inputs.iter().enumerate() {
        let node = Node::Input(k);
        let mut rng = Rng::new(id, node_key(node));
        let at = pos2(x_sense + rng.range(-7.0, 7.0), row_y(i, w.inputs.len()));
        let mut n = new_neuron(node, Role::Sensory, at, (in_gap * 0.42).clamp(2.6, 5.0), &mut rng);
        n.label = Some(pos2(LABEL_W - 6.0, at.y));
        index.insert(node, neurons.len());
        neurons.push(n);
    }
    let out_gap = (height - 28.0) / w.outputs.len().max(1) as f32;
    for (i, &k) in w.outputs.iter().enumerate() {
        let node = Node::Output(k);
        let mut rng = Rng::new(id, node_key(node));
        let at = pos2(x_motor + rng.range(-7.0, 7.0), row_y(i, w.outputs.len()));
        let radius = (out_gap * 0.45).clamp(3.0, 6.0);
        let mut n = new_neuron(node, Role::Motor, at, radius, &mut rng);
        n.branches = grow_dendrites(&mut rng, at, radius, (out_gap * 1.3).clamp(7.0, 22.0), Some(std::f32::consts::PI));
        n.label = Some(pos2(width - LABEL_W + 6.0, at.y));
        index.insert(node, neurons.len());
        neurons.push(n);
    }

    // The hidden neurons scattered through the middle, each keeping its place (a slot picked
    // from its own number) as long as it stays in the sketch.
    let (zx0, zx1) = (LABEL_W + 66.0, (width - LABEL_W - 66.0).max(LABEL_W + 146.0));
    let (zy0, zy1) = (24.0, (height - 24.0).max(48.0));
    let slots = ((w.hidden.len() as f32 * 1.7).ceil() as usize).max(4);
    let cols = ((slots as f32 * (zx1 - zx0) / (zy1 - zy0)).sqrt().round() as usize).max(2);
    let rows = slots.div_ceil(cols).max(1);
    let (cw, ch) = ((zx1 - zx0) / cols as f32, (zy1 - zy0) / rows as f32);
    let mut taken = vec![false; cols * rows];
    let mut hidden: Vec<usize> = w.hidden.clone();
    hidden.sort_unstable();
    for &k in &hidden {
        let node = Node::Hidden(k);
        let mut rng = Rng::new(id, node_key(node));
        let mut s = (rng.next() % taken.len() as u64) as usize;
        while taken[s] {
            s = (s + 1) % taken.len();
        }
        taken[s] = true;
        let (c, r) = (s % cols, s / cols);
        let at = pos2(
            zx0 + (c as f32 + 0.5 + rng.range(-0.28, 0.28)) * cw,
            zy0 + (r as f32 + 0.5 + rng.range(-0.28, 0.28)) * ch,
        );
        let radius = (cw.min(ch) * 0.17).clamp(5.0, 9.0) * rng.range(0.85, 1.15);
        let mut n = new_neuron(node, Role::Hidden, at, radius, &mut rng);
        n.branches = grow_dendrites(&mut rng, at, radius, (cw.min(ch) * 0.46).clamp(11.0, 44.0), None);
        index.insert(node, neurons.len());
        neurons.push(n);
    }

    let mut max = [1e-6f32; 3];
    for e in &w.edges {
        max[kind(e)] = max[kind(e)].max(e.weight.abs());
    }
    let mut paths: Vec<Path> = Vec::new();
    let mut synapses: Vec<Synapse> = Vec::new();
    let mut trunks = Vec::new();
    // How often each dendrite point is used, to spread synapses out.
    let mut used: HashMap<(usize, usize, usize), u32> = HashMap::new();

    // Synapses: each connection lands on the dendrite point of its target nearest the source.
    let mut by_source: HashMap<usize, Vec<Landing>> = HashMap::new();
    for e in &w.edges {
        let (Some(&s), Some(&t)) = (index.get(&e.from), index.get(&e.to)) else {
            continue;
        };
        let src = neurons[s].at;
        let tn = &neurons[t];
        let mut best: Option<(f32, usize, usize)> = None;
        for (b, br) in tn.branches.iter().enumerate() {
            for p in (br.own + 1).max(2)..br.from_soma.len() {
                let pen = *used.get(&(t, b, p)).unwrap_or(&0) as f32 * 7.0;
                let d = br.from_soma[p].distance(src) + pen;
                if best.is_none_or(|(bd, _, _)| d < bd) {
                    best = Some((d, b, p));
                }
            }
        }
        let (at, outward, b, p) = match best {
            Some((_, b, p)) => {
                *used.entry((t, b, p)).or_default() += 1;
                let pts = &tn.branches[b].from_soma;
                (pts[p], (pts[p] - pts[p - 1]).normalized(), b, p)
            }
            // A neuron without dendrites takes the synapse on its cell body.
            None => {
                let away = (src - tn.at).normalized();
                (tn.at + away * tn.radius, away, usize::MAX, 0)
            }
        };
        let weight = e.weight / max[kind(e)];
        by_source.entry(s).or_default().push((t, at, outward, weight, b, p));
    }

    // Axons: a hillock leaving the soma toward its targets, a trunk, then a collateral to each
    // synapse, arriving along the dendrite.
    let mut sources: Vec<usize> = by_source.keys().copied().collect();
    sources.sort_unstable();
    for s in sources {
        let targets = &by_source[&s];
        let me = neurons[s].at;
        let mut rng = Rng::new(id, node_key(neurons[s].node) ^ 0xA807);
        let mean = targets.iter().fold(Vec2::ZERO, |acc, t| acc + t.1.to_vec2()) / targets.len() as f32;
        let mut heading = (mean.to_pos2() - me).normalized();
        if neurons[s].role == Role::Sensory {
            heading = (heading + vec2(1.5, 0.0)).normalized();
        }
        if !heading.x.is_finite() {
            heading = vec2(1.0, 0.0);
        }
        let hill = me + heading * (neurons[s].radius + 5.0);
        let reach = (mean.to_pos2() - hill).length();
        let fork = hill + heading * (reach * 0.28).clamp(4.0, 60.0) + perp(heading) * rng.range(-8.0, 8.0);
        let trunk = curve(me, hill, hill + heading * 6.0, fork, 6, 0.0, 0.0);
        let strongest = targets.iter().fold(0.0f32, |m, t| m.max(t.3.abs()));
        let mut first = true;
        for &(t, at, outward, weight, b, p) in targets {
            let len = at.distance(fork).max(1.0);
            let collateral = if t == s {
                // An autapse: the axon curls back onto its own dendrite.
                let side = perp(heading) * 26.0;
                curve(fork, fork + heading * 30.0 + side, at + outward * 30.0 + side, at, 14, 2.0, rng.range(0.0, 6.0))
            } else {
                let toward = (at - fork).normalized();
                curve(
                    fork,
                    fork + (heading + toward).normalized() * len * 0.35,
                    at + outward * (len * 0.3).min(40.0),
                    at,
                    (len / 18.0).clamp(5.0, 14.0) as usize,
                    (len * 0.05).min(6.0),
                    rng.range(0.0, 6.0),
                )
            };
            let mut pts = trunk.clone();
            pts.extend_from_slice(&collateral[1..]);
            let axon = paths.len();
            paths.push(Path::new(pts));
            if first {
                trunks.push((axon, trunk.len(), strongest));
                first = false;
            }
            // The postsynaptic potential runs from the synapse along the dendrite to the soma.
            let back: Vec<Pos2> = if b == usize::MAX {
                vec![at, neurons[t].at]
            } else {
                neurons[t].branches[b].from_soma[..=p].iter().rev().copied().collect()
            };
            let dendrite = paths.len();
            paths.push(Path::new(back));
            neurons[s].out.push(synapses.len());
            synapses.push(Synapse { at, weight, source: s, target: t, axon, dendrite });
        }
    }

    // Receptor endings feeding the senses, end plates at the ends of the motor axons.
    for n in &mut neurons {
        let mut rng = Rng::new(id, node_key(n.node) ^ 0x5E45);
        match n.role {
            Role::Sensory => {
                let ending = pos2(LABEL_W + 10.0, n.at.y + rng.range(-1.5, 1.5));
                let pts = curve(ending, ending + vec2(10.0, 0.0), n.at - vec2(12.0, 0.0), n.at, 8, 1.5, rng.range(0.0, 6.0));
                n.receptor = Some(paths.len());
                paths.push(Path::new(pts));
            }
            Role::Motor => {
                let plate = pos2(width - LABEL_W - 4.0, n.at.y + rng.range(-1.5, 1.5));
                let hill = n.at + vec2(n.radius + 4.0, 0.0);
                let pts = curve(n.at, hill, plate - vec2(14.0, 0.0), plate, 8, 1.5, rng.range(0.0, 6.0));
                n.end_plate = Some((paths.len(), plate));
                paths.push(Path::new(pts));
            }
            Role::Hidden => {}
        }
    }

    Layout {
        topology,
        style,
        size,
        neurons,
        synapses,
        paths,
        trunks,
        mesh: Arc::new(egui::Mesh::default()),
        origin: Pos2::ZERO,
        ppp: 0.0,
    }
}

/// Scales a layout's synapse weights to a fresh sketch with the same connections.
fn restyle(layout: &mut Layout, w: &Wiring, style: u64) {
    let mut max = [1e-6f32; 3];
    for e in &w.edges {
        max[kind(e)] = max[kind(e)].max(e.weight.abs());
    }
    // Synapses were made in the sketch's edge order, grouped by source; match them up again.
    let index: HashMap<Node, usize> = layout.neurons.iter().enumerate().map(|(i, n)| (n.node, i)).collect();
    let mut weights: HashMap<(usize, usize), f32> = HashMap::new();
    for e in &w.edges {
        if let (Some(&s), Some(&t)) = (index.get(&e.from), index.get(&e.to)) {
            weights.insert((s, t), e.weight / max[kind(e)]);
        }
    }
    for s in 0..layout.neurons.len() {
        for &j in &layout.neurons[s].out {
            let syn = &mut layout.synapses[j];
            if let Some(&wt) = weights.get(&(s, syn.target)) {
                syn.weight = wt;
            }
        }
    }
    for (path, _, strongest) in &mut layout.trunks {
        let s = layout.neurons.iter().position(|n| n.out.first().is_some_and(|&j| layout.synapses[j].axon == *path));
        if let Some(s) = s {
            *strongest = layout.neurons[s].out.iter().fold(0.0f32, |m, &j| m.max(layout.synapses[j].weight.abs()));
        }
    }
    layout.style = style;
    layout.ppp = 0.0;
}

/// The tissue as shapes, in layout coordinates: the out-of-focus background, dendrites,
/// axons, synapses, receptor endings and end plates.
fn tissue(l: &Layout, id: u64) -> Vec<Shape> {
    let mut shapes = Vec::new();
    let mut rng = Rng::new(id, 0xB10B);
    for _ in 0..28 {
        let at = pos2(rng.range(0.0, l.size.x), rng.range(0.0, l.size.y));
        let r = rng.range(6.0, 34.0);
        let c = if rng.f() < 0.5 { rgba(40, 70, 90, 16) } else { rgba(70, 40, 80, 14) };
        shapes.push(Shape::circle_filled(at, r, c));
    }
    let dendrite = |a: u8| rgba(110, 205, 185, a);
    for n in &l.neurons {
        // One line per branch; the forks, thinner each time, give the taper.
        for br in &n.branches {
            let own = br.from_soma[br.own..].to_vec();
            shapes.push(Shape::line(own, Stroke::new(br.width * 0.8, dendrite(195))));
        }
    }
    let axon = |a: u8| rgba(175, 200, 235, a);
    for &(path, trunk, strongest) in &l.trunks {
        let pts = &l.paths[path].pts[..trunk];
        let w = 0.7 + 1.8 * strongest;
        // The hillock: thick where it leaves the soma.
        shapes.push(Shape::line(pts[..3.min(pts.len())].to_vec(), Stroke::new(w + 1.3, axon(140))));
        shapes.push(Shape::line(pts[2.min(pts.len() - 1)..].to_vec(), Stroke::new(w, axon(85))));
    }
    for n in &l.neurons {
        // Once a neuron, not once a synapse: with thousands of each, that's millions of looks.
        let trunk = l.trunks.iter().find(|t| n.out.first().is_some_and(|&f| l.synapses[f].axon == t.0)).map_or(0, |t| t.1);
        for &j in &n.out {
            let syn = &l.synapses[j];
            let path = &l.paths[syn.axon];
            let pts = path.pts[trunk.saturating_sub(1)..].to_vec();
            shapes.push(Shape::line(pts, Stroke::new(0.35 + 1.2 * syn.weight.abs(), axon(52))));
        }
    }
    for syn in &l.synapses {
        shapes.push(Shape::circle_filled(syn.at, 1.3 + 1.6 * syn.weight.abs(), sign_color(syn.weight, 170)));
    }
    for n in &l.neurons {
        let mut rng = Rng::new(id, node_key(n.node) ^ 0xE4D);
        if let Some(r) = n.receptor {
            let pts = &l.paths[r].pts;
            shapes.push(Shape::line(pts.clone(), Stroke::new(1.1, dendrite(140))));
            let ending = pts[0];
            for k in 0..3 {
                let a = std::f32::consts::PI + (k as f32 - 1.0) * 0.55 + rng.range(-0.15, 0.15);
                let tip = ending + dir(a) * rng.range(5.0, 8.0);
                shapes.push(Shape::line_segment([ending, tip], Stroke::new(0.8, dendrite(140))));
                shapes.push(Shape::circle_filled(tip, 1.6, rgba(150, 235, 190, 190)));
            }
        }
        if let Some((path, plate)) = n.end_plate {
            shapes.push(Shape::line(l.paths[path].pts.clone(), Stroke::new(1.2, axon(130))));
            // The muscle it drives, and the terminal branches spread over it.
            shapes.push(Shape::rect_filled(
                egui::Rect::from_center_size(plate + vec2(3.0, 0.0), vec2(7.0, 9.0)),
                2.0,
                rgba(190, 90, 110, 55),
            ));
            for k in 0..3 {
                let a = (k as f32 - 1.0) * 0.7 + rng.range(-0.15, 0.15);
                let tip = plate + dir(a) * rng.range(3.0, 5.0);
                shapes.push(Shape::line_segment([plate, tip], Stroke::new(0.9, axon(150))));
                shapes.push(Shape::circle_filled(tip, 1.5, rgba(235, 200, 160, 200)));
            }
        }
    }
    shapes
}

/// How busy a neuron is, 0..1, from its live value: senses by magnitude, neurons and actions by
/// how far toward +1 their output is.
fn activity(role: Role, v: f32) -> f32 {
    match role {
        Role::Sensory => v.abs().min(1.0),
        _ => ((v + 1.0) * 0.5).clamp(0.0, 1.0),
    }
}

fn value(c: &CreatureDetail, node: Node) -> f32 {
    match node {
        Node::Input(k) => c.inputs.get(k).copied().unwrap_or(0.0),
        Node::Hidden(k) => c.hidden.get(k).copied().unwrap_or(0.0),
        Node::Output(k) => c.outputs.get(k).copied().unwrap_or(0.0),
    }
}

impl NeuralView {
    /// Moves the firing state onto a new layout: neurons keep their rhythm, and spikes in flight
    /// carry on along the same connection where it still exists (the sketch is redrawn every
    /// second, and learning shuffles which connections are the strongest).
    fn carry_over(&mut self, new: &Layout, id: u64) {
        let n = new.neurons.len();
        let mut rng = Rng::new(id, 0xF12E);
        let mut phase: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        let mut fired = vec![-10.0; n];
        let mut psp = vec![(-10.0, 0.0); n];
        let mut plate_lit = vec![-10.0; n];
        let mut synapse_lit = vec![-10.0; new.synapses.len()];
        let mut pulses = Vec::new();
        if let Some(old) = &self.layout
            && self.phase.len() == old.neurons.len()
        {
            let at: HashMap<Node, usize> = new.neurons.iter().enumerate().map(|(i, m)| (m.node, i)).collect();
            let remap: Vec<Option<usize>> = old.neurons.iter().map(|m| at.get(&m.node).copied()).collect();
            for (i, &to) in remap.iter().enumerate() {
                if let Some(to) = to {
                    phase[to] = self.phase[i];
                    fired[to] = self.fired[i];
                    psp[to] = self.psp[i];
                    plate_lit[to] = self.plate_lit[i];
                }
            }
            let pairs: HashMap<(usize, usize), usize> =
                new.synapses.iter().enumerate().map(|(j, y)| ((y.source, y.target), j)).collect();
            let same = |j: usize| {
                let y = &old.synapses[j];
                pairs.get(&(remap[y.source]?, remap[y.target]?)).copied()
            };
            for (j, &lit) in self.synapse_lit.iter().enumerate() {
                if let Some(k) = same(j) {
                    synapse_lit[k] = lit;
                }
            }
            for p in self.pulses.drain(..) {
                let moved = match (p.kind, p.arrive) {
                    (PulseKind::Receptor, Arrive::Fire(i)) => remap[i]
                        .and_then(|i| Some((new.neurons[i].receptor?, Arrive::Fire(i)))),
                    (_, Arrive::EndPlate(i)) => remap[i]
                        .and_then(|i| Some((new.neurons[i].end_plate?.0, Arrive::EndPlate(i)))),
                    (PulseKind::Axon, Arrive::Synapse(j)) => same(j).map(|k| (new.synapses[k].axon, Arrive::Synapse(k))),
                    (PulseKind::Dendrite(sign), Arrive::Soma(..)) => old
                        .synapses
                        .iter()
                        .position(|y| y.dendrite == p.path)
                        .and_then(same)
                        .map(|k| (new.synapses[k].dendrite, Arrive::Soma(new.synapses[k].target, sign))),
                    _ => None,
                };
                if let Some((path, arrive)) = moved {
                    pulses.push(Pulse { path, arrive, ..p });
                }
            }
        }
        (self.phase, self.fired, self.psp, self.plate_lit, self.synapse_lit, self.pulses) =
            (phase, fired, psp, plate_lit, synapse_lit, pulses);
    }

    /// Draws the neural view into `rect`, `now` seconds into the app's run.
    pub fn show(&mut self, ui: &mut egui::Ui, rect: egui::Rect, resp: &egui::Response, c: &CreatureDetail, now: f64) {
        let w = &*c.wiring;
        let (topology, style) = keys(w, c.id, rect.size());
        if self.layout.as_ref().is_none_or(|l| l.topology != topology) {
            let l = build(w, c.id, rect.size(), topology, style);
            self.carry_over(&l, c.id);
            self.layout = Some(l);
        }
        let Some(l) = self.layout.as_mut() else {
            return;
        };
        if l.style != style {
            restyle(l, w, style);
        }
        let ppp = ui.ctx().pixels_per_point();
        if l.ppp != ppp {
            let mut tess = egui::epaint::Tessellator::new(ppp, egui::epaint::TessellationOptions::default(), [1, 1], Vec::new());
            let mut mesh = egui::Mesh::default();
            for shape in tissue(l, c.id) {
                tess.tessellate_shape(shape, &mut mesh);
            }
            l.mesh = Arc::new(mesh);
            l.origin = Pos2::ZERO;
            l.ppp = ppp;
        }
        if l.origin != rect.min {
            Arc::make_mut(&mut l.mesh).translate(rect.min - l.origin);
            l.origin = rect.min;
        }
        let o = rect.min.to_vec2();

        // Fire neurons at their rates and move the pulses on.
        let dt = (now - self.last).clamp(0.0, 0.1) as f32;
        self.last = now;
        let mut events: Vec<Arrive> = Vec::new();
        for (i, n) in l.neurons.iter().enumerate() {
            let a = activity(n.role, value(c, n.node));
            self.phase[i] += (0.12 + 4.0 * a * a) * dt;
            if self.phase[i] >= 1.0 {
                self.phase[i] = (self.phase[i] - 1.0).min(0.99);
                match n.role {
                    // A sense starts at its receptor and fires on reaching the soma.
                    Role::Sensory => {
                        if let Some(r) = n.receptor
                            && self.pulses.len() < MAX_PULSES
                        {
                            self.pulses.push(Pulse { path: r, born: now, speed: RECEPTOR_SPEED, kind: PulseKind::Receptor, arrive: Arrive::Fire(i) });
                        }
                    }
                    _ => events.push(Arrive::Fire(i)),
                }
            }
        }
        self.pulses.retain(|p| {
            let done = (now - p.born) as f32 * p.speed >= l.paths[p.path].len();
            if done {
                events.push(p.arrive);
            }
            !done
        });
        for e in events {
            match e {
                Arrive::Fire(i) => {
                    self.fired[i] = now;
                    let n = &l.neurons[i];
                    if let Some((path, _)) = n.end_plate
                        && self.pulses.len() < MAX_PULSES
                    {
                        self.pulses.push(Pulse { path, born: now, speed: AXON_SPEED, kind: PulseKind::Axon, arrive: Arrive::EndPlate(i) });
                    }
                    for &j in &n.out {
                        if self.pulses.len() < MAX_PULSES {
                            self.pulses.push(Pulse { path: l.synapses[j].axon, born: now, speed: AXON_SPEED, kind: PulseKind::Axon, arrive: Arrive::Synapse(j) });
                        }
                    }
                }
                Arrive::Synapse(j) => {
                    self.synapse_lit[j] = now;
                    let syn = &l.synapses[j];
                    if self.pulses.len() < MAX_PULSES {
                        let sign = syn.weight.signum() * (0.3 + 0.7 * syn.weight.abs());
                        self.pulses.push(Pulse {
                            path: syn.dendrite,
                            born: now,
                            speed: DENDRITE_SPEED,
                            kind: PulseKind::Dendrite(sign),
                            arrive: Arrive::Soma(syn.target, sign),
                        });
                    }
                }
                Arrive::Soma(t, s) => self.psp[t] = (now, s),
                Arrive::EndPlate(i) => self.plate_lit[i] = now,
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BACKGROUND);
        painter.add(Shape::Mesh(l.mesh.clone()));

        // Synapses and end plates that just received a spike.
        for (j, syn) in l.synapses.iter().enumerate() {
            let age = (now - self.synapse_lit[j]) as f32;
            if age < 0.3 {
                let k = 1.0 - age / 0.3;
                painter.circle_filled(syn.at + o, 2.5 + 4.0 * k, sign_color(syn.weight, (150.0 * k) as u8));
            }
        }
        for (i, n) in l.neurons.iter().enumerate() {
            if let Some((_, plate)) = n.end_plate {
                let age = (now - self.plate_lit[i]) as f32;
                if age < 0.35 {
                    let k = 1.0 - age / 0.35;
                    painter.rect_filled(
                        egui::Rect::from_center_size(plate + vec2(3.0, 0.0) + o, vec2(7.0, 9.0)),
                        2.0,
                        rgba(255, 140, 150, (170.0 * k) as u8),
                    );
                    painter.circle_filled(plate + o, 3.0 + 3.0 * k, rgba(255, 230, 200, (200.0 * k) as u8));
                }
            }
        }

        // Action potentials: a bright bead with a short tail; postsynaptic potentials glow warm
        // or cool, dimmer and slower.
        for p in &self.pulses {
            let path = &l.paths[p.path];
            let s = (now - p.born) as f32 * p.speed;
            let (core, halo, r) = match p.kind {
                PulseKind::Axon => (rgba(235, 250, 255, 255), rgba(120, 210, 255, 70), 1.9),
                PulseKind::Receptor => (rgba(220, 255, 225, 240), rgba(120, 240, 170, 60), 1.6),
                PulseKind::Dendrite(sign) => {
                    let a = (100.0 + 120.0 * sign.abs()) as u8;
                    (sign_color(sign, a), sign_color(sign, a / 3), 1.5)
                }
            };
            painter.circle_filled(path.at(s - 6.0) + o, r * 0.8, halo.gamma_multiply(0.7));
            let at = path.at(s) + o;
            painter.circle_filled(at, r * 2.2, halo);
            painter.circle_filled(at, r, core);
        }

        // Cell bodies: tinted by role, glowing with activity, flashing when they fire, and
        // rimmed warm or cool for a moment when a postsynaptic potential arrives.
        for (i, n) in l.neurons.iter().enumerate() {
            let a = activity(n.role, value(c, n.node));
            let flash = (1.0 - (now - self.fired[i]) as f32 / 0.15).max(0.0);
            let base = match n.role {
                Role::Sensory => [70.0, 150.0, 110.0],
                Role::Hidden => [110.0, 95.0, 170.0],
                Role::Motor => [175.0, 110.0, 80.0],
            };
            let lit = |k: usize| (base[k] * (0.35 + 0.65 * a) + (255.0 - base[k] * (0.35 + 0.65 * a)) * flash) as u8;
            let fill = rgba(lit(0), lit(1), lit(2), 235);
            let (psp_at, psp_sign) = self.psp[i];
            let psp = (1.0 - (now - psp_at) as f32 / 0.2).max(0.0);
            let rim = if psp > 0.0 { sign_color(psp_sign, (120.0 + 120.0 * psp) as u8) } else { rgba(200, 230, 225, 90) };
            let center = n.at + o;
            if flash > 0.0 {
                painter.circle_filled(center, n.radius * (1.6 + 1.2 * flash), rgba(200, 230, 255, (60.0 * flash) as u8));
            }
            let pts: Vec<Pos2> = n.outline.iter().map(|v| center + *v).collect();
            painter.add(Shape::convex_polygon(pts, fill, Stroke::new(0.9 + psp, rim)));
            painter.circle_filled(center + n.nucleus, n.radius * 0.38, rgba(25, 20, 40, 150));
        }

        // Names of the senses and actions.
        let font = egui::FontId::proportional(10.5);
        let text = rgba(185, 195, 205, 230);
        for n in &l.neurons {
            if let Some(at) = n.label {
                match n.node {
                    Node::Input(k) => {
                        painter.text(at + o, egui::Align2::RIGHT_CENTER, crate::ui::short_input(k), font.clone(), text);
                    }
                    Node::Output(k) => {
                        painter.text(at + o, egui::Align2::LEFT_CENTER, crate::ui::short_output(k), font.clone(), text);
                    }
                    Node::Hidden(_) => {}
                }
            }
        }

        if let Some(pointer) = resp.hover_pos()
            && let Some(n) = l
                .neurons
                .iter()
                .min_by(|a, b| (a.at + o).distance(pointer).total_cmp(&(b.at + o).distance(pointer)))
            && (n.at + o).distance(pointer) < n.radius + 5.0
        {
            let v = value(c, n.node);
            let name = match n.node {
                Node::Input(k) => format!("sensory neuron: {}", evolve_sim::brain::input_label(k)),
                Node::Output(k) => format!("motor neuron: {}", evolve_sim::brain::output_label(k)),
                Node::Hidden(k) => format!("neuron {k}"),
            };
            let i = l.neurons.iter().position(|m| m.node == n.node).unwrap_or(0);
            let rate = 0.12 + 4.0 * activity(n.role, v).powi(2);
            let synapses = n.out.len();
            let incoming = l.synapses.iter().filter(|s| s.target == i).count();
            resp.clone().on_hover_text_at_pointer(format!(
                "{name}: {v:+.2}\nfiring about {rate:.1} times a second\n{incoming} synapses in, {synapses} out (shown)"
            ));
        }
    }
}
