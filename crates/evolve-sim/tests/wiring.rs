//! The brain sketch the brain view draws: as much of the brain as asked for, the strongest first.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use evolve_sim::brain::{BrainGenes, N_IN, N_OUT};
use evolve_sim::rng::Rng;
use evolve_sim::wiring::{Edge, Node, SketchLimits, Wiring, sketch_brain};

fn kind(e: &Edge) -> usize {
    match (e.from, e.to) {
        (Node::Input(_), _) => 0,
        (_, Node::Output(_)) => 2,
        _ => 1,
    }
}

/// Every connection of each kind among what `w` shows, as (from, to, weight).
fn candidates(b: &BrainGenes, learned: Option<&[f32]>, w: &Wiring) -> [Vec<(Node, Node, f32)>; 3] {
    let h = b.h;
    let out = |k: usize, i: usize| b.w_out.0[k * h + i].to_f32() + learned.map_or(0.0, |l| l[k * h + i]);
    let mut all: [Vec<(Node, Node, f32)>; 3] = Default::default();
    for &i in &w.hidden {
        for &k in &w.inputs {
            all[0].push((Node::Input(k), Node::Hidden(i), b.w_in.0[i * N_IN + k].to_f32()));
        }
        for &j in &w.hidden {
            all[1].push((Node::Hidden(j), Node::Hidden(i), b.w_rec.0[i * h + j].to_f32()));
        }
    }
    for &k in &w.outputs {
        for &i in &w.hidden {
            all[2].push((Node::Hidden(i), Node::Output(k), out(k, i)));
        }
    }
    all
}

/// Checks a sketch against the brain: only what it shows, as many connections as asked for and
/// shared by kind as promised, each kind strongest first, and nothing left out stronger than what
/// was kept.
fn check(b: &BrainGenes, learned: Option<&[f32]>, w: &Wiring, limits: SketchLimits) {
    assert_eq!(w.total_hidden, b.h);
    assert_eq!(w.hidden.len(), limits.neurons.min(b.h));
    assert!(w.hidden.windows(2).all(|p| p[0] < p[1]), "hidden neurons in order");
    let all = candidates(b, learned, w);
    let n: Vec<usize> = all.iter().map(Vec::len).collect();
    let total: usize = n.iter().sum();
    let k = limits.connections.min(total);
    assert_eq!(w.edges.len(), k, "as many connections as asked for");
    let k_in = k * n[0] / total;
    let k_out = k * n[2] / total;
    let by_kind: Vec<Vec<&Edge>> = (0..3).map(|t| w.edges.iter().filter(|e| kind(e) == t).collect()).collect();
    assert_eq!([by_kind[0].len(), by_kind[1].len(), by_kind[2].len()], [k_in, k - k_in - k_out, k_out]);
    // Senses into neurons, then neurons into neurons, then neurons into actions.
    assert!(w.edges.windows(2).all(|p| kind(&p[0]) <= kind(&p[1])));
    let shown: HashSet<usize> = w.hidden.iter().copied().collect();
    for (t, edges) in by_kind.iter().enumerate() {
        assert!(edges.windows(2).all(|p| p[0].weight.abs() >= p[1].weight.abs()), "kind {t} strongest first");
        for e in edges {
            for node in [e.from, e.to] {
                if let Node::Hidden(i) = node {
                    assert!(shown.contains(&i));
                }
            }
        }
        let kept: HashSet<(Node, Node)> = edges.iter().map(|e| (e.from, e.to)).collect();
        assert_eq!(kept.len(), edges.len(), "no connection twice");
        let weakest_kept = edges.last().map_or(f32::INFINITY, |e| e.weight.abs());
        let strongest_left = all[t].iter().filter(|c| !kept.contains(&(c.0, c.1))).map(|c| c.2.abs()).fold(0.0, f32::max);
        // Output weights with learned offsets are picked to f16 precision.
        assert!(strongest_left <= weakest_kept * 1.001, "kind {t}: left out {strongest_left} > kept {weakest_kept}");
        let weights: HashMap<(Node, Node), f32> = all[t].iter().map(|c| ((c.0, c.1), c.2)).collect();
        for e in edges {
            assert_eq!(weights[&(e.from, e.to)], e.weight);
        }
    }
}

#[test]
fn sketch_keeps_to_its_limits_strongest_first() {
    let mut rng = Rng::new(3);
    let b = BrainGenes::random(&mut rng, 300);
    let inputs: Vec<usize> = (0..N_IN).step_by(2).collect();
    let outputs: Vec<usize> = (0..N_OUT).collect();
    let learned: Vec<f32> = (0..b.w_out.0.len()).map(|_| rng.normal() * 0.05).collect();
    for limits in [
        SketchLimits::default(),
        SketchLimits { neurons: 100, connections: 5_000 },
        SketchLimits { neurons: usize::MAX, connections: 20_000 },
        SketchLimits { neurons: 7, connections: 1_000_000 },
        SketchLimits { neurons: 50, connections: 0 },
    ] {
        for l in [None, Some(&learned[..])] {
            let w = sketch_brain(&b, l, inputs.clone(), outputs.clone(), limits);
            check(&b, l, &w, limits);
        }
    }
}

#[test]
fn sketch_shows_the_neurons_most_tied_to_senses_and_actions() {
    let mut rng = Rng::new(8);
    let b = BrainGenes::random(&mut rng, 200);
    let inputs: Vec<usize> = (0..N_IN).collect();
    let outputs: Vec<usize> = (0..N_OUT).collect();
    let w = sketch_brain(&b, None, inputs.clone(), outputs.clone(), SketchLimits { neurons: 25, connections: 100 });
    let score = |i: usize| {
        let out: f32 = outputs.iter().map(|&k| b.w_out.0[k * b.h + i].to_f32().abs()).sum();
        let inp: f32 = inputs.iter().map(|&k| b.w_in.0[i * N_IN + k].to_f32().abs()).sum();
        out + 0.5 * inp
    };
    let shown: HashSet<usize> = w.hidden.iter().copied().collect();
    let least_shown = w.hidden.iter().map(|&i| score(i)).fold(f32::INFINITY, f32::min);
    let most_left = (0..b.h).filter(|i| !shown.contains(i)).map(score).fold(0.0, f32::max);
    assert!(least_shown >= most_left);
}

#[test]
fn sketch_is_the_same_every_time_and_counts_what_it_learned() {
    let mut rng = Rng::new(12);
    let b = BrainGenes::random(&mut rng, 150);
    let inputs: Vec<usize> = (0..N_IN).collect();
    let outputs: Vec<usize> = (0..N_OUT).collect();
    let limits = SketchLimits { neurons: 60, connections: 3_000 };
    let a = sketch_brain(&b, None, inputs.clone(), outputs.clone(), limits);
    let again = sketch_brain(&b, None, inputs.clone(), outputs.clone(), limits);
    assert_eq!(a.hidden, again.hidden);
    let key = |w: &Wiring| w.edges.iter().map(|e| (e.from, e.to, e.weight.to_bits())).collect::<Vec<_>>();
    assert_eq!(key(&a), key(&again));

    // A big learned offset on one output weight puts that connection first among the actions'.
    let i = a.hidden[3];
    let mut learned = vec![0.0; b.w_out.0.len()];
    learned[5 * b.h + i] = -40.0;
    let w = sketch_brain(&b, Some(&learned), inputs, outputs, limits);
    let first = w.edges.iter().find(|e| matches!(e.to, Node::Output(_))).unwrap();
    assert_eq!((first.from, first.to), (Node::Hidden(i), Node::Output(5)));
    assert_eq!(first.weight, b.w_out.0[5 * b.h + i].to_f32() - 40.0);
}

/// What a sketch of a whole 4096-neuron brain costs (run with --nocapture to see), and that it's
/// right at that size too.
#[test]
fn sketch_of_a_big_brain() {
    let mut rng = Rng::new(21);
    let b = BrainGenes::random(&mut rng, 4096);
    let inputs: Vec<usize> = (0..N_IN).collect();
    let outputs: Vec<usize> = (0..N_OUT).collect();
    for (neurons, connections) in [(40, 320), (usize::MAX, 20_000), (usize::MAX, 50_000), (usize::MAX, 100_000)] {
        let limits = SketchLimits { neurons, connections };
        let start = Instant::now();
        let w = sketch_brain(&b, None, inputs.clone(), outputs.clone(), limits);
        let took = start.elapsed();
        eprintln!(
            "sketch of 4096 neurons to {} neurons, {connections} connections: {:.1} ms ({} threads)",
            w.hidden.len(),
            took.as_secs_f64() * 1000.0,
            rayon::current_num_threads()
        );
        assert_eq!(w.edges.len(), connections);
        if neurons == usize::MAX {
            let rec = w.edges.iter().filter(|e| kind(e) == 1).count();
            // Neuron-to-neuron connections are most of a big brain, so most of the picture.
            assert!(rec * 100 > connections * 90, "{rec} of {connections}");
            assert!(w.edges.iter().any(|e| kind(e) == 0) && w.edges.iter().any(|e| kind(e) == 2));
        }
    }
}
