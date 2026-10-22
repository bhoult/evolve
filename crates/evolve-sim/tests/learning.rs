//! Lifetime learning: the rule learns, and parents pass some of it on.

use evolve_sim::brain::{self, BrainGenes, BrainState, Learning, N_IN, OutputBias};
use evolve_sim::rng::Rng;

/// A brain rewarded for how high one output goes, exploring with noise on that output, should
/// learn to push it up. With learning off it shouldn't change.
fn train(lr_scale: f32) -> (f32, f32) {
    let mut rng = Rng::new(5);
    let g = BrainGenes::random(&mut rng, 16);
    let mut st = BrainState::new(&g);
    let x: [f32; N_IN] = std::array::from_fn(|_| rng.signed() * 0.5);
    let dt = 1.0 / 30.0;
    let k = 3;
    let (mut early, mut late) = (0.0, 0.0);
    let ticks = 6000;
    for t in 0..ticks {
        let lr = brain::learning_signal(&mut st, brain::LEARNING_RATE * lr_scale, dt);
        let learn = Learning {
            lr,
            trace_decay: brain::per_tick_decay(0.3, dt),
            plastic_decay: 1.0,
            pipelined: false,
        };
        brain::step_cpu(&g, OutputBias::default(), &mut st, &x, learn);
        st.noise[k] = 0.2 * rng.normal();
        let action = (st.outputs[k] + st.noise[k]).clamp(-1.0, 1.0);
        st.reward = action;
        if t < 300 {
            early += st.outputs[k] / 300.0;
        }
        if t >= ticks - 300 {
            late += st.outputs[k] / 300.0;
        }
    }
    (early, late)
}

#[test]
fn reward_with_exploration_teaches_the_rewarded_action() {
    let (early, late) = train(1.0);
    eprintln!("learning on: output {early:+.3} -> {late:+.3}");
    assert!(late > early + 0.3 && late > 0.5, "should learn to raise the rewarded output");
    let (early, late) = train(0.0);
    eprintln!("learning off: output {early:+.3} -> {late:+.3}");
    assert!((late - early).abs() < 0.05, "without learning nothing should change");
}

#[test]
fn parents_pass_on_a_fraction_of_what_they_learned() {
    let mut rng = Rng::new(8);
    let mut g = BrainGenes::random(&mut rng, 6);
    let before: Vec<f32> = g.w_out.0.iter().map(|w| w.to_f32()).collect();
    let learned: Vec<f32> = (0..before.len()).map(|_| rng.signed()).collect();
    g.absorb(&learned, 0.5);
    for ((w, b), l) in g.w_out.0.iter().zip(&before).zip(&learned) {
        assert!((w.to_f32() - (b + 0.5 * l)).abs() < 2e-3);
    }
}

#[test]
fn the_gpu_hands_back_what_a_brain_learned() {
    use evolve_sim::gpu::GpuBrains;
    let mut gpu = match GpuBrains::new() {
        Ok(g) => g,
        Err(e) => {
            println!("no GPU, skipping: {e}");
            return;
        }
    };
    let mut rng = Rng::new(9);
    let mut want = Vec::new();
    for (id, h) in [(1u64, 3usize), (2, 64), (3, 301)] {
        let g = BrainGenes::random(&mut rng, h);
        let mut st = BrainState::new(&g);
        st.plastic = (0..st.plastic.len()).map(|_| rng.signed()).collect();
        gpu.register(id, &g, Some(&st)).unwrap();
        want.push((id, st.plastic));
    }
    let got = gpu.read_plastic(&[3, 1, 2, 99]);
    assert_eq!(got.len(), 3);
    for (id, p) in want {
        assert_eq!(got[&id], p);
    }
}
