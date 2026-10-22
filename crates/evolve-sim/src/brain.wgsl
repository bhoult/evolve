// Every creature's recurrent brain, evaluated in two batched dispatches per brain tick.
// Must compute exactly what `brain::step_cpu` computes.
//
// Weights are f16 pairs packed in u32 words. Per creature (in f16 elements):
//   W_in  h x IN_PAD   (row i: neuron i's sensor weights, padded with a zero)
//   W_rec h x hp       (row i: neuron i's weights from every neuron, padded to even)
//   W_out N_OUT x hp
// and in the f32 `aux` arena (offsets mirror `Layout` in gpu.rs):
//   P     N_OUT x hp   lifetime-learned offsets on W_out
//   E     N_OUT x hp   eligibility traces
//   o_prev N_OUT       last tick's outputs
//   bias  hp
//   b_out N_OUT
// Per brain in `inputs`: the senses (IN_PAD), then the exploration noise on each output (N_OUT).

struct Meta {
    w_off: u32,     // first word of this brain in the weight page
    aux_off: u32,   // first float in aux
    h_off: u32,     // first float in the hidden-state arrays
    h: u32,
    hp: u32,
    row_start: u32, // first hidden row of this brain within the page dispatch
    batch: u32,     // index into inputs / outputs
    lr: f32,        // learning rate times reward for this tick
    bias_eat: f32,
    bias_attack: f32,
    bias_mate: f32,
    trace_decay: f32,
}

struct Params {
    meta_start: u32,
    meta_count: u32,
    rows: u32,
    plastic_decay: f32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> aux: array<f32>;
@group(0) @binding(2) var<storage, read> h_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_next: array<f32>;
@group(0) @binding(4) var<storage, read> inputs: array<f32>;
@group(0) @binding(5) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(6) var<storage, read> metas: array<Meta>;
@group(0) @binding(7) var<uniform> params: Params;

const WG: u32 = 64u;
// Filled in from brain.rs when the shader is built.
const IN_PAD: u32 = {IN_PAD}u;
const IN_STRIDE: u32 = {IN_STRIDE}u;
const N_OUT: u32 = {N_OUT}u;
const EAT: u32 = {EAT}u;
const ATTACK: u32 = {ATTACK}u;
const MATE: u32 = {MATE}u;

fn aux_traces(m: Meta) -> u32 {
    return m.aux_off + N_OUT * m.hp;
}
fn aux_last_out(m: Meta) -> u32 {
    return m.aux_off + 2u * N_OUT * m.hp;
}
fn aux_bias(m: Meta) -> u32 {
    return aux_last_out(m) + N_OUT;
}
fn aux_b_out(m: Meta) -> u32 {
    return aux_bias(m) + m.hp;
}

var<workgroup> partial: array<f32, 64>;

fn reduce(t: u32, v: f32) -> f32 {
    partial[t] = v;
    workgroupBarrier();
    var s = WG / 2u;
    loop {
        if s == 0u {
            break;
        }
        if t < s {
            partial[t] += partial[t + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    return partial[0];
}

// The brain owning global hidden row `row`: the last meta whose row_start <= row.
fn find_meta(row: u32) -> u32 {
    var lo = params.meta_start;
    var hi = params.meta_start + params.meta_count - 1u;
    loop {
        if lo >= hi {
            break;
        }
        let mid = (lo + hi + 1u) / 2u;
        if metas[mid].row_start <= row {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    return lo;
}

// One workgroup per hidden neuron: h' = tanh(W_in x + W_rec h + b).
@compute @workgroup_size(64)
fn hidden(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let row = wid.x + wid.y * nwg.x;
    if row >= params.rows {
        return;
    }
    let m = metas[find_meta(row)];
    let i = row - m.row_start;
    var acc = 0.0;
    let in_w = m.w_off + i * (IN_PAD / 2u);
    let x0 = m.batch * IN_STRIDE;
    for (var w = t; w < IN_PAD / 2u; w += WG) {
        let p = unpack2x16float(weights[in_w + w]);
        acc += p.x * inputs[x0 + 2u * w] + p.y * inputs[x0 + 2u * w + 1u];
    }
    let rec_w = m.w_off + (m.h * IN_PAD + i * m.hp) / 2u;
    for (var w = t; w < m.hp / 2u; w += WG) {
        let p = unpack2x16float(weights[rec_w + w]);
        acc += p.x * h_prev[m.h_off + 2u * w] + p.y * h_prev[m.h_off + 2u * w + 1u];
    }
    let total = reduce(t, acc);
    if t == 0u {
        let b = aux[aux_bias(m) + i];
        h_next[m.h_off + i] = tanh(total + b);
    }
}

// One workgroup per (brain, output): the learning step (traces take in last tick's exploration
// times the hidden activity behind it, then P moves along the traces by lr and fades), then
// o = tanh((W_out + P) h' + b_out + leaning).
@compute @workgroup_size(64)
fn output(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let row = wid.x + wid.y * nwg.x;
    if row >= params.meta_count * N_OUT {
        return;
    }
    let m = metas[params.meta_start + row / N_OUT];
    let k = row % N_OUT;
    let o_at = aux_last_out(m) + k;
    let noise = inputs[m.batch * IN_STRIDE + IN_PAD + k];
    let lr = m.lr;
    let decay = params.plastic_decay;
    let p0 = m.aux_off + k * m.hp;
    let e0 = aux_traces(m) + k * m.hp;
    let w0 = m.w_off + (m.h * IN_PAD + m.h * m.hp + k * m.hp) / 2u;
    var acc = 0.0;
    for (var w = t; w < m.hp / 2u; w += WG) {
        let wp = unpack2x16float(weights[w0 + w]);
        let i0 = 2u * w;
        // The reward that just came in was earned by the action before this tick's noise
        // (outputs are acted on a tick late), so the learned weights follow the traces as they
        // were, and only then do the traces take in the new noise.
        let ea0 = aux[e0 + i0];
        let eb0 = aux[e0 + i0 + 1u];
        aux[e0 + i0] = ea0 * m.trace_decay + noise * h_prev[m.h_off + i0];
        aux[e0 + i0 + 1u] = eb0 * m.trace_decay + noise * h_prev[m.h_off + i0 + 1u];
        let pa = clamp((aux[p0 + i0] + lr * ea0) * decay, -2.0, 2.0);
        let pb = clamp((aux[p0 + i0 + 1u] + lr * eb0) * decay, -2.0, 2.0);
        aux[p0 + i0] = pa;
        aux[p0 + i0 + 1u] = pb;
        acc += (wp.x + pa) * h_next[m.h_off + i0] + (wp.y + pb) * h_next[m.h_off + i0 + 1u];
    }
    let total = reduce(t, acc);
    if t == 0u {
        var b = aux[aux_b_out(m) + k];
        if k == EAT {
            b += m.bias_eat;
        } else if k == ATTACK {
            b += m.bias_attack;
        } else if k == MATE {
            b += m.bias_mate;
        }
        let o = tanh(total + b);
        outputs[m.batch * N_OUT + k] = o;
        aux[o_at] = o;
    }
}
