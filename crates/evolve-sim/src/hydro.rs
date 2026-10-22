//! Rain on the land: where it runs, where it collects, and what it wears away.
//!
//! Every couple of simulated seconds the island is drained toward the sea: a priority flood from
//! the coast gives every land point a downhill neighbour to pass its water to (water in a hollow
//! fills it and spills over the lowest edge), and adding up the rain from everything uphill gives
//! the flow at each point. Enough flow makes a stream; hollows hold ponds while the ground is
//! soaked. Now and then the flow cuts its channel deeper (the "stream power" law: faster on
//! steeper ground with more water behind it, and faster on ground trampled bare), solved
//! implicitly downstream-first so it never digs a pit or overshoots. Trampling fades as the
//! ground grows back.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::params::SimParams;
use crate::terrain::Terrain;

/// Stream depth per square root of flow (flow counted in m² of hillside in steady rain), m.
const STREAM_DEPTH: f32 = 0.01;
/// Flow too thin to show or drink from: a film on the ground, not a stream.
const FILM: f32 = 0.02;
const MAX_STREAM_DEPTH: f32 = 0.8;
/// Recent rain keeps streams running for a while after it stops, at this share of the rain.
const BASE_FLOW: f32 = 0.35;
/// Soil wets in a few minutes of rain and dries over half an hour.
const WETTING_TIME: f32 = 120.0;
const DRYING_TIME: f32 = 1800.0;

#[derive(Clone, Copy)]
struct Entry {
    level: f32,
    k: u32,
}

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    // Lowest level first (BinaryHeap is a max-heap), ties by index so the order is repeatable.
    fn cmp(&self, o: &Self) -> Ordering {
        o.level.total_cmp(&self.level).then(o.k.cmp(&self.k))
    }
}

const NONE: u32 = u32::MAX;

/// How the island drains: for every grid point the neighbour its water goes to (NONE for the
/// sea and the edge of the world), the order points were reached from the sea (each after its
/// receiver), and the level water fills hollows to.
pub struct Drainage {
    pub receiver: Vec<u32>,
    pub order: Vec<u32>,
    pub filled: Vec<f32>,
}

pub fn drainage(t: &Terrain, water_level: f32) -> Drainage {
    let (n, h) = (t.res, &t.heights);
    let count = n * n;
    let mut receiver = vec![NONE; count];
    let mut filled = h.clone();
    let mut seen = vec![false; count];
    let mut order = Vec::with_capacity(count);
    let mut heap = BinaryHeap::with_capacity(count / 4);
    for k in 0..count {
        let (i, j) = (k / n, k % n);
        if h[k] <= water_level || i == 0 || j == 0 || i == n - 1 || j == n - 1 {
            seen[k] = true;
            heap.push(Entry { level: h[k], k: k as u32 });
        }
    }
    while let Some(Entry { level, k }) = heap.pop() {
        let k = k as usize;
        order.push(k as u32);
        let (i, j) = (k / n, k % n);
        for (di, dj) in NEIGHBOURS {
            let (ii, jj) = (i as isize + di, j as isize + dj);
            if ii < 0 || jj < 0 || ii >= n as isize || jj >= n as isize {
                continue;
            }
            let m = ii as usize * n + jj as usize;
            if seen[m] {
                continue;
            }
            seen[m] = true;
            receiver[m] = k as u32;
            // A point lower than where its water leaves is in a hollow: it fills to that level.
            filled[m] = h[m].max(level);
            heap.push(Entry { level: filled[m], k: m as u32 });
        }
    }
    Drainage { receiver, order, filled }
}

const NEIGHBOURS: [(isize, isize); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

/// Moves the water and soil on for `dt` seconds of `rain` (0 to 1) after a spell whose
/// `soak` (recent rain, 0 to 1) is given. With `erode`, also wears channels for that long.
/// Returns whether the ground's shape changed.
pub fn update(t: &mut Terrain, p: &SimParams, rain: f32, soak: f32, dt: f32, erode: Option<f32>) -> bool {
    t.ensure_layers();
    let (n, wl) = (t.res, p.water_level);
    let count = n * n;
    let d = drainage(t, wl);

    // Flow: the rain landing on each point plus everything uphill of it, in m² of hillside in
    // steady rain.
    let fall = rain.max(BASE_FLOW * soak) * p.runoff * t.cell() * t.cell();
    let mut flow: Vec<f32> = (0..count).map(|k| if t.heights[k] > wl { fall } else { 0.0 }).collect();
    for &k in d.order.iter().rev() {
        let r = d.receiver[k as usize];
        if r != NONE {
            flow[r as usize] += flow[k as usize];
        }
    }

    // Streams where enough flows, ponds in hollows while the ground is soaked.
    let ponding = (1.5 * soak).clamp(0.0, 1.0);
    let wet_k = 1.0 - (-dt / WETTING_TIME).exp();
    let dry_k = 1.0 - (-dt / DRYING_TIME).exp();
    for k in 0..count {
        if t.heights[k] <= wl {
            t.water[k] = 0.0;
            t.moisture[k] = 1.0;
            continue;
        }
        let stream = (STREAM_DEPTH * flow[k].sqrt() - FILM).clamp(0.0, MAX_STREAM_DEPTH);
        let pond = (d.filled[k] - t.heights[k]) * ponding;
        t.water[k] = stream.max(if pond > FILM { pond } else { 0.0 });
        let target = if t.water[k] > 0.0 { 1.0 } else { rain.max(0.5 * soak) };
        let m = &mut t.moisture[k];
        *m += (target - *m) * if target > *m { wet_k } else { dry_k };
    }

    // Trampled ground grows back.
    let recover = (-dt * std::f32::consts::LN_2 / (p.wear_recovery.max(0.01) * 60.0)).exp();
    for w in &mut t.wear {
        *w *= recover;
    }

    let mut changed = false;
    if let Some(edt) = erode
        && p.erosion_rate > 0.0
    {
        changed = erode_channels(t, p, &d, &flow, edt);
    }
    t.version += 1;
    changed
}

/// Cuts channels for `dt` seconds of the given flow, downstream first, each point toward its
/// (already lowered) receiver: h' = (h + F h_r') / (1 + F), F = K sqrt(flow) dt / distance, with
/// K set so 100 m² of flow at a 1-in-10 slope cuts `erosion_rate` m an hour.
fn erode_channels(t: &mut Terrain, p: &SimParams, d: &Drainage, flow: &[f32], dt: f32) -> bool {
    let (n, wl) = (t.res, p.water_level);
    let cell = t.cell();
    let rate = p.erosion_rate / 3600.0;
    let floor = wl - 0.3;
    let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0, usize::MAX, 0);
    for &k in &d.order {
        let k = k as usize;
        let r = d.receiver[k];
        // Only streams cut (not the thin film of rain on every slope), and not the sea, the
        // world's edge or the bottoms of hollows.
        if r == NONE || t.water[k] <= 0.0 || t.heights[k] <= floor || d.filled[k] > t.heights[k] + 1e-4 {
            continue;
        }
        let r = r as usize;
        let hr = t.heights[r];
        let h = t.heights[k];
        if h <= hr {
            continue;
        }
        let dist = if r / n != k / n && r % n != k % n { cell * std::f32::consts::SQRT_2 } else { cell };
        let f = rate * (1.0 + 3.0 * t.worn(k)) * (flow[k] / 100.0).sqrt() * 10.0 * dt / dist;
        let room = (p.erosion_max_depth - t.eroded[k]).max(0.0);
        let cut = (h - (h + f * hr) / (1.0 + f)).min(room).min(h - floor).max(0.0);
        if cut < 1e-6 {
            continue;
        }
        t.heights[k] -= cut;
        t.eroded[k] += cut;
        let (i, j) = (k / n, k % n);
        (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(j), j1.max(j));
    }
    if i0 == usize::MAX {
        return false;
    }
    t.mark_dirty(i0, i1, j0, j1);
    true
}
