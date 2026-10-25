//! Rain on the land: where it runs, where it collects, and what it wears away.
//!
//! Every couple of simulated seconds the island is drained toward the sea: a priority flood from
//! the coast gives every land point a downhill neighbour to pass its water to (water in a hollow
//! fills it and spills over the lowest edge), and adding up the rain from everything uphill gives
//! the flow at each point. Enough flow makes a stream. Hollows hold lakes and ponds: the rain on
//! everything draining into a hollow collects there, levels out, evaporates slowly and spills
//! over the rim when full, on downhill into the next hollow or the sea (see `fill_lakes`). A
//! spring's water creeps downhill from it, fills the first hollow in its way and only then runs
//! on over the rim (see `run_springs`). Now
//! and then the flow cuts its channel deeper (the "stream power" law: faster on
//! steeper ground with more water behind it, and faster on ground trampled bare), solved
//! implicitly downstream-first so it never digs a pit or overshoots. Trampling fades as the
//! ground grows back.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use std::sync::{Arc, Mutex};

use crate::params::SimParams;
use crate::terrain::{Layer, Terrain};

/// Stream depth per square root of flow (flow counted in m² of hillside in steady rain), m.
const STREAM_DEPTH: f32 = 0.01;
/// Flow too thin to show or drink from: a film on the ground, not a stream.
const FILM: f32 = 0.02;
const MAX_STREAM_DEPTH: f32 = 0.8;
/// Recent rain keeps streams running for a while after it stops, at this share of the rain.
const BASE_FLOW: f32 = 0.35;
/// An eternal spring's stream: as much as runs off this many m² of hillside in steady rain,
/// whatever the weather.
pub const SPRING_FLOW: f32 = 3000.0;
/// What a spring pours into the hollow it runs into, m³/s (about ten times what it would gather
/// from that much rain, so a spring makes a pond within minutes).
const SPRING_FILL: f32 = 0.35;
/// How fast a spring's water creeps on over dry ground, m/s.
const SPRING_SPEED: f32 = 0.5;
/// Its stream thins out over this much of its leading edge, m.
const SPRING_EDGE: f32 = 6.0;
/// A hollow within this of its rim is full, m.
const BRIM: f32 = 0.005;
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
    let mut filled: Vec<f32> = h.to_vec();
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

/// Where the water in lakes went in one update, m³.
#[derive(Clone, Copy, Debug, Default)]
pub struct Budget {
    /// Gathered into hollows from the rain on everything draining into them.
    pub rained: f32,
    pub evaporated: f32,
    /// Spilled over a rim (or left a hollow that's no longer there) and run off into the sea.
    pub to_sea: f32,
    /// Whether the ground's shape changed (erosion).
    pub reshaped: bool,
}

/// Moves the water and soil on for `dt` seconds of `rain` (0 to 1) after a spell whose
/// `soak` (recent rain, 0 to 1) is given. With `erode`, also wears channels for that long.
/// The finest spacing the water is worked out at, m. On a finer terrain, drainage, lakes,
/// streams and erosion run on a grid this coarse (kept in `Terrain::coarse`, so its lakes stay
/// put), and the water, soil moisture and erosion are spread back over the fine grid. Water needs
/// no finer detail, and a priority flood over the 25 million points of a 5000-a-side terrain
/// every two simulated seconds held the world to about a tenth of real time.
pub const HYDRO_CELL: f32 = 1.0;

/// Water shallower than this, m, isn't spread from the coarse grid onto a finer one (nothing
/// notices it: creatures drink, see water and plants drown from 3 cm deep).
const VISIBLE: f32 = 0.005;

/// How many points a side the water is worked out on for a terrain: its own, or a coarser grid
/// no finer than `HYDRO_CELL`.
pub fn grid_res(t: &Terrain) -> usize {
    let stride = (HYDRO_CELL / t.cell()).ceil().max(1.0) as usize;
    (t.res - 1).div_ceil(stride) + 1
}

/// Rain on the land for `dt` seconds (see the module docs), on a grid no finer than `HYDRO_CELL`.
pub fn update(t: &mut Terrain, p: &SimParams, rain: f32, soak: f32, dt: f32, erode: Option<f32>) -> Budget {
    t.ensure_layers();
    let res = grid_res(t);
    if res == t.res {
        t.coarse = None;
        return update_grid(t, p, rain, soak, dt, erode);
    }
    // The coarse grid's water is worked out on another thread while the world steps on, and
    // spread back at the next update (waiting for it only if it isn't done): working it out took
    // about 55 ms of every update on an 800 m world. Always one update behind, so the world
    // doesn't depend on how fast the thread was.
    let done = t.water_job.take().and_then(|j| j.0.lock().ok()?.take()).and_then(|h| h.join().ok());
    let (budget, mut c) = match done {
        Some(Coarse { grid, budget, heights, eroded }) if grid.res == res && grid.size == t.size => {
            spread_back(t, &grid, &heights, &eroded, budget.reshaped, p, dt);
            t.version += 1;
            (budget, grid)
        }
        // The first update (or a new terrain): the coarse grid kept from before, or a new one.
        _ => (
            Budget::default(),
            match t.coarse.take() {
                Some(c) if c.res == res && c.size == t.size => c,
                _ => {
                    t.wet_cells.clear();
                    Box::new(t.resampled(res))
                }
            },
        ),
    };
    // Its ground is taken afresh from the fine grid, which creatures dig and pile; its water
    // stays as it was.
    {
        let fine = &*t;
        let points: Vec<(f32, f32)> = (0..res * res).map(|k| c.point(k)).collect();
        let sample = |src: &[f32]| points.par_iter().map(|&(x, z)| fine.sample(src, x, z).0).collect::<Vec<f32>>().into();
        c.heights = sample(&fine.heights);
        c.ground = sample(&fine.ground);
        c.eroded = sample(&fine.eroded);
        c.wear = sample(&fine.wear);
        c.springs = fine.springs_on(&c);
    }
    let (heights, eroded) = (c.heights.clone(), c.eroded.clone());
    let p = p.clone();
    let job = std::thread::Builder::new()
        .name("evolve-water".into())
        .spawn(move || {
            let budget = update_grid(&mut c, &p, rain, soak, dt, erode);
            Coarse { grid: c, budget, heights, eroded }
        })
        .expect("a thread for the water");
    t.water_job = Some(WaterJob(Arc::new(Mutex::new(Some(job)))));
    budget
}

/// The coarse grid's water being worked out on another thread (see `update`).
#[derive(Clone, Debug)]
pub struct WaterJob(Arc<Mutex<Option<std::thread::JoinHandle<Coarse>>>>);

/// A worked-out coarse grid, with its ground as it was before (for what erosion cut).
#[derive(Debug)]
pub struct Coarse {
    grid: Box<Terrain>,
    budget: Budget,
    heights: Layer,
    eroded: Layer,
}

/// Spreads a worked-out coarse grid `c` back onto the fine terrain `t`; `heights` and `eroded`
/// are `c`'s from before it was worked out.
fn spread_back(t: &mut Terrain, c: &Terrain, heights: &Layer, eroded: &Layer, reshaped: bool, p: &SimParams, dt: f32) {
    // Back onto the fine grid, only where something changed: spreading everything over all 25
    // million points of a 5000-a-side terrain took half a second every update. Water lies at the
    // level the coarse grid has it (a lake's surface is flat however bumpy its bed), in coarse
    // cells with water now or last time (to dry out where it's gone); erosion goes where it cut;
    // soil moisture and trampling, which change slowly, are brought up to date a stripe of rows at
    // a time.
    let n = t.res;
    let surface: Vec<f32> = c.heights.iter().zip(&c.water).map(|(h, w)| h + w).collect();
    let pool_surface: Vec<f32> = c.heights.iter().zip(&c.pooled).map(|(h, w)| h + w).collect();
    let dh: Vec<f32> = c.heights.iter().zip(heights.iter()).map(|(a, b)| a - b).collect();
    let de: Vec<f32> = c.eroded.iter().zip(eroded.iter()).map(|(a, b)| a - b).collect();
    let (cres, cells) = (c.res, c.res - 1);
    let corners = |layer: &[f32], ci: usize, cj: usize, f: &dyn Fn(f32) -> bool| {
        let k = ci * cres + cj;
        f(layer[k]) || f(layer[k + 1]) || f(layer[k + cres]) || f(layer[k + cres + 1])
    };
    // Wet: water deep enough to see (`VISIBLE`) or a pond at one of its corners. Rain leaves a
    // film of a few mm on most slopes, which nothing notices and which put a fifth of the
    // island to be written every update.
    let wet_now: Vec<bool> = (0..cells * cells)
        .map(|q| corners(&c.water, q / cells, q % cells, &|w| w > VISIBLE) || corners(&c.pooled, q / cells, q % cells, &|w| w > 0.0))
        .collect();
    let first = t.wet_cells.len() != wet_now.len();
    let write: Vec<bool> = wet_now.iter().enumerate().map(|(q, &w)| w || first || t.wet_cells[q]).collect();
    let cut: Vec<bool> = (0..cells * cells)
        .map(|q| reshaped && (corners(&dh, q / cells, q % cells, &|d| d != 0.0) || corners(&de, q / cells, q % cells, &|d| d != 0.0)))
        .collect();
    // A stripe of about 1.5 million points each update.
    let stripes = (n * n).div_ceil(1_500_000).max(1);
    t.stripe_age.resize(stripes, 0.0);
    for a in &mut t.stripe_age {
        *a += dt;
    }
    let stripe = t.stripe % stripes;
    t.stripe = t.stripe.wrapping_add(1);
    let rows = stripe * n / stripes..(stripe + 1) * n / stripes;
    let age = std::mem::take(&mut t.stripe_age[stripe]);
    let recover = (-age * std::f32::consts::LN_2 / (p.wear_recovery.max(0.01) * 60.0)).exp();
    let cr = c;
    let (cell, half, cc) = (t.cell(), t.half(), c.cell());
    let coarse_of = |v: f32| (((v + half) / cc).floor().max(0.0) as usize).min(cells - 1);
    // The fine columns in each coarse column: cols[cj]..cols[cj + 1].
    let mut cols = vec![n; cells + 1];
    for j in (0..n).rev() {
        cols[coarse_of(j as f32 * cell - half)] = j;
    }
    for cj in (0..cells).rev() {
        cols[cj] = cols[cj].min(cols[cj + 1]);
    }
    let Terrain { heights: th, eroded: te, water: tw, pooled: tp, moisture: tm, wear: twr, .. } = t;
    th.par_chunks_mut(n)
        .zip(te.par_chunks_mut(n))
        .zip(tw.par_chunks_mut(n))
        .zip(tp.par_chunks_mut(n))
        .zip(tm.par_chunks_mut(n))
        .zip(twr.par_chunks_mut(n))
        .enumerate()
        .for_each(|(i, (((((h, e), w), pl), m), wr))| {
            let ci = coarse_of(i as f32 * cell - half);
            let fz = (i as f32 * cell / cc - ci as f32).clamp(0.0, 1.0);
            let in_stripe = rows.contains(&i);
            for cj in 0..cells {
                let q = ci * cells + cj;
                let (wq, cq) = (write[q], cut[q]);
                if !(wq || cq || in_stripe) {
                    continue;
                }
                // Each fine point in a coarse cell is interpolated from that cell's corners.
                let k = ci * cres + cj;
                let corner = |l: &[f32]| [l[k], l[k + 1], l[k + cres], l[k + cres + 1]];
                let lerp = |v: [f32; 4], fx: f32| (v[0] + (v[1] - v[0]) * fx) * (1.0 - fz) + (v[2] + (v[3] - v[2]) * fx) * fz;
                let top = |v: [f32; 4]| v[0].max(v[1]).max(v[2]).max(v[3]);
                let (dhc, dec) = if cq { (corner(&dh), corner(&de)) } else { ([0.0; 4], [0.0; 4]) };
                let (wc, sc, pc, psc, mc) = (corner(&cr.water), corner(&surface), corner(&cr.pooled), corner(&pool_surface), corner(&cr.moisture));
                let (wet, deepest, pond) = (wet_now[q], top(wc), top(pc));
                for j in cols[cj]..cols[cj + 1] {
                    let fx = (j as f32 * cell / cc - cj as f32).clamp(0.0, 1.0);
                    if cq {
                        h[j] += lerp(dhc, fx);
                        e[j] += lerp(dec, fx);
                    }
                    if wq {
                        w[j] = if wet { (lerp(sc, fx) - h[j]).max(0.0).min(deepest) } else { 0.0 };
                        pl[j] = if pond > 0.0 { (lerp(psc, fx) - h[j]).max(0.0) } else { 0.0 };
                    }
                    if in_stripe {
                        m[j] = lerp(mc, fx);
                        wr[j] *= recover;
                    }
                }
            }
        });
    t.wet_cells = wet_now;
    if reshaped {
        // The physics ground follows where the wear has added up (see `EROSION_REBUILD`).
        let (tiles, cells) = (t.tiles(), t.tile_cells());
        t.erosion_drift.resize(tiles * tiles, 0.0);
        // This time's deepest wear in each piece, added up over the times (never less than the
        // deepest a point has worn since).
        let mut deepest = vec![0.0f32; tiles * tiles];
        for (k, &d) in dh.iter().enumerate() {
            if d != 0.0 {
                let (x, z) = c.point(k);
                let f = t.nearest(x, z);
                let tile = ((f / n) / cells).min(tiles - 1) * tiles + ((f % n) / cells).min(tiles - 1);
                deepest[tile] = deepest[tile].max(d.abs());
            }
        }
        for (k, drift) in t.erosion_drift.iter_mut().enumerate() {
            *drift += deepest[k];
            if *drift > crate::terrain::EROSION_REBUILD {
                *drift = 0.0;
                t.worn_tiles.push(k);
            }
        }
    }
}

fn update_grid(t: &mut Terrain, p: &SimParams, rain: f32, soak: f32, dt: f32, erode: Option<f32>) -> Budget {
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

    let mut budget = fill_lakes(t, p, &d, fall, &mut flow, dt);

    // Streams where enough flows, and the lakes.
    let wet_k = 1.0 - (-dt / WETTING_TIME).exp();
    let dry_k = 1.0 - (-dt / DRYING_TIME).exp();
    for k in 0..count {
        if t.heights[k] <= wl {
            t.water[k] = 0.0;
            t.moisture[k] = 1.0;
            continue;
        }
        let stream = (STREAM_DEPTH * flow[k].sqrt() - FILM).clamp(0.0, MAX_STREAM_DEPTH);
        let pool = t.pooled[k];
        t.water[k] = stream.max(if pool > FILM { pool } else { 0.0 });
        let target = if t.water[k] > 0.0 { 1.0 } else { rain.max(0.5 * soak) };
        let m = &mut t.moisture[k];
        *m += (target - *m) * if target > *m { wet_k } else { dry_k };
    }

    // Trampled ground grows back.
    let recover = (-dt * std::f32::consts::LN_2 / (p.wear_recovery.max(0.01) * 60.0)).exp();
    for w in &mut t.wear {
        *w *= recover;
    }

    if let Some(edt) = erode
        && p.erosion_rate > 0.0
    {
        budget.reshaped = erode_channels(t, p, &d, &flow, edt);
    }
    t.version += 1;
    budget
}

/// Points this much below their hollow's rim are in it.
const HOLLOW: f32 = 1e-4;

/// Lakes and ponds. A hollow (a basin) is a connected set of points below the level the
/// priority flood filled them to, all sharing that level: its rim, where it spills. Each update:
/// - the water already in each basin (`pooled`, summed) stays in it; water standing where the
///   ground has changed so there's no longer a hollow runs downhill to the next basin;
/// - the rain on everything draining into a basin (`fall`, in m² of hillside, times
///   `rain_rate`) is added, and evaporation over its wet area taken away;
/// - basins are filled from the highest rim down, so what spills over one reaches the next;
/// - each basin's water is levelled: one surface height whose depth over the basin's ground adds
///   up to its volume, capped at the rim.
///
/// A basin with two pits under its rim is levelled as one (both fill together), a fair likeness
/// at this scale.
fn fill_lakes(t: &mut Terrain, p: &SimParams, d: &Drainage, fall: f32, flow: &mut [f32], dt: f32) -> Budget {
    let (n, wl) = (t.res, p.water_level);
    let count = n * n;
    let area = t.cell() * t.cell();
    let h = &t.heights;
    let mut budget = Budget::default();

    // Basins: connected hollow points with the same rim level.
    let mut label = vec![NONE; count];
    let mut basins: Vec<Vec<u32>> = Vec::new();
    let in_hollow = |k: usize| d.filled[k] > h[k] + HOLLOW && h[k] > wl;
    let mut stack = Vec::new();
    for start in 0..count {
        if label[start] != NONE || !in_hollow(start) {
            continue;
        }
        let b = basins.len() as u32;
        let rim = d.filled[start];
        let mut cells = Vec::new();
        label[start] = b;
        stack.push(start);
        while let Some(k) = stack.pop() {
            cells.push(k as u32);
            let (i, j) = (k / n, k % n);
            for (di, dj) in NEIGHBOURS {
                let (ii, jj) = (i as isize + di, j as isize + dj);
                if ii < 0 || jj < 0 || ii >= n as isize || jj >= n as isize {
                    continue;
                }
                let m = ii as usize * n + jj as usize;
                if label[m] == NONE && in_hollow(m) && d.filled[m] == rim {
                    label[m] = b;
                    stack.push(m);
                }
            }
        }
        basins.push(cells);
    }

    // Where water at each point ends up: its own basin, the first basin downhill, or the sea.
    let mut down = vec![NONE; count];
    for &k in &d.order {
        let k = k as usize;
        down[k] = if label[k] != NONE {
            label[k]
        } else {
            let r = d.receiver[k];
            if r == NONE { NONE } else { down[r as usize] }
        };
    }
    // Where each basin spills: wherever water leaving it over the rim goes. (A rim point level
    // with the rim can pass water back in; that isn't where it spills.)
    let mut spill_to = vec![NONE; basins.len()];
    for (b, cells) in basins.iter().enumerate() {
        for &k in cells {
            let r = d.receiver[k as usize];
            if r != NONE && label[r as usize] != b as u32 && down[r as usize] != b as u32 {
                spill_to[b] = down[r as usize];
                break;
            }
        }
    }

    // Water already standing, and the rain gathered.
    let rate = p.rain_rate.max(0.0) / 1000.0 / 3600.0;
    let mut volume = vec![0.0f64; basins.len()];
    let mut wet_area = vec![0.0f32; basins.len()];
    for k in 0..count {
        let pool = t.pooled[k];
        let to = down[k];
        if pool > 0.0 {
            match (label[k] != NONE, to) {
                (true, b) => wet_area[b as usize] += area,
                (false, NONE) => budget.to_sea += pool * area,
                _ => {}
            }
            if to != NONE {
                volume[to as usize] += (pool * area) as f64;
            }
        }
        if to != NONE && h[k] > wl {
            let v = fall * rate * dt;
            volume[to as usize] += v as f64;
            budget.rained += v;
        }
    }
    let full: Vec<bool> = basins
        .iter()
        .map(|cells| {
            let rim = d.filled[cells[0] as usize];
            cells.iter().any(|&k| h[k as usize] + t.pooled[k as usize] >= rim - BRIM)
        })
        .collect();
    let cell = t.cell();
    let (fed, reached) = run_springs((&t.springs, &mut t.spring_reach), (n, cell), d, &label, &full, &mut volume, flow, dt);

    // Fill from the highest rim down, passing what spills over on. (Water spills to a rim no
    // higher, but one exactly as high may already be done: it's levelled again.)
    let mut by_rim: Vec<usize> = (0..basins.len()).collect();
    by_rim.sort_by(|&a, &b| d.filled[basins[b][0] as usize].total_cmp(&d.filled[basins[a][0] as usize]));
    let mut queue: std::collections::VecDeque<usize> = by_rim.into();
    let mut done = vec![false; basins.len()];
    let mut relevels = 0;
    let evap = p.evaporation.max(0.0) / 1000.0 / 3600.0 * dt;
    t.pooled.fill(0.0);
    let mut heights: Vec<f32> = Vec::new();
    while let Some(b) = queue.pop_front() {
        let cells = &basins[b];
        if !done[b] {
            let gone = ((evap * wet_area[b]) as f64).min(volume[b]);
            budget.evaporated += gone as f32;
            volume[b] -= gone;
            done[b] = true;
        }
        let vol = volume[b];
        let rim = d.filled[cells[0] as usize];
        heights.clear();
        heights.extend(cells.iter().map(|&k| h[k as usize]));
        heights.sort_by(f32::total_cmp);
        let capacity: f64 = heights.iter().map(|&x| ((rim - x) * area) as f64).sum();
        let level = if vol >= capacity {
            let over = vol - capacity;
            volume[b] = capacity;
            match spill_to[b] {
                NONE => budget.to_sea += over as f32,
                // Re-levelling is bounded, in case odd ground makes basins spill into each other.
                to if done[to as usize] && relevels >= basins.len() => budget.to_sea += over as f32,
                // What a spring's hollow spills soaks into the dry ground ahead of its stream
                // until the stream gets to the next hollow.
                to if fed[b] && !reached[to as usize] => budget.to_sea += over as f32,
                to => {
                    volume[to as usize] += over;
                    if done[to as usize] {
                        relevels += 1;
                        queue.push_back(to as usize);
                    }
                }
            }
            rim
        } else {
            // The surface over the lowest j points: (vol/area + their heights summed) / j.
            let per = vol / area as f64;
            let mut sum = 0.0f64;
            let mut level = rim;
            for (j, &x) in heights.iter().enumerate() {
                sum += x as f64;
                let l = (per + sum) / (j + 1) as f64;
                if heights.get(j + 1).is_none_or(|&next| l <= next as f64) {
                    level = (l as f32).min(rim);
                    break;
                }
            }
            level
        };
        for &k in cells {
            let k = k as usize;
            t.pooled[k] = (level - h[k]).max(0.0);
        }
    }
    budget
}

/// Each spring's water, followed down its path as far as it has got (`Terrain::spring_reach`):
/// it runs as a stream (added to `flow`, thinning toward its leading edge) into the first hollow
/// in its way and pours into it (`volume`); while that hollow isn't full it goes no further, and
/// once it is, on over the rim through whatever is downhill. Where nothing stops it, it creeps
/// on at `SPRING_SPEED`. Returns which hollows springs feed, and which their water has reached.
#[allow(clippy::too_many_arguments)]
fn run_springs(
    (springs, spring_reach): (&[u32], &mut std::collections::HashMap<u32, f32>),
    (n, cell): (usize, f32),
    d: &Drainage,
    label: &[u32],
    full: &[bool],
    volume: &mut [f64],
    flow: &mut [f32],
    dt: f32,
) -> (Vec<bool>, Vec<bool>) {
    let mut fed = vec![false; full.len()];
    let mut reached = vec![false; full.len()];
    spring_reach.retain(|k, _| springs.contains(k));
    for &s in springs {
        let reach = spring_reach.get(&s).copied().unwrap_or(0.0);
        let (mut k, mut dist, mut first) = (s as usize, 0.0f32, true);
        // Where something stops it (a hollow still filling, or the sea), if anything does.
        let mut stopped = None;
        while dist <= reach && k < label.len() {
            let b = label[k];
            if b != NONE {
                let b = b as usize;
                if first {
                    volume[b] += (SPRING_FILL * dt) as f64;
                    first = false;
                }
                fed[b] = true;
                reached[b] = true;
                if !full[b] {
                    stopped = Some(dist);
                    break;
                }
            } else {
                flow[k] += SPRING_FLOW * ((reach - dist) / SPRING_EDGE).clamp(0.15, 1.0);
            }
            let r = d.receiver[k];
            if r == NONE {
                stopped = Some(dist);
                break;
            }
            let r = r as usize;
            // (Across a full hollow it's at once over the far rim: its surface is level.)
            if b == NONE {
                dist += if r / n != k / n && r % n != k % n { cell * std::f32::consts::SQRT_2 } else { cell };
            }
            k = r;
        }
        let on = reach + SPRING_SPEED * dt;
        spring_reach.insert(s, stopped.map_or(on, |at: f32| on.min(at + cell)));
    }
    (fed, reached)
}

/// Cuts channels for `dt` seconds of the given flow, downstream first, each point toward its
/// (already lowered) receiver: h' = (h + F h_r') / (1 + F), F = K sqrt(flow) dt / distance, with
/// K set so 100 m² of flow at a 1-in-10 slope cuts `erosion_rate` m an hour.
fn erode_channels(t: &mut Terrain, p: &SimParams, d: &Drainage, flow: &[f32], dt: f32) -> bool {
    let (n, wl) = (t.res, p.water_level);
    let cell = t.cell();
    let steepest = p.slump_angle.min(89.0).to_radians().tan() * cell;
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
        // A bed can't be cut deeper below its banks than they'd stand (`slump_angle`): water
        // wears valleys, not slots a creature can't climb out of.
        let (i, j) = (k / n, k % n);
        let bank = [(i > 0).then(|| k - n), (i + 1 < n).then(|| k + n), (j > 0).then(|| k - 1), (j + 1 < n).then(|| k + 1)]
            .into_iter()
            .flatten()
            .map(|q| t.heights[q])
            .fold(h, f32::max);
        let cut = (h - (h + f * hr) / (1.0 + f)).min(room).min(h - floor).min(h - (bank - steepest)).max(0.0);
        if cut < 1e-6 {
            continue;
        }
        t.heights[k] -= cut;
        t.eroded[k] += cut;
        (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(j), j1.max(j));
    }
    if i0 == usize::MAX {
        return false;
    }
    t.mark_dirty(i0, i1, j0, j1);
    true
}
