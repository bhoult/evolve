use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use rapier3d::prelude::{Array2, ColliderBuilder, SharedShape};
use serde::{Deserialize, Serialize};

use crate::math::Vec3;
use crate::noise::Perlin;
use crate::params::SimParams;

/// Trampling (kg·m/m²) at which ground is about two-thirds worn: a 50 kg creature walking over
/// the same spot a few dozen times.
pub const WEAR_SCALE: f32 = 1200.0;

/// A square heightfield centered on the origin: an island in an ocean. Rows (i) run along z,
/// columns (j) along x, matching parry's heightfield layout, so `height_at` agrees with the
/// physics surface.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Terrain {
    pub size: f32,
    pub res: usize,
    /// Row-major: `heights[i * res + j]`.
    pub heights: Layer,
    /// Patchiness of plant growth in [0, 1], same layout as `heights`.
    pub fertility: Layer,
    /// The ground as the world made it, before creatures dug or piled earth; empty in saves from
    /// before earthworks (then it's the current ground).
    #[serde(default)]
    pub ground: Layer,
    /// How far running water has worn the ground down from `ground`, m. Kept apart from what
    /// creatures dug, so a stream bed isn't counted as their earthworks.
    #[serde(default)]
    pub eroded: Layer,
    /// Trampling: body mass times distance moved over each grid point, per m², fading as the
    /// ground recovers. See `worn`.
    #[serde(default)]
    pub wear: Layer,
    /// Standing or running water on the ground (streams, lakes, ponds, puddles), m deep: the
    /// deeper of `pooled` and the stream running there.
    #[serde(default)]
    pub water: Layer,
    /// Water held in hollows (lakes, ponds, puddles), m deep. Unlike streams, which follow the
    /// rain, it stays until it evaporates or spills over (see `hydro`).
    #[serde(default)]
    pub pooled: Layer,
    /// How wet the soil is, 0 (dry) to 1 (soaked).
    #[serde(default)]
    pub moisture: Layer,
    /// Eternal springs, by grid point: each pours out water forever (see `hydro`).
    #[serde(default)]
    pub springs: Vec<u32>,
    /// How far down its path each spring's water has got, m (by grid point; see `hydro`).
    #[serde(default)]
    pub spring_reach: std::collections::HashMap<u32, f32>,
    /// Patches of the grid (rows i0..=i1, columns j0..=j1, inclusive) changed since the last
    /// `take_dirty`, kept apart unless they nearly touch: one box around them all made creatures
    /// digging at opposite ends of a big island settle and rebuild the whole island.
    #[serde(skip)]
    pub dirty: Vec<(usize, usize, usize, usize)>,
    /// Bumped whenever the shape changes so renderers know to rebuild.
    pub version: u64,
    /// On a terrain finer than `hydro::HYDRO_CELL`, the grid the water is worked out on (its
    /// lakes stay there between updates; see `hydro::update`). Rebuilt from this one when lost.
    #[serde(skip)]
    pub coarse: Option<Box<Terrain>>,
    /// On a big terrain, how far erosion has worn each piece of the physics ground (row by row,
    /// at its deepest, m) since it was last rebuilt, and the pieces to rebuild for it.
    #[serde(skip)]
    pub erosion_drift: Vec<f32>,
    /// Where the last `slump` left earth still sliding, to go on with next time.
    #[serde(skip)]
    pub sliding: Vec<(usize, usize, usize, usize)>,
    /// Seconds each block of `REST_BLOCK` grid cells a side has gone undisturbed (dug, piled,
    /// slid, raised...), row by row: only ground left alone `earth_rest` heals.
    #[serde(default)]
    pub quiet: Vec<f32>,
    /// For spreading the coarse grid's water back (see `hydro::update`): which of its cells had
    /// water last time, the stripe of rows whose soil moisture and trampling are brought up to
    /// date next, and how long since each stripe last was.
    #[serde(skip)]
    pub wet_cells: Vec<bool>,
    #[serde(skip)]
    pub stripe: usize,
    #[serde(skip)]
    pub stripe_age: Vec<f32>,
    /// The coarse grid's water being worked out on another thread, to be spread back at the next
    /// update (see `hydro::update`).
    #[serde(skip)]
    pub water_job: Option<crate::hydro::WaterJob>,
    /// The physics ground samples every this many grid points (see `set_collision_cell`); 0 or
    /// 1 is every one.
    #[serde(skip)]
    pub collision_stride: usize,
    #[serde(skip)]
    pub worn_tiles: Vec<usize>,
}

/// Cells along a side of each piece the physics ground is split into, on a terrain wider than
/// `ONE_PIECE` cells. Reshaping the ground (digging, slumps, erosion) rebuilds only the pieces it
/// touched: rebuilding one heightfield of 25 million points in every physics world, up to twice
/// a simulated second, stalled big worlds, and pieces of 512 still took tens of ms each.
pub const GROUND_TILE: usize = 128;
/// Grid cells along a side of each block whose quiet time decides whether it heals (see
/// `Terrain::quiet`).
pub const REST_BLOCK: usize = 16;

/// The widest terrain (in cells) whose physics ground stays in one piece, as it always was.
pub const ONE_PIECE: usize = 512;
/// How far erosion may wear a piece of a big terrain's ground (m, at its deepest) before the
/// physics ground there is rebuilt: physics needs no millimetres, and rebuilding every piece
/// with a channel in it every 30 s stalled the world.
pub const EROSION_REBUILD: f32 = 0.05;

/// One layer of the grid (heights, water...), shared between copies of the terrain until one of
/// them changes it, when that one gets its own. The screen takes a copy of the terrain several
/// times a second, and copying every layer of a big one each time (800 MB at 5000 a side) paused
/// the world.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layer(Arc<Vec<f32>>);

impl Layer {
    /// Whether two copies still share this layer (neither has changed it since).
    pub fn same(&self, other: &Layer) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for Layer {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        &self.0
    }
}

impl DerefMut for Layer {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        Arc::make_mut(&mut self.0)
    }
}

impl<'a> IntoIterator for &'a Layer {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Layer {
    type Item = &'a mut f32;
    type IntoIter = std::slice::IterMut<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        Arc::make_mut(&mut self.0).iter_mut()
    }
}

impl From<Vec<f32>> for Layer {
    fn from(v: Vec<f32>) -> Self {
        Layer(Arc::new(v))
    }
}

impl Serialize for Layer {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.as_slice().serialize(s)
    }
}

impl<'de> Deserialize<'de> for Layer {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Vec::<f32>::deserialize(d).map(Layer::from)
    }
}

/// A placeholder 2x2 flat terrain, used before the first real snapshot arrives.
impl Default for Terrain {
    fn default() -> Self {
        Terrain {
            size: 1.0,
            res: 2,
            heights: vec![0.0; 4].into(),
            fertility: vec![0.0; 4].into(),
            ground: vec![0.0; 4].into(),
            eroded: vec![0.0; 4].into(),
            wear: vec![0.0; 4].into(),
            water: vec![0.0; 4].into(),
            pooled: vec![0.0; 4].into(),
            moisture: vec![0.0; 4].into(),
            springs: Vec::new(),
            spring_reach: Default::default(),
            dirty: Vec::new(),
            version: 0,
            coarse: None,
            erosion_drift: Vec::new(),
            worn_tiles: Vec::new(),
            sliding: Vec::new(),
            quiet: Vec::new(),
            wet_cells: Vec::new(),
            stripe: 0,
            stripe_age: Vec::new(),
            water_job: None,
            collision_stride: 1,
        }
    }
}

impl Terrain {
    pub fn generate(p: &SimParams, version: u64) -> Self {
        let res = p.terrain_resolution.clamp(9, 5000) as usize;
        let size = p.world_size.max(10.0);
        let perlin = Perlin::new(p.seed);
        let fert_noise = Perlin::new(p.seed ^ 0xF00D);
        let coast_noise = Perlin::new(p.seed ^ 0xC0A5);
        let feature = p.terrain_feature_size.max(1.0);
        let th = p.terrain_height;
        let wl = p.water_level;
        let radius = size * 0.5 * p.island_size.max(0.05);
        // The sea floor just past the beach, sloping deeper toward the edges.
        let shelf = wl - (th * 0.4).max(5.0);
        let mut heights = vec![0.0; res * res];
        let mut fertility = vec![0.0; res * res];
        for i in 0..res {
            for j in 0..res {
                let x = (j as f32 / (res - 1) as f32 - 0.5) * size;
                let z = (i as f32 / (res - 1) as f32 - 0.5) * size;
                let n = perlin.fbm(x / feature, z / feature, 5, p.terrain_roughness);
                let t = (n * 0.9 + 0.5).clamp(0.0, 1.0);
                // Distance from the middle in island radii. Hills push the coast out into
                // headlands and low ground lets the sea in, so the shore follows the land.
                let wobble = coast_noise.fbm(x / (radius * 0.9), z / (radius * 0.9), 3, 0.5);
                let r = (x * x + z * z).sqrt() / radius - wobble * p.coast_roughness * 3.2 - (t - 0.5) * 0.4;
                // Land sits a little above the water so lakes are rare; a gentle dome gives the
                // island a middle.
                let dome = (1.0 - r * r).max(0.0) * th * 0.12;
                let land = wl - 1.5 + t.powf(1.6) * th * 0.9 + dome;
                let sea = shelf - (r - 1.0).max(0.0) * th * 0.6;
                let inland = 1.0 - smoothstep(0.62, 1.0, r);
                heights[i * res + j] = sea + (land - sea) * inland;
                let f = fert_noise.fbm(x / 25.0, z / 25.0, 3, 0.5);
                fertility[i * res + j] = (f * 1.2 + 0.6).clamp(0.05, 1.0);
            }
        }
        let n = res * res;
        Self {
            size,
            res,
            ground: heights.clone().into(),
            heights: heights.into(),
            fertility: fertility.into(),
            eroded: vec![0.0; n].into(),
            wear: vec![0.0; n].into(),
            water: vec![0.0; n].into(),
            pooled: vec![0.0; n].into(),
            moisture: vec![0.0; n].into(),
            springs: Vec::new(),
            spring_reach: Default::default(),
            version,
            dirty: Vec::new(),
            coarse: None,
            erosion_drift: Vec::new(),
            worn_tiles: Vec::new(),
            sliding: Vec::new(),
            quiet: Vec::new(),
            wet_cells: Vec::new(),
            stripe: 0,
            stripe_age: Vec::new(),
            water_job: None,
            collision_stride: 1,
        }
    }

    /// Fills in layers missing from saves made before they existed.
    pub fn ensure_layers(&mut self) {
        let n = self.heights.len();
        if self.ground.len() != n {
            self.ground = self.heights.clone();
        }
        for layer in [&mut self.eroded, &mut self.wear, &mut self.water, &mut self.pooled, &mut self.moisture] {
            if layer.len() != n {
                *layer = vec![0.0; n].into();
            }
        }
    }

    /// The ground as nature has it now: as the world made it, less what water has worn away.
    /// Creatures dig and pile relative to this.
    pub fn natural(&self, k: usize) -> f32 {
        self.ground.get(k).map_or(self.heights[k], |g| g - self.eroded.get(k).copied().unwrap_or(0.0))
    }

    /// How trampled a grid point looks and feels, 0 (untouched) to 1 (a bare path).
    pub fn worn(&self, k: usize) -> f32 {
        1.0 - (-self.wear.get(k).copied().unwrap_or(0.0) / WEAR_SCALE).exp()
    }

    /// A layer of this grid (`heights`, `water`...) at a spot, interpolated over its cell, and the
    /// largest of the cell's four corners.
    pub fn sample(&self, layer: &[f32], x: f32, z: f32) -> (f32, f32) {
        let (i, j, fx, fz) = self.cell_coords(x, z);
        let r = self.res;
        let (a, b) = (layer[i * r + j], layer[i * r + j + 1]);
        let (c, d) = (layer[(i + 1) * r + j], layer[(i + 1) * r + j + 1]);
        let v = (a * (1.0 - fx) + b * fx) * (1.0 - fz) + (c * (1.0 - fx) + d * fx) * fz;
        (v, a.max(b).max(c).max(d))
    }

    /// Where grid point `k` is, (x, z).
    pub fn point(&self, k: usize) -> (f32, f32) {
        let c = self.cell();
        ((k % self.res) as f32 * c - self.half(), (k / self.res) as f32 * c - self.half())
    }

    /// The grid point nearest a spot.
    pub fn nearest(&self, x: f32, z: f32) -> usize {
        let n = (self.res - 1) as f32;
        let j = ((x / self.size + 0.5) * n).round().clamp(0.0, n) as usize;
        let i = ((z / self.size + 0.5) * n).round().clamp(0.0, n) as usize;
        i * self.res + j
    }

    /// This terrain sampled onto a coarser grid of `res` a side over the same ground: every layer
    /// but fertility, and the springs at their nearest points.
    pub fn resampled(&self, res: usize) -> Terrain {
        let mut t = Terrain { size: self.size, res, ..Terrain::default() };
        let n = res * res;
        let points: Vec<(f32, f32)> = (0..n).map(|k| t.point(k)).collect();
        let layer = |src: &[f32]| points.iter().map(|&(x, z)| self.sample(src, x, z).0).collect::<Vec<f32>>().into();
        t.heights = layer(&self.heights);
        t.ground = layer(&self.ground);
        t.eroded = layer(&self.eroded);
        t.wear = layer(&self.wear);
        t.water = layer(&self.water);
        t.pooled = layer(&self.pooled);
        t.moisture = layer(&self.moisture);
        t.fertility = vec![0.0; n].into();
        t.springs = self.springs_on(&t);
        t
    }

    /// This grid's springs at their nearest points of another grid over the same ground.
    pub fn springs_on(&self, other: &Terrain) -> Vec<u32> {
        let mut s: Vec<u32> = self
            .springs
            .iter()
            .map(|&k| {
                let (x, z) = self.point(k as usize);
                other.nearest(x, z) as u32
            })
            .collect();
        s.sort_unstable();
        s.dedup();
        s
    }

    /// How close a spot is to a stream, pond or lake (not the sea): 1 at the water's edge, down
    /// to 0 at `reach` m or more. Looks about every metre, so it costs the same on any grid.
    pub fn near_water(&self, x: f32, z: f32, reach: f32) -> f32 {
        if self.water.len() != self.heights.len() || reach <= 0.0 {
            return 0.0;
        }
        let (cell, n) = (self.cell(), self.res as isize);
        let step = (1.0 / cell).ceil().max(1.0) as usize;
        let r = (reach / cell).ceil() as isize;
        let k = self.nearest(x, z) as isize;
        let (i, j) = (k / n, k % n);
        let mut best = f32::MAX;
        for di in (-r..=r).step_by(step) {
            for dj in (-r..=r).step_by(step) {
                let (ii, jj) = (i + di, j + dj);
                if ii < 0 || jj < 0 || ii >= n || jj >= n || self.water[(ii * n + jj) as usize] <= 0.03 {
                    continue;
                }
                best = best.min((di * di + dj * dj) as f32);
            }
        }
        (1.0 - best.sqrt() * cell / reach).max(0.0)
    }

    /// Water on the ground at a spot, m deep (streams, ponds, puddles; not the sea).
    pub fn water_at(&self, x: f32, z: f32) -> f32 {
        if self.water.len() != self.heights.len() {
            return 0.0;
        }
        let (i, j, fx, fz) = self.cell_coords(x, z);
        let r = self.res;
        let w = &self.water;
        let a = w[i * r + j] * (1.0 - fx) + w[i * r + j + 1] * fx;
        let b = w[(i + 1) * r + j] * (1.0 - fx) + w[(i + 1) * r + j + 1] * fx;
        a * (1.0 - fz) + b * fz
    }

    /// Height of the water surface over a spot: the sea, or a lake, pond or stream standing
    /// higher than it.
    pub fn water_surface(&self, x: f32, z: f32, sea: f32) -> f32 {
        let w = self.water_at(x, z);
        if w > 0.02 { sea.max(self.height_at(x, z) + w) } else { sea }
    }

    /// The grid point nearest a spot.
    pub fn nearest_point(&self, x: f32, z: f32) -> usize {
        let (i, j, fx, fz) = self.cell_coords(x, z);
        let i = if fz > 0.5 { i + 1 } else { i };
        let j = if fx > 0.5 { j + 1 } else { j };
        i * self.res + j
    }

    /// Moves earth at a spot: `volume` m³, positive to pile it up, negative to dig it out, spread
    /// over the four grid points around the spot. Ground can be dug at most `max_dig` m below and
    /// piled at most `max_pile` m above where the world made it. Returns the volume actually moved
    /// (with the sign of the request).
    pub fn reshape(&mut self, x: f32, z: f32, volume: f32, max_dig: f32, max_pile: f32) -> f32 {
        self.ensure_layers();
        let (i, j, fx, fz) = self.cell_coords(x, z);
        let r = self.res;
        let area = self.cell() * self.cell();
        let corners = [
            (i, j, (1.0 - fx) * (1.0 - fz)),
            (i, j + 1, fx * (1.0 - fz)),
            (i + 1, j, (1.0 - fx) * fz),
            (i + 1, j + 1, fx * fz),
        ];
        let mut moved = 0.0;
        for (ci, cj, w) in corners {
            let k = ci * r + cj;
            let natural = self.natural(k);
            let (lo, hi) = (natural - max_dig, natural + max_pile);
            let (now, want) = (self.heights[k], volume * w / area);
            // Only ever moves the way asked: ground already past a limit (the limits are live
            // params) stays put rather than snapping back.
            let h = if want < 0.0 { (now + want).max(lo.min(now)) } else { (now + want).min(hi.max(now)) };
            moved += (h - self.heights[k]) * area;
            self.heights[k] = h;
        }
        if moved != 0.0 {
            self.mark_dirty(i, i + 1, j, j + 1);
        }
        moved
    }

    /// Loose earth slides: wherever two neighboring grid points differ by more than a slope of
    /// `angle` degrees and at least one of them was dug, piled or torn up, earth moves from the
    /// higher to the lower until the slope holds (within the dig and pile limits). Holes become
    /// bowls and heaps become mounds, while the steep places the world was made with stay as
    /// they are. Works outward from the region changed since the last `take_dirty`, at most
    /// `passes` sweeps, and marks what it moved as changed. Returns whether it settled.
    pub fn slump(&mut self, angle: f32, max_dig: f32, max_pile: f32, passes: usize) -> bool {
        if angle >= 90.0 || self.res < 2 {
            return true;
        }
        self.sliding.clear();
        let mut settled = true;
        for patch in self.dirty.clone() {
            settled &= self.slump_patch(patch, angle, max_dig, max_pile, passes);
        }
        settled
    }

    /// `slump` over one changed patch and whatever slides beyond it.
    fn slump_patch(&mut self, (i0, i1, j0, j1): (usize, usize, usize, usize), angle: f32, max_dig: f32, max_pile: f32, passes: usize) -> bool {
        self.ensure_layers();
        let (n, cell) = (self.res, self.cell());
        let lim = angle.to_radians().tan() * cell;
        let grow = |(a, b, c, d): (usize, usize, usize, usize)| (a.saturating_sub(1), (b + 1).min(n - 1), c.saturating_sub(1), (d + 1).min(n - 1));
        let mut region = grow((i0, i1, j0, j1));
        let mut moved_any: Option<(usize, usize, usize, usize)> = None;
        for _ in 0..passes {
            let mut moved: Option<(usize, usize, usize, usize)> = None;
            let (a, b, c, d) = region;
            for i in a..=b {
                for j in c..=d {
                    let k = i * n + j;
                    for (k2, i2, j2) in [(k + 1, i, j + 1), (k + n, i + 1, j)] {
                        if i2 >= n || j2 >= n {
                            continue;
                        }
                        let diff = self.heights[k] - self.heights[k2];
                        if diff.abs() <= lim {
                            continue;
                        }
                        let (hi, lo) = if diff > 0.0 { (k, k2) } else { (k2, k) };
                        // Only loose earth moves: piled earth sliding off, or the sides of a hole
                        // falling in (as far as it needs filling). Untouched steep ground stays,
                        // and a slide can't set off more than was dug or piled.
                        let loose = (self.heights[hi] - self.natural(hi)).max(0.0) + (self.natural(lo) - self.heights[lo]).max(0.0);
                        let fall = ((diff.abs() - lim) * 0.5)
                            .min(loose)
                            .min(self.heights[hi] - (self.natural(hi) - max_dig))
                            .min(self.natural(lo) + max_pile - self.heights[lo]);
                        if fall <= 1e-5 {
                            continue;
                        }
                        self.heights[hi] -= fall;
                        self.heights[lo] += fall;
                        let (r0, r1, c0, c1) = moved.unwrap_or((i, i, j, j));
                        moved = Some((r0.min(i), r1.max(i2), c0.min(j), c1.max(j2)));
                    }
                }
            }
            let Some(m) = moved else {
                if let Some((a, b, c, d)) = moved_any {
                    self.mark_dirty(a, b, c, d);
                }
                return true;
            };
            moved_any = Some(match moved_any {
                Some((a, b, c, d)) => (a.min(m.0), b.max(m.1), c.min(m.2), d.max(m.3)),
                None => m,
            });
            region = grow(m);
        }
        if let Some((a, b, c, d)) = moved_any {
            self.mark_dirty(a, b, c, d);
            self.sliding.push((a, b, c, d));
        }
        false
    }

    /// Dug and piled ground left alone at least `rest` seconds weathers back toward the land as
    /// the world made it (less what water has worn away), halfway in `half_life` seconds.
    /// Returns whether anything changed.
    pub fn heal(&mut self, dt: f32, half_life: f32, rest: f32) -> bool {
        if self.ground.len() != self.heights.len() {
            return false;
        }
        use rayon::prelude::*;
        self.ensure_quiet();
        for q in &mut self.quiet {
            *q = (*q + dt).min(f32::MAX);
        }
        let keep = (-dt * std::f32::consts::LN_2 / half_life.max(1.0)).exp();
        let (n, tiles, cells, blocks) = (self.res, self.tiles(), self.tile_cells(), self.rest_blocks());
        // Row by row on every core: each row's changed columns and its biggest change.
        let (ground, eroded, quiet) = (&self.ground, &self.eroded, &self.quiet);
        let rows: Vec<Option<(usize, usize, f32)>> = self
            .heights
            .par_chunks_mut(n)
            .enumerate()
            .map(|(i, row)| {
                let mut span: Option<(usize, usize, f32)> = None;
                let resting = &quiet[(i / REST_BLOCK) * blocks..(i / REST_BLOCK + 1) * blocks];
                for (j, h) in row.iter_mut().enumerate() {
                    let k = i * n + j;
                    let natural = ground[k] - eroded.get(k).copied().unwrap_or(0.0);
                    let off = *h - natural;
                    if off == 0.0 || resting[j / REST_BLOCK] < rest {
                        continue;
                    }
                    // The last millimeters go at once, so healed ground stops counting as dug.
                    let new = if off.abs() * keep < 0.002 { natural } else { natural + off * keep };
                    let moved = (*h - new).abs();
                    *h = new;
                    span = Some(span.map_or((j, j, moved), |(a, b, d)| (a.min(j), b.max(j), d.max(moved))));
                }
                span
            })
            .collect();
        if rows.iter().all(Option::is_none) {
            return false;
        }
        if tiles > 1 {
            // On a big terrain healing is spread over the whole island: like erosion, it rebuilds
            // a piece of the physics ground once it has moved it enough (`EROSION_REBUILD`), and
            // doesn't set slopes settling (one box around everything healed did, over 25 million
            // points, for most of a second).
            let mut deepest = vec![0.0f32; tiles * tiles];
            for (i, r) in rows.iter().enumerate() {
                if let Some((a, b, d)) = *r {
                    let ti = (i / cells).min(tiles - 1);
                    for tj in (a / cells).min(tiles - 1)..=(b / cells).min(tiles - 1) {
                        deepest[ti * tiles + tj] = deepest[ti * tiles + tj].max(d);
                    }
                }
            }
            self.erosion_drift.resize(tiles * tiles, 0.0);
            for (k, drift) in self.erosion_drift.iter_mut().enumerate() {
                *drift += deepest[k];
                if *drift > EROSION_REBUILD {
                    *drift = 0.0;
                    self.worn_tiles.push(k);
                }
            }
            self.version += 1;
            return true;
        }
        let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0, usize::MAX, 0);
        for (i, r) in rows.iter().enumerate() {
            if let Some((a, b, _)) = *r {
                (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(a), j1.max(b));
            }
        }
        self.mark_reshaped(i0, i1, j0, j1);
        true
    }

    /// Raises the ground around (x, z) by `profile(dx, dz)` m at each grid point within `reach`
    /// (negative lowers it). With `remake`, the world itself is reshaped: the ground "as the
    /// world made it" moves with it, so it looks undisturbed and creatures dig relative to the
    /// new land. Without, it's torn-up earth (bare soil, like what creatures dig). Returns the
    /// volume added, m³.
    pub fn lift(&mut self, x: f32, z: f32, reach: f32, remake: bool, profile: impl Fn(f32, f32) -> f32) -> f32 {
        self.ensure_layers();
        let (n, cell) = (self.res, self.cell());
        let col = |v: f32| ((v / self.size + 0.5) * (n - 1) as f32).round() as isize;
        let span = (reach / cell).ceil() as isize;
        let (ci, cj) = (col(z), col(x));
        let (i0, i1) = ((ci - span).max(0) as usize, ((ci + span) as usize).min(n - 1));
        let (j0, j1) = ((cj - span).max(0) as usize, ((cj + span) as usize).min(n - 1));
        let at = |k: usize| (k as f32 / (n - 1) as f32 - 0.5) * self.size;
        let mut added = 0.0;
        for i in i0..=i1 {
            for j in j0..=j1 {
                let (dx, dz) = (at(j) - x, at(i) - z);
                if dx.hypot(dz) > reach {
                    continue;
                }
                let dh = profile(dx, dz);
                if dh == 0.0 {
                    continue;
                }
                let k = i * n + j;
                let h = (self.heights[k] + dh).clamp(-60.0, 150.0);
                let dh = h - self.heights[k];
                self.heights[k] = h;
                if remake {
                    self.ground[k] += dh;
                }
                added += dh * cell * cell;
            }
        }
        self.mark_dirty(i0, i1, j0, j1);
        added
    }

    /// How far creatures have moved the ground at a grid point, m (positive: piled).
    pub fn disturbance(&self, k: usize) -> f32 {
        if self.ground.len() != self.heights.len() {
            return 0.0;
        }
        self.heights[k] - self.natural(k)
    }

    /// Marks grid rows i0..=i1 and columns j0..=j1 as reshaped.
    pub fn mark_dirty(&mut self, i0: usize, i1: usize, j0: usize, j1: usize) {
        self.disturbed(i0, i1, j0, j1);
        self.mark_reshaped(i0, i1, j0, j1);
    }

    /// Blocks of `REST_BLOCK` cells along each side of the grid.
    fn rest_blocks(&self) -> usize {
        self.res.div_ceil(REST_BLOCK)
    }

    /// Fills in quiet times missing from saves made before they were kept: long left alone.
    fn ensure_quiet(&mut self) {
        let b = self.rest_blocks();
        if self.quiet.len() != b * b {
            self.quiet = vec![f32::MAX; b * b];
        }
    }

    /// The ground in rows i0..=i1, columns j0..=j1 was disturbed just now: it rests again before
    /// it heals.
    fn disturbed(&mut self, i0: usize, i1: usize, j0: usize, j1: usize) {
        self.ensure_quiet();
        let b = self.rest_blocks();
        for bi in i0 / REST_BLOCK..=(i1 / REST_BLOCK).min(b - 1) {
            self.quiet[bi * b + j0 / REST_BLOCK..=bi * b + (j1 / REST_BLOCK).min(b - 1)].fill(0.0);
        }
    }

    /// Marks rows i0..=i1 and columns j0..=j1 as reshaped, for slopes to settle and the physics
    /// ground to follow, without counting as a disturbance (healing reshapes ground too).
    fn mark_reshaped(&mut self, i0: usize, i1: usize, j0: usize, j1: usize) {
        // Patches this close (in cells) or overlapping become one.
        const NEAR: usize = 8;
        let mut r = (i0, i1, j0, j1);
        loop {
            let before = self.dirty.len();
            self.dirty.retain(|&(a, b, c, d)| {
                let touch = a <= r.1 + NEAR && r.0 <= b + NEAR && c <= r.3 + NEAR && r.2 <= d + NEAR;
                if touch {
                    r = (r.0.min(a), r.1.max(b), r.2.min(c), r.3.max(d));
                }
                !touch
            });
            if self.dirty.len() == before {
                break;
            }
        }
        self.dirty.push(r);
        if self.dirty.len() > 4096 {
            // Too many to keep apart: merge those in the same 64-cell block. (Merging into bigger
            // blocks, with hundreds of creatures digging, had slopes settle over millions of
            // points twice a second.)
            let mut blocks: std::collections::BTreeMap<(usize, usize), (usize, usize, usize, usize)> = Default::default();
            for x in self.dirty.drain(..) {
                blocks
                    .entry((x.0 / 64, x.2 / 64))
                    .and_modify(|y| *y = (x.0.min(y.0), x.1.max(y.1), x.2.min(y.2), x.3.max(y.3)))
                    .or_insert(x);
            }
            self.dirty.extend(blocks.into_values());
        }
        self.version += 1;
    }

    /// The patches changed since the last call.
    pub fn take_dirty(&mut self) -> Vec<(usize, usize, usize, usize)> {
        std::mem::take(&mut self.dirty)
    }

    pub fn cell(&self) -> f32 {
        self.size / (self.res - 1) as f32
    }

    pub fn half(&self) -> f32 {
        self.size * 0.5
    }

    pub fn collider(&self) -> ColliderBuilder {
        ColliderBuilder::new(self.shape())
    }

    /// The physics surface: a heightfield over the grid.
    pub fn shape(&self) -> SharedShape {
        let res = self.res;
        let h = &self.heights;
        let arr = Array2::from_fn(res, res, |i, j| h[i * res + j]);
        SharedShape::heightfield(arr, Vec3::new(self.size, 1.0, self.size))
    }

    /// Cells along a side of each piece of the physics ground (see `GROUND_TILE`).
    pub fn tile_cells(&self) -> usize {
        let stride = self.stride();
        if (self.res - 1).div_ceil(stride) <= ONE_PIECE { self.res - 1 } else { GROUND_TILE * stride }
    }

    /// How many grid points apart the physics ground's samples are.
    pub fn stride(&self) -> usize {
        self.collision_stride.max(1)
    }

    /// Spaces the physics ground's samples `cell` m apart, or as close as the grid allows (every
    /// point, at `cell` 0 or below the grid's spacing): a part resting on ground sampled every
    /// 0.16 m has many more triangles under it to collide with than one on ground every 0.5 m,
    /// for detail no body can feel. The drawn ground keeps every point.
    pub fn set_collision_cell(&mut self, cell: f32) {
        self.collision_stride = if cell > 0.0 { (cell / self.cell()).ceil().max(1.0) as usize } else { 1 };
    }

    /// Height of the physics ground at a spot: the grid's own (`height_at`) unless the physics
    /// ground is sampled coarser (`set_collision_cell`), when it's that surface's.
    pub fn collision_height_at(&self, x: f32, z: f32) -> f32 {
        let stride = self.stride();
        if stride == 1 {
            return self.height_at(x, z);
        }
        let (n, cells, last) = (self.res - 1, self.tile_cells(), self.tiles() - 1);
        let c = self.cell();
        let gx = ((x + self.half()) / c).clamp(0.0, n as f32);
        let gz = ((z + self.half()) / c).clamp(0.0, n as f32);
        let (ti, tj) = (((gz as usize) / cells).min(last), ((gx as usize) / cells).min(last));
        let (i0, j0) = (ti * cells, tj * cells);
        let (i1, j1) = ((i0 + cells).min(n), (j0 + cells).min(n));
        let (mi, mj) = ((i1 - i0).div_ceil(stride), (j1 - j0).div_ceil(stride));
        // Where in the piece's own grid of samples, as `tile_shape` lays them out.
        let u = (gx - j0 as f32) / (j1 - j0) as f32 * mj as f32;
        let v = (gz - i0 as f32) / (i1 - i0) as f32 * mi as f32;
        let (b, a) = ((u.floor() as usize).min(mj - 1), (v.floor() as usize).min(mi - 1));
        let (fx, fz) = (u - b as f32, v - a as f32);
        let at = |a: usize, b: usize| {
            self.height_at(j0 as f32 * c - self.half() + (j1 - j0) as f32 * c * b as f32 / mj as f32, i0 as f32 * c - self.half() + (i1 - i0) as f32 * c * a as f32 / mi as f32)
        };
        let (h00, h01, h10, h11) = (at(a, b), at(a, b + 1), at(a + 1, b), at(a + 1, b + 1));
        if fx + fz <= 1.0 {
            h00 + fx * (h01 - h00) + fz * (h10 - h00)
        } else {
            h11 + (1.0 - fx) * (h10 - h11) + (1.0 - fz) * (h01 - h11)
        }
    }

    /// How many pieces the physics ground is split into along each side.
    pub fn tiles(&self) -> usize {
        (self.res - 1).div_ceil(self.tile_cells())
    }

    /// Piece (ti, tj) of the physics surface: a heightfield over grid rows ti*cells to
    /// (ti+1)*cells (`tile_cells`) and the same columns (neighbours share their edge), sampled
    /// every `stride` points, and where its center sits. A terrain of one piece sampled at every
    /// point gets exactly `shape()`.
    pub fn tile_shape(&self, ti: usize, tj: usize) -> (Vec3, SharedShape) {
        let (n, cells) = (self.res - 1, self.tile_cells());
        let (i0, j0) = (ti * cells, tj * cells);
        let (i1, j1) = ((i0 + cells).min(n), (j0 + cells).min(n));
        let (res, h, c) = (self.res, &self.heights, self.cell());
        let (w, d) = ((j1 - j0) as f32 * c, (i1 - i0) as f32 * c);
        let center = Vec3::new(j0 as f32 * c - self.half() + w * 0.5, 0.0, i0 as f32 * c - self.half() + d * 0.5);
        let stride = self.stride();
        let arr = if stride == 1 {
            Array2::from_fn(i1 - i0 + 1, j1 - j0 + 1, |a, b| h[(i0 + a) * res + j0 + b])
        } else {
            let (mi, mj) = ((i1 - i0).div_ceil(stride), (j1 - j0).div_ceil(stride));
            let (x0, z0) = (center.x - w * 0.5, center.z - d * 0.5);
            Array2::from_fn(mi + 1, mj + 1, |a, b| self.height_at(x0 + w * b as f32 / mj as f32, z0 + d * a as f32 / mi as f32))
        };
        (center, SharedShape::heightfield(arr, Vec3::new(w, 1.0, d)))
    }

    /// The pieces of the physics ground that grid rows i0..=i1 and columns j0..=j1 lie in.
    pub fn tiles_over(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> impl Iterator<Item = (usize, usize)> {
        let (last, cells) = (self.tiles() - 1, self.tile_cells());
        let span = move |a: usize, b: usize| (a.saturating_sub(1) / cells).min(last)..=(b / cells).min(last);
        let cols = span(j0, j1);
        span(i0, i1).flat_map(move |ti| cols.clone().map(move |tj| (ti, tj)))
    }

    fn cell_coords(&self, x: f32, z: f32) -> (usize, usize, f32, f32) {
        let n = (self.res - 1) as f32;
        let gx = ((x / self.size + 0.5) * n).clamp(0.0, n);
        let gz = ((z / self.size + 0.5) * n).clamp(0.0, n);
        // The last cell's far edge belongs to it (an f32 can't stay just short of it on a large
        // grid: 4999 - 1e-4 rounds to 4999).
        let j = (gx.floor() as usize).min(self.res - 2);
        let i = (gz.floor() as usize).min(self.res - 2);
        (i, j, gx - j as f32, gz - i as f32)
    }

    /// Height of the triangulated surface, matching parry's default diagonal (p10 to p01).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let (i, j, fx, fz) = self.cell_coords(x, z);
        let r = self.res;
        let h00 = self.heights[i * r + j];
        let h01 = self.heights[i * r + j + 1];
        let h10 = self.heights[(i + 1) * r + j];
        let h11 = self.heights[(i + 1) * r + j + 1];
        if fx + fz <= 1.0 {
            h00 + fx * (h01 - h00) + fz * (h10 - h00)
        } else {
            h11 + (1.0 - fx) * (h10 - h11) + (1.0 - fz) * (h01 - h11)
        }
    }

    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let e = self.cell() * 0.5;
        let dx = self.height_at(x + e, z) - self.height_at(x - e, z);
        let dz = self.height_at(x, z + e) - self.height_at(x, z - e);
        Vec3::new(-dx, 2.0 * e, -dz).normalize()
    }

    pub fn fertility_at(&self, x: f32, z: f32) -> f32 {
        let (i, j, fx, fz) = self.cell_coords(x, z);
        let r = self.res;
        let f = &self.fertility;
        let a = f[i * r + j] * (1.0 - fx) + f[i * r + j + 1] * fx;
        let b = f[(i + 1) * r + j] * (1.0 - fx) + f[(i + 1) * r + j + 1] * fx;
        a * (1.0 - fz) + b * fz
    }

    /// How well plants grow at a spot, given the current water level.
    pub fn plant_suitability(&self, x: f32, z: f32, water_level: f32, terrain_height: f32) -> f32 {
        let h = self.height_at(x, z);
        let above = h - water_level;
        if above < 0.15 {
            return 0.0;
        }
        // Lush just above the shore, thinning out toward the peaks.
        let span = (terrain_height * 0.8).max(2.0);
        let elev = (1.0 - (above / span)).clamp(0.05, 1.0);
        let slope = self.normal_at(x, z).y.clamp(0.0, 1.0);
        // Nothing grows in a stream or a pond, and little on a trampled path.
        if self.water_at(x, z) > 0.03 {
            return 0.0;
        }
        let worn = self.worn(self.nearest_point(x, z));
        elev * slope * slope * self.fertility_at(x, z) * (1.0 - 0.85 * worn)
    }

    pub fn contains(&self, x: f32, z: f32, margin: f32) -> bool {
        let h = self.half() - margin;
        x.abs() <= h && z.abs() <= h
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
