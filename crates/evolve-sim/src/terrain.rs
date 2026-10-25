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
    pub heights: Vec<f32>,
    /// Patchiness of plant growth in [0, 1], same layout as `heights`.
    pub fertility: Vec<f32>,
    /// The ground as the world made it, before creatures dug or piled earth; empty in saves from
    /// before earthworks (then it's the current ground).
    #[serde(default)]
    pub ground: Vec<f32>,
    /// How far running water has worn the ground down from `ground`, m. Kept apart from what
    /// creatures dug, so a stream bed isn't counted as their earthworks.
    #[serde(default)]
    pub eroded: Vec<f32>,
    /// Trampling: body mass times distance moved over each grid point, per m², fading as the
    /// ground recovers. See `worn`.
    #[serde(default)]
    pub wear: Vec<f32>,
    /// Standing or running water on the ground (streams, lakes, ponds, puddles), m deep: the
    /// deeper of `pooled` and the stream running there.
    #[serde(default)]
    pub water: Vec<f32>,
    /// Water held in hollows (lakes, ponds, puddles), m deep. Unlike streams, which follow the
    /// rain, it stays until it evaporates or spills over (see `hydro`).
    #[serde(default)]
    pub pooled: Vec<f32>,
    /// How wet the soil is, 0 (dry) to 1 (soaked).
    #[serde(default)]
    pub moisture: Vec<f32>,
    /// Eternal springs, by grid point: each pours out water forever (see `hydro`).
    #[serde(default)]
    pub springs: Vec<u32>,
    /// How far down its path each spring's water has got, m (by grid point; see `hydro`).
    #[serde(default)]
    pub spring_reach: std::collections::HashMap<u32, f32>,
    /// Grid rows and columns changed since the last `take_dirty` (i0, i1, j0, j1, inclusive).
    #[serde(skip)]
    pub dirty: Option<(usize, usize, usize, usize)>,
    /// Bumped whenever the shape changes so renderers know to rebuild.
    pub version: u64,
}

/// A placeholder 2x2 flat terrain, used before the first real snapshot arrives.
impl Default for Terrain {
    fn default() -> Self {
        Terrain {
            size: 1.0,
            res: 2,
            heights: vec![0.0; 4],
            fertility: vec![0.0; 4],
            ground: vec![0.0; 4],
            eroded: vec![0.0; 4],
            wear: vec![0.0; 4],
            water: vec![0.0; 4],
            pooled: vec![0.0; 4],
            moisture: vec![0.0; 4],
            springs: Vec::new(),
            spring_reach: Default::default(),
            dirty: None,
            version: 0,
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
            ground: heights.clone(),
            heights,
            fertility,
            eroded: vec![0.0; n],
            wear: vec![0.0; n],
            water: vec![0.0; n],
            pooled: vec![0.0; n],
            moisture: vec![0.0; n],
            springs: Vec::new(),
            spring_reach: Default::default(),
            version,
            dirty: None,
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
                *layer = vec![0.0; n];
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
        let Some((i0, i1, j0, j1)) = self.dirty else {
            return true;
        };
        if angle >= 90.0 || self.res < 2 {
            return true;
        }
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
        }
        false
    }

    /// Dug and piled ground weathers back toward the land as the world made it (less what water
    /// has worn away), halfway in `half_life` seconds. Returns whether anything changed.
    pub fn heal(&mut self, dt: f32, half_life: f32) -> bool {
        if self.ground.len() != self.heights.len() {
            return false;
        }
        let keep = (-dt * std::f32::consts::LN_2 / half_life.max(1.0)).exp();
        let n = self.res;
        let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0, usize::MAX, 0);
        for k in 0..self.heights.len() {
            let natural = self.natural(k);
            let off = self.heights[k] - natural;
            if off == 0.0 {
                continue;
            }
            // The last millimeters go at once, so healed ground stops counting as dug.
            self.heights[k] = if off.abs() * keep < 0.002 { natural } else { natural + off * keep };
            let (i, j) = (k / n, k % n);
            (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(j), j1.max(j));
        }
        if i0 == usize::MAX {
            return false;
        }
        self.mark_dirty(i0, i1, j0, j1);
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
        self.dirty = Some(match self.dirty {
            Some((a, b, c, d)) => (a.min(i0), b.max(i1), c.min(j0), d.max(j1)),
            None => (i0, i1, j0, j1),
        });
        self.version += 1;
    }

    /// The region changed since the last call, if any.
    pub fn take_dirty(&mut self) -> Option<(usize, usize, usize, usize)> {
        self.dirty.take()
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
