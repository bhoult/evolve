use rapier3d::prelude::{Array2, ColliderBuilder, SharedShape};
use serde::{Deserialize, Serialize};

use crate::math::Vec3;
use crate::noise::Perlin;
use crate::params::SimParams;

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
            dirty: None,
            version: 0,
        }
    }
}

impl Terrain {
    pub fn generate(p: &SimParams, version: u64) -> Self {
        let res = p.terrain_resolution.clamp(9, 1025) as usize;
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
        Self {
            size,
            res,
            ground: heights.clone(),
            heights,
            fertility,
            version,
            dirty: None,
        }
    }

    /// Moves earth at a spot: `volume` m³, positive to pile it up, negative to dig it out, spread
    /// over the four grid points around the spot. Ground can be dug at most `max_dig` m below and
    /// piled at most `max_pile` m above where the world made it. Returns the volume actually moved
    /// (with the sign of the request).
    pub fn reshape(&mut self, x: f32, z: f32, volume: f32, max_dig: f32, max_pile: f32) -> f32 {
        if self.ground.len() != self.heights.len() {
            self.ground = self.heights.clone();
        }
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
            let (lo, hi) = (self.ground[k] - max_dig, self.ground[k] + max_pile);
            let want = volume * w / area;
            let h = (self.heights[k] + want).clamp(lo, hi.max(lo));
            moved += (h - self.heights[k]) * area;
            self.heights[k] = h;
        }
        if moved != 0.0 {
            let (i1, j1) = (i + 1, j + 1);
            self.dirty = Some(match self.dirty {
                Some((a, b, c, d)) => (a.min(i), b.max(i1), c.min(j), d.max(j1)),
                None => (i, i1, j, j1),
            });
            self.version += 1;
        }
        moved
    }

    /// How far the ground at a grid point is from where the world made it, m (positive: piled).
    pub fn disturbance(&self, k: usize) -> f32 {
        self.ground.get(k).map_or(0.0, |g| self.heights[k] - g)
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
        let gx = ((x / self.size + 0.5) * n).clamp(0.0, n - 1e-4);
        let gz = ((z / self.size + 0.5) * n).clamp(0.0, n - 1e-4);
        let j = gx.floor() as usize;
        let i = gz.floor() as usize;
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
        elev * slope * slope * self.fertility_at(x, z)
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
