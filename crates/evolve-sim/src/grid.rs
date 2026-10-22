//! Uniform grid over the XZ plane for nearest-neighbor lookups of plants, meat and creatures.

use crate::math::Vec3;

pub struct Grid {
    cell: f32,
    half: f32,
    n: usize,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    pub fn new(world_size: f32, cell: f32) -> Self {
        let n = ((world_size / cell).ceil() as usize).max(1);
        Self {
            cell,
            half: world_size * 0.5,
            n,
            cells: vec![Vec::new(); n * n],
        }
    }

    fn coord(&self, v: f32) -> usize {
        (((v + self.half) / self.cell).floor().max(0.0) as usize).min(self.n - 1)
    }

    pub fn clear(&mut self) {
        for c in &mut self.cells {
            c.clear();
        }
    }

    pub fn insert(&mut self, pos: Vec3, index: u32) {
        let (i, j) = (self.coord(pos.z), self.coord(pos.x));
        self.cells[i * self.n + j].push(index);
    }

    pub fn rebuild(&mut self, positions: impl Iterator<Item = Vec3>) {
        self.clear();
        for (k, p) in positions.enumerate() {
            self.insert(p, k as u32);
        }
    }

    /// Calls `f(index)` for every item in cells overlapping the square around `pos`.
    pub fn for_each_near(&self, pos: Vec3, radius: f32, mut f: impl FnMut(u32)) {
        let i0 = self.coord(pos.z - radius);
        let i1 = self.coord(pos.z + radius);
        let j0 = self.coord(pos.x - radius);
        let j1 = self.coord(pos.x + radius);
        for i in i0..=i1 {
            for j in j0..=j1 {
                for &k in &self.cells[i * self.n + j] {
                    f(k);
                }
            }
        }
    }

    /// Nearest item within `radius` (by horizontal distance) that passes `accept`.
    pub fn nearest(
        &self,
        pos: Vec3,
        radius: f32,
        position_of: impl Fn(u32) -> Vec3,
        accept: impl Fn(u32) -> bool,
    ) -> Option<(u32, f32)> {
        let mut best: Option<(u32, f32)> = None;
        let r2 = radius * radius;
        self.for_each_near(pos, radius, |k| {
            let p = position_of(k);
            let dx = p.x - pos.x;
            let dz = p.z - pos.z;
            let d2 = dx * dx + dz * dz;
            if d2 <= r2 && best.is_none_or(|(_, b)| d2 < b) && accept(k) {
                best = Some((k, d2));
            }
        });
        best.map(|(k, d2)| (k, d2.sqrt()))
    }
}
