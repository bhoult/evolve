//! Uniform grid over the XZ plane for nearest-neighbor lookups of plants, meat and creatures.

use crate::math::Vec3;

/// The items are sorted by cell into one list (a counting sort), so building a grid allocates
/// three vectors rather than one per cell: about a dozen grids are built every brain tick, and on
/// an 800 m world each has 25,600 cells, which as a vector apiece cost most of a step.
pub struct Grid {
    cell: f32,
    half: f32,
    n: usize,
    /// The items in cell c are `items[start[c]..start[c + 1]]`.
    start: Vec<u32>,
    items: Vec<u32>,
    /// Items inserted one at a time since the last rebuild, with their cells.
    extra: Vec<(u32, u32)>,
}

impl Grid {
    pub fn new(world_size: f32, cell: f32) -> Self {
        let n = ((world_size / cell).ceil() as usize).max(1);
        Self {
            cell,
            half: world_size * 0.5,
            n,
            start: vec![0; n * n + 1],
            items: Vec::new(),
            extra: Vec::new(),
        }
    }

    fn coord(&self, v: f32) -> usize {
        (((v + self.half) / self.cell).floor().max(0.0) as usize).min(self.n - 1)
    }

    fn cell_of(&self, pos: Vec3) -> usize {
        self.coord(pos.z) * self.n + self.coord(pos.x)
    }

    pub fn clear(&mut self) {
        self.start.fill(0);
        self.items.clear();
        self.extra.clear();
    }

    pub fn insert(&mut self, pos: Vec3, index: u32) {
        let c = self.cell_of(pos) as u32;
        self.extra.push((c, index));
    }

    pub fn rebuild(&mut self, positions: impl Iterator<Item = Vec3>) {
        let cells: Vec<u32> = positions.map(|p| self.cell_of(p) as u32).collect();
        self.extra.clear();
        // Count each cell's items, then where each cell's run starts, then place them.
        self.start.fill(0);
        for &c in &cells {
            self.start[c as usize + 1] += 1;
        }
        for c in 1..self.start.len() {
            self.start[c] += self.start[c - 1];
        }
        self.items.clear();
        self.items.resize(cells.len(), 0);
        let mut next = self.start.clone();
        for (k, &c) in cells.iter().enumerate() {
            let at = &mut next[c as usize];
            self.items[*at as usize] = k as u32;
            *at += 1;
        }
    }

    /// Calls `f(index)` for every item in cells overlapping the square around `pos`.
    pub fn for_each_near(&self, pos: Vec3, radius: f32, mut f: impl FnMut(u32)) {
        let i0 = self.coord(pos.z - radius);
        let i1 = self.coord(pos.z + radius);
        let j0 = self.coord(pos.x - radius);
        let j1 = self.coord(pos.x + radius);
        for i in i0..=i1 {
            let row = i * self.n;
            let (a, b) = (self.start[row + j0] as usize, self.start[row + j1 + 1] as usize);
            for &k in &self.items[a..b] {
                f(k);
            }
        }
        for &(c, k) in &self.extra {
            let (i, j) = (c as usize / self.n, c as usize % self.n);
            if (i0..=i1).contains(&i) && (j0..=j1).contains(&j) {
                f(k);
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
