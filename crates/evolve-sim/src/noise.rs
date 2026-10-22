//! 2D gradient (Perlin) noise and fractal sums of it, for terrain and plant fertility.

use crate::rng::Rng;

pub struct Perlin {
    perm: [u8; 512],
}

impl Perlin {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut p: Vec<u8> = (0..=255).collect();
        for i in (1..256).rev() {
            let j = rng.below(i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Self { perm }
    }

    fn grad(hash: u8, x: f32, y: f32) -> f32 {
        match hash & 7 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x,
            5 => -x,
            6 => y,
            _ => -y,
        }
    }

    /// Roughly in [-1, 1].
    pub fn noise(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor();
        let yi = y.floor();
        let xf = x - xi;
        let yf = y - yi;
        let xi = (xi as i32 & 255) as usize;
        let yi = (yi as i32 & 255) as usize;
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let u = fade(xf);
        let v = fade(yf);
        let p = &self.perm;
        let aa = p[p[xi] as usize + yi];
        let ab = p[p[xi] as usize + yi + 1];
        let ba = p[p[xi + 1] as usize + yi];
        let bb = p[p[xi + 1] as usize + yi + 1];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x1 = lerp(Self::grad(aa, xf, yf), Self::grad(ba, xf - 1.0, yf), u);
        let x2 = lerp(
            Self::grad(ab, xf, yf - 1.0),
            Self::grad(bb, xf - 1.0, yf - 1.0),
            u,
        );
        lerp(x1, x2, v) * 0.75
    }

    /// Fractal Brownian motion, normalized to roughly [-1, 1].
    pub fn fbm(&self, x: f32, y: f32, octaves: u32, persistence: f32) -> f32 {
        let mut sum = 0.0;
        let mut amp = 1.0;
        let mut freq = 1.0;
        let mut norm = 0.0;
        for o in 0..octaves {
            // Offset each octave so their lattices don't line up at the origin.
            let off = o as f32 * 17.13;
            sum += amp * self.noise(x * freq + off, y * freq - off);
            norm += amp;
            amp *= persistence;
            freq *= 2.0;
        }
        sum / norm
    }
}
