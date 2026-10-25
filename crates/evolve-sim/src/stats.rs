use serde::{Deserialize, Serialize};

use crate::genome::Genome;
use crate::params::SimParams;
use crate::rng::Rng;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    pub births: u64,
    pub rained: u64,
    pub deaths_starved: u64,
    pub deaths_killed: u64,
    pub deaths_old: u64,
    pub deaths_drowned: u64,
    #[serde(default)]
    pub deaths_thirst: u64,
    pub deaths_other: u64,
    pub max_generation: u32,
    /// Moving averages over recent deaths: what a lifetime achieved.
    pub life_food: f32,
    pub life_distance: f32,
    pub life_children: f32,
    pub life_age: f32,
    pub dead_count: u64,
    /// Lives let into the gene pool, what their scores were made of (summed `Sim::score_parts`:
    /// food, kills, children, travel), and how many each part made the most of.
    #[serde(default)]
    pub admitted: u64,
    #[serde(default)]
    pub admitted_parts: [f32; 4],
    #[serde(default)]
    pub admitted_mainly: [u64; 4],
    /// Creatures meant to fall from the zoo that came from the world's gene pool instead, there
    /// being no zoo to copy (a headless run, a test).
    #[serde(default)]
    pub rain_without_zoo: u64,
}

impl Stats {
    pub fn record_life(&mut self, food: f32, distance: f32, children: f32, age: f32) {
        let a = if self.dead_count < 20 {
            1.0 / (self.dead_count + 1) as f32
        } else {
            0.03
        };
        self.life_food += (food - self.life_food) * a;
        self.life_distance += (distance - self.life_distance) * a;
        self.life_children += (children - self.life_children) * a;
        self.life_age += (age - self.life_age) * a;
        self.dead_count += 1;
    }

    pub fn record_admitted(&mut self, parts: [f32; 4]) {
        self.admitted += 1;
        for (sum, x) in self.admitted_parts.iter_mut().zip(parts) {
            *sum += x;
        }
        let main = (0..4).max_by(|&a, &b| parts[a].total_cmp(&parts[b])).unwrap_or(0);
        self.admitted_mainly[main] += 1;
    }

    pub fn deaths(&self) -> u64 {
        self.deaths_starved
            + self.deaths_killed
            + self.deaths_old
            + self.deaths_drowned
            + self.deaths_thirst
            + self.deaths_other
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Sample {
    pub time: f32,
    pub population: u32,
    pub plants: u32,
    pub meat: u32,
    pub mean_generation: f32,
    pub max_generation: u32,
    pub mean_diet: f32,
    pub mean_neurons: f32,
    pub mean_limbs: f32,
    pub mean_mass: f32,
    pub life_food: f32,
    pub life_distance: f32,
    pub births: u64,
    pub species: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolEntry {
    pub score: f32,
    pub generation: u32,
    pub genome: Genome,
    /// Lives its score is the average of: one, until exact copies are retested (see
    /// `GenePool::retest`).
    #[serde(default = "one")]
    pub samples: u32,
}

fn one() -> u32 {
    1
}

/// Hall of fame of the most successful genomes seen recently. Creatures that rain from the sky
/// are mostly bred from here, which keeps evolution moving while the population is too small to
/// sustain itself.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenePool {
    pub entries: Vec<PoolEntry>,
}

pub const POOL_SIZE: usize = 48;

/// Lives a pool genome is judged on before it stops being retested.
pub const RETEST_SAMPLES: u32 = 3;

impl GenePool {
    pub fn consider(&mut self, score: f32, generation: u32, genome: &Genome) {
        // The same genome can come back through migration; keep one copy.
        let fp = genome.fingerprint();
        if self.entries.iter().any(|e| e.genome.fingerprint() == fp) {
            return;
        }
        let full = self.entries.len() >= POOL_SIZE;
        if full && self.entries.last().is_some_and(|e| e.score >= score) {
            return;
        }
        let at = self.entries.partition_point(|e| e.score >= score);
        self.entries.insert(
            at,
            PoolEntry {
                score,
                generation,
                genome: genome.clone(),
                samples: 1,
            },
        );
        self.entries.truncate(POOL_SIZE);
    }

    /// The pool's entries grouped by kind, best first: each group's best entry (the entries are
    /// kept best first) and how many are within `distance` of it (see `Genome::distance`).
    pub fn kinds(&self, distance: f32) -> Vec<(usize, usize)> {
        let mut kinds: Vec<(usize, usize)> = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            match kinds.iter_mut().find(|(k, _)| self.entries[*k].genome.distance(&e.genome) < distance) {
                Some(k) => k.1 += 1,
                None => kinds.push((i, 1)),
            }
        }
        kinds
    }

    /// An exact copy of a genome the pool has judged on too few lives, to live again (and its
    /// entry's fingerprint, to find it by when it dies). The better of two such entries, so the
    /// ones that shape the most rain are checked first. A single life mostly measures luck: where
    /// it fell, what was within reach. Without retests the pool filled with the luckiest and bred
    /// from them, and its creatures did no better than random ones.
    pub fn retest(&self, rng: &mut Rng) -> Option<(Genome, u32, u64)> {
        let open: Vec<usize> = (0..self.entries.len()).filter(|&i| self.entries[i].samples < RETEST_SAMPLES).collect();
        if open.is_empty() {
            return None;
        }
        let i = open[rng.below(open.len())].min(open[rng.below(open.len())]);
        let e = &self.entries[i];
        Some((e.genome.clone(), e.generation, e.genome.fingerprint()))
    }

    /// Folds a retested life's score into its genome's average. False if the genome has left the
    /// pool meanwhile.
    pub fn retested(&mut self, fingerprint: u64, score: f32) -> bool {
        let Some(i) = self.entries.iter().position(|e| e.genome.fingerprint() == fingerprint) else {
            return false;
        };
        let mut e = self.entries.remove(i);
        e.score = (e.score * e.samples as f32 + score) / (e.samples + 1) as f32;
        e.samples += 1;
        let at = self.entries.partition_point(|x| x.score >= e.score);
        self.entries.insert(at, e);
        true
    }

    /// Whether a life with this score would make it in (before checking for duplicates).
    pub fn would_take(&self, score: f32) -> bool {
        self.entries.len() < POOL_SIZE || self.entries.last().is_some_and(|e| e.score < score)
    }

    /// Old achievements fade so the pool tracks what works in the current world.
    pub fn age(&mut self, factor: f32) {
        for e in &mut self.entries {
            e.score *= factor;
        }
    }

    fn tournament(&self, rng: &mut Rng) -> usize {
        let n = self.entries.len();
        let a = rng.below(n);
        let b = rng.below(n);
        a.min(b) // entries are sorted best-first
    }

    /// A child of the pool's best: a cross of two of them, or a copy of one, then mutated. True if
    /// it's a cross. Only two close enough to interbreed in the world are crossed: the pool holds
    /// lineages from every world, and a cross of strangers mixes limbs and neurons that have
    /// nothing to do with each other.
    pub fn breed(&self, rng: &mut Rng, p: &SimParams) -> Option<(Genome, u32, bool)> {
        if self.entries.is_empty() {
            return None;
        }
        let (i, k) = (self.tournament(rng), self.tournament(rng));
        let (a, b) = (&self.entries[i], &self.entries[k]);
        let cross = i != k && rng.coin() && a.genome.distance(&b.genome) < p.species_threshold;
        let mut g = if cross {
            Genome::crossover(&a.genome, &b.genome, rng)
        } else {
            a.genome.clone()
        };
        g.mutate(rng, p);
        Some((g, a.generation.max(b.generation) + 1, cross))
    }
}
