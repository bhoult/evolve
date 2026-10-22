//! The zoo: the best creatures from every world, kept on disk across sessions, ready to be
//! dropped into any world. Each creature is one gzipped genome file plus a line in `index.json`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};

use crate::body;
use crate::genome::Genome;
use crate::params::SimParams;

/// EVOZOO02 genomes have multi-jointed bodies; EVOZOO01 ones are upgraded as they're read.
const MAGIC: &[u8; 8] = b"EVOZOO02";
const LEGACY_MAGIC: &[u8; 8] = b"EVOZOO01";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZooMeta {
    pub id: u64,
    pub name: String,
    pub score: f32,
    pub generation: u32,
    pub neurons: usize,
    pub limbs: usize,
    pub mass: f32,
    pub diet: f32,
    pub color: [f32; 3],
    /// Which world it came from.
    pub source: String,
    pub saved_unix: u64,
    /// Pinned creatures are never pruned.
    pub pinned: bool,
    pub fingerprint: u64,
    /// Which way `score` was worked out (see `SCORE_VERSION`); scores of different versions
    /// don't compare.
    #[serde(default)]
    pub score_version: u32,
}

/// 1: eating, hunting, breeding (by energy given) and travelling. 0 (before 2026-10-22) counted
/// 4 points per child however little it was given, which tiny do-nothing breeders exploited.
pub const SCORE_VERSION: u32 = 1;

pub type SharedZoo = Arc<Mutex<Zoo>>;

pub struct Zoo {
    dir: PathBuf,
    pub entries: Vec<ZooMeta>,
    /// Bumped on every change, so readers know when to refresh.
    pub version: u64,
    next_id: u64,
}

/// A pronounceable name derived from the genome, so the same creature always gets the same one.
fn name_for(fp: u64, generation: u32) -> String {
    const C: &[&str] = &["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "sh", "th"];
    const V: &[&str] = &["a", "e", "i", "o", "u", "ai", "oo", "ia"];
    let mut x = fp;
    let mut s = String::new();
    for _ in 0..2 + (fp % 2) as usize {
        s.push_str(C[(x % C.len() as u64) as usize]);
        x /= C.len() as u64;
        s.push_str(V[(x % V.len() as u64) as usize]);
        x /= V.len() as u64;
    }
    let mut c = s.chars();
    let s = match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => s,
    };
    format!("{s} g{generation}")
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Zoo {
    /// `$EVOLVE_ZOO_DIR`, or `zoo` next to the saves directory.
    pub fn default_dir() -> PathBuf {
        if let Some(d) = std::env::var_os("EVOLVE_ZOO_DIR") {
            return PathBuf::from(d);
        }
        let saves = crate::save::default_dir();
        saves.parent().map(|p| p.join("zoo")).unwrap_or_else(|| PathBuf::from("zoo"))
    }

    pub fn open(dir: PathBuf) -> Self {
        let entries: Vec<ZooMeta> = std::fs::read(dir.join("index.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let mut zoo = Self {
            next_id: entries.iter().map(|e| e.id + 1).max().unwrap_or(1),
            dir,
            entries,
            version: 1,
        };
        // Forget entries whose genome file has gone missing.
        let before = zoo.entries.len();
        let dir = zoo.dir.clone();
        zoo.entries.retain(|e| genome_path(&dir, e.id).exists());
        if zoo.entries.len() != before {
            let _ = zoo.save_index();
        }
        zoo
    }

    pub fn shared(dir: PathBuf) -> SharedZoo {
        Arc::new(Mutex::new(Self::open(dir)))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Stores a genome unless an identical one is already here. Returns the new entry's id.
    pub fn add(
        &mut self,
        genome: &Genome,
        score: f32,
        generation: u32,
        source: &str,
        pinned: bool,
        max_entries: usize,
    ) -> Result<Option<u64>, String> {
        let fp = genome.fingerprint();
        if let Some(e) = self.entries.iter_mut().find(|e| e.fingerprint == fp) {
            if pinned && !e.pinned {
                e.pinned = true;
                self.version += 1;
                self.save_index()?;
            }
            return Ok(None);
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("create {}: {e}", self.dir.display()))?;
        let id = self.next_id;
        self.next_id += 1;
        write_genome(&genome_path(&self.dir, id), genome)?;
        let plan = body::body_plan(genome);
        self.entries.push(ZooMeta {
            id,
            name: name_for(fp, generation),
            score,
            generation,
            neurons: genome.brain.h,
            limbs: plan.len() - 1,
            mass: body::plan_mass(&plan, SimParams::default().body_density),
            diet: genome.traits.diet,
            color: genome.color(),
            source: source.to_string(),
            saved_unix: unix_now(),
            pinned,
            fingerprint: fp,
            score_version: SCORE_VERSION,
        });
        self.prune(max_entries);
        self.version += 1;
        self.save_index()?;
        Ok(Some(id))
    }

    pub fn genome(&self, id: u64) -> Result<Genome, String> {
        read_genome(&genome_path(&self.dir, id))
    }

    pub fn get(&self, id: u64) -> Option<&ZooMeta> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn remove(&mut self, id: u64) {
        if let Some(i) = self.entries.iter().position(|e| e.id == id) {
            self.entries.remove(i);
            let _ = std::fs::remove_file(genome_path(&self.dir, id));
            self.version += 1;
            let _ = self.save_index();
        }
    }

    pub fn set_pinned(&mut self, id: u64, pinned: bool) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.pinned = pinned;
            self.version += 1;
            let _ = self.save_index();
        }
    }

    /// Drops unpinned creatures beyond `max` entries: those scored the old way first, then the
    /// weakest.
    fn prune(&mut self, max: usize) {
        while self.entries.iter().filter(|e| !e.pinned).count() > max.max(1) {
            let Some(i) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.pinned)
                .min_by(|a, b| a.1.score_version.cmp(&b.1.score_version).then(a.1.score.total_cmp(&b.1.score)))
                .map(|(i, _)| i)
            else {
                break;
            };
            let e = self.entries.remove(i);
            let _ = std::fs::remove_file(genome_path(&self.dir, e.id));
        }
    }

    fn save_index(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        let tmp = self.dir.join("index.json.tmp");
        let json = serde_json::to_vec_pretty(&self.entries).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, self.dir.join("index.json")).map_err(|e| e.to_string())
    }
}

fn genome_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("{id}.genome"))
}

fn write_genome(path: &Path, g: &Genome) -> Result<(), String> {
    let tmp = path.with_extension("genome.tmp");
    let file = File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    let mut w = BufWriter::new(file);
    w.write_all(MAGIC).map_err(|e| e.to_string())?;
    let mut gz = GzEncoder::new(w, Compression::fast());
    rmp_serde::encode::write_named(&mut gz, g).map_err(|e| format!("encode: {e}"))?;
    gz.finish().and_then(|mut w| w.flush()).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn read_genome(path: &Path) -> Result<Genome, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC && &magic != LEGACY_MAGIC {
        return Err(format!("{} is from a different version of evolve", path.display()));
    }
    let mut g: Genome = rmp_serde::decode::from_read(GzDecoder::new(r)).map_err(|e| format!("decode: {e}"))?;
    g.upgrade();
    Ok(g)
}

/// Reads one zoo creature's genome straight from the zoo directory.
pub fn load_genome(dir: &Path, id: u64) -> Result<Genome, String> {
    read_genome(&genome_path(dir, id))
}

/// Starts a fresh world off with creatures from the zoo: `zoo_seed_fraction` of its minimum
/// population, as pairs of copies of randomly chosen zoo creatures (so each pair can breed),
/// and the same genomes in its gene pool so creatures falling from the sky come from them too.
/// The rest of the population rains down at random as usual.
pub fn seed_world(zoo: &SharedZoo, sim: &mut crate::Sim) {
    let want = (sim.params.min_population as f32 * sim.params.zoo_seed_fraction).round() as usize;
    if want == 0 {
        return;
    }
    let (dir, ids): (PathBuf, Vec<u64>) = match zoo.lock() {
        Ok(z) => (z.dir.clone(), z.entries.iter().map(|e| e.id).collect()),
        Err(_) => return,
    };
    if ids.is_empty() {
        return;
    }
    let mut placed = 0;
    let mut tries = 0;
    while placed < want && tries < want * 3 {
        tries += 1;
        let id = ids[sim.rng.below(ids.len())];
        let Ok(genome) = load_genome(&dir, id) else {
            continue;
        };
        let copies = 2.min(want - placed) as u32;
        sim.drop_genome(&genome, 0, None, copies);
        sim.pool.consider(0.0, 0, &genome);
        placed += copies as usize;
    }
    sim.log(format!("{placed} creatures from the zoo"));
}

/// Puts a world's current best genome (the top of its gene pool) in the zoo.
pub fn store_best(zoo: &SharedZoo, sim: &crate::Sim, world: usize) {
    let Some(best) = sim.pool.entries.first() else {
        return;
    };
    let source = if world == 0 {
        format!("seed {}, {:.0} min", sim.params.seed, sim.time / 60.0)
    } else {
        format!("background world {world} (seed {}), {:.0} min", sim.params.seed, sim.time / 60.0)
    };
    if let Ok(mut z) = zoo.lock() {
        let max = sim.params.zoo_max_entries as usize;
        if let Err(e) = z.add(&best.genome, best.score, best.generation, &source, false, max) {
            eprintln!("evolve: zoo: {e}");
        }
    }
}
