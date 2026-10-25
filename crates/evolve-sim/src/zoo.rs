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
use crate::genome::{Genome, Shape};
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
    /// Body plan and traits, to tell kinds apart without reading the genome (filled in from the
    /// genome file for entries stored before it was kept).
    #[serde(default)]
    pub shape: Option<Shape>,
}

/// 6: as 5, with food counted only when eaten at least `score_food_away` (4 m) from where the
/// creature fell or was born (2026-10-24); before, a big body that landed beside plants filled
/// its tank and won, the pool bred for sitting still, and 10 hours of evolution left creatures
/// eating and travelling half what random ones did. 5: as 4, plus a point for every 10 m closed on food it had room to eat (2026-10-24); before, the
/// pool mostly rewarded landing beside a plant, and 300 kg flailers that starved within 40 s
/// with plants 2-5 m away won. 4: as 3, with bodies that fell from the sky leaving no meat of their own (2026-10-23); before,
/// the population floor rained free carcasses, and huge limbless carrion-eaters that never moved
/// won. 3: as 2, earned with servo motors (2026-10-23), whose limbs can push the body; before, a
/// planted leg could barely push, so limbless bodies won. 2: earned with children's bodies paid
/// for. 1 (2026-10-22) was earned while a child's body was free and its carcass still left meat,
/// which heaps of meat-eaters living on their own dead exploited. 0 (before 2026-10-22) counted 4
/// points per child however little it was given, which tiny do-nothing breeders exploited.
pub const SCORE_VERSION: u32 = 6;

pub type SharedZoo = Arc<Mutex<Zoo>>;

pub struct Zoo {
    dir: PathBuf,
    pub entries: Vec<ZooMeta>,
    /// Bumped on every change, so readers know when to refresh.
    pub version: u64,
    next_id: u64,
    /// Held while this process owns the zoo. Without it (another evolve has the zoo open) the
    /// zoo is read-only, so two processes never overwrite each other's index.
    lock: Option<File>,
}

/// Pruned and removed creatures are moved here rather than deleted, up to this many.
const TRASH_KEEP: usize = 500;

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
        let lock = take_lock(&dir);
        let index = dir.join("index.json");
        let (mut entries, damaged): (Vec<ZooMeta>, bool) = match std::fs::read(&index) {
            Ok(b) => match serde_json::from_slice(&b) {
                Ok(v) => (v, false),
                Err(e) => {
                    eprintln!("evolve: zoo: {} is unreadable ({e}); rebuilding it from the genome files", index.display());
                    (Vec::new(), true)
                }
            },
            Err(_) => (Vec::new(), false),
        };
        if damaged && lock.is_some() {
            let aside = dir.join(format!("index.json.bad-{}", unix_now()));
            if let Err(e) = std::fs::rename(&index, &aside) {
                eprintln!("evolve: zoo: couldn't set {} aside: {e}", index.display());
            }
        }
        // Forget entries whose genome file has gone missing, and take in genome files the index
        // doesn't list (all of them, if the index was lost or damaged).
        let before = entries.len();
        entries.retain(|e| genome_path(&dir, e.id).exists());
        let mut changed = entries.len() != before;
        let listed: std::collections::HashSet<u64> = entries.iter().map(|e| e.id).collect();
        for id in genome_ids(&dir) {
            if listed.contains(&id) {
                continue;
            }
            match read_genome(&genome_path(&dir, id)) {
                Ok(g) => {
                    entries.push(meta_for(id, &g, 0.0, 0, "recovered", 0));
                    changed = true;
                }
                Err(e) => eprintln!("evolve: zoo: skipping {id}.genome: {e}"),
            }
        }
        entries.sort_by_key(|e| e.id);
        let next_id = entries
            .iter()
            .map(|e| e.id)
            .chain(genome_ids(&dir))
            .chain(genome_ids(&dir.join("trash")))
            .max()
            .map_or(1, |m| m + 1);
        let zoo = Self {
            next_id,
            dir,
            entries,
            version: 1,
            lock,
        };
        if zoo.lock.is_none() {
            eprintln!("evolve: zoo: another evolve has {} open; this one won't change it", zoo.dir.display());
        } else if changed {
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
        self.writable()?;
        let score = if score.is_finite() { score } else { 0.0 };
        let fp = genome.fingerprint();
        if let Some(e) = self.entries.iter_mut().find(|e| e.fingerprint == fp) {
            // Already here; a score worked out the old way is replaced by this one.
            let rescore = e.score_version < SCORE_VERSION;
            if rescore {
                e.score = score;
                e.score_version = SCORE_VERSION;
            }
            if rescore || (pinned && !e.pinned) {
                e.pinned |= pinned;
                self.version += 1;
                self.save_index()?;
            }
            return Ok(None);
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("create {}: {e}", self.dir.display()))?;
        let id = self.next_id;
        self.next_id += 1;
        write_genome(&genome_path(&self.dir, id), genome)?;
        let mut meta = meta_for(id, genome, score, generation, source, SCORE_VERSION);
        meta.pinned = pinned;
        self.entries.push(meta);
        self.fill_shapes();
        self.prune_alike(max_entries);
        self.version += 1;
        self.save_index()?;
        Ok(Some(id))
    }

    /// Offers a genome from a world. It joins as a new kind if its body and traits are at least
    /// `kind_distance` from every unpinned creature here (see `Genome::distance`); otherwise it
    /// takes the place of the most alike one if it scored better (or that one was scored the old
    /// way), and is turned away if not. Over `max_entries`, the most redundant go first (see
    /// `prune_alike`), so the zoo fills with different creatures rather than copies of a few.
    /// Returns the new entry's id, if it got in.
    pub fn offer(
        &mut self,
        genome: &Genome,
        score: f32,
        generation: u32,
        source: &str,
        max_entries: usize,
        kind_distance: f32,
    ) -> Result<Option<u64>, String> {
        self.writable()?;
        let score = if score.is_finite() { score } else { 0.0 };
        if self.entries.iter().any(|e| e.fingerprint == genome.fingerprint()) {
            return self.add(genome, score, generation, source, false, max_entries);
        }
        self.fill_shapes();
        let shape = genome.shape();
        let alike = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.pinned)
            .filter_map(|(i, e)| Some((i, e.shape.as_ref()?.distance(&shape))))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, d)) = alike
            && d < kind_distance
        {
            let e = &self.entries[i];
            if e.score_version == SCORE_VERSION && e.score >= score {
                return Ok(None);
            }
            let old = self.entries.remove(i).id;
            self.trash([old]);
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("create {}: {e}", self.dir.display()))?;
        let id = self.next_id;
        self.next_id += 1;
        write_genome(&genome_path(&self.dir, id), genome)?;
        self.entries.push(meta_for(id, genome, score, generation, source, SCORE_VERSION));
        self.prune_alike(max_entries);
        self.version += 1;
        self.save_index()?;
        Ok(Some(id))
    }

    /// Reads the shape of every entry stored without one (once; it's kept in the index after).
    fn fill_shapes(&mut self) {
        let mut filled = false;
        for e in self.entries.iter_mut().filter(|e| e.shape.is_none()) {
            if let Ok(g) = read_genome(&genome_path(&self.dir, e.id)) {
                e.shape = Some(g.shape());
                filled = true;
            }
        }
        if filled {
            let _ = self.save_index();
        }
    }

    /// Drops unpinned creatures beyond `max` entries: those scored the old way first (weakest
    /// first), then the weaker of the two most alike, again and again, so every kind keeps its
    /// best and, when there are more kinds than room, the closest kinds merge first. What stays is
    /// the best of the most different creatures, not just the best.
    fn prune_alike(&mut self, max: usize) {
        let excess = self.entries.iter().filter(|e| !e.pinned).count().saturating_sub(max.max(1));
        if excess == 0 {
            return;
        }
        let n = self.entries.len();
        // How far apart every two are, worked out once (NaN where a shape is missing).
        let apart: Vec<f32> = (0..n * n)
            .map(|k| match (&self.entries[k / n].shape, &self.entries[k % n].shape) {
                (Some(a), Some(b)) if k / n != k % n => a.distance(b),
                _ => f32::NAN,
            })
            .collect();
        let mut kept = vec![true; n];
        let mut gone = Vec::new();
        for _ in 0..excess {
            let e = &self.entries;
            let free = |i: usize| kept[i] && !e[i].pinned;
            let weakest = |old: bool| {
                (0..n)
                    .filter(|&i| free(i) && (!old || e[i].score_version < SCORE_VERSION))
                    .min_by(|&a, &b| e[a].score.total_cmp(&e[b].score))
            };
            let i = weakest(true).or_else(|| {
                // Each unpinned entry's nearest neighbour; of the closest pair, the weaker goes
                // (or the unpinned one, beside a pinned creature).
                let mut best: Option<(f32, usize)> = None;
                for i in (0..n).filter(|&i| free(i)) {
                    for j in (0..n).filter(|&j| kept[j]) {
                        let d = apart[i * n + j];
                        let weaker = e[j].pinned || e[i].score <= e[j].score;
                        if !d.is_nan() && weaker && best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, i));
                        }
                    }
                }
                best.map(|(_, i)| i)
            });
            // No shapes to compare: the weakest.
            let Some(i) = i.or_else(|| weakest(false)) else {
                break;
            };
            kept[i] = false;
            gone.push(e[i].id);
        }
        let mut k = 0;
        self.entries.retain(|_| {
            k += 1;
            kept[k - 1]
        });
        self.trash(gone);
    }

    pub fn genome(&self, id: u64) -> Result<Genome, String> {
        read_genome(&genome_path(&self.dir, id))
    }

    pub fn get(&self, id: u64) -> Option<&ZooMeta> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Takes a creature out of the zoo (its genome goes to the trash).
    pub fn remove(&mut self, id: u64) -> Result<(), String> {
        self.writable()?;
        if let Some(i) = self.entries.iter().position(|e| e.id == id) {
            self.entries.remove(i);
            self.trash([id]);
            self.version += 1;
            self.save_index()?;
        }
        Ok(())
    }

    /// Keeps `n` creatures, the best of each of the most different kinds (see `prune_alike`),
    /// and every pinned one, moving the rest to the trash. Returns how many went.
    pub fn keep_best(&mut self, n: usize) -> Result<usize, String> {
        self.writable()?;
        let before = self.entries.len();
        self.fill_shapes();
        self.prune_alike(n);
        let gone = before - self.entries.len();
        if gone > 0 {
            self.version += 1;
            self.save_index()?;
        }
        Ok(gone)
    }

    pub fn set_pinned(&mut self, id: u64, pinned: bool) {
        if self.writable().is_err() {
            return;
        }
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.pinned = pinned;
            self.version += 1;
            let _ = self.save_index();
        }
    }

    /// Whether this process may change the zoo.
    pub fn read_only(&self) -> bool {
        self.lock.is_none()
    }

    fn writable(&self) -> Result<(), String> {
        if self.read_only() {
            Err(format!("another evolve has {} open", self.dir.display()))
        } else {
            Ok(())
        }
    }

    /// Moves creatures' genomes into `trash/`, keeping only the newest `TRASH_KEEP` there.
    fn trash(&self, ids: impl IntoIterator<Item = u64>) {
        let trash = self.dir.join("trash");
        let made = std::fs::create_dir_all(&trash).is_ok();
        let mut moved = false;
        for id in ids {
            if made && std::fs::rename(genome_path(&self.dir, id), genome_path(&trash, id)).is_ok() {
                moved = true;
            } else {
                let _ = std::fs::remove_file(genome_path(&self.dir, id));
            }
        }
        if !moved {
            return;
        }
        let mut old: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&trash)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|f| Some((f.metadata().ok()?.modified().ok()?, f.path())))
            .collect();
        if old.len() > TRASH_KEEP {
            old.sort();
            for (_, p) in &old[..old.len() - TRASH_KEEP] {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    fn save_index(&self) -> Result<(), String> {
        self.writable()?;
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

/// The ids of the `<id>.genome` files in a directory.
fn genome_ids(dir: &Path) -> Vec<u64> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|f| f.file_name().to_str()?.strip_suffix(".genome")?.parse().ok())
        .collect()
}

/// Locks `<dir>/lock` for this process, or returns None if another process holds it.
fn take_lock(dir: &Path) -> Option<File> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("evolve: zoo: create {}: {e}", dir.display());
        return None;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("lock"))
        .ok()?;
    file.try_lock().ok().map(|()| file)
}

fn meta_for(id: u64, genome: &Genome, score: f32, generation: u32, source: &str, score_version: u32) -> ZooMeta {
    let fp = genome.fingerprint();
    let plan = body::body_plan(genome);
    ZooMeta {
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
        pinned: false,
        fingerprint: fp,
        score_version,
        shape: Some(genome.shape()),
    }
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
/// Only creatures scored under today's rules are chosen: ones that won under older rules (free
/// children, limbs that couldn't push) would only bring back what those rules rewarded. The rest
/// of the population rains down at random as usual.
pub fn seed_world(zoo: &SharedZoo, sim: &mut crate::Sim) {
    let want = (sim.params.min_population as f32 * sim.params.zoo_seed_fraction).round() as usize;
    if want == 0 {
        return;
    }
    let (dir, ids): (PathBuf, Vec<u64>) = match zoo.lock() {
        Ok(z) => (z.dir.clone(), z.entries.iter().filter(|e| e.score_version == SCORE_VERSION).map(|e| e.id).collect()),
        Err(_) => return,
    };
    if ids.is_empty() {
        sim.log("Nothing in the zoo from this version yet to seed the world with");
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

/// Zoo creatures a world keeps to hand for creatures falling from the zoo (`rain_from_zoo`).
const RAIN_SAMPLE: usize = 16;

/// Refills a world's sample of zoo creatures to rain from (`Sim::zoo_sample`) when the zoo has
/// changed since: `RAIN_SAMPLE` picked at random, which spreads across kinds since the zoo keeps
/// one of each (see `Zoo::offer`). Only creatures scored under today's rules, as for
/// `seed_world`. Cheap when nothing changed, so it can be called often.
pub fn refresh_rain_sample(zoo: &SharedZoo, sim: &mut crate::Sim) {
    if sim.params.rain_from_zoo <= 0.0 {
        return;
    }
    let (dir, mut ids, version) = match zoo.lock() {
        Ok(z) if z.version != sim.zoo_sample_version || sim.zoo_sample.is_empty() => (
            z.dir.clone(),
            z.entries.iter().filter(|e| e.score_version == SCORE_VERSION).map(|e| (e.id, e.generation)).collect::<Vec<_>>(),
            z.version,
        ),
        _ => return,
    };
    sim.zoo_sample_version = version;
    // Its own random numbers, so when the zoo changes doesn't change the world's.
    let mut rng = crate::rng::Rng::new(version ^ sim.tick);
    let mut sample = Vec::with_capacity(RAIN_SAMPLE);
    while sample.len() < RAIN_SAMPLE && !ids.is_empty() {
        let (id, generation) = ids.swap_remove(rng.below(ids.len()));
        if let Ok(g) = load_genome(&dir, id) {
            sample.push((g, generation));
        }
    }
    sim.zoo_sample = sample;
}

/// Offers the zoo the best genome of each kind in a world's gene pool (see `Zoo::offer`).
pub fn store_best(zoo: &SharedZoo, sim: &crate::Sim, world: usize) {
    if sim.pool.entries.is_empty() {
        return;
    }
    let source = if world == 0 {
        format!("seed {}, {:.0} min", sim.params.seed, sim.time / 60.0)
    } else {
        format!("background world {world} (seed {}), {:.0} min", sim.params.seed, sim.time / 60.0)
    };
    let p = &sim.params;
    if let Ok(mut z) = zoo.lock() {
        for (best, _) in sim.pool.kinds(p.zoo_kind_distance) {
            let e = &sim.pool.entries[best];
            if let Err(e) = z.offer(&e.genome, e.score, e.generation, &source, p.zoo_max_entries as usize, p.zoo_kind_distance) {
                eprintln!("evolve: zoo: {e}");
                break;
            }
        }
    }
}
