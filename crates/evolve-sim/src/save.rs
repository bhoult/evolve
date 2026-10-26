//! Save files: the whole `Sim`, physics state included, as MessagePack in zstd-compressed blocks.
//! Saving mid-stride and loading resumes exactly where it left off.
//!
//! The world is cut into sections (the terrain, the gene pool, each physics world, the creatures
//! a few at a time, and the rest), each turned into bytes on a core of its own, and each section
//! is compressed in blocks, also in parallel; loading undoes both the same way. The world waits
//! only while it is turned into bytes: `begin` hands back a `Saving` that compresses and writes
//! the file on other threads while the world runs on.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;
use rapier3d::prelude::PhysicsWorld;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body::PartRef;
use crate::genome::Genome;
use crate::params::SimParams;
use crate::rng::Rng;
use crate::sim::{Creature, Food, LifeStats, Obstacle, Shard, Sim};
use crate::stats::{GenePool, Sample, Stats};
use crate::terrain::Terrain;

/// EVOLVE07: the world in sections, compressed in blocks.
const MAGIC: &[u8; 8] = b"EVOLVE07";
/// EVOLVE06: the same world as one gzipped stream. Written on one core a few bytes at a time, a
/// 2.7 GB save held the world up for 100 s.
const GZIP_MAGIC: &[u8; 8] = b"EVOLVE06";
/// Older saves still load, their creatures rebuilt from their genomes where they stood (see
/// `import_legacy`): EVOLVE05 had one physics world, EVOLVE04 separate joint constraints (which
/// pumped energy into bodies with many joints), EVOLVE03 single-block bodies with one-hinge
/// limbs.
const LEGACY_MAGICS: [&[u8; 8]; 3] = [b"EVOLVE05", b"EVOLVE04", b"EVOLVE03"];
pub const EXTENSION: &str = "evo";

/// Bytes of a section compressed as one piece.
const BLOCK: usize = 16 << 20;
/// Creatures turned into bytes together, as one section.
const CREATURES_PER_SECTION: usize = 16;
/// zstd's fastest level: on a save, level 3 was half again as slow for under 1% smaller.
const LEVEL: i32 = 1;

/// Saves the whole world and waits until it's written. Takes `&mut` because brain state is
/// first pulled back from the GPU.
pub fn save(sim: &mut Sim, path: &Path) -> Result<u64, String> {
    begin(sim, path)?.wait()
}

/// A save being compressed and written on other threads.
pub struct Saving {
    path: PathBuf,
    /// How long the world waited while it was turned into bytes.
    pub held: Duration,
    /// When the world was let go to run on while this is written.
    pub let_go: Instant,
    thread: JoinHandle<Result<u64, String>>,
}

impl Saving {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_finished(&self) -> bool {
        self.thread.is_finished()
    }

    /// Waits for the file to be written: its size in bytes.
    pub fn wait(self) -> Result<u64, String> {
        self.thread.join().unwrap_or_else(|_| Err("writing it panicked".into()))
    }
}

/// Starts saving the whole world: turns it into bytes (the world waits for that, a second or so
/// for a big one), then compresses and writes them on other threads.
pub fn begin(sim: &mut Sim, path: &Path) -> Result<Saving, String> {
    let start = Instant::now();
    sim.sync_brains();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let sections = encode(sim)?;
    let held = start.elapsed();
    let target = path.to_path_buf();
    let thread = std::thread::Builder::new()
        .name("evolve-save".into())
        .spawn(move || write(&target, sections))
        .map_err(|e| format!("start writing: {e}"))?;
    Ok(Saving {
        path: path.to_path_buf(),
        held,
        let_go: Instant::now(),
        thread,
    })
}

/// A world turned into bytes: the rest of it, the terrain, the gene pool, each physics world, then
/// the creatures `CREATURES_PER_SECTION` at a time.
struct Sections {
    shards: u32,
    creature_sections: u32,
    bytes: Vec<Vec<u8>>,
}

fn encode(sim: &mut Sim) -> Result<Sections, String> {
    // The big parts come out of the world while the rest of it is turned into bytes, and go back
    // before anything else can happen to it.
    let terrain = std::mem::take(&mut sim.terrain);
    let pool = std::mem::take(&mut sim.pool);
    let shards = std::mem::take(&mut sim.shards);
    let creatures = std::mem::take(&mut sim.creatures);
    let rest = rmp_serde::to_vec_named(&*sim);
    let chunks: Vec<&[Creature]> = creatures.chunks(CREATURES_PER_SECTION).collect();
    let parts: Result<Vec<Vec<u8>>, _> = (0..2 + shards.len() + chunks.len())
        .into_par_iter()
        .map(|k| match k {
            0 => rmp_serde::to_vec_named(&terrain),
            1 => rmp_serde::to_vec_named(&pool),
            k if k < 2 + shards.len() => rmp_serde::to_vec_named(&shards[k - 2]),
            k => rmp_serde::to_vec_named(chunks[k - 2 - shards.len()]),
        })
        .collect();
    let counts = (shards.len() as u32, chunks.len() as u32);
    drop(chunks);
    sim.terrain = terrain;
    sim.pool = pool;
    sim.shards = shards;
    sim.creatures = creatures;
    let mut bytes = vec![rest.map_err(|e| format!("encode: {e}"))?];
    bytes.extend(parts.map_err(|e| format!("encode: {e}"))?);
    Ok(Sections {
        shards: counts.0,
        creature_sections: counts.1,
        bytes,
    })
}

/// Threads that compress saves: a quarter of the cores, so the world running on meanwhile keeps
/// the rest.
fn save_threads() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get() / 4).max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("evolve-save-{i}"))
            .build()
            .expect("save threads")
    })
}

/// A section's bytes compressed `block` bytes at a time.
fn compress(raw: &[u8], block: usize) -> std::io::Result<Vec<Vec<u8>>> {
    save_threads().install(|| raw.par_chunks(block).map(|b| zstd::bulk::compress(b, LEVEL)).collect())
}

/// Undoes `compress`: `len` bytes from blocks of `block` bytes each (the last one shorter).
fn decompress(len: usize, blocks: &[&[u8]], block: usize) -> Result<Vec<u8>, String> {
    if blocks.len() != len.div_ceil(block) {
        return Err(format!("{} blocks for {len} bytes", blocks.len()));
    }
    let mut raw = vec![0u8; len];
    raw.par_chunks_mut(block).zip(blocks.par_iter()).try_for_each(|(out, z)| {
        match zstd::bulk::decompress_to_buffer(z, out) {
            Ok(n) if n == out.len() => Ok(()),
            Ok(n) => Err(format!("a block came out {n} bytes, not {}", out.len())),
            Err(e) => Err(format!("decompress: {e}")),
        }
    })?;
    Ok(raw)
}

/// Writes the file: the magic; the block size, the counts of physics worlds and creature sections
/// and the number of sections; then each section as its length, its number of blocks and each
/// block's length and bytes (all little-endian). A section at a time, each one's bytes dropped
/// once written, so a save never holds the whole world twice over besides the world itself.
fn write(path: &Path, sections: Sections) -> Result<u64, String> {
    // Written beside the target and renamed, so a crash never leaves a half-written save.
    let tmp = path.with_extension("evo.tmp");
    let result = (|| -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(MAGIC)?;
        for n in [BLOCK as u32, sections.shards, sections.creature_sections, sections.bytes.len() as u32] {
            w.write_all(&n.to_le_bytes())?;
        }
        for raw in sections.bytes {
            let blocks = compress(&raw, BLOCK)?;
            w.write_all(&(raw.len() as u64).to_le_bytes())?;
            drop(raw);
            w.write_all(&(blocks.len() as u32).to_le_bytes())?;
            for b in &blocks {
                w.write_all(&(b.len() as u32).to_le_bytes())?;
                w.write_all(b)?;
            }
        }
        w.into_inner().map_err(|e| e.into_error())?.sync_all()
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("write {}: {e}", tmp.display()));
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("rename: {e}"))?;
    Ok(std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
}

pub fn load(path: &Path) -> Result<Sim, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).map_err(|e| format!("read: {e}"))?;
    let mut sim = if &magic == MAGIC {
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| format!("read: {e}"))?;
        read_sections(&data)?
    } else if &magic == GZIP_MAGIC || LEGACY_MAGICS.contains(&&magic) {
        // Unpacked whole before decoding: decoding straight from the gzip stream read it a few
        // bytes at a time.
        let mut data = Vec::new();
        GzDecoder::new(std::io::BufReader::new(file))
            .read_to_end(&mut data)
            .map_err(|e| format!("read: {e}"))?;
        if &magic != GZIP_MAGIC {
            let old: LegacySim = rmp_serde::from_slice(&data).map_err(|e| format!("decode: {e}"))?;
            return Ok(import_legacy(old));
        }
        rmp_serde::from_slice(&data).map_err(|e| format!("decode: {e}"))?
    } else {
        return Err(if magic.starts_with(b"EVOLVE") {
            format!("{} was saved by a different version of evolve", path.display())
        } else {
            format!("{} is not an evolve save", path.display())
        });
    };
    sim.after_load();
    Ok(sim)
}

/// A section decoded.
enum Part {
    Rest(Box<Sim>),
    Terrain(Box<Terrain>),
    Pool(GenePool),
    Shard(Box<Shard>),
    Creatures(Vec<Creature>),
}

/// The world from what `write` wrote after the magic.
fn read_sections(data: &[u8]) -> Result<Sim, String> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], String> {
        let s = data.get(at..at + n).ok_or("the save is cut short")?;
        at += n;
        Ok(s)
    };
    let mut u32_at = || -> Result<u32, String> { Ok(u32::from_le_bytes(take(4)?.try_into().unwrap())) };
    let block = u32_at()? as usize;
    let shards = u32_at()? as usize;
    let creature_sections = u32_at()? as usize;
    let count = u32_at()? as usize;
    if block == 0 || count != 3 + shards + creature_sections {
        return Err(format!("a save of {count} sections, for {shards} physics worlds and {creature_sections} lots of creatures"));
    }
    // Where each section's blocks lie.
    let mut sections: Vec<(usize, Vec<&[u8]>)> = Vec::with_capacity(count);
    for _ in 0..count {
        let len = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let z = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            blocks.push(take(z)?);
        }
        sections.push((len, blocks));
    }
    let parts: Vec<Part> = sections
        .par_iter()
        .enumerate()
        .map(|(k, (len, blocks))| {
            let raw = decompress(*len, blocks, block)?;
            let decoded = match k {
                0 => rmp_serde::from_slice(&raw).map(|s| Part::Rest(Box::new(s))),
                1 => rmp_serde::from_slice(&raw).map(|t| Part::Terrain(Box::new(t))),
                2 => rmp_serde::from_slice(&raw).map(Part::Pool),
                k if k < 3 + shards => rmp_serde::from_slice(&raw).map(|s| Part::Shard(Box::new(s))),
                _ => rmp_serde::from_slice(&raw).map(Part::Creatures),
            };
            decoded.map_err(|e| format!("decode: {e}"))
        })
        .collect::<Result<_, String>>()?;
    let mut parts = parts.into_iter();
    let Some(Part::Rest(sim)) = parts.next() else { unreachable!("the first section is the rest of the world") };
    let mut sim = *sim;
    for part in parts {
        match part {
            Part::Terrain(t) => sim.terrain = *t,
            Part::Pool(p) => sim.pool = p,
            Part::Shard(s) => sim.shards.push(*s),
            Part::Creatures(c) => sim.creatures.extend(c),
            Part::Rest(_) => unreachable!("only the first section is the rest of the world"),
        }
    }
    Ok(sim)
}

/// The parts of an older save worth keeping. Its creatures' joints (and, for EVOLVE03, brain
/// states) don't fit today's bodies, so they are rebuilt from their (upgraded) genomes where they
/// stood.
#[derive(Deserialize)]
struct LegacySim {
    params: SimParams,
    world: PhysicsWorld,
    terrain: Terrain,
    creatures: Vec<LegacyCreature>,
    plants: Vec<Food>,
    meat: Vec<Food>,
    obstacles: Vec<Obstacle>,
    rng: Rng,
    tick: u64,
    time: f64,
    next_id: u64,
    stats: Stats,
    history: Vec<Sample>,
    pool: GenePool,
}

#[derive(Deserialize)]
struct LegacyCreature {
    genome: Genome,
    parts: Vec<PartRef>,
    generation: u32,
    parents: Option<(u64, u64)>,
    energy: f32,
    age: f32,
    life: LifeStats,
}

fn import_legacy(old: LegacySim) -> Sim {
    let mut params = old.params;
    // Water drag used to be all linear at 400; it's now 150 plus drag that grows with speed.
    if params.water_drag == 400.0 {
        params.water_drag = SimParams::default().water_drag;
    }
    let mut sim = Sim::new(params);
    sim.install_terrain(old.terrain);
    sim.plants = old.plants;
    sim.meat = old.meat;
    sim.rng = old.rng;
    sim.tick = old.tick;
    sim.time = old.time;
    sim.next_id = old.next_id;
    sim.stats = old.stats;
    sim.history = old.history;
    sim.pool = old.pool;
    for e in &mut sim.pool.entries {
        e.genome.upgrade();
    }
    for o in &old.obstacles {
        if let Some(b) = old.world.bodies.get(o.body) {
            let (axis, angle) = b.rotation().to_axis_angle();
            let yaw = if axis.y >= 0.0 { angle } else { -angle };
            sim.add_obstacle(b.translation(), o.shape, o.dynamic, yaw);
        }
    }
    for c in old.creatures {
        let Some(at) = c.parts.first().and_then(|p| old.world.bodies.get(p.body)).map(|b| b.translation())
        else {
            continue;
        };
        let mut genome = c.genome;
        genome.upgrade();
        let id = sim.spawn_creature(genome, at, 0.3, c.energy, c.generation, c.parents);
        if let Some(n) = sim.creature_mut(id) {
            n.age = c.age;
            n.life = c.life;
        }
    }
    sim.refresh_weather();
    sim.log("Loaded a save from an older version: creatures were rebuilt from their genomes");
    sim
}

/// `$XDG_DATA_HOME/evolve/saves`, or `~/.local/share/evolve/saves`.
pub fn default_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("evolve").join("saves")
}

/// Save files in `dir`, newest first.
pub fn list(dir: &Path) -> Vec<(PathBuf, std::time::SystemTime, u64)> {
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == EXTENSION))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((e.path(), m.modified().ok()?, m.len()))
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_come_back_as_they_went() {
        let mut rng = Rng::new(3);
        // Random bytes don't compress, so each block is stored about as it was; a long run of one
        // byte does.
        let mut raw: Vec<u8> = (0..2500).map(|_| rng.range(0.0, 256.0) as u8).collect();
        raw.extend(std::iter::repeat_n(7u8, 2000));
        for block in [1000, 4500, 10_000] {
            let blocks = compress(&raw, block).unwrap();
            assert_eq!(blocks.len(), raw.len().div_ceil(block));
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            assert_eq!(decompress(raw.len(), &refs, block).unwrap(), raw);
        }
        assert!(compress(&[], 1000).unwrap().is_empty(), "an empty section has no blocks");
        assert_eq!(decompress(0, &[], 1000).unwrap(), Vec::<u8>::new());
        let blocks = compress(&raw, 1000).unwrap();
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        assert!(decompress(raw.len(), &refs[1..], 1000).is_err(), "a block missing is an error");
        assert!(decompress(raw.len() - 1, &refs, 1000).is_err(), "and so is a wrong length");
    }
}
