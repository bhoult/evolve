//! Save files: the whole `Sim`, physics state included, as gzipped MessagePack.
//! Saving mid-stride and loading resumes exactly where it left off.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

use rapier3d::prelude::PhysicsWorld;
use serde::Deserialize;

use crate::body::PartRef;
use crate::genome::Genome;
use crate::params::SimParams;
use crate::rng::Rng;
use crate::sim::{Food, LifeStats, Obstacle, Sim};
use crate::stats::{GenePool, Sample, Stats};
use crate::terrain::Terrain;

/// EVOLVE04: multi-jointed bodies (spines, knees, swivels) and thirst.
const MAGIC: &[u8; 8] = b"EVOLVE04";
/// Single-block bodies with one-hinge limbs. Still loads: see `import_legacy`.
const LEGACY_MAGIC: &[u8; 8] = b"EVOLVE03";
pub const EXTENSION: &str = "evo";

/// Saves the whole world. Takes `&mut` because brain state is first pulled back from the GPU.
pub fn save(sim: &mut Sim, path: &Path) -> Result<u64, String> {
    sim.sync_brains();
    let sim = &*sim;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    // Write beside the target and rename, so a crash never leaves a half-written save.
    let tmp = path.with_extension("evo.tmp");
    let result = (|| -> Result<(), String> {
        let file = File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
        let mut w = BufWriter::new(file);
        w.write_all(MAGIC).map_err(|e| e.to_string())?;
        let mut gz = GzEncoder::new(w, Compression::fast());
        rmp_serde::encode::write_named(&mut gz, sim).map_err(|e| format!("encode: {e}"))?;
        gz.finish()
            .and_then(|mut w| w.flush())
            .map_err(|e| format!("write: {e}"))
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("rename: {e}"))?;
    Ok(std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
}

pub fn load(path: &Path) -> Result<Sim, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).map_err(|e| format!("read: {e}"))?;
    let gz = GzDecoder::new(r);
    if &magic == LEGACY_MAGIC {
        let old: LegacySim = rmp_serde::decode::from_read(gz).map_err(|e| format!("decode: {e}"))?;
        return Ok(import_legacy(old));
    }
    if &magic != MAGIC {
        return Err(if magic.starts_with(b"EVOLVE") {
            format!("{} was saved by a different version of evolve", path.display())
        } else {
            format!("{} is not an evolve save", path.display())
        });
    }
    let mut sim: Sim = rmp_serde::decode::from_read(gz).map_err(|e| format!("decode: {e}"))?;
    sim.after_load();
    Ok(sim)
}

/// The parts of an EVOLVE03 save worth keeping. Its creatures' joints and brain states don't fit
/// today's bodies, so they are rebuilt from their (upgraded) genomes where they stood.
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
    sim.weather = crate::weather::at(&sim.params, sim.time);
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
