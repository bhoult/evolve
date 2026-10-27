//! The record: how every world has done over time, kept in a SQLite database that outlives the
//! worlds, so a run can be looked back on across restarts, loads and experiments (the reports
//! screen graphs it; the MCP server's `stats_history` reads it).
//!
//! Each world adds a sample every `WINDOW` simulated seconds: for each kind of creature (by
//! where it came from) the raw sums over the lives that ended in the window (deaths, minutes
//! lived, meters travelled, muscle energy...), and for the world as a whole a few counts and
//! levels. Rates are worked out when read, from the sums, so windows add up exactly into longer
//! buckets. Settings changes, loads and experiments are kept as events to mark the graphs.
//!
//! Nothing is recorded until `start` is called (the app does; tests and headless runs don't).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, params};

use crate::observe::Obituary;
use crate::params::ParamRef;
use crate::sim::{DeathCause, Origin, Sim};

/// Simulated seconds each sample covers.
pub const WINDOW: f64 = 5.0 * 60.0;

/// Seconds apart (wall clock) that changes to the same setting are folded into one event, so a
/// slider dragged across its range is one change, not a hundred.
const FOLD_SECS: i64 = 10;

/// What a world is called in the record.
#[derive(Clone, Debug)]
pub struct Recording {
    /// Empty: "World, seed N".
    pub name: String,
    /// "main", "island" or "arm".
    pub kind: &'static str,
}

impl Default for Recording {
    fn default() -> Self {
        Self { name: String::new(), kind: "main" }
    }
}

#[derive(Clone, Debug)]
struct WorldTag {
    uid: u64,
    name: String,
    kind: &'static str,
    seed: u64,
}

enum Msg {
    Sample { world: WorldTag, sim_min: f64, wall: i64, rows: Vec<(&'static str, &'static str, f64)> },
    Event { world: WorldTag, sim_min: f64, wall: i64, text: String },
    Param { world: WorldTag, sim_min: f64, wall: i64, name: String, old: String, new: String },
    Flush(Sender<()>),
}

static TX: OnceLock<Sender<Msg>> = OnceLock::new();

/// `$EVOLVE_STATS_DB`, or `stats.sqlite` next to the saves directory.
pub fn default_path() -> PathBuf {
    if let Some(p) = std::env::var_os("EVOLVE_STATS_DB") {
        return PathBuf::from(p);
    }
    let saves = crate::save::default_dir();
    saves.parent().map(|p| p.join("stats.sqlite")).unwrap_or_else(|| PathBuf::from("stats.sqlite"))
}

/// Starts recording to the database at `path` (made if missing), on a thread of its own.
pub fn start(path: &Path) -> Result<(), String> {
    let conn = open(path)?;
    let (tx, rx) = channel();
    TX.set(tx).map_err(|_| "already recording".to_string())?;
    std::thread::Builder::new()
        .name("evolve-record".into())
        .spawn(move || writer(conn, rx))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Whether `start` has been called.
pub fn active() -> bool {
    TX.get().is_some()
}

/// Waits (a second or two at most) for what's been sent to be written, before quitting.
pub fn flush() {
    if let Some(tx) = TX.get() {
        let (done, wait) = channel();
        if tx.send(Msg::Flush(done)).is_ok() {
            let _ = wait.recv_timeout(Duration::from_secs(2));
        }
    }
}

/// Opens (making it and its folder if need be) the record for reading and writing.
pub fn open(path: &Path) -> Result<Connection, String> {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let conn = Connection::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    conn.busy_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         CREATE TABLE IF NOT EXISTS worlds (
             id INTEGER PRIMARY KEY,
             uid TEXT NOT NULL UNIQUE,
             name TEXT NOT NULL,
             kind TEXT NOT NULL,
             seed INTEGER NOT NULL,
             first_wall INTEGER NOT NULL,
             last_wall INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS samples (
             world INTEGER NOT NULL,
             metric TEXT NOT NULL,
             grp TEXT NOT NULL,
             sim_min REAL NOT NULL,
             wall INTEGER NOT NULL,
             value REAL NOT NULL,
             PRIMARY KEY (world, metric, grp, sim_min)
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS events (
             id INTEGER PRIMARY KEY,
             world INTEGER NOT NULL,
             sim_min REAL NOT NULL,
             wall INTEGER NOT NULL,
             text TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS events_by_world ON events (world, sim_min);",
    )
    .map_err(|e| e.to_string())?;
    Ok(conn)
}

/// Opens the record read-only (for the reports screen and the MCP server), or None if there
/// isn't one yet.
pub fn open_read(path: &Path) -> Option<Connection> {
    let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(path, flags).ok()?;
    let _ = conn.busy_timeout(Duration::from_secs(2));
    Some(conn)
}

/// A wall time (Unix seconds) as local "MM-DD HH:MM".
pub fn local_time_text(wall: i64) -> String {
    // Safe: localtime_r only writes the struct it's given.
    let tm = unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        let t = wall as libc::time_t;
        libc::localtime_r(&t, &mut tm);
        tm
    };
    format!("{:02}-{:02} {:02}:{:02}", tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min)
}

pub fn now_wall() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// A world id no other world has had: the clock and the process's random hasher keys mixed.
fn fresh_uid() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish().max(1)
}

fn writer(mut conn: Connection, rx: Receiver<Msg>) {
    let mut ids: HashMap<u64, i64> = HashMap::new();
    // The last event for each (world, setting): its row, when it was last changed, and the
    // value before the change began.
    let mut folding: HashMap<(i64, String), (i64, i64, String)> = HashMap::new();
    while let Ok(first) = rx.recv() {
        let mut batch = vec![first];
        while let Ok(m) = rx.try_recv() {
            batch.push(m);
        }
        let mut flushed = Vec::new();
        let result = (|| -> rusqlite::Result<()> {
            let tx = conn.transaction()?;
            for m in batch {
                match m {
                    Msg::Sample { world, sim_min, wall, rows } => {
                        let id = world_id(&tx, &mut ids, &world, wall)?;
                        let mut put = tx.prepare_cached(
                            "INSERT OR REPLACE INTO samples (world, metric, grp, sim_min, wall, value) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        )?;
                        for (grp, metric, value) in rows {
                            put.execute(params![id, metric, grp, sim_min, wall, value])?;
                        }
                    }
                    Msg::Event { world, sim_min, wall, text } => {
                        let id = world_id(&tx, &mut ids, &world, wall)?;
                        tx.execute("INSERT INTO events (world, sim_min, wall, text) VALUES (?1, ?2, ?3, ?4)", params![id, sim_min, wall, text])?;
                    }
                    Msg::Param { world, sim_min, wall, name, old, new } => {
                        let id = world_id(&tx, &mut ids, &world, wall)?;
                        let key = (id, name.clone());
                        match folding.get_mut(&key) {
                            Some((row, last, first_old)) if wall - *last <= FOLD_SECS => {
                                tx.execute("UPDATE events SET text = ?1 WHERE id = ?2", params![format!("{name}: {first_old} → {new}"), *row])?;
                                *last = wall;
                            }
                            _ => {
                                tx.execute(
                                    "INSERT INTO events (world, sim_min, wall, text) VALUES (?1, ?2, ?3, ?4)",
                                    params![id, sim_min, wall, format!("{name}: {old} → {new}")],
                                )?;
                                folding.insert(key, (tx.last_insert_rowid(), wall, old));
                            }
                        }
                    }
                    Msg::Flush(done) => flushed.push(done),
                }
            }
            tx.commit()
        })();
        if let Err(e) = result {
            eprintln!("evolve: stats record: {e}");
        }
        for done in flushed {
            let _ = done.send(());
        }
    }
}

fn world_id(conn: &Connection, ids: &mut HashMap<u64, i64>, w: &WorldTag, wall: i64) -> rusqlite::Result<i64> {
    let uid = format!("{:016x}", w.uid);
    if let Some(&id) = ids.get(&w.uid) {
        conn.execute("UPDATE worlds SET last_wall = ?1 WHERE id = ?2", params![wall, id])?;
        return Ok(id);
    }
    conn.execute(
        "INSERT INTO worlds (uid, name, kind, seed, first_wall, last_wall) VALUES (?1, ?2, ?3, ?4, ?5, ?5)
         ON CONFLICT (uid) DO UPDATE SET name = excluded.name, kind = excluded.kind, last_wall = excluded.last_wall",
        params![uid, w.name, w.kind, w.seed as i64, wall],
    )?;
    let id = conn.query_row("SELECT id FROM worlds WHERE uid = ?1", [&uid], |r| r.get(0))?;
    ids.insert(w.uid, id);
    Ok(id)
}

/// The kinds of creature samples are kept for: where they came from (`Origin`), and "sky" (all
/// that fell), "pool" (bred from the gene pool: copies and crosses) and "all".
pub const GROUPS: &[(&str, &str)] = &[
    ("all", "all creatures"),
    ("sky", "fell from the sky"),
    ("born", "born in the world"),
    ("pool", "bred from the gene pool"),
    ("pool_copy", "gene-pool copies"),
    ("pool_cross", "gene-pool crosses"),
    ("random", "made at random"),
    ("zoo", "zoo copies"),
    ("retest", "gene-pool retests"),
    ("dropped", "dropped in"),
];

fn groups_of(origin: Origin, rained: bool) -> impl Iterator<Item = &'static str> {
    let own = match origin {
        Origin::Born => Some("born"),
        Origin::Random => Some("random"),
        Origin::PoolClone => Some("pool_copy"),
        Origin::PoolCross => Some("pool_cross"),
        Origin::Zoo => Some("zoo"),
        Origin::Retest => Some("retest"),
        Origin::Dropped => Some("dropped"),
        Origin::Unknown => None,
    };
    let pool = matches!(origin, Origin::PoolClone | Origin::PoolCross).then_some("pool");
    let sky = rained.then_some("sky");
    ["all"].into_iter().chain(own).chain(pool).chain(sky)
}

/// How a metric is worked out from the raw sums in a bucket of samples.
#[derive(Clone, Copy, Debug)]
pub enum Calc {
    /// Sum of one over sum of another, times a scale.
    Ratio(&'static str, &'static str, f64),
    /// Sum of a count over the minutes the samples cover.
    PerMinute(&'static str),
    /// Mean of a level, weighted by the samples' deaths when they have any.
    Mean(&'static str),
}

#[derive(Clone, Copy, Debug)]
pub struct Metric {
    pub key: &'static str,
    pub label: &'static str,
    /// Whether it's a measure of the world as a whole (group "world") rather than per kind.
    pub world: bool,
    pub calc: Calc,
}

const fn per_kind(key: &'static str, label: &'static str, calc: Calc) -> Metric {
    Metric { key, label, world: false, calc }
}

const fn whole(key: &'static str, label: &'static str, calc: Calc) -> Metric {
    Metric { key, label, world: true, calc }
}

/// Everything the record can graph.
pub const METRICS: &[Metric] = &[
    per_kind("travel", "travel, m per minute of life", Calc::Ratio("distance", "lived_min", 1.0)),
    per_kind("cost", "muscle energy per 100 kg per 100 m", Calc::Ratio("muscle", "kg_m", 1e4)),
    per_kind("age_mean", "lifespan, mean s", Calc::Ratio("lived_min", "deaths", 60.0)),
    per_kind("age_p50", "lifespan, median s", Calc::Mean("age_p50")),
    per_kind("age_p90", "lifespan, p90 s", Calc::Mean("age_p90")),
    per_kind("arrivals", "arrivals per minute (births, sky spawns)", Calc::PerMinute("arrivals")),
    per_kind("deaths", "deaths per minute", Calc::PerMinute("deaths")),
    per_kind("eaten", "eaten per minute of life", Calc::Ratio("eaten", "lived_min", 1.0)),
    per_kind("eaten_away", "eaten away from landing, per minute", Calc::Ratio("eaten_away", "lived_min", 1.0)),
    per_kind("children", "children per life", Calc::Ratio("children", "deaths", 1.0)),
    per_kind("kills", "kills per life", Calc::Ratio("kills", "deaths", 1.0)),
    per_kind("mass", "body mass at death, mean kg", Calc::Ratio("mass", "deaths", 1.0)),
    per_kind("neurons", "brain neurons at death, mean", Calc::Ratio("neurons", "deaths", 1.0)),
    per_kind("starved", "starved, % of deaths", Calc::Ratio("starved", "deaths", 100.0)),
    per_kind("killed", "killed, % of deaths", Calc::Ratio("killed", "deaths", 100.0)),
    per_kind("drowned", "drowned, % of deaths", Calc::Ratio("drowned", "deaths", 100.0)),
    per_kind("old_age", "died of old age, % of deaths", Calc::Ratio("old_age", "deaths", 100.0)),
    whole("births", "births per minute", Calc::PerMinute("births")),
    whole("sky", "sky spawns per minute", Calc::PerMinute("sky")),
    whole("sky_share", "sky spawns, % of arrivals", Calc::Ratio("sky", "arrivals", 100.0)),
    whole("alive", "creatures alive", Calc::Mean("alive")),
    whole("born_alive", "creatures alive that were born here", Calc::Mean("born_alive")),
    whole("plants", "plants", Calc::Mean("plants")),
    whole("mass_p50", "body mass of the living, median kg", Calc::Mean("mass_p50")),
    whole("neurons_p50", "brain neurons of the living, median", Calc::Mean("neurons_p50")),
    whole("brain_price", "brain price", Calc::Mean("brain_price")),
    whole("pool_score", "gene pool, mean score", Calc::Mean("pool_score")),
    whole("generation", "generation of the living, max", Calc::Mean("generation")),
];

pub fn metric(key: &str) -> Option<&'static Metric> {
    METRICS.iter().find(|m| m.key == key)
}

impl Sim {
    /// Called every step: once `WINDOW` has passed since the last sample, records one.
    pub(crate) fn record_stats(&mut self) {
        let Some(tx) = TX.get() else {
            return;
        };
        let Some(from) = self.recorded_to else {
            // Loaded or made just now: the census starts empty, so the first window starts here.
            self.recorded_to = Some(self.time);
            return;
        };
        if self.time < from + WINDOW {
            return;
        }
        let rows = self.window_rows(from);
        self.recorded_to = Some(self.time);
        let _ = tx.send(Msg::Sample { world: self.world_tag(), sim_min: self.time / 60.0, wall: now_wall(), rows });
    }

    fn world_tag(&mut self) -> WorldTag {
        if self.world_uid == 0 {
            self.world_uid = fresh_uid();
        }
        let name = if self.recording.name.is_empty() { format!("World, seed {}", self.params.seed) } else { self.recording.name.clone() };
        WorldTag { uid: self.world_uid, name, kind: self.recording.kind, seed: self.params.seed }
    }

    /// Notes something that happened to the world (a load, an experiment), for the graphs.
    pub fn record_event(&mut self, text: impl Into<String>) {
        if let Some(tx) = TX.get() {
            let _ = tx.send(Msg::Event { world: self.world_tag(), sim_min: self.time / 60.0, wall: now_wall(), text: text.into() });
        }
    }

    /// Notes each setting that differs between `old` and the world's settings now.
    pub(crate) fn record_param_changes(&mut self, old: &crate::SimParams) {
        let Some(tx) = TX.get() else {
            return;
        };
        let values = |p: &crate::SimParams| {
            let mut v = Vec::new();
            p.clone().visit(&mut |info, r| {
                let s = match r {
                    ParamRef::F32(x) => format!("{x}"),
                    ParamRef::U32(x) => format!("{x}"),
                    ParamRef::U64(x) => format!("{x}"),
                    ParamRef::Bool(x) => format!("{x}"),
                };
                v.push((info.name, s));
            });
            v
        };
        let changed: Vec<_> = values(old)
            .into_iter()
            .zip(values(&self.params))
            .filter(|(a, b)| a.1 != b.1)
            .map(|(a, b)| (a.0, a.1, b.1))
            .collect();
        if changed.is_empty() {
            return;
        }
        let (world, sim_min, wall) = (self.world_tag(), self.time / 60.0, now_wall());
        for (name, old, new) in changed {
            let _ = tx.send(Msg::Param { world: world.clone(), sim_min, wall, name: name.to_string(), old, new });
        }
    }

    /// The raw sums for lives that ended, and arrivals, since `from`; and the world's levels now.
    fn window_rows(&self, from: f64) -> Vec<(&'static str, &'static str, f64)> {
        let mut out = Vec::new();
        let minutes = (self.time - from) / 60.0;
        let deaths: Vec<&Obituary> = self.census.deaths.iter().filter(|d| d.time >= from).collect();
        for &(grp, _) in GROUPS {
            let d: Vec<&&Obituary> = deaths.iter().filter(|o| groups_of(o.origin, o.rained).any(|g| g == grp)).collect();
            let arrivals = if grp == "born" {
                self.census.births.iter().filter(|b| b.time >= from).count()
            } else {
                self.census.sky.iter().filter(|s| s.0 >= from && groups_of(s.2, true).any(|g| g == grp)).count()
            };
            out.push((grp, "minutes", minutes));
            out.push((grp, "arrivals", arrivals as f64));
            if d.is_empty() {
                out.push((grp, "deaths", 0.0));
                continue;
            }
            let sum = |f: &dyn Fn(&Obituary) -> f32| d.iter().map(|o| f(o) as f64).sum::<f64>();
            let count = |c: DeathCause| d.iter().filter(|o| o.cause == c).count() as f64;
            let mut ages: Vec<f32> = d.iter().map(|o| o.age).collect();
            ages.sort_by(f32::total_cmp);
            let at = |q: f32| ages[((ages.len() - 1) as f32 * q).round() as usize] as f64;
            out.extend([
                (grp, "deaths", d.len() as f64),
                (grp, "lived_min", sum(&|o| o.age) / 60.0),
                (grp, "distance", sum(&|o| o.life.distance)),
                (grp, "muscle", sum(&|o| o.life.spent.muscle)),
                (grp, "kg_m", sum(&|o| o.mass * o.life.distance)),
                (grp, "eaten", sum(&|o| o.life.food())),
                (grp, "eaten_away", sum(&|o| o.life.earned)),
                (grp, "children", sum(&|o| o.life.children as f32)),
                (grp, "kills", sum(&|o| o.life.kills as f32)),
                (grp, "mass", sum(&|o| o.mass)),
                (grp, "neurons", sum(&|o| o.neurons as f32)),
                (grp, "starved", count(DeathCause::Starved)),
                (grp, "killed", count(DeathCause::Killed)),
                (grp, "drowned", count(DeathCause::Drowned)),
                (grp, "old_age", count(DeathCause::OldAge)),
                (grp, "age_p50", at(0.5)),
                (grp, "age_p90", at(0.9)),
            ]);
        }
        let median = |mut v: Vec<f32>| {
            v.sort_by(f32::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0) as f64
        };
        let births = self.census.births.iter().filter(|b| b.time >= from).count() as f64;
        let sky = self.census.sky.iter().filter(|s| s.0 >= from).count() as f64;
        let pool = &self.pool.entries;
        out.extend([
            ("world", "minutes", minutes),
            ("world", "births", births),
            ("world", "sky", sky),
            ("world", "arrivals", births + sky),
            ("world", "alive", self.creatures.len() as f64),
            ("world", "born_alive", self.creatures.iter().filter(|c| !c.rained).count() as f64),
            ("world", "plants", self.plants.len() as f64),
            ("world", "mass_p50", median(self.creatures.iter().map(|c| c.mass).collect())),
            ("world", "neurons_p50", median(self.creatures.iter().map(|c| c.genome.brain.h as f32).collect())),
            ("world", "brain_price", self.brain_price as f64),
            ("world", "pool_score", pool.iter().map(|e| e.score as f64).sum::<f64>() / pool.len().max(1) as f64),
            ("world", "generation", self.creatures.iter().map(|c| c.generation).max().unwrap_or(0) as f64),
        ]);
        out
    }
}

/// A world in the record.
#[derive(Clone, Debug)]
pub struct WorldRow {
    pub id: i64,
    pub uid: u64,
    pub name: String,
    pub kind: String,
    pub first_wall: i64,
    pub last_wall: i64,
}

/// Every world in the record, most recently active first.
pub fn worlds(conn: &Connection) -> Vec<WorldRow> {
    let Ok(mut q) = conn.prepare("SELECT id, uid, name, kind, first_wall, last_wall FROM worlds ORDER BY last_wall DESC") else {
        return Vec::new();
    };
    q.query_map([], |r| {
        let uid: String = r.get(1)?;
        Ok(WorldRow {
            id: r.get(0)?,
            uid: u64::from_str_radix(&uid, 16).unwrap_or(0),
            name: r.get(2)?,
            kind: r.get(3)?,
            first_wall: r.get(4)?,
            last_wall: r.get(5)?,
        })
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}

/// The record's id for the world with this uid, if it has one.
pub fn world_by_uid(conn: &Connection, uid: u64) -> Option<i64> {
    conn.query_row("SELECT id FROM worlds WHERE uid = ?1", [format!("{uid:016x}")], |r| r.get(0)).optional().ok().flatten()
}

/// One point of a series: the bucket's middle in sim minutes, its mean wall time, the value.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub sim_min: f64,
    pub wall: f64,
    pub value: f64,
}

/// `metric` for `grp` of world `world`, in buckets of `bucket_min` simulated minutes (samples
/// are `WINDOW` long, so anything shorter is one sample a point), from `since_wall` on.
pub fn series(conn: &Connection, world: i64, metric: &Metric, grp: &str, bucket_min: f64, since_wall: i64) -> Vec<Point> {
    let grp = if metric.world { "world" } else { grp };
    // The two raw sums each sample gives; a level of the world as a whole stands alone.
    let (a, b) = match metric.calc {
        Calc::Ratio(a, b, _) => (a, b),
        Calc::PerMinute(a) => (a, "minutes"),
        Calc::Mean(a) if metric.world => (a, a),
        Calc::Mean(a) => (a, "deaths"),
    };
    let Ok(mut q) = conn.prepare_cached(
        "SELECT sim_min, wall, metric, value FROM samples
         WHERE world = ?1 AND grp = ?2 AND metric IN (?3, ?4) AND wall >= ?5 ORDER BY sim_min",
    ) else {
        return Vec::new();
    };
    // Per bucket: sum of a (weighted, for means), sum of b (or the weights), sum of wall times,
    // samples, sum of sim minutes.
    let mut buckets: std::collections::BTreeMap<i64, [f64; 5]> = Default::default();
    let bucket_min = bucket_min.max(WINDOW / 60.0);
    let rows = q.query_map(params![world, grp, a, b, since_wall], |r| {
        Ok((r.get::<_, f64>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?, r.get::<_, f64>(3)?))
    });
    let Ok(rows) = rows else {
        return Vec::new();
    };
    // A sample's a and b arrive as two rows; pair them by sim minute.
    let mut pending: HashMap<u64, (Option<f64>, Option<f64>, i64)> = HashMap::new();
    for (sim_min, wall, m, v) in rows.flatten() {
        let e = pending.entry(sim_min.to_bits()).or_insert((None, None, wall));
        if m == a {
            e.0 = Some(v);
        }
        if m == b {
            e.1 = Some(v);
        }
        let (Some(va), Some(vb)) = (e.0, e.1) else {
            continue;
        };
        // Samples are stamped at their window's end: bucket them by its middle.
        let k = ((sim_min - WINDOW / 120.0) / bucket_min).floor() as i64;
        let s = buckets.entry(k).or_insert([0.0; 5]);
        match metric.calc {
            Calc::Mean(_) => {
                // Weighted by deaths for per-kind levels (a window with one death says little).
                let w = if metric.world { 1.0 } else { vb };
                s[0] += va * w;
                s[1] += w;
            }
            _ => {
                s[0] += va;
                s[1] += vb;
            }
        }
        s[2] += wall as f64;
        s[3] += 1.0;
        s[4] += sim_min;
        pending.remove(&sim_min.to_bits());
    }
    let scale = match metric.calc {
        Calc::Ratio(_, _, k) => k,
        _ => 1.0,
    };
    buckets
        .values()
        .filter(|s| s[1] > 0.0)
        .map(|s| Point { sim_min: s[4] / s[3], wall: s[2] / s[3], value: s[0] / s[1] * scale })
        .collect()
}

/// A world's events from `since_wall` on: (sim minute, wall time, text).
pub fn events(conn: &Connection, world: i64, since_wall: i64) -> Vec<(f64, i64, String)> {
    let Ok(mut q) = conn.prepare_cached("SELECT sim_min, wall, text FROM events WHERE world = ?1 AND wall >= ?2 ORDER BY sim_min") else {
        return Vec::new();
    };
    q.query_map(params![world, since_wall], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::{BirthNote, Obituary};
    use crate::sim::LifeStats;

    fn death(time: f64, origin: Origin, age: f32, distance: f32, muscle: f32, mass: f32) -> Obituary {
        let life = LifeStats {
            distance,
            spent: crate::sim::Spending { muscle, ..Default::default() },
            ..Default::default()
        };
        Obituary {
            time,
            rained: origin != Origin::Born,
            origin,
            lineage: 0,
            cause: DeathCause::Starved,
            age,
            mass,
            parts: 1,
            neurons: 100,
            life,
            energy_left: 0.0,
            nearest_plant: f32::INFINITY,
            foreign: 0.0,
        }
    }

    #[test]
    fn windows_add_up_into_buckets_and_setting_changes_fold() {
        let path = std::env::temp_dir().join(format!("evolve-record-test-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let (tx, rx) = channel();
        let conn = open(&path).unwrap();
        let writer = std::thread::spawn(move || writer(conn, rx));
        let mut sim = Sim::new(crate::SimParams {
            min_population: 0,
            world_size: 60.0,
            terrain_resolution: 33,
            plant_max_count: 0,
            stones: 0,
            tree_max_count: 0,
            ..Default::default()
        });
        let world = sim.world_tag();
        // Two windows: a pool copy that walked 60 m in 2 minutes, then a born one 20 m in 2.
        for (k, (origin, distance)) in [(Origin::PoolClone, 60.0), (Origin::Born, 20.0)].into_iter().enumerate() {
            let from = k as f64 * WINDOW;
            sim.time = from + WINDOW;
            sim.census.deaths.push_back(death(from + 10.0, origin, 120.0, distance, 30.0, 100.0));
            sim.census.births.push_back(BirthNote { time: from + 20.0, lineage: 1, mass: 50.0, foreign: 0.0 });
            sim.census.sky.push_back((from + 30.0, 80.0, Origin::Random));
            sim.census.sky.push_back((from + 40.0, 80.0, Origin::PoolCross));
            let rows = sim.window_rows(from);
            tx.send(Msg::Sample { world: world.clone(), sim_min: sim.time / 60.0, wall: 1000 + k as i64, rows }).unwrap();
        }
        for (wall, new) in [(2000, "2"), (2005, "3"), (2100, "4")] {
            let (name, old) = ("score_child".to_string(), "1".to_string());
            tx.send(Msg::Param { world: world.clone(), sim_min: 10.0, wall, name, old, new: new.into() }).unwrap();
        }
        let (done, wait) = channel();
        tx.send(Msg::Flush(done)).unwrap();
        wait.recv().unwrap();

        let read = open_read(&path).unwrap();
        let id = world_by_uid(&read, sim.world_uid).unwrap();
        let one = |key: &str, grp: &str, bucket: f64| series(&read, id, metric(key).unwrap(), grp, bucket, 0);
        // Window by window, then both in one bucket: 80 m over 4 minutes of life.
        let travel: Vec<f64> = one("travel", "all", 5.0).iter().map(|p| p.value).collect();
        assert_eq!(travel, vec![30.0, 10.0]);
        assert_eq!(one("travel", "all", 10.0)[0].value, 20.0);
        assert_eq!(one("travel", "pool", 10.0)[0].value, 30.0);
        assert!(one("travel", "zoo", 10.0).is_empty());
        // 30 muscle over 100 kg x 60 m, per 100 kg per 100 m.
        assert!((one("cost", "pool", 5.0)[0].value - 30.0 / 6000.0 * 1e4).abs() < 1e-9);
        assert_eq!(one("age_mean", "all", 10.0)[0].value, 120.0);
        // Sky spawns and births per minute: one of each kind per 5-minute window.
        assert_eq!(one("sky", "", 10.0)[0].value, 0.4);
        assert_eq!(one("births", "", 10.0)[0].value, 0.2);
        assert_eq!(one("arrivals", "random", 10.0)[0].value, 0.2);
        assert_eq!(one("arrivals", "born", 10.0)[0].value, 0.2);
        assert!((one("sky_share", "", 10.0)[0].value - 200.0 / 3.0).abs() < 1e-9);
        assert_eq!(one("alive", "", 5.0).len(), 2);
        // Two changes 5 s apart are one event from where it started; one 95 s later is its own.
        let texts: Vec<String> = events(&read, id, 0).into_iter().map(|e| e.2).collect();
        assert_eq!(texts, vec!["score_child: 1 → 3", "score_child: 1 → 4"]);

        drop(tx);
        writer.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }
}
