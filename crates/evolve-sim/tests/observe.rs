//! The observatory (census, reports, watching a creature), the MCP server that serves it, and the
//! parts of the life score and gene pool it helped put right.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use evolve_sim::brain::output;
use evolve_sim::genome::Genome;
use evolve_sim::math::Vec3;
use evolve_sim::rng::Rng;
use evolve_sim::sim::{DeathCause, Origin};
use evolve_sim::stats::GenePool;
use evolve_sim::{Sim, SimParams};

fn quiet_params() -> SimParams {
    SimParams {
        min_population: 0,
        stones: 0,
        tree_max_count: 0,
        terrain_resolution: 65,
        parallel_worlds: 0,
        ..SimParams::default()
    }
}

#[test]
fn the_census_counts_lives_by_where_they_came_from() {
    let mut sim = Sim::new(SimParams { min_population: 6, rain_interval: 0.1, ..quiet_params() });
    for _ in 0..60 {
        sim.step();
    }
    assert_eq!(sim.creatures.len(), 6);
    assert!(sim.creatures.iter().all(|c| c.rained && c.lineage == 0 && c.origin == Origin::Random));
    assert_eq!(sim.census.sky.len(), 6);
    let ids: Vec<u64> = sim.creatures.iter().map(|c| c.id).collect();
    for &id in &ids[..4] {
        sim.kill(id, DeathCause::Starved);
    }
    sim.step();
    assert_eq!(sim.census.deaths.len(), 4);
    assert!(sim.census.deaths.iter().all(|d| d.cause == DeathCause::Starved && d.origin == Origin::Random));
    let report = sim.report(600.0);
    assert!(report.contains("From the sky, made at random: 4 deaths"), "{report}");
    assert!(report.contains("starved 100%"), "{report}");
    assert!(sim.summary_line(600.0).contains("alive"));
}

#[test]
fn a_child_carries_on_its_parents_line() {
    let mut sim = Sim::new(quiet_params());
    let mut rng = Rng::new(5);
    let g = Genome::random(&mut rng, &sim.params);
    let at = sim.random_land_point(10.0);
    let a = sim.spawn_creature(g.clone(), at, 0.5, 50.0, 0, None);
    let b = sim.spawn_creature(g.clone(), at + Vec3::new(3.0, 0.0, 0.0), 0.5, 50.0, 0, None);
    let child = sim.spawn_creature(g.clone(), at + Vec3::new(0.0, 0.0, 3.0), 0.5, 50.0, 1, Some((a, b)));
    let grandchild = sim.spawn_creature(g, at + Vec3::new(3.0, 0.0, 3.0), 0.5, 50.0, 2, Some((child, a)));
    let line = |id| sim.creature(id).map(|c| (c.lineage, c.origin)).unwrap();
    assert_eq!(line(a), (0, Origin::Dropped));
    assert_eq!(line(child), (1, Origin::Born));
    assert_eq!(line(grandchild), (2, Origin::Born));
}

/// The life score credits meters closed on food while there's room to eat, not jumps when the
/// nearest food changes, and not hunger (a creature from the sky starts with the same energy
/// whatever its size, so weighting by hunger would favor big bodies).
#[test]
fn closing_in_on_food_counts_toward_the_life_score() {
    let mut sim = Sim::new(SimParams { plant_spawn_rate: 0.0, plant_spread_rate: 0.0, ..quiet_params() });
    sim.plants.clear();
    let mut rng = Rng::new(11);
    let g = Genome::random(&mut rng, &sim.params);
    let at = sim.random_land_point(30.0);
    // Half full: hungry enough to count, with enough to live through the test.
    let id = sim.spawn_creature(g, at, 0.0, f32::MAX, 0, None);
    let c = sim.creature_mut(id).unwrap();
    c.energy = 0.5 * c.max_energy;
    for _ in 0..60 {
        sim.step();
    }
    let home = sim.torso_of(sim.creature(id).unwrap()).unwrap().translation();
    // Anywhere a plant will grow; it's moved where the test wants it below.
    let plant = (0..200)
        .find_map(|k| sim.add_plant(home + Vec3::new(2.0 + k as f32 * 0.5, 0.0, 0.0), 20.0))
        .expect("somewhere to put a plant");
    let approached = |sim: &Sim| sim.creature(id).unwrap().life.approached;
    let set_gap = |sim: &mut Sim, gap: f32| {
        let torso = sim.torso_of(sim.creature(id).unwrap()).unwrap().translation();
        let k = sim.plants.iter().position(|f| f.id == plant).unwrap();
        sim.plants[k].pos = torso + Vec3::new(gap, 0.0, 0.0);
        sim.plants[k].energy = 20.0;
    };
    for _ in 0..30 {
        sim.step();
    }
    let before = approached(&sim);
    // The food comes 12 m closer, a little each tick (as if the creature walked to it).
    let ticks = 240;
    for t in 0..=ticks {
        set_gap(&mut sim, 18.0 - 12.0 * t as f32 / ticks as f32);
        sim.step();
    }
    let gained = approached(&sim) - before;
    assert!((gained - 12.0).abs() < 1.5, "closed 12 m, credited {gained}");
    // Food that turns up 10 m off in an instant is a new target, not a walk.
    let before = approached(&sim);
    for _ in 0..20 {
        set_gap(&mut sim, 16.0);
        sim.step();
    }
    let jumped = approached(&sim) - before;
    assert!(jumped.abs() < 0.5, "a jump counted {jumped}");
    let c = sim.creature(id).unwrap();
    let parts = sim.score_parts(c);
    assert!(parts[3] >= c.life.approached / 10.0 - 1e-3);
}

/// Food counts toward the life score only once a creature has gone to get it: a meal where it
/// fell earns nothing (big bodies that landed beside plants used to fill their tanks and win).
#[test]
fn only_food_a_creature_went_to_get_counts_toward_its_score() {
    let mut sim = Sim::new(SimParams { plant_spawn_rate: 0.0, plant_spread_rate: 0.0, ..quiet_params() });
    sim.plants.clear();
    let mut rng = Rng::new(11);
    let mut g = Genome::random(&mut rng, &sim.params);
    // Always biting at whatever is at its mouth.
    g.brain.b_out[output::EAT] = 10.0;
    let at = sim.random_land_point(30.0);
    let id = sim.spawn_creature(g, at, 0.0, f32::MAX, 0, None);
    let c = sim.creature_mut(id).unwrap();
    c.energy = 0.3 * c.max_energy;
    for _ in 0..60 {
        sim.step();
    }
    let plant = (0..200)
        .find_map(|k| sim.add_plant(at + Vec3::new(2.0 + k as f32 * 0.5, 0.0, 0.0), 20.0))
        .expect("somewhere to put a plant");
    let meal = |sim: &mut Sim| {
        for _ in 0..90 {
            let c = sim.creature(id).unwrap();
            let mouth = sim.torso_of(c).unwrap().position().transform_point(Vec3::new(c.torso_half().x, 0.0, 0.0));
            let k = sim.plants.iter().position(|f| f.id == plant).unwrap();
            sim.plants[k].pos = mouth;
            sim.plants[k].energy = 20.0;
            sim.step();
        }
    };
    meal(&mut sim);
    let c = sim.creature(id).unwrap();
    assert!(c.life.food() > 10.0, "it ate {}", c.life.food());
    assert_eq!(c.life.earned, 0.0);
    assert_eq!(sim.score_parts(c)[0], 0.0);
    // As if it had walked 10 m from where it fell.
    sim.creature_mut(id).unwrap().home += Vec3::new(10.0, 0.0, 0.0);
    let before = sim.creature(id).unwrap().life.food();
    meal(&mut sim);
    let c = sim.creature(id).unwrap();
    let ate = c.life.food() - before;
    assert!(ate > 10.0, "it ate {ate}");
    assert!((c.life.earned - ate).abs() < 1e-3, "ate {ate}, earned {}", c.life.earned);
    assert!((sim.score_parts(c)[0] - ate / sim.params.plant_max_energy).abs() < 1e-3);
}

#[test]
fn only_kills_made_away_from_where_the_killer_landed_count_toward_its_score() {
    let mut sim = Sim::new(SimParams { score_kill: 3.0, ..quiet_params() });
    let mut rng = Rng::new(12);
    let at = sim.random_land_point(30.0);
    let killer = sim.spawn_creature(Genome::random(&mut rng, &sim.params), at, 0.0, f32::MAX, 0, None);
    let victims: Vec<u64> = (0..2)
        .map(|k| sim.spawn_creature(Genome::random(&mut rng, &sim.params), at + Vec3::new(0.0, 0.0, 3.0 + 3.0 * k as f32), 0.0, f32::MAX, 0, None))
        .collect();
    for _ in 0..30 {
        sim.step();
    }
    let kill = |sim: &mut Sim, victim: u64| {
        let v = sim.creature_mut(victim).unwrap();
        v.health = -50.0;
        v.last_attacker = Some(killer);
        v.attacked_ago = 0.0;
        // Deaths are noticed on the brain's beat.
        for _ in 0..10 {
            sim.step();
        }
        assert!(sim.creature(victim).is_none());
    };
    // Biting whatever landed beside it.
    kill(&mut sim, victims[0]);
    let c = sim.creature(killer).unwrap();
    assert_eq!((c.life.kills, c.life.kills_away), (1, 0));
    assert_eq!(sim.score_parts(c)[1], 0.0);
    // As if it had walked 10 m from where it fell to make this one.
    sim.creature_mut(killer).unwrap().home += Vec3::new(10.0, 0.0, 0.0);
    kill(&mut sim, victims[1]);
    let c = sim.creature(killer).unwrap();
    assert_eq!((c.life.kills, c.life.kills_away), (2, 1));
    assert_eq!(sim.score_parts(c)[1], 3.0);
}

#[test]
fn the_gene_pool_crosses_only_genomes_that_could_interbreed() {
    let p = SimParams::default();
    let mut rng = Rng::new(21);
    let a = Genome::random(&mut rng, &p);
    let strangers = {
        let mut b = Genome::random(&mut rng, &p);
        while a.distance(&b) < p.species_threshold {
            b = Genome::random(&mut rng, &p);
        }
        b
    };
    let kin = {
        let mut b = a.clone();
        b.traits.hue = (b.traits.hue + 0.01).fract();
        b
    };
    assert!(a.distance(&kin) < p.species_threshold);
    let crosses = |other: &Genome, rng: &mut Rng| {
        let mut pool = GenePool::default();
        pool.consider(2.0, 1, &a);
        pool.consider(1.0, 1, other);
        (0..400).filter(|_| pool.breed(rng, &p).unwrap().2).count()
    };
    assert_eq!(crosses(&strangers, &mut rng), 0, "strangers were crossed");
    assert!(crosses(&kin, &mut rng) > 20, "kin were never crossed");
}

/// A genome that got into the pool on one lucky life falls back once exact copies of it, living
/// again, do no better; a retested copy is never let in a second time as a new entry.
#[test]
fn the_gene_pool_retests_genomes_judged_on_one_life() {
    let mut sim = Sim::new(SimParams { rain_from_pool: 1.0, pool_retest: 1.0, ..quiet_params() });
    let mut rng = Rng::new(31);
    let lucky = Genome::random(&mut rng, &sim.params);
    let steady = Genome::random(&mut rng, &sim.params);
    sim.pool.consider(10.0, 3, &lucky);
    sim.pool.consider(4.0, 3, &steady);
    let fp = lucky.fingerprint();
    let id = sim.rain_creature(None);
    let c = sim.creature(id).unwrap();
    assert_eq!(c.origin, Origin::Retest);
    let entry = c.retest_of.expect("a retest knows what it retests");
    assert!(entry == fp || entry == steady.fingerprint());
    // Two more lives of the lucky one, scoring nothing.
    assert!(sim.pool.retested(fp, 0.0));
    assert!(sim.pool.retested(fp, 0.0));
    let e = &sim.pool.entries;
    assert_eq!(e.len(), 2);
    assert_eq!(e[0].genome.fingerprint(), steady.fingerprint(), "the steady one leads now");
    assert!((e[1].score - 10.0 / 3.0).abs() < 1e-4 && e[1].samples == 3);
    // Once the steady one has had its retests too, nothing is left to retest, and rain is bred
    // as usual.
    let s = steady.fingerprint();
    assert!(sim.pool.retested(s, 4.0) && sim.pool.retested(s, 4.0));
    assert!(sim.pool.retest(&mut rng).is_none());
    // A retest that dies is averaged in, not added as a new entry.
    sim.kill(id, DeathCause::Starved);
    sim.step();
    assert_eq!(sim.pool.entries.len(), 2);
    assert!(sim.pool.entries.iter().any(|e| e.samples == 4));
}

#[test]
fn a_watched_creature_is_described_when_the_watch_ends() {
    let mut sim = Sim::new(quiet_params());
    let mut rng = Rng::new(9);
    let g = Genome::random(&mut rng, &sim.params);
    let at = sim.random_land_point(10.0);
    let id = sim.spawn_creature(g, at, 0.5, 60.0, 0, None);
    let (tx, rx) = std::sync::mpsc::channel();
    sim.watch(id, 2.0, tx);
    for _ in 0..(2.5 * 60.0) as usize {
        sim.step();
    }
    let text = rx.try_recv().expect("the watch should be over").unwrap();
    assert!(text.contains(&format!("Watched #{id} for 2.0 s")), "{text}");
    assert!(text.contains("Torso: upright"), "{text}");
    assert!(sim.watches.is_empty());
    let (tx, rx) = std::sync::mpsc::channel();
    sim.watch(999_999, 2.0, tx);
    assert!(rx.try_recv().unwrap().is_err());
}

#[test]
fn png_pictures_inflate_back_to_their_pixels() {
    let (w, h) = (5u32, 3u32);
    let rgb: Vec<u8> = (0..w * h * 3).map(|k| (k * 7 % 256) as u8).collect();
    let png = evolve_sim::mcp::png_rgb(w, h, &rgb);
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&png[12..16], b"IHDR");
    assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), w);
    assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), h);
    let mut crc = flate2::Crc::new();
    crc.update(&png[12..29]);
    assert_eq!(crc.sum(), u32::from_be_bytes(png[29..33].try_into().unwrap()));
    let len = u32::from_be_bytes(png[33..37].try_into().unwrap()) as usize;
    assert_eq!(&png[37..41], b"IDAT");
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(&png[41..41 + len]).read_to_end(&mut raw).unwrap();
    let row = (w * 3) as usize;
    assert_eq!(raw.len(), h as usize * (row + 1));
    for (r, line) in raw.chunks(row + 1).enumerate() {
        assert_eq!(line[0], 0);
        assert_eq!(&line[1..], &rgb[r * row..(r + 1) * row]);
    }
    assert!(png.ends_with(&[0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82]));
}

/// One HTTP request to the server; the status code and the JSON that came back.
fn post(port: u16, body: &str, origin: Option<&str>) -> (u16, serde_json::Value) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let origin = origin.map(|o| format!("Origin: {o}\r\n")).unwrap_or_default();
    write!(
        s,
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n{origin}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut r = BufReader::new(s);
    let mut status = String::new();
    r.read_line(&mut status).unwrap();
    let code = status.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut len = 0;
    loop {
        let mut h = String::new();
        r.read_line(&mut h).unwrap();
        if h.trim().is_empty() {
            break;
        }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; len];
    r.read_exact(&mut body).unwrap();
    (code, serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null))
}

fn call(port: u16, tool: &str, args: serde_json::Value) -> serde_json::Value {
    let msg = serde_json::json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": tool, "arguments": args}});
    let (code, v) = post(port, &msg.to_string(), None);
    assert_eq!(code, 200);
    v["result"].clone()
}

fn text(result: &serde_json::Value) -> String {
    result["content"][0]["text"].as_str().unwrap_or_default().to_string()
}

#[test]
fn the_mcp_server_answers_an_assistant() {
    let dir = std::env::temp_dir().join(format!("evolve-mcp-zoo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: the only test in this binary that starts a runner, which reads EVOLVE_ZOO_DIR.
    unsafe { std::env::set_var("EVOLVE_ZOO_DIR", &dir) };
    let runner = evolve_sim::Runner::start(Sim::new(SimParams { min_population: 5, ..quiet_params() }));
    runner.send(evolve_sim::Control::Speed(0.0));
    let port = evolve_sim::mcp::serve(0, runner.sender(), None).unwrap();

    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#;
    let (code, v) = post(port, init, None);
    assert_eq!(code, 200);
    assert_eq!(v["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(v["result"]["serverInfo"]["name"], "evolve");
    let (code, _) = post(port, r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#, None);
    assert_eq!(code, 202);
    let (code, _) = post(port, init, Some("http://example.com"));
    assert_eq!(code, 403, "web pages are turned away");
    let (_, v) = post(port, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#, None);
    let names: Vec<&str> = v["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"world_report") && names.contains(&"watch_creature"), "{names:?}");
    assert!(!names.contains(&"screenshot"), "no pictures without the app");

    // Wait for the creatures to fall.
    let start = std::time::Instant::now();
    while runner.snapshot().creatures.len() < 5 {
        assert!(start.elapsed().as_secs() < 30, "no creatures arrived");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let summary = text(&call(port, "worlds_summary", serde_json::json!({})));
    assert!(summary.starts_with("World 0 (on screen): t "), "{summary}");
    let report = text(&call(port, "world_report", serde_json::json!({"minutes": 1})));
    assert!(report.contains("Food standing"), "{report}");
    let list = text(&call(port, "list_creatures", serde_json::json!({"sort_by": "mass", "count": 2})));
    let id: u64 = list.split('#').nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap();
    assert!(text(&call(port, "creature", serde_json::json!({"id": id}))).contains("Genes: size"));
    let watched = text(&call(port, "watch_creature", serde_json::json!({"id": id, "seconds": 1})));
    assert!(watched.contains(&format!("Watched #{id}")), "{watched}");
    let bad = call(port, "creature", serde_json::json!({"id": 123456789}));
    assert_eq!(bad["isError"], true);
    let set = text(&call(port, "set_param", serde_json::json!({"name": "min_population", "value": 7})));
    assert!(set.contains("now: min_population = 7"), "{set}");
    assert_eq!(call(port, "set_param", serde_json::json!({"name": "no_such", "value": 1}))["isError"], true);
    let saved = call(port, "save_world", serde_json::json!({"path": "relative.evo"}));
    assert_eq!(saved["isError"], true, "only absolute .evo paths");
    let path = dir.join("world.evo");
    let saved = text(&call(port, "save_world", serde_json::json!({"path": path.to_str().unwrap()})));
    assert!(saved.starts_with("saved") && path.exists(), "{saved}");
    let again = call(port, "save_world", serde_json::json!({"path": path.to_str().unwrap()}));
    assert_eq!(again["isError"], true, "it won't overwrite");
    drop(runner);
    let _ = std::fs::remove_dir_all(&dir);
}
