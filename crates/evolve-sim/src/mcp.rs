//! A Model Context Protocol server, so an AI assistant can look into a running world: reports
//! on how lives go, lists and descriptions of creatures, a creature followed for a few seconds
//! to see how it moves, parameters read and set, the world saved, and (in the app) a picture.
//!
//! It speaks MCP's "streamable HTTP" transport in its simplest form (every request answered
//! with one JSON response, no event streams) on 127.0.0.1 only, and turns away requests from web
//! pages (any Origin header other than localhost). Register it with
//! `claude mcp add --transport http evolve http://127.0.0.1:7311/mcp`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::time::Duration;

use serde_json::{Value, json};

use crate::params::ParamRef;
use crate::runner::Control;
use crate::sim::Sim;

pub const DEFAULT_PORT: u16 = 7311;

/// What a tool returns: text, or text with a PNG picture.
pub type Reply = Sender<Result<String, String>>;

/// A question for one world, answered on the world's own thread between steps.
#[derive(Clone, Debug)]
pub enum Ask {
    Report { minutes: f64 },
    Summary,
    Creatures { sort: String, count: usize },
    Creature { id: u64 },
    Watch { id: u64, seconds: f32 },
    Params { filter: String },
    /// How many background worlds are running (answered by the runner).
    Worlds,
    SetParam { name: String, value: String },
    /// Several at once: all are checked before any is set (answered by the runner).
    SetParams(Vec<(String, String)>),
    /// Rebuild the landscape of the world on screen from the current settings, keeping all life.
    RegenerateTerrain,
    /// Start the world on screen over with the current settings (and restart the background
    /// worlds), as the New world button does.
    NewWorld,
    Save(PathBuf),
}

/// A picture of the world from the app (there is none in the headless runner).
pub struct Shot {
    /// Look at this creature, or else at the scene as the user sees it.
    pub creature: Option<u64>,
    pub reply: Sender<Result<Vec<u8>, String>>,
}

impl Sim {
    /// Answers the questions any world can answer about itself.
    pub fn answer(&mut self, ask: Ask, reply: Reply) {
        let text = match ask {
            Ask::Report { minutes } => Ok(self.report(minutes * 60.0)),
            Ask::Summary => Ok(self.summary_line(600.0)),
            Ask::Creatures { sort, count } => self.list_creatures(&sort, count),
            Ask::Creature { id } => self.creature_details(id),
            Ask::Watch { id, seconds } => return self.watch(id, seconds, reply),
            Ask::Params { filter } => Ok(self.params_text(&filter)),
            Ask::Worlds | Ask::SetParam { .. } | Ask::SetParams(_) | Ask::RegenerateTerrain | Ask::NewWorld | Ask::Save(_) => {
                Err("the runner answers that".into())
            }
        };
        let _ = reply.send(text);
    }

    fn list_creatures(&self, sort: &str, count: usize) -> Result<String, String> {
        let key = |c: &crate::sim::Creature| -> f32 {
            match sort {
                "age" => c.age,
                "mass" => c.mass,
                "lineage" => c.lineage as f32,
                "distance" => c.life.distance,
                "children" => c.life.children as f32,
                "kills" => c.life.kills as f32,
                "energy" => c.energy,
                "food" => c.life.food(),
                "neurons" => c.genome.brain.h as f32,
                "generation" => c.generation as f32,
                _ => self.life_score(c),
            }
        };
        let known = ["score", "age", "mass", "lineage", "distance", "children", "kills", "energy", "food", "neurons", "generation"];
        if !known.contains(&sort) {
            return Err(format!("sort_by must be one of {}", known.join(", ")));
        }
        let mut v: Vec<&crate::sim::Creature> = self.creatures.iter().collect();
        v.sort_by(|a, b| key(b).total_cmp(&key(a)));
        let mut out = format!("{} creatures alive; the top {} by {sort}:\n", v.len(), count.min(v.len()));
        for c in v.into_iter().take(count) {
            out.push_str(&self.describe(c));
            out.push('\n');
        }
        Ok(out)
    }

    fn creature_details(&self, id: u64) -> Result<String, String> {
        let c = self.creature(id).ok_or_else(|| format!("no creature #{id} in this world"))?;
        let t = &c.genome.traits;
        let pos = self.torso_of(c).map(|b| b.translation()).unwrap_or(c.last_pos);
        let s = &c.life.spent;
        let mut out = String::new();
        out.push_str(&self.describe(c));
        out.push_str(&format!(
            "\nBody: {}\nFrom: {}\nAt ({:.1}, {:.1}, {:.1}); health {:.0}/{:.0}, water {:.0}%, breath {:.0}%, mating drive {:.2}, carrying earth {:.2} m3.\n",
            crate::observe::body_summary(c),
            match c.parents {
                Some((a, b)) if a == b => format!("born to #{a} alone"),
                Some((a, b)) => format!("born to #{a} and #{b}"),
                None => "fell from the sky".into(),
            },
            pos.x,
            pos.y,
            pos.z,
            c.health,
            c.max_health,
            c.hydration * 100.0,
            c.oxygen * 100.0,
            c.mate_drive,
            c.soil
        ));
        out.push_str(&format!(
            "Genes: size {:.2}, diet {:.2} (0 plants, 1 meat), aggression {:.2}, libido {:.2}, appetite {:.2}; brain {} neurons ({} connections).\n",
            t.size,
            t.diet,
            t.aggression,
            t.libido,
            t.appetite,
            c.genome.brain.h,
            c.genome.brain.synapses()
        ));
        out.push_str(&format!(
            "Life so far: ate {:.1} plants and {:.1} meat; spent basal {:.1}, brain {:.1}, muscle {:.1}, healing {:.1}, digging {:.1}, calls and attacks {:.1}; gave children {:.1}; {} attacks.\n",
            c.life.plant_eaten, c.life.meat_eaten, s.basal, s.brain, s.muscle, s.healing, s.digging, s.other, c.life.invested, c.life.attacks
        ));
        let o = |k: usize| c.output(k);
        use crate::brain::output;
        out.push_str(&format!(
            "Wants now: eat {:.2}, attack {:.2}, mate {:.2}, call {:.2}, dig {:.2}, pile {:.2}, grip {:.2} (above 0 means yes).\n",
            o(output::EAT),
            o(output::ATTACK),
            o(output::MATE),
            o(output::CALL),
            o(output::DIG),
            o(output::PILE),
            o(output::GRIP)
        ));
        Ok(out)
    }

    pub(crate) fn params_text(&mut self, filter: &str) -> String {
        let mut out = String::new();
        let filter = filter.to_lowercase();
        self.params.visit(&mut |info, r| {
            let hit = filter.is_empty() || info.name.contains(&filter) || info.group.to_lowercase().contains(&filter);
            if !hit {
                return;
            }
            let value = match r {
                ParamRef::F32(x) => format!("{x}"),
                ParamRef::U32(x) => format!("{x}"),
                ParamRef::U64(x) => format!("{x}"),
                ParamRef::Bool(x) => format!("{x}"),
            };
            let doc = info.doc.split_whitespace().collect::<Vec<_>>().join(" ");
            out.push_str(&format!("{} = {value}  [{}; {} to {}] {doc}\n", info.name, info.group, info.min, info.max));
        });
        if out.is_empty() {
            out = format!("no parameter matches {filter}");
        }
        out
    }
}

/// Starts the server on its own thread and returns its port (port 0 picks a free one). Fails if
/// the port is taken (another evolve is serving).
pub fn serve(port: u16, runner: Sender<Control>, shots: Option<Sender<Shot>>) -> std::io::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let port = listener.local_addr()?.port();
    std::thread::Builder::new().name("evolve-mcp".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let (runner, shots) = (runner.clone(), shots.clone());
            let _ = std::thread::Builder::new().name("evolve-mcp-call".into()).spawn(move || {
                let _ = handle(stream, &runner, shots.as_ref());
            });
        }
    })?;
    Ok(port)
}

fn handle(stream: TcpStream, runner: &Sender<Control>, shots: Option<&Sender<Shot>>) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut out = stream;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let mut parts = line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
        let mut length = 0usize;
        let mut origin_ok = true;
        let mut keep_alive = true;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h)? == 0 {
                return Ok(());
            }
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            let Some((name, value)) = h.split_once(':') else { continue };
            let (name, value) = (name.trim().to_ascii_lowercase(), value.trim());
            match name.as_str() {
                "content-length" => length = value.parse().unwrap_or(0),
                "origin" => {
                    origin_ok = ["http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"]
                        .iter()
                        .any(|o| value == *o || value.starts_with(&format!("{o}:")))
                }
                "connection" => keep_alive = !value.eq_ignore_ascii_case("close"),
                _ => {}
            }
        }
        let mut body = vec![0u8; length.min(1 << 20)];
        reader.read_exact(&mut body)?;
        let (status, payload) = if !origin_ok {
            ("403 Forbidden", None)
        } else if !path.starts_with("/mcp") {
            ("404 Not Found", None)
        } else if method == "POST" {
            match serde_json::from_slice::<Value>(&body) {
                Ok(msg) => match rpc(&msg, runner, shots) {
                    Some(v) => ("200 OK", Some(v)),
                    None => ("202 Accepted", None),
                },
                Err(e) => ("400 Bad Request", Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": e.to_string()}}))),
            }
        } else if method == "DELETE" {
            ("200 OK", None)
        } else {
            // No server-sent event stream: everything comes back in the POST's response.
            ("405 Method Not Allowed", None)
        };
        let body = payload.map(|v| v.to_string()).unwrap_or_default();
        write!(
            out,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}\r\n{body}",
            body.len(),
            if status.starts_with("405") { "Allow: POST, DELETE\r\n" } else { "" }
        )?;
        out.flush()?;
        if !keep_alive {
            return Ok(());
        }
    }
}

/// Handles one JSON-RPC message; None for notifications, which get no answer.
fn rpc(msg: &Value, runner: &Sender<Control>, shots: Option<&Sender<Shot>>) -> Option<Value> {
    let id = msg.get("id").cloned()?;
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(json!({}));
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": params.get("protocolVersion").and_then(Value::as_str).unwrap_or("2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "evolve", "version": env!("CARGO_PKG_VERSION")},
            "instructions": INSTRUCTIONS,
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tools(shots.is_some())})),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            Ok(match call(name, &args, runner, shots) {
                Ok(content) => json!({"content": content}),
                Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
            })
        }
        _ => Err(json!({"code": -32601, "message": format!("no method {method}")})),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    })
}

const INSTRUCTIONS: &str = "evolve is a 3D evolution simulator. Creatures (bodies of boxes and capsules jointed by servo motors, each run by a recurrent neural network on the GPU) live on an island, eat plants and meat, fight, and breed. When there are fewer than min_population, new ones fall from the sky, mostly bred from a pool of past successes. The world on screen is world 0; background worlds (1, 2, ...) evolve in parallel and trade their best genomes. Start with worlds_summary or world_report; watch_creature shows how one moves.";

fn tools(pictures: bool) -> Value {
    let world = json!({"type": "integer", "description": "0 for the world on screen (the default), 1 and up for background worlds", "minimum": 0});
    let mut list = vec![
        json!({
            "name": "worlds_summary",
            "description": "One line per world on how the last 10 simulated minutes went: population, births vs creatures from the sky, deaths, age at death, food eaten, travel, longest line of births, body and brain sizes.",
            "inputSchema": {"type": "object", "properties": {}},
        }),
        json!({
            "name": "world_report",
            "description": "A full report on one world: food standing, births and arrivals from the sky, deaths by cause and age, where energy comes from and goes (basal, brain, muscle, healing...), travel and its muscle cost, how close the starved were to food, the living's bodies and brains, lineages, and the best creatures alive.",
            "inputSchema": {"type": "object", "properties": {
                "world": world,
                "minutes": {"type": "number", "description": "simulated minutes to look back over (default 10, at most 30)", "minimum": 0.5, "maximum": 30},
            }},
        }),
        json!({
            "name": "list_creatures",
            "description": "The living creatures of a world, best first by a measure, one line each (id, life score, mass, parts, neurons, generation, births in a row since the sky, age, energy, food, children, kills, travel).",
            "inputSchema": {"type": "object", "properties": {
                "world": world,
                "sort_by": {"type": "string", "enum": ["score", "age", "mass", "lineage", "distance", "children", "kills", "energy", "food", "neurons", "generation"], "description": "default score"},
                "count": {"type": "integer", "minimum": 1, "maximum": 200, "description": "default 20"},
            }},
        }),
        json!({
            "name": "creature",
            "description": "Everything about one living creature: body, ancestry, genes, state, lifetime energy accounts, and what its brain wants right now.",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}, "world": world}, "required": ["id"]},
        }),
        json!({
            "name": "watch_creature",
            "description": "Follows a creature for some simulated seconds and describes how it moves: net travel and path, speed, whether it moves head first, turning, how upright it stays and how high, which limb tips touch the ground and how often they step, whether it closes on food, energy spent, and its track second by second. Takes as long as the world needs to run that many seconds.",
            "inputSchema": {"type": "object", "properties": {
                "id": {"type": "integer"},
                "world": world,
                "seconds": {"type": "number", "minimum": 0.5, "maximum": 120, "description": "default 10"},
            }, "required": ["id"]},
        }),
        json!({
            "name": "get_params",
            "description": "Current parameter values with their ranges and meaning; filter by part of a name or group (e.g. 'brain', 'Population', 'plant').",
            "inputSchema": {"type": "object", "properties": {"filter": {"type": "string"}}},
        }),
        json!({
            "name": "set_param",
            "description": "Changes a parameter in every world (as the parameters panel does). Some, like the island's shape, only take effect in a new world.",
            "inputSchema": {"type": "object", "properties": {
                "name": {"type": "string"},
                "value": {"type": ["string", "number", "boolean"]},
            }, "required": ["name", "value"]},
        }),
        json!({
            "name": "set_params",
            "description": "Changes several parameters in every world at once, e.g. {\"values\": {\"min_population\": 300, \"rain_from_zoo\": 0.4}}. All are checked before any is set. Ones that shape a new world (island, terrain_resolution...) take effect with regenerate_terrain or new_world.",
            "inputSchema": {"type": "object", "properties": {
                "values": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean"]}},
            }, "required": ["values"]},
        }),
        json!({
            "name": "regenerate_terrain",
            "description": "Rebuilds the landscape of the world on screen from the current settings (world_size, terrain_resolution, island shape...), keeping every creature, as the Regenerate terrain button does. Large terrains take a while and run slowly: every doubling of terrain_resolution is four times the cells.",
            "inputSchema": {"type": "object", "properties": {}},
        }),
        json!({
            "name": "new_world",
            "description": "Discards the world on screen and starts over with the current settings, seeded from the zoo, and restarts the background worlds, as the New world button does. Save it first (save_world) to keep it.",
            "inputSchema": {"type": "object", "properties": {"confirm": {"type": "boolean", "description": "must be true"}}, "required": ["confirm"]},
        }),
        json!({
            "name": "save_world",
            "description": "Saves the world on screen to a new .evo file (it won't overwrite one), e.g. to study it offline with the diagnostic tests.",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "description": "absolute path ending in .evo"}}, "required": ["path"]},
        }),
    ];
    if pictures {
        list.push(json!({
            "name": "screenshot",
            "description": "A picture of the world on screen: close behind a creature if one is given (the view returns to what it was), otherwise the view as it is.",
            "inputSchema": {"type": "object", "properties": {"creature": {"type": "integer"}}},
        }));
    }
    Value::Array(list)
}

/// How long to wait for a world to answer. A watch takes as long as the world needs to run it.
const WAIT: Duration = Duration::from_secs(600);

fn ask(runner: &Sender<Control>, world: usize, ask: Ask) -> Result<String, String> {
    let (tx, rx) = mpsc::channel();
    runner.send(Control::Ask { world, ask, reply: tx }).map_err(|_| "the simulation has stopped".to_string())?;
    rx.recv_timeout(WAIT).map_err(|_| "the world didn't answer in time".to_string())?
}

fn call(name: &str, args: &Value, runner: &Sender<Control>, shots: Option<&Sender<Shot>>) -> Result<Vec<Value>, String> {
    let world = args.get("world").and_then(Value::as_u64).unwrap_or(0) as usize;
    let text = |s: String| vec![json!({"type": "text", "text": s})];
    let id = || args.get("id").and_then(Value::as_u64).ok_or("id must be a creature number");
    match name {
        "worlds_summary" => {
            let n: usize = ask(runner, 0, Ask::Worlds)?.parse().unwrap_or(0);
            let mut out = String::new();
            for w in 0..=n {
                let line = ask(runner, w, Ask::Summary).unwrap_or_else(|e| e);
                let label = if w == 0 { "World 0 (on screen)".to_string() } else { format!("World {w}") };
                out.push_str(&format!("{label}: {line}\n"));
            }
            Ok(text(out))
        }
        "world_report" => {
            let minutes = args.get("minutes").and_then(Value::as_f64).unwrap_or(10.0).clamp(0.5, 30.0);
            ask(runner, world, Ask::Report { minutes }).map(text)
        }
        "list_creatures" => {
            let sort = args.get("sort_by").and_then(Value::as_str).unwrap_or("score").to_string();
            let count = args.get("count").and_then(Value::as_u64).unwrap_or(20).clamp(1, 200) as usize;
            ask(runner, world, Ask::Creatures { sort, count }).map(text)
        }
        "creature" => ask(runner, world, Ask::Creature { id: id()? }).map(text),
        "watch_creature" => {
            let seconds = args.get("seconds").and_then(Value::as_f64).unwrap_or(10.0) as f32;
            ask(runner, world, Ask::Watch { id: id()?, seconds }).map(text)
        }
        "get_params" => {
            let filter = args.get("filter").and_then(Value::as_str).unwrap_or("").to_string();
            ask(runner, 0, Ask::Params { filter }).map(text)
        }
        "set_param" => {
            let name = args.get("name").and_then(Value::as_str).ok_or("name is required")?.to_string();
            let value = match args.get("value") {
                Some(Value::String(s)) => s.clone(),
                Some(v @ (Value::Number(_) | Value::Bool(_))) => v.to_string(),
                _ => return Err("value is required".into()),
            };
            ask(runner, 0, Ask::SetParam { name, value }).map(text)
        }
        "set_params" => {
            let values = args.get("values").and_then(Value::as_object).ok_or("values is required: an object of name: value")?;
            let pairs = values
                .iter()
                .map(|(k, v)| match v {
                    Value::String(s) => Ok((k.clone(), s.clone())),
                    Value::Number(_) | Value::Bool(_) => Ok((k.clone(), v.to_string())),
                    _ => Err(format!("{k}: give a number, true/false or text")),
                })
                .collect::<Result<Vec<_>, String>>()?;
            ask(runner, 0, Ask::SetParams(pairs)).map(text)
        }
        "regenerate_terrain" => ask(runner, 0, Ask::RegenerateTerrain).map(text),
        "new_world" => {
            if args.get("confirm").and_then(Value::as_bool) != Some(true) {
                return Err("new_world discards the world on screen; pass confirm: true".into());
            }
            ask(runner, 0, Ask::NewWorld).map(text)
        }
        "save_world" => {
            let path = PathBuf::from(args.get("path").and_then(Value::as_str).ok_or("path is required")?);
            if !path.is_absolute() || path.extension().is_none_or(|e| e != "evo") {
                return Err("path must be absolute and end in .evo".into());
            }
            if path.exists() {
                return Err(format!("{} already exists", path.display()));
            }
            ask(runner, 0, Ask::Save(path)).map(text)
        }
        "screenshot" => {
            let shots = shots.ok_or("pictures need the app; this is the headless runner")?;
            let creature = args.get("creature").and_then(Value::as_u64);
            let (tx, rx) = mpsc::channel();
            shots.send(Shot { creature, reply: tx }).map_err(|_| "the app has stopped".to_string())?;
            let png = rx.recv_timeout(Duration::from_secs(30)).map_err(|_| "no picture came back".to_string())??;
            Ok(vec![json!({"type": "image", "data": base64(&png), "mimeType": "image/png"})])
        }
        _ => Err(format!("no tool named {name}")),
    }
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(T[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A PNG of 8-bit RGB pixels, row by row from the top.
pub fn png_rgb(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    let chunk = |out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut crc = flate2::Crc::new();
        crc.update(kind);
        crc.update(data);
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        out.extend_from_slice(&crc.sum().to_be_bytes());
    };
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bits, RGB, deflate, no filter choice, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
    let row = width as usize * 3;
    for r in rgb.chunks(row).take(height as usize) {
        let _ = z.write_all(&[0]);
        let _ = z.write_all(r);
    }
    chunk(&mut out, b"IDAT", &z.finish().unwrap_or_default());
    chunk(&mut out, b"IEND", &[]);
    out
}
