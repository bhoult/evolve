//! The Reports window: graphs of the stats record (`evolve_sim::record`) over time, for any
//! world it holds, filtered by metric, kind of creature and time range, with settings changes,
//! loads and experiments marked.

use bevy_egui::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, VLine};
use evolve_sim::record::{self, Connection, METRICS, Point, WorldRow};

/// How far back the graphs go, in hours of wall clock (None: everything).
const RANGES: [(Option<f64>, &str); 5] = [(Some(6.0), "6 h"), (Some(24.0), "24 h"), (Some(48.0), "48 h"), (Some(168.0), "7 days"), (None, "all")];
/// Simulated minutes per point.
const BUCKETS: [(f64, &str); 5] = [(5.0, "5 min"), (15.0, "15 min"), (30.0, "30 min"), (60.0, "1 h"), (120.0, "2 h")];
/// Real seconds between rereading the record while the window is open.
const REFRESH: f64 = 15.0;

/// Per metric shown: per kind of creature, its points.
type Graphs = Vec<(&'static str, Vec<(&'static str, Vec<Point>)>)>;

pub struct Reports {
    pub open: bool,
    /// Behind a lock only because the UI's state must be shareable between threads.
    conn: Option<std::sync::Mutex<Connection>>,
    worlds: Vec<WorldRow>,
    /// The record's id of the world shown; None follows the world on screen.
    world: Option<i64>,
    metrics: Vec<&'static str>,
    groups: Vec<&'static str>,
    range: usize,
    bucket: usize,
    /// Against the wall clock, or against the world's own (simulated) time.
    by_clock: bool,
    series: Graphs,
    events: Vec<(f64, i64, String)>,
    shown: Option<i64>,
    read_at: f64,
    dirty: bool,
}

impl Default for Reports {
    fn default() -> Self {
        Self {
            open: false,
            conn: None,
            worlds: Vec::new(),
            world: None,
            metrics: vec!["travel", "cost", "age_mean", "births", "sky"],
            groups: vec!["all", "pool", "born"],
            range: 2,
            bucket: 2,
            by_clock: true,
            series: Vec::new(),
            events: Vec::new(),
            shown: None,
            read_at: f64::NEG_INFINITY,
            dirty: true,
        }
    }
}

/// A steady color for each kind of creature, the same in every graph.
fn group_color(grp: &str) -> egui::Color32 {
    let rgb = egui::Color32::from_rgb;
    match grp {
        "all" => rgb(230, 230, 230),
        "sky" => rgb(120, 170, 255),
        "born" => rgb(250, 160, 60),
        "pool" => rgb(110, 210, 110),
        "pool_copy" => rgb(60, 160, 90),
        "pool_cross" => rgb(170, 230, 120),
        "random" => rgb(200, 120, 220),
        "zoo" => rgb(240, 220, 90),
        "retest" => rgb(120, 220, 220),
        "dropped" => rgb(220, 110, 120),
        _ => rgb(200, 200, 200),
    }
}

impl Reports {
    /// Rereads the record if the filters changed or it's been a while.
    fn read(&mut self, current_uid: u64, now: f64) {
        if !self.dirty && now - self.read_at < REFRESH {
            return;
        }
        self.read_at = now;
        self.dirty = false;
        if self.conn.is_none() {
            self.conn = record::open_read(&record::default_path()).map(std::sync::Mutex::new);
        }
        let Some(Ok(conn)) = self.conn.as_ref().map(|c| c.lock()) else {
            return;
        };
        let conn = &*conn;
        self.worlds = record::worlds(conn);
        let following = record::world_by_uid(conn, current_uid).or_else(|| self.worlds.iter().find(|w| w.kind == "main").map(|w| w.id));
        let Some(world) = self.world.or(following) else {
            return;
        };
        self.shown = Some(world);
        let since = RANGES[self.range].0.map_or(0, |h| record::now_wall() - (h * 3600.0) as i64);
        let bucket = BUCKETS[self.bucket].0;
        self.series = self
            .metrics
            .iter()
            .filter_map(|&key| {
                let m = record::metric(key)?;
                let groups: Vec<&'static str> = if m.world { vec!["world"] } else { self.groups.clone() };
                let lines = groups.into_iter().map(|g| (g, record::series(conn, world, m, g, bucket, since))).collect();
                Some((key, lines))
            })
            .collect();
        self.events = record::events(conn, world, since);
    }
}

pub fn reports_window(ctx: &egui::Context, st: &mut Reports, current_uid: u64, now: f64) {
    if !st.open {
        return;
    }
    st.read(current_uid, now);
    let mut open = st.open;
    egui::Window::new("Reports")
        .open(&mut open)
        .default_size([760.0, 640.0])
        .vscroll(true)
        .show(ctx, |ui| filters_and_graphs(ui, st));
    st.open = open;
}

fn world_text(w: &WorldRow) -> String {
    format!("{} ({}), {} to {}", w.name, w.kind, record::local_time_text(w.first_wall), record::local_time_text(w.last_wall))
}

fn filters_and_graphs(ui: &mut egui::Ui, st: &mut Reports) {
    if st.conn.is_none() {
        ui.label(format!(
            "Nothing recorded yet: the record ({}) starts with the first five simulated minutes.",
            record::default_path().display()
        ));
        return;
    }
    let mut dirty = false;
    ui.horizontal_wrapped(|ui| {
        ui.label("World");
        let current = st.worlds.iter().find(|w| Some(w.id) == st.shown).map(world_text).unwrap_or_default();
        egui::ComboBox::from_id_salt("reports world").width(380.0).selected_text(current).show_ui(ui, |ui| {
            dirty |= ui.selectable_value(&mut st.world, None, "the world on screen").changed();
            for w in &st.worlds {
                dirty |= ui.selectable_value(&mut st.world, Some(w.id), world_text(w)).changed();
            }
        });
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Last");
        for (k, (_, name)) in RANGES.iter().enumerate() {
            dirty |= ui.selectable_value(&mut st.range, k, *name).changed();
        }
        ui.separator();
        ui.label("Points every");
        for (k, (_, name)) in BUCKETS.iter().enumerate() {
            dirty |= ui.selectable_value(&mut st.bucket, k, *name).changed();
        }
        ui.label("of world time");
        ui.separator();
        dirty |= ui.selectable_value(&mut st.by_clock, true, "by clock").changed();
        dirty |= ui.selectable_value(&mut st.by_clock, false, "by world time").changed();
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Creatures");
        for &(g, label) in record::GROUPS {
            let mut on = st.groups.contains(&g);
            let text = egui::RichText::new(label).color(group_color(g));
            if ui.checkbox(&mut on, text).changed() {
                st.groups.retain(|&x| x != g);
                if on {
                    st.groups.push(g);
                }
                dirty = true;
            }
        }
    });
    egui::CollapsingHeader::new(format!("Metrics ({} shown)", st.metrics.len())).id_salt("reports metrics").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            for m in METRICS {
                let mut on = st.metrics.contains(&m.key);
                if ui.checkbox(&mut on, m.label).changed() {
                    st.metrics.retain(|&x| x != m.key);
                    if on {
                        // Keep the order of the list, so graphs don't jump about.
                        st.metrics.push(m.key);
                        st.metrics.sort_by_key(|k| METRICS.iter().position(|x| x.key == *k));
                    }
                    dirty = true;
                }
            }
        });
    });
    st.dirty |= dirty;
    if st.series.is_empty() {
        ui.label("Pick a metric to graph.");
        return;
    }

    let by_clock = st.by_clock;
    // Event times on the graphs' x axis.
    let marks: Vec<(f64, &str)> = st.events.iter().map(|(sim_min, wall, text)| (if by_clock { *wall as f64 } else { sim_min / 60.0 }, text.as_str())).collect();
    for (key, lines) in &st.series {
        let Some(m) = record::metric(key) else {
            continue;
        };
        ui.add_space(4.0);
        ui.label(egui::RichText::new(m.label).strong());
        let near_event = |x: f64, span: f64| -> Option<&str> {
            marks.iter().filter(|(t, _)| (t - x).abs() < span).min_by(|a, b| (a.0 - x).abs().total_cmp(&(b.0 - x).abs())).map(|e| e.1)
        };
        let span = if by_clock { 1800.0 } else { 0.5 };
        let mut plot = Plot::new(("reports", *key))
            .height(170.0)
            .legend(Legend::default())
            .allow_scroll(false)
            .link_axis("reports x", [true, false])
            .link_cursor("reports x", [true, false])
            .label_formatter(move |name, p| {
                let at = if by_clock { record::local_time_text(p.x as i64) } else { format!("{:.1} h", p.x) };
                let mut s = if name.is_empty() { format!("{at}\n{:.2}", p.y) } else { format!("{name}\n{at}\n{:.2}", p.y) };
                if let Some(e) = near_event(p.x, span) {
                    s.push_str(&format!("\n⚑ {e}"));
                }
                s
            });
        plot = if by_clock {
            plot.x_axis_formatter(|mark, _| record::local_time_text(mark.value as i64))
        } else {
            plot.x_axis_label("world time, hours")
        };
        plot.show(ui, |p| {
            for (x, _) in &marks {
                p.vline(VLine::new("", *x).color(egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40)));
            }
            for (grp, points) in lines {
                if points.is_empty() {
                    continue;
                }
                let pts: PlotPoints = points.iter().map(|q| [if by_clock { q.wall } else { q.sim_min / 60.0 }, q.value]).collect();
                let name = if *grp == "world" { m.label.to_string() } else { record::GROUPS.iter().find(|g| g.0 == *grp).map_or(*grp, |g| g.1).to_string() };
                let color = if *grp == "world" { egui::Color32::from_rgb(230, 230, 230) } else { group_color(grp) };
                p.line(Line::new(name, pts).color(color));
            }
        });
    }
    if !st.events.is_empty() {
        ui.add_space(6.0);
        egui::CollapsingHeader::new(format!("Changes and events ({})", st.events.len())).id_salt("reports events").show(ui, |ui| {
            for (sim_min, wall, text) in st.events.iter().rev() {
                ui.label(format!("{}  (world {:.1} h)  {text}", record::local_time_text(*wall), sim_min / 60.0));
            }
        });
    }
}
