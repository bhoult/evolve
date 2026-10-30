//! The Load window: how much of the CPU, the GPU, RAM and GPU memory this program uses against
//! everything else, and how fast the world runs, over the last minutes or hours (from
//! `evolve_sim::machine`, which looks every second or two).

use std::time::SystemTime;

use bevy_egui::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use evolve_sim::machine::{self, Sample};

/// evolve's part and everything else's, the same in every graph (a pair that stays apart for
/// colour-blind eyes and on the dark theme). Text keeps the theme's own colours.
const OURS: egui::Color32 = egui::Color32::from_rgb(0x4F, 0x8B, 0xF0);
const OTHERS: egui::Color32 = egui::Color32::from_rgb(0xC9, 0x76, 0x28);
const SINGLE: egui::Color32 = egui::Color32::from_rgb(230, 230, 230);

pub struct LoadView {
    pub open: bool,
    /// Minutes shown.
    span: f64,
    samples: Vec<Sample>,
    looks: u64,
}

impl Default for LoadView {
    fn default() -> Self {
        Self { open: false, span: 10.0, samples: Vec::new(), looks: u64::MAX }
    }
}

pub fn load_window(ctx: &egui::Context, v: &mut LoadView) {
    if !v.open {
        return;
    }
    // Copied again only when there's a new look (every second or two), not every frame.
    let looks = machine::looks();
    if looks != v.looks {
        v.samples = machine::history();
        v.looks = looks;
    }
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
    let mut open = v.open;
    egui::Window::new("Load").open(&mut open).default_width(540.0).default_height(720.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Last");
            for (minutes, name) in [(2.0, "2 min"), (10.0, "10 min"), (60.0, "1 hour"), (120.0, "2 hours")] {
                ui.selectable_value(&mut v.span, minutes, name);
            }
        });
        let span = v.span;
        let shown: Vec<&Sample> = v.samples.iter().filter(|s| now - s.at <= span * 60.0).collect();
        let Some(last) = shown.last().copied() else {
            ui.label("Looking at the computer (every second or two)...");
            return;
        };
        let graph = Graph { shown: &shown, now, span };
        egui::ScrollArea::vertical().show(ui, |ui| {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            graph.stacked(
                ui,
                "cpu",
                &format!(
                    "CPU, % of all {cores} cores: evolve {:.0}%, other programs {:.0}%",
                    last.cpu_ours * 100.0,
                    last.cpu_others * 100.0
                ),
                |s| Some((s.cpu_ours as f64 * 100.0, s.cpu_others as f64 * 100.0)),
                Some(100.0),
                "%",
            );
            match last.gpu_ours.zip(last.gpu_others) {
                Some((ours, others)) => graph.stacked(
                    ui,
                    "gpu",
                    &format!("GPU, % of its time: evolve {:.0}%, other programs {:.0}%", ours * 100.0, others * 100.0),
                    |s| s.gpu_ours.zip(s.gpu_others).map(|(o, r)| (o as f64 * 100.0, r as f64 * 100.0)),
                    Some(100.0),
                    "%",
                ),
                None => {
                    ui.weak("GPU: not known (it's read through nvidia-smi)");
                }
            }
            graph.stacked(
                ui,
                "ram",
                &format!(
                    "RAM, GB of {:.0}: evolve {:.1}, other programs {:.1}",
                    last.ram_total, last.ram_ours, last.ram_others
                ),
                |s| Some((s.ram_ours as f64, s.ram_others as f64)),
                Some(last.ram_total as f64),
                " GB",
            );
            if let (Some(ours), Some(others), Some(total)) = (last.vram_ours, last.vram_others, last.vram_total) {
                graph.stacked(
                    ui,
                    "vram",
                    &format!("GPU memory, GB of {total:.0}: evolve {ours:.1}, other programs {others:.1}"),
                    |s| s.vram_ours.zip(s.vram_others).map(|(o, r)| (o as f64, r as f64)),
                    Some(total as f64),
                    " GB",
                );
            }
            graph.single(
                ui,
                "speed",
                &format!("Speed of the world, × real time: {:.2}×", last.speed),
                |s| s.speed as f64,
                None,
                "×",
            );
            graph.single(
                ui,
                "pace",
                &format!("Low priority's pace, % of full speed: {:.0}%", last.pace * 100.0),
                |s| s.pace as f64 * 100.0,
                Some(100.0),
                "%",
            );
            ui.weak("Time runs left to right, in minutes before now. Hover a graph to read it.");
        });
    });
    v.open = open;
}

struct Graph<'a> {
    shown: &'a [&'a Sample],
    now: f64,
    span: f64,
}

impl Graph<'_> {
    fn plot(&self, id: &str, top: Option<f64>, unit: &'static str) -> Plot<'static> {
        let mut plot = Plot::new(("load", id.to_string()))
            .height(110.0)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .link_cursor("load x", [true, false])
            .include_x(-self.span)
            .include_x(0.0)
            .include_y(0.0)
            .x_axis_formatter(|mark, _| format!("{:.0}", -mark.value))
            .label_formatter(move |name, p| {
                let value = if unit == "×" { format!("{:.2}×", p.y) } else { format!("{:.1}{unit}", p.y) };
                let at = format!("{:.1} min ago", -p.x);
                if name.is_empty() { format!("{at}\n{value}") } else { format!("{name}\n{at}\n{value}") }
            });
        if let Some(t) = top {
            plot = plot.include_y(t);
        }
        plot
    }

    fn x(&self, s: &Sample) -> f64 {
        (s.at - self.now) / 60.0
    }

    /// evolve's part and the other programs' stacked on it: everything is drawn filled, then
    /// evolve's part over its lower part, so the orange left showing is the rest's.
    fn stacked(
        &self,
        ui: &mut egui::Ui,
        id: &str,
        title: &str,
        parts: impl Fn(&Sample) -> Option<(f64, f64)>,
        top: Option<f64>,
        unit: &'static str,
    ) {
        ui.add_space(4.0);
        ui.label(egui::RichText::new(title).strong());
        let points = |f: &dyn Fn(f64, f64) -> f64| -> PlotPoints<'static> {
            self.shown.iter().filter_map(|s| parts(s).map(|(o, r)| [self.x(s), f(o, r)])).collect()
        };
        let all = points(&|o, r| o + r);
        let ours = points(&|o, _| o);
        self.plot(id, top, unit).legend(Legend::default()).show(ui, |p| {
            p.line(Line::new("other programs", all).color(OTHERS).width(1.5).fill(0.0).fill_alpha(0.55));
            p.line(Line::new("evolve", ours).color(OURS).width(1.5).fill(0.0).fill_alpha(1.0));
        });
    }

    /// One series: the title names it, so no legend.
    fn single(&self, ui: &mut egui::Ui, id: &str, title: &str, value: impl Fn(&Sample) -> f64, top: Option<f64>, unit: &'static str) {
        ui.add_space(4.0);
        ui.label(egui::RichText::new(title).strong());
        let points: PlotPoints<'static> = self.shown.iter().map(|s| [self.x(s), value(s)]).collect();
        self.plot(id, top, unit).show(ui, |p| {
            p.line(Line::new("", points).color(SINGLE).width(2.0));
        });
    }
}
