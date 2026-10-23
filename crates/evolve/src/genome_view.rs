//! The genome window: every gene of the selected creature, where it sits in its possible range,
//! and what it does to the body and the behavior.

use bevy_egui::egui::{self, RichText};
use evolve_sim::SimParams;
use evolve_sim::brain::{N_IN, N_OUT};
use evolve_sim::genome::{
    ANGLE, CPG_FREQ, LIMB_LENGTH, LIMB_LIMIT, LIMB_RADIUS, LIMB_STRENGTH, LimbGene, MAX_LIMB_SEGMENTS, MAX_SPINE, Range, SIZE,
    SPINE_LENGTH, SPINE_LIMIT, TAPER, TORSO_HALF, TRACE_TIME, UNIT, ZERO_ONE,
};
use evolve_sim::snapshot::CreatureDetail;

use crate::help::num;

const BAR: f32 = 64.0;

pub fn genome_window(ctx: &egui::Context, c: &CreatureDetail, open: &mut bool, p: &SimParams) {
    egui::Window::new(format!("Genome of #{}", c.id))
        .id(egui::Id::new("genome"))
        .open(open)
        .default_pos([12.0, 90.0])
        .default_size([440.0, 640.0])
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.add(
                    egui::Label::new(format!(
                        "Genes are the recipe for a creature's body and instincts, fixed for its life. A child's genes are a mix of its parents', then each gene has a {}% chance of changing a little. The bars show where each gene sits in the range it can take.",
                        num(p.mutation_rate * 100.0)
                    ))
                    .wrap(),
                );
                ui.add_space(4.0);
                egui::CollapsingHeader::new("Body").default_open(true).show(ui, |ui| body(ui, c));
                egui::CollapsingHeader::new("Spine").default_open(true).show(ui, |ui| spine(ui, c, p));
                egui::CollapsingHeader::new("Limbs").default_open(true).show(ui, |ui| limbs(ui, c, p));
                egui::CollapsingHeader::new("Instincts and upkeep").default_open(true).show(ui, |ui| traits(ui, c, p));
                egui::CollapsingHeader::new("Learning").default_open(true).show(ui, |ui| learning(ui, c));
                egui::CollapsingHeader::new("Voice").default_open(false).show(ui, |ui| voice(ui, c));
                egui::CollapsingHeader::new("Brain").default_open(true).show(ui, |ui| brain(ui, c, p));
            });
        });
}

/// Where `v` sits in `r`, 0..1.
fn at(v: f32, r: Range) -> Option<f32> {
    Some(((v - r.lo) / (r.hi - r.lo)).clamp(0.0, 1.0))
}

fn deg(radians: f32) -> String {
    format!("{:.0}°", radians.to_degrees())
}

/// One gene: a bar for where it sits in its range (if it has one), its name and value, and what it
/// does underneath.
fn gene(ui: &mut egui::Ui, frac: Option<f32>, name: &str, value: impl Into<String>, meaning: impl Into<String>) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(BAR, 8.0), egui::Sense::hover());
        if let Some(f) = frac {
            let v = ui.visuals();
            ui.painter().rect_filled(rect, 2.0, v.extreme_bg_color);
            let mut fill = rect;
            fill.set_width((rect.width() * f).max(2.0));
            ui.painter().rect_filled(fill, 2.0, v.selection.bg_fill);
        }
        ui.label(RichText::new(name).strong());
        ui.label(value.into());
    });
    let meaning = meaning.into();
    if !meaning.is_empty() {
        ui.horizontal_top(|ui| {
            ui.add_space(BAR + ui.spacing().item_spacing.x);
            ui.add(egui::Label::new(RichText::new(meaning).weak()).wrap());
        });
    }
    ui.add_space(2.0);
}

fn body(ui: &mut egui::Ui, c: &CreatureDetail) {
    let s = c.traits.size;
    gene(
        ui,
        at(s, SIZE),
        "Size",
        format!("{s:.2}x"),
        format!(
            "Scales every part; mass grows with its cube. This body weighs {:.0} kg and can store {:.0} energy.",
            c.mass, c.max_energy
        ),
    );
    let h = c.torso.half;
    gene(
        ui,
        at(h[0], TORSO_HALF[0]),
        "Torso length",
        format!("{:.2} m", 2.0 * h[0] * s),
        "Front to back. The torso is the head end: the eyes and the mouth are at its front, and it bites with that end.",
    );
    gene(ui, at(h[1], TORSO_HALF[1]), "Torso height", format!("{:.2} m", 2.0 * h[1] * s), "");
    gene(ui, at(h[2], TORSO_HALF[2]), "Torso width", format!("{:.2} m", 2.0 * h[2] * s), "");
    ui.label(format!(
        "Built: {} body segment{}, {} limb segments, {} hinges and {} swivels.",
        c.trunk,
        if c.trunk == 1 { "" } else { "s" },
        c.limbs,
        c.hinges,
        c.swivels
    ));
}

fn strength_note(p: &SimParams, strength: f32) -> String {
    format!(
        "Motor strength: up to {:.1} N·m per kg of body at each joint.",
        p.motor_torque_per_kg * strength
    )
}

fn spine(ui: &mut egui::Ui, c: &CreatureDetail, p: &SimParams) {
    let sp = &c.spine;
    let n = sp.segments as usize;
    gene(
        ui,
        Some(n as f32 / MAX_SPINE as f32),
        "Segments",
        format!("{n} of {MAX_SPINE}"),
        if n == 0 {
            "None: the body is a single block. A mutation can add segments.".to_string()
        } else {
            format!("{n} segment{} trailing behind the torso, each joined to the one in front.", if n == 1 { "" } else { "s" })
        },
    );
    if n == 0 {
        return;
    }
    gene(ui, at(sp.length, SPINE_LENGTH), "Length", format!("{:.0}%", sp.length * 100.0), "Each segment's length, relative to the torso's.");
    gene(
        ui,
        at(sp.taper, TAPER),
        "Taper",
        format!("{:.2}", sp.taper),
        format!("Each segment is {:.0}% as thick as the one in front.", sp.taper * 100.0),
    );
    let a = sp.axis_angle;
    let quarter = std::f32::consts::FRAC_PI_2;
    let way = if (a - quarter).abs() < quarter / 4.0 {
        "up and down, like a caterpillar"
    } else if a < quarter / 4.0 || a > 3.5 * quarter {
        "side to side, like a snake"
    } else {
        "diagonally, between a snake's sideways wave and a caterpillar's up and down"
    };
    gene(
        ui,
        at(a, ANGLE),
        "Bend direction",
        deg(a),
        format!("The joints bend {way}."),
    );
    gene(
        ui,
        None,
        "Joints",
        if sp.swivel { "swivels" } else { "hinges" },
        if sp.swivel {
            format!("Bend any way, but each unit of work costs {}x a hinge's.", num(p.swivel_energy_cost))
        } else {
            "Bend one way (the direction above). The cheap kind.".to_string()
        },
    );
    gene(ui, at(sp.limit, SPINE_LIMIT), "Range", format!("±{}", deg(sp.limit)), "How far each spine joint can bend.");
    gene(ui, at(sp.strength, LIMB_STRENGTH), "Strength", format!("{:.2}x", sp.strength), strength_note(p, sp.strength));
}

fn limb_noun(lg: &LimbGene, pair: bool) -> String {
    let n = (lg.segments as usize).clamp(1, MAX_LIMB_SEGMENTS);
    let noun = if lg.swivel_knees && n > 1 {
        "tentacle"
    } else {
        match n {
            1 => "stick",
            2 => "two-part limb",
            _ => "three-part limb",
        }
    };
    let paddle = lg.paddle >= 0.2;
    match (pair, paddle) {
        (true, true) => format!("a pair of {noun}s with paddles"),
        (true, false) => format!("a pair of {noun}s"),
        (false, true) => format!("a {noun} with a paddle"),
        (false, false) => format!("a {noun}"),
    }
}

fn face(attach: [f32; 3]) -> &'static str {
    let [x, y, z] = attach.map(f32::abs);
    if x >= y && x >= z {
        if attach[0] >= 0.0 { "front" } else { "back" }
    } else if y >= z {
        if attach[1] >= 0.0 { "top" } else { "underside" }
    } else {
        "side"
    }
}

fn limbs(ui: &mut egui::Ui, c: &CreatureDetail, p: &SimParams) {
    ui.add(
        egui::Label::new(
            "Six limb genes, each on or off. A limb grows from a body segment or from the tip of an earlier limb, and can be mirrored into a pair. Genes that are off still mutate, and a mutation can switch one back on, settings and all.",
        )
        .wrap(),
    );
    // Which genes grew, and which as pairs: a limb on a missing limb doesn't grow, and one on a
    // mirrored limb is mirrored with it.
    let n = c.limb_genes.len();
    let mut grown = vec![false; n];
    let mut paired = vec![false; n];
    for (i, lg) in c.limb_genes.iter().enumerate() {
        if !lg.present {
            continue;
        }
        if lg.parent == 0 {
            grown[i] = true;
            paired[i] = lg.mirror;
        } else {
            let pg = lg.parent as usize - 1;
            if pg < i && grown[pg] {
                grown[i] = true;
                paired[i] = paired[pg];
            }
        }
    }
    for (i, lg) in c.limb_genes.iter().enumerate() {
        let title = if grown[i] {
            let place = if lg.parent == 0 {
                let seg = (lg.segment as usize).min(c.trunk.saturating_sub(1));
                let on = if seg == 0 { "the torso".to_string() } else { format!("body segment {seg}") };
                format!("on {on}'s {}", face(lg.attach))
            } else {
                format!("on the tip of limb {}", lg.parent)
            };
            format!("Limb {}: {} {place}", i + 1, limb_noun(lg, paired[i]))
        } else if lg.present {
            format!("Limb {}: on, but not grown (limb {} isn't there)", i + 1, lg.parent)
        } else {
            format!("Limb {}: off", i + 1)
        };
        egui::CollapsingHeader::new(title)
            .id_salt(("limb gene", i))
            .default_open(grown[i])
            .show(ui, |ui| limb(ui, lg, c, p));
    }
}

fn limb(ui: &mut egui::Ui, lg: &LimbGene, c: &CreatureDetail, p: &SimParams) {
    let s = c.traits.size;
    let n = (lg.segments as usize).clamp(1, MAX_LIMB_SEGMENTS);
    gene(
        ui,
        None,
        "Grows from",
        match (lg.parent, (lg.segment as usize).min(c.trunk.saturating_sub(1))) {
            (0, 0) => "the torso".to_string(),
            (0, seg) => format!("body segment {seg}"),
            (k, _) => format!("the tip of limb {k}"),
        },
        if lg.parent == 0 {
            if lg.mirror { "Mirrored: a matching limb grows on the other side." } else { "Not mirrored: a single limb." }
        } else {
            "A limb on a limb: mirrored if the one it grows on is."
        },
    );
    gene(
        ui,
        Some((n - 1) as f32 / (MAX_LIMB_SEGMENTS - 1) as f32),
        "Segments",
        format!("{n}"),
        "1 is a stick, 2 a limb with a knee, 3 with an ankle as well.",
    );
    gene(ui, at(lg.length, LIMB_LENGTH), "Length", format!("{:.2} m", lg.length * s), "Of the first segment.");
    gene(ui, at(lg.radius, LIMB_RADIUS), "Thickness", format!("{:.2} m", 2.0 * lg.radius * s), "");
    gene(
        ui,
        at(lg.taper, TAPER),
        "Taper",
        format!("{:.2}", lg.taper),
        format!("Each further segment is {:.0}% as long and thick as the one before.", lg.taper * 100.0),
    );
    gene(
        ui,
        None,
        "Base joint",
        if lg.swivel_root { "swivel" } else { "hinge" },
        if lg.swivel_root {
            format!(
                "A hip or shoulder that swings any way, at {}x a hinge's energy per unit of work.",
                num(p.swivel_energy_cost)
            )
        } else {
            format!("Swings one way, in the direction set by its axis gene ({} around the limb).", deg(lg.axis_angle))
        },
    );
    gene(ui, at(lg.limit, LIMB_LIMIT), "Base range", format!("±{}", deg(lg.limit)), "How far the base joint can swing.");
    if n > 1 {
        gene(
            ui,
            None,
            "Knees",
            if lg.swivel_knees { "swivels" } else { "hinges" },
            if lg.swivel_knees {
                "Bend any way: with a swivel base too, the limb is a tentacle."
            } else {
                "Bend mostly one way, a quarter as far the other, like real knees and elbows."
            },
        );
        gene(ui, at(lg.knee_limit, LIMB_LIMIT), "Knee range", format!("±{}", deg(lg.knee_limit)), "");
        gene(
            ui,
            at(lg.curl, UNIT),
            "Curl",
            format!("{:+.2}", lg.curl),
            format!("The knees rest bent {} and bend mostly toward that side.", deg(lg.curl.abs() * 0.8)),
        );
    }
    gene(ui, at(lg.strength, LIMB_STRENGTH), "Strength", format!("{:.2}x", lg.strength), strength_note(p, lg.strength));
    gene(
        ui,
        at(lg.paddle, ZERO_ONE),
        "Tip",
        if lg.paddle >= 0.2 { format!("paddle {:.2}", lg.paddle) } else { format!("round ({:.2})", lg.paddle) },
        "From 0.2 up the tip is a flat paddle or foot, which pushes hard against water when swept face-first.",
    );
}

fn traits(ui: &mut egui::Ui, c: &CreatureDetail, p: &SimParams) {
    let t = &c.traits;
    gene(
        ui,
        at(t.diet, ZERO_ONE),
        "Diet",
        format!("{:.0}% meat", t.diet * 100.0),
        format!(
            "Plants give it {:.0}% of their energy and meat {:.0}% (at least 5% each); the rest is wasted.",
            (1.0 - t.diet).max(0.05) * 100.0,
            t.diet.max(0.05) * 100.0
        ),
    );
    gene(
        ui,
        at(t.aggression, UNIT),
        "Aggression",
        format!("{:+.2}", t.aggression),
        "Added to its brain's attack output: above 0 it bites whatever its head touches unless its brain holds it back.",
    );
    gene(
        ui,
        at(t.libido, UNIT),
        "Libido",
        format!("{:+.2}", t.libido),
        format!(
            "Added to its mate output, and sets how fast its mating drive builds ({:.2}x).",
            2f32.powf(t.libido.clamp(-1.0, 1.0))
        ),
    );
    gene(ui, at(t.appetite, UNIT), "Appetite", format!("{:+.2}", t.appetite), "Added to its eat output.");
    gene(
        ui,
        at(t.armor, ZERO_ONE),
        "Armor",
        format!("{:.2}", t.armor),
        format!(
            "Bites do {:.0}% less damage; resting metabolism is {:.0}% higher.",
            60.0 * t.armor,
            50.0 * t.armor
        ),
    );
    gene(
        ui,
        at(t.breath, ZERO_ONE),
        "Lungs",
        format!("{:.2}", t.breath),
        format!(
            "Holds its breath {:.0} s underwater at rest and breathes back {:.0}% of a breath a second; resting metabolism is {:.0}% higher.",
            p.breath_time * (0.3 + 1.7 * t.breath),
            p.breath_rate * (0.6 + 0.8 * t.breath) * 100.0,
            30.0 * t.breath
        ),
    );
    gene(
        ui,
        at(t.cpg_freq, CPG_FREQ),
        "Rhythm",
        format!("{:.2} Hz", t.cpg_freq),
        "An internal clock its brain senses: a beat to time a gait or a swim stroke to.",
    );
    gene(ui, at(t.hue, ZERO_ONE), "Color", format!("{:.2}", t.hue), "Its lineage's hue. It drifts slowly, so relatives look alike.");
}

fn learning(ui: &mut egui::Ui, c: &CreatureDetail) {
    let t = &c.traits;
    gene(ui, at(t.plasticity, ZERO_ONE), "Plasticity", format!("{:.2}", t.plasticity), "How fast it learns during its life.");
    gene(
        ui,
        at(t.explore, ZERO_ONE),
        "Explore",
        format!("{:.2}", t.explore),
        "How much it varies its movements to find better ones. 0 repeats itself exactly; too much spoils a good gait.",
    );
    gene(
        ui,
        at(t.trace_time, TRACE_TIME),
        "Memory",
        format!("{:.1} s", t.trace_time),
        "How far back a reward can reach to credit the movements that led to it.",
    );
    gene(
        ui,
        at(t.seek, ZERO_ONE),
        "Seek",
        format!("{:.2}", t.seek),
        "How good it feels to close in on food when hungry, water when thirsty and a mate when ready. This is what points its learning at them.",
    );
}

fn voice(ui: &mut egui::Ui, c: &CreatureDetail) {
    let v = c.traits.voice;
    gene(
        ui,
        at(v[0], ZERO_ONE),
        "Pitch",
        format!("{:.0} Hz", 110.0 * 10f32.powf(v[0].clamp(0.0, 1.0))),
        "Its natural pitch; its brain can call higher or lower.",
    );
    gene(ui, at(v[1], ZERO_ONE), "Timbre", format!("{:.2}", v[1]), "From pure to metallic and bright.");
    gene(ui, at(v[2], ZERO_ONE), "Rasp", format!("{:.2}", v[2]), "From clean to rasping.");
    gene(ui, at(v[3], ZERO_ONE), "Warble", format!("{:.2}", v[3]), "From steady to wavering.");
}

fn brain(ui: &mut egui::Ui, c: &CreatureDetail, p: &SimParams) {
    let (lo, hi) = (p.brain_min_neurons.max(1) as f32, p.brain_max_neurons.max(2) as f32);
    let frac = ((c.neurons as f32).max(lo).ln() - lo.ln()) / (hi.ln() - lo.ln()).max(1e-6);
    gene(
        ui,
        Some(frac.clamp(0.0, 1.0)),
        "Neurons",
        format!("{}", c.neurons),
        format!(
            "Hidden neurons, between {} and {} (the bar is on a log scale). About {:.0}% of births resize a brain by some 20%.",
            p.brain_min_neurons,
            p.brain_max_neurons,
            p.brain_resize_rate * 100.0
        ),
    );
    let synapses = if c.synapses >= 1_000_000 {
        format!("{:.2} million", c.synapses as f64 / 1e6)
    } else {
        format!("{:.1} thousand", c.synapses as f64 / 1e3)
    };
    gene(
        ui,
        None,
        "Connections",
        synapses,
        format!(
            "Its wiring: {N_IN} senses feed {} hidden neurons, each connected to every other, which drive {N_OUT} actions. Every connection's weight is a gene. Children mix their parents' brains neuron by neuron. It costs {:.4} energy a second to run.",
            c.neurons, c.brain_cost
        ),
    );
    gene(
        ui,
        None,
        "Learned",
        format!("{:.0}% of its inherited wiring", 100.0 * c.learned / c.inherited.max(1e-6)),
        format!(
            "What it has learned in life sits on top of its genes, not in them, but {:.0}% of it is written into the genes it passes on.",
            p.learned_inheritance * 100.0
        ),
    );
}
