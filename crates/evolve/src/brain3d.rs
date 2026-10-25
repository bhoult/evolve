//! The 3D brain: every neuron of a creature's brain as a point in a brain-shaped volume, its
//! senses on a disc behind and its actions on a disc in front (as the network view has senses
//! on the left and actions on the right, seen from the side at first), and the strongest
//! connections as lines, orange exciting and blue inhibiting, fainter the weaker. Points glow
//! with their live activity. Drag turns it, the wheel zooms, and it turns slowly by itself when
//! left alone.
//!
//! A hidden neuron has no place of its own (the brain computes the same whatever order its
//! neurons are in), so each is put where its number says: an even scatter through the volume,
//! the same for a neuron of that number in any brain. It's drawn with egui in software: points
//! projected each frame and the lines batched into one mesh (`Lines`), so 100,000 stay cheap.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_egui::egui::{self, Color32, Pos2, Shape, vec2};
use evolve_sim::snapshot::CreatureDetail;
use evolve_sim::wiring::{Edge, Node, Wiring};

const BACKGROUND: Color32 = Color32::from_rgb(9, 10, 18);
/// How far the eye is from the middle of the brain, in brain half-lengths.
const EYE: f32 = 3.4;
/// The farthest anything is from the middle (the rims of the discs).
const REACH: f32 = 1.3;
/// Where the discs of senses and actions sit, behind and in front of the brain, and how big.
const FACE: f32 = 1.22;
const DISC: f32 = 0.42;
/// Seconds after the last drag or zoom before it turns by itself, and how fast, radians a second.
const IDLE: f64 = 2.5;
const SPIN: f32 = 0.22;
/// Past this many lines they're drawn without soft edges: half the vertices, and in a crowd of
/// faint lines nobody can tell.
const FEATHER_UP_TO: usize = 20_000;

/// Where every neuron sits, in brain space: x from the back (-1) to the front (+1), y up, z
/// across, with the hemispheres either side of z = 0.
pub struct Layout {
    /// Every hidden neuron, by number.
    pub hidden: Vec<[f32; 3]>,
    /// The senses and actions shown, in the sketch's order.
    pub inputs: Vec<[f32; 3]>,
    pub outputs: Vec<[f32; 3]>,
}

/// Point `i` of an even scatter through the unit ball: Roberts' R3 sequence (the golden ratio's
/// three-dimensional cousin) picks a direction and a depth, so neurons close in number land far
/// apart and the first n fill the ball evenly for any n.
fn in_ball(i: usize) -> [f32; 3] {
    const G: f64 = 1.220_744_084_605_759_5;
    let t = i as f64 + 1.0;
    let u = |k: i32| (0.5 + t / G.powi(k)).fract() as f32;
    let z = 1.0 - 2.0 * u(1);
    let s = (1.0 - z * z).max(0.0).sqrt();
    let a = std::f32::consts::TAU * u(2);
    // Cube root: as many neurons in each shell as it has room for.
    let r = u(3).cbrt();
    [s * a.cos() * r, s * a.sin() * r, z * r]
}

/// The unit ball shaped like a brain: longer than it is wide, flatter underneath, a little
/// narrower in front, and split down the middle into two hemispheres.
fn brain_shape([x, y, z]: [f32; 3]) -> [f32; 3] {
    let y = y * if y < 0.0 { 0.55 } else { 0.72 };
    let z = z * 0.82 * (1.0 - 0.12 * x);
    [x, y, z + 0.05 * z.signum()]
}

/// Point `j` of `m` on a sunflower disc facing along x at `x`.
fn disc(j: usize, m: usize, x: f32) -> [f32; 3] {
    let r = DISC * ((j as f32 + 0.5) / m.max(1) as f32).sqrt();
    let a = j as f32 * 2.399_963;
    [x, r * a.sin() * 0.8, r * a.cos()]
}

/// Where a brain of `hidden` neurons and the `inputs` senses and `outputs` actions shown sit.
pub fn layout(hidden: usize, inputs: usize, outputs: usize) -> Layout {
    Layout {
        hidden: (0..hidden).map(|i| brain_shape(in_ball(i))).collect(),
        inputs: (0..inputs).map(|j| disc(j, inputs, -FACE)).collect(),
        outputs: (0..outputs).map(|j| disc(j, outputs, FACE)).collect(),
    }
}

/// Thin lines batched into one mesh, each a strip with a pixel of soft edge either side (as
/// egui draws lines) or without, skipping egui's work per shape: 100,000 lines take a few
/// milliseconds rather than most of a frame.
pub(crate) struct Lines {
    pub mesh: egui::Mesh,
    /// A physical pixel, in points.
    pixel: f32,
    feather: bool,
}

impl Lines {
    /// Lines into `mesh` (emptied first, so its memory is reused).
    pub fn new(mut mesh: egui::Mesh, pixels_per_point: f32, feather: bool) -> Self {
        mesh.clear();
        Self { mesh, pixel: 1.0 / pixels_per_point, feather }
    }

    /// A line through `pts`, `width` points wide, each point its own color. A line thinner than
    /// a pixel is drawn a pixel wide and fainter.
    pub fn path(&mut self, pts: &[(Pos2, Color32)], width: f32) {
        if pts.len() < 2 {
            return;
        }
        let px = self.pixel;
        let (w, fade) = if width < px { (px, width / px) } else { (width, 1.0) };
        let lanes: u32 = if self.feather { 4 } else { 2 };
        let base = self.mesh.vertices.len() as u32;
        for i in 0..pts.len() {
            let d = pts[(i + 1).min(pts.len() - 1)].0 - pts[i.saturating_sub(1)].0;
            let n = vec2(-d.y, d.x) / d.length().max(1e-6);
            let (p, c) = pts[i];
            let c = if fade < 1.0 { c.gamma_multiply(fade) } else { c };
            if self.feather {
                let (inner, outer) = (n * (w - px).max(0.0) * 0.5, n * (w + px) * 0.5);
                self.mesh.colored_vertex(p + outer, Color32::TRANSPARENT);
                self.mesh.colored_vertex(p + inner, c);
                self.mesh.colored_vertex(p - inner, c);
                self.mesh.colored_vertex(p - outer, Color32::TRANSPARENT);
            } else {
                self.mesh.colored_vertex(p + n * w * 0.5, c);
                self.mesh.colored_vertex(p - n * w * 0.5, c);
            }
        }
        for i in 0..pts.len() as u32 - 1 {
            let (a, b) = (base + i * lanes, base + (i + 1) * lanes);
            for k in 0..lanes - 1 {
                self.mesh.add_triangle(a + k, a + k + 1, b + k);
                self.mesh.add_triangle(a + k + 1, b + k + 1, b + k);
            }
        }
    }
}

/// Warm for excitation, cool for inhibition (as in the neural view).
fn sign_color(excites: bool, alpha: u8) -> Color32 {
    if excites {
        Color32::from_rgba_unmultiplied(255, 160, 60, alpha)
    } else {
        Color32::from_rgba_unmultiplied(80, 150, 255, alpha)
    }
}

/// Which kind of connection: weights are scaled per kind, as in the other views, so each kind's
/// strongest reads as strong.
fn kind(e: &Edge) -> usize {
    match (e.from, e.to) {
        (Node::Input(_), _) => 0,
        (_, Node::Output(_)) => 2,
        _ => 1,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Sense,
    Hidden,
    Action,
}

/// How busy a neuron is, 0..1, as in the neural view: senses by magnitude, neurons and actions
/// by how far toward +1 they are.
fn activity(role: Role, v: f32) -> f32 {
    match role {
        Role::Sense => v.abs().min(1.0),
        _ => ((v + 1.0) * 0.5).clamp(0.0, 1.0),
    }
}

/// A count for people: "50,000", or "13.4 M" past a million.
pub fn count(n: usize) -> String {
    if n >= 1_000_000 {
        return format!("{:.1} M", n as f64 / 1e6);
    }
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// The 3D view's state: how it's turned, and what it has worked out from the current sketch.
pub struct Brain3d {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    /// When it was last turned or zoomed by hand.
    touched: f64,
    last: f64,
    /// Every point (hidden neurons by number, then the senses and actions shown) for a brain of
    /// (neurons, senses, actions).
    points: Vec<[f32; 3]>,
    points_for: (usize, usize, usize),
    /// The sketch's connections as (from, to, strength 0..1, excites), weakest first so the
    /// strongest are drawn on top, and the sketch they're from.
    edges: Vec<(u32, u32, f32, bool)>,
    edges_of: Option<Arc<Wiring>>,
    mesh: Arc<egui::Mesh>,
}

impl Default for Brain3d {
    fn default() -> Self {
        Self {
            // Seen from the side, a little from above and behind: back on the left.
            yaw: -0.5,
            pitch: 0.32,
            zoom: 1.0,
            touched: f64::NEG_INFINITY,
            last: 0.0,
            points: Vec::new(),
            points_for: (0, 0, 0),
            edges: Vec::new(),
            edges_of: None,
            mesh: Arc::default(),
        }
    }
}

impl Brain3d {
    /// Draws the 3D brain into `rect` (allocated by the caller, which `resp` answers for), `now`
    /// seconds into the app's run.
    pub fn show(&mut self, ui: &mut egui::Ui, rect: egui::Rect, resp: &egui::Response, c: &CreatureDetail, now: f64) {
        let w = &c.wiring;
        let (nh, ni, no) = (w.total_hidden, w.inputs.len(), w.outputs.len());
        if self.points_for != (nh, ni, no) || self.points.is_empty() {
            let l = layout(nh, ni, no);
            self.points = l.hidden.into_iter().chain(l.inputs).chain(l.outputs).collect();
            self.points_for = (nh, ni, no);
        }
        if self.edges_of.as_ref().is_none_or(|e| !Arc::ptr_eq(e, w)) {
            let senses: HashMap<usize, usize> = w.inputs.iter().enumerate().map(|(j, &k)| (k, nh + j)).collect();
            let actions: HashMap<usize, usize> = w.outputs.iter().enumerate().map(|(j, &k)| (k, nh + ni + j)).collect();
            let at = |n: Node| match n {
                Node::Hidden(k) => (k < nh).then_some(k as u32),
                Node::Input(k) => senses.get(&k).map(|&i| i as u32),
                Node::Output(k) => actions.get(&k).map(|&i| i as u32),
            };
            let mut max = [1e-6f32; 3];
            for e in &w.edges {
                max[kind(e)] = max[kind(e)].max(e.weight.abs());
            }
            self.edges = w
                .edges
                .iter()
                .filter_map(|e| Some((at(e.from)?, at(e.to)?, e.weight.abs() / max[kind(e)], e.weight >= 0.0)))
                .collect();
            self.edges.sort_by(|a, b| a.2.total_cmp(&b.2));
            self.edges_of = Some(w.clone());
        }

        let dt = (now - self.last).clamp(0.0, 0.1) as f32;
        self.last = now;
        if resp.dragged() {
            let d = resp.drag_delta();
            self.yaw += d.x * 0.01;
            self.pitch = (self.pitch + d.y * 0.01).clamp(-1.45, 1.45);
            self.touched = now;
        }
        if resp.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            if scroll != 0.0 || pinch != 1.0 {
                self.zoom = (self.zoom * (scroll * 0.0025).exp() * pinch).clamp(0.5, 8.0);
                self.touched = now;
            }
        }
        if now - self.touched > IDLE {
            self.yaw += SPIN * dt;
        }

        // Turn (about the vertical, then tilt), then perspective. Zoom is the lens, so the eye
        // never ends up inside the brain.
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let center = rect.center();
        let focal = rect.width().min(rect.height()) * 0.46 * (EYE - REACH) / REACH * self.zoom;
        let proj: Vec<(Pos2, f32)> = self
            .points
            .iter()
            .map(|&[x, y, z]| {
                let x1 = x * cy + z * sy;
                let z1 = z * cy - x * sy;
                let y2 = y * cp - z1 * sp;
                let z2 = y * sp + z1 * cp;
                let depth = EYE - z2;
                (center + vec2(x1, -y2) * (focal / depth), depth)
            })
            .collect();
        // Farther is dimmer.
        let fog = |d: f32| 1.0 - 0.6 * ((d - (EYE - REACH)) / (2.0 * REACH)).clamp(0.0, 1.0);

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BACKGROUND);

        // A crowd of lines gets fainter each, so a hundred thousand don't white out the picture.
        let density = (3000.0 / self.edges.len().max(1) as f32).sqrt().clamp(0.15, 1.0);
        let ppp = ui.ctx().pixels_per_point();
        // Last frame's mesh, if egui is done with it, so its memory is reused.
        let mesh = Arc::try_unwrap(std::mem::take(&mut self.mesh)).unwrap_or_default();
        let mut lines = Lines::new(mesh, ppp, self.edges.len() <= FEATHER_UP_TO);
        for &(a, b, s, excites) in &self.edges {
            let ((pa, da), (pb, db)) = (proj[a as usize], proj[b as usize]);
            let alpha = 255.0 * (0.12 + 0.88 * s) * density;
            lines.path(&[(pa, sign_color(excites, (alpha * fog(da)) as u8)), (pb, sign_color(excites, (alpha * fog(db)) as u8))], 0.5 + 1.3 * s);
        }
        self.mesh = Arc::new(lines.mesh);
        painter.add(Shape::Mesh(self.mesh.clone()));

        // The neurons, far ones first, so near ones cover them.
        let role = |i: usize| if i < nh { Role::Hidden } else if i < nh + ni { Role::Sense } else { Role::Action };
        let value = |i: usize| match role(i) {
            Role::Hidden => c.hidden.get(i).copied().unwrap_or(0.0),
            Role::Sense => c.inputs.get(w.inputs[i - nh]).copied().unwrap_or(0.0),
            Role::Action => c.outputs.get(w.outputs[i - nh - ni]).copied().unwrap_or(0.0),
        };
        let mut order: Vec<usize> = (0..proj.len()).collect();
        order.sort_by(|&i, &j| proj[j].1.total_cmp(&proj[i].1));
        let mut shapes = Vec::with_capacity(order.len() + 2);
        let grow = self.zoom.sqrt().min(2.0);
        for &i in &order {
            let (p, d) = proj[i];
            let r = role(i);
            let a = activity(r, value(i));
            // From dim to bright with activity: violet neurons, green senses, amber actions.
            let (dim, bright, size) = match r {
                Role::Hidden => ([70.0, 62.0, 120.0], [230.0, 220.0, 255.0], 0.9),
                Role::Sense => ([50.0, 120.0, 85.0], [170.0, 255.0, 205.0], 1.8),
                Role::Action => ([150.0, 90.0, 60.0], [255.0, 205.0, 150.0], 1.8),
            };
            let k = fog(d);
            let ch = |n: usize| ((dim[n] + (bright[n] - dim[n]) * a) * k) as u8;
            let radius = (size + 1.5 * a) * grow * EYE / d;
            shapes.push(Shape::circle_filled(p, radius, Color32::from_rgb(ch(0), ch(1), ch(2))));
        }
        painter.extend(shapes);

        // Names over the two discs.
        let font = egui::FontId::proportional(11.0);
        let text = Color32::from_rgba_unmultiplied(185, 195, 205, 200);
        for (x, name) in [(-FACE, "senses"), (FACE, "actions")] {
            let [px, py, pz] = [x, DISC * 0.8 + 0.1, 0.0];
            let x1 = px * cy + pz * sy;
            let z1 = pz * cy - px * sy;
            let y2 = py * cp - z1 * sp;
            let z2 = py * sp + z1 * cp;
            painter.text(center + vec2(x1, -y2) * (focal / (EYE - z2)), egui::Align2::CENTER_BOTTOM, name, font.clone(), text);
        }

        if let Some(pointer) = resp.hover_pos()
            && !resp.dragged()
            && let Some(&i) = order
                .iter()
                .filter(|&&i| proj[i].0.distance(pointer) < 6.0)
                .min_by(|&&i, &&j| proj[i].0.distance(pointer).total_cmp(&proj[j].0.distance(pointer)))
        {
            let name = match role(i) {
                Role::Hidden => format!("neuron {i}"),
                Role::Sense => evolve_sim::brain::input_label(w.inputs[i - nh]),
                Role::Action => evolve_sim::brain::output_label(w.outputs[i - nh - ni]),
            };
            let links = self.edges.iter().filter(|e| e.0 as usize == i || e.1 as usize == i).count();
            resp.clone().on_hover_text_at_pointer(format!("{name}: {:+.2}\n{links} connections shown", value(i)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_egui::egui::pos2;

    #[test]
    fn layout_is_the_same_every_time() {
        let a = layout(3186, 70, 40);
        let b = layout(3186, 70, 40);
        assert_eq!(a.hidden, b.hidden);
        assert_eq!(a.inputs, b.inputs);
        assert_eq!(a.outputs, b.outputs);
        // A neuron keeps its place in a bigger brain.
        assert_eq!(layout(4096, 70, 40).hidden[..3186], a.hidden[..]);
    }

    #[test]
    fn neurons_fill_the_brain_and_senses_and_actions_face_apart() {
        let l = layout(4096, 70, 40);
        for p in &l.hidden {
            assert!(p.iter().all(|v| v.is_finite()));
            assert!(p[0].abs() <= 1.0 && p[1].abs() <= 0.72 && p[2].abs() <= 1.0, "{p:?}");
            // Nothing on the midline: the hemispheres are apart.
            assert!(p[2].abs() >= 0.05, "{p:?}");
        }
        // Evenly spread: no two neurons on top of each other, and both halves well filled.
        let mut keys: Vec<[i32; 3]> = l.hidden.iter().map(|p| p.map(|v| (v * 1e4) as i32)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 4096);
        let front = l.hidden.iter().filter(|p| p[0] > 0.0).count();
        assert!((1900..2200).contains(&front), "{front} in front");
        assert!(l.inputs.iter().all(|p| p[0] == -FACE));
        assert!(l.outputs.iter().all(|p| p[0] == FACE));
        assert!(l.inputs.iter().chain(&l.outputs).all(|p| p[1].hypot(p[2]) <= DISC));
    }

    #[test]
    fn counts_read_well() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(50_000), "50,000");
        assert_eq!(count(100_000), "100,000");
        assert_eq!(count(13_400_000), "13.4 M");
    }

    #[test]
    fn lines_make_one_strip_each() {
        let mut lines = Lines::new(egui::Mesh::default(), 1.0, false);
        lines.path(&[(pos2(0.0, 0.0), Color32::WHITE), (pos2(10.0, 0.0), Color32::WHITE)], 2.0);
        assert_eq!(lines.mesh.vertices.len(), 4);
        assert_eq!(lines.mesh.indices.len(), 6);
        let mut soft = Lines::new(lines.mesh, 1.0, true);
        assert!(soft.mesh.vertices.is_empty());
        soft.path(&[(pos2(0.0, 0.0), Color32::WHITE), (pos2(5.0, 5.0), Color32::WHITE), (pos2(10.0, 0.0), Color32::WHITE)], 2.0);
        assert_eq!(soft.mesh.vertices.len(), 12);
        assert_eq!(soft.mesh.indices.len(), 2 * 3 * 6);
        assert!(soft.mesh.is_valid());
    }

    /// What batching a frame of 100,000 lines costs (run with --nocapture to see).
    #[test]
    fn a_hundred_thousand_lines() {
        let l = layout(4096, 70, 40);
        let pts: Vec<Pos2> = l.hidden.iter().map(|p| pos2(400.0 + p[0] * 300.0, 300.0 - p[1] * 300.0)).collect();
        let mut mesh = egui::Mesh::default();
        for feather in [false, true] {
            let mut took = Vec::new();
            for _ in 0..5 {
                let start = std::time::Instant::now();
                let mut lines = Lines::new(mesh, 1.0, feather);
                for k in 0..100_000 {
                    let (a, b) = (pts[k % 4096], pts[(k * 7919 + 13) % 4096]);
                    lines.path(&[(a, Color32::WHITE), (b, Color32::WHITE)], 1.2);
                }
                mesh = lines.mesh;
                took.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            let best = took.iter().copied().fold(f64::INFINITY, f64::min);
            eprintln!("100,000 lines{}: {best:.1} ms, {} vertices", if feather { " with soft edges" } else { "" }, mesh.vertices.len());
            assert_eq!(mesh.vertices.len(), 100_000 * if feather { 8 } else { 4 });
        }
    }
}
