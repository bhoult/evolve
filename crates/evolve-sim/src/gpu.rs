//! Brains on the GPU. Every creature's recurrent network is evaluated in two batched compute
//! dispatches per brain tick (all hidden neurons of all creatures, then all outputs), and the
//! results are read back one tick later, so the GPU works on brains while the CPU runs physics.
//!
//! Weights live in growable buffer "pages" (a storage binding is limited to 2 GB on most GPUs);
//! lifetime-learned weights, biases and hidden state live in f32 arenas. `brain::step_cpu` is the
//! reference these kernels must match.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::brain::{BrainGenes, BrainState, Learning, N_IN, N_OUT, OutputBias};

const IN_PAD: usize = (N_IN + 1) & !1;
/// Per brain in the inputs buffer: the senses (padded), then the exploration noise on each output.
const IN_STRIDE: usize = IN_PAD + N_OUT;
const UNIT: u64 = 4;
/// Brains beyond this many bytes of GPU memory are trimmed rather than allocated.
pub const DEFAULT_BUDGET_BYTES: u64 = 10 << 30;

const _: () = assert!(IN_PAD >= N_IN && IN_PAD % 2 == 0 && N_OUT % 2 == 0);

pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

fn pad2(n: usize) -> usize {
    (n + 1) & !1
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    h: usize,
    hp: usize,
}

impl Layout {
    fn new(h: usize) -> Self {
        Self { h, hp: pad2(h) }
    }
    /// f16 elements of weights.
    fn len(&self) -> usize {
        self.h * IN_PAD + self.h * self.hp + N_OUT * self.hp
    }
    fn words(&self) -> u64 {
        (self.len() / 2) as u64
    }
    // The aux block, in f32 elements (brain.wgsl mirrors these): learned offsets P and
    // eligibility traces E (N_OUT x hp each), last outputs, hidden biases, output biases.
    fn traces(&self) -> usize {
        N_OUT * self.hp
    }
    fn last_out(&self) -> usize {
        2 * N_OUT * self.hp
    }
    fn bias(&self) -> usize {
        self.last_out() + N_OUT
    }
    fn b_out(&self) -> usize {
        self.bias() + self.hp
    }
    fn aux_len(&self) -> usize {
        self.b_out() + N_OUT
    }
    /// GPU memory one brain takes: weights, aux block and both hidden-state copies.
    fn bytes(&self) -> u64 {
        (self.words() + self.aux_len() as u64 + 2 * self.hp as u64) * UNIT
    }
}

fn pack_weights(g: &BrainGenes, l: Layout) -> Vec<u16> {
    let (h, hp) = (l.h, l.hp);
    let mut v = Vec::with_capacity(l.len());
    for i in 0..h {
        v.extend(g.w_in.0[i * N_IN..(i + 1) * N_IN].iter().map(|x| x.to_bits()));
        v.extend(std::iter::repeat_n(0u16, IN_PAD - N_IN));
    }
    for i in 0..h {
        v.extend(g.w_rec.0[i * h..(i + 1) * h].iter().map(|x| x.to_bits()));
        v.extend(std::iter::repeat_n(0u16, hp - h));
    }
    for k in 0..N_OUT {
        v.extend(g.w_out.0[k * h..(k + 1) * h].iter().map(|x| x.to_bits()));
        v.extend(std::iter::repeat_n(0u16, hp - h));
    }
    debug_assert_eq!(v.len(), l.len());
    let _ = f16::ZERO;
    v
}

fn pack_aux(g: &BrainGenes, l: Layout, state: Option<&BrainState>) -> Vec<f32> {
    let (h, hp) = (l.h, l.hp);
    let mut v = vec![0.0f32; l.aux_len()];
    if let Some(s) = state {
        for k in 0..N_OUT {
            if s.plastic.len() == N_OUT * h {
                v[k * hp..k * hp + h].copy_from_slice(&s.plastic[k * h..(k + 1) * h]);
            }
            if s.traces.len() == N_OUT * h {
                let e = l.traces() + k * hp;
                v[e..e + h].copy_from_slice(&s.traces[k * h..(k + 1) * h]);
            }
        }
        if s.outputs.len() == N_OUT {
            v[l.last_out()..l.last_out() + N_OUT].copy_from_slice(&s.outputs);
        }
    }
    v[l.bias()..l.bias() + h].copy_from_slice(&g.bias);
    v[l.b_out()..l.b_out() + N_OUT].copy_from_slice(&g.b_out);
    v
}

/// First-fit allocator over units of a buffer.
#[derive(Clone, Debug)]
struct Alloc {
    cap: u64,
    free: Vec<(u64, u64)>,
}

impl Alloc {
    fn new(cap: u64) -> Self {
        Self {
            cap,
            free: if cap > 0 { vec![(0, cap)] } else { Vec::new() },
        }
    }

    fn alloc(&mut self, n: u64) -> Option<u64> {
        let i = self.free.iter().position(|&(_, l)| l >= n)?;
        let (s, l) = self.free[i];
        if l == n {
            self.free.remove(i);
        } else {
            self.free[i] = (s + n, l - n);
        }
        Some(s)
    }

    fn release(&mut self, s: u64, n: u64) {
        if n == 0 {
            return;
        }
        let i = self.free.partition_point(|&(fs, _)| fs < s);
        self.free.insert(i, (s, n));
        // Coalesce with the next block, then the previous one.
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }

    fn grow(&mut self, new_cap: u64) {
        let old = self.cap;
        self.cap = new_cap;
        self.release(old, new_cap - old);
    }

    /// Space at the very end that is free (so growing adds to it).
    fn free_tail(&self) -> u64 {
        match self.free.last() {
            Some(&(s, l)) if s + l == self.cap => l,
            _ => 0,
        }
    }
}

const ARENA_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// One or more same-shaped buffers sharing an allocator, grown by copying.
struct Arena {
    bufs: Vec<wgpu::Buffer>,
    alloc: Alloc,
    max_units: u64,
    label: &'static str,
}

impl Arena {
    fn new(device: &wgpu::Device, label: &'static str, copies: usize, units: u64, max_units: u64) -> Self {
        let bufs = (0..copies)
            .map(|_| make_buffer(device, label, units * UNIT, ARENA_USAGE))
            .collect();
        Self {
            bufs,
            alloc: Alloc::new(units),
            max_units,
            label,
        }
    }

    fn bytes(&self) -> u64 {
        self.alloc.cap * UNIT * self.bufs.len() as u64
    }

    /// Allocates `n` units, growing the buffers if needed. Returns None if it can't fit.
    /// `grew` is set when the buffers were replaced (bind groups must be rebuilt).
    fn alloc(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, n: u64, grew: &mut bool) -> Option<u64> {
        if let Some(o) = self.alloc.alloc(n) {
            return Some(o);
        }
        let need = self.alloc.cap - self.alloc.free_tail() + n;
        if need > self.max_units {
            return None;
        }
        let new_cap = (self.alloc.cap * 2).max(need).max(1 << 16).min(self.max_units);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(self.label) });
        let mut new_bufs = Vec::with_capacity(self.bufs.len());
        for old in &self.bufs {
            let nb = make_buffer(device, self.label, new_cap * UNIT, ARENA_USAGE);
            if self.alloc.cap > 0 {
                enc.copy_buffer_to_buffer(old, 0, &nb, 0, self.alloc.cap * UNIT);
            }
            new_bufs.push(nb);
        }
        // Submit the copy now, so later writes into the new buffers land after it.
        queue.submit([enc.finish()]);
        self.bufs = new_bufs;
        self.alloc.grow(new_cap);
        *grew = true;
        self.alloc.alloc(n)
    }

    fn release(&mut self, off: u64, n: u64) {
        self.alloc.release(off, n);
    }
}

fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn make_buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Meta {
    w_off: u32,
    aux_off: u32,
    h_off: u32,
    h: u32,
    hp: u32,
    row_start: u32,
    batch: u32,
    lr: f32,
    bias_eat: f32,
    bias_attack: f32,
    bias_mate: f32,
    trace_decay: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    meta_start: u32,
    meta_count: u32,
    rows: u32,
    plastic_decay: f32,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    page: usize,
    w_off: u64,
    aux_off: u64,
    h_off: u64,
    layout: Layout,
}

pub struct BatchItem {
    pub id: u64,
    pub inputs: [f32; N_IN],
    /// Exploration noise on the outputs the brain last produced (what it acted on).
    pub noise: [f32; N_OUT],
    pub learn: Learning,
    pub bias: OutputBias,
}

pub struct SavedState {
    pub hidden: Vec<f32>,
    pub plastic: Vec<f32>,
    pub traces: Vec<f32>,
    pub outputs: Vec<f32>,
}

/// The selected creature's brain, read back each tick for the inspector and brain view.
pub struct SelectedState {
    pub id: u64,
    pub hidden: Vec<f32>,
    /// Learned output-weight offsets, N_OUT x h.
    pub plastic: Vec<f32>,
}

pub struct Collected {
    pub ids: Vec<u64>,
    pub outputs: Vec<[f32; N_OUT]>,
    /// The creature that was selected when this tick was submitted.
    pub selected: Option<SelectedState>,
    /// GPU time spent on this tick's brains, if the GPU can time itself.
    pub gpu_ms: Option<f32>,
}

struct InFlight {
    ids: Vec<u64>,
    selected: Option<(u64, usize)>,
    out_bytes: u64,
    total_bytes: u64,
    timed: bool,
    ready: Arc<AtomicBool>,
}

/// A buffer that is recreated (bigger) when it runs out of room.
struct Grow {
    buf: wgpu::Buffer,
    cap: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl Grow {
    fn new(device: &wgpu::Device, label: &'static str, cap: u64, usage: wgpu::BufferUsages) -> Self {
        Self {
            buf: make_buffer(device, label, cap, usage),
            cap: cap.max(16),
            usage,
            label,
        }
    }

    fn ensure(&mut self, device: &wgpu::Device, need: u64) -> bool {
        if need <= self.cap {
            return false;
        }
        self.cap = need.next_power_of_two();
        self.buf = make_buffer(device, self.label, self.cap, self.usage);
        true
    }
}

pub struct GpuBrains {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter_name: String,
    hidden_pipe: wgpu::ComputePipeline,
    output_pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pages: Vec<Arena>,
    page_max_units: u64,
    aux: Arena,
    hstate: Arena,
    inputs: Grow,
    outputs: Grow,
    metas: Grow,
    staging: Grow,
    params: Vec<wgpu::Buffer>,
    timer: Option<(wgpu::QuerySet, wgpu::Buffer, f32)>,
    bind_groups: Vec<[wgpu::BindGroup; 2]>,
    dirty: bool,
    /// Which hidden-state buffer holds the latest activations.
    parity: usize,
    slots: HashMap<u64, Slot>,
    /// Bytes the living brains take up (see `live_bytes`).
    live: u64,
    in_flight: Option<InFlight>,
    held: Option<Collected>,
    pub budget_bytes: u64,
}

impl GpuBrains {
    /// Opens the most powerful GPU. Returns an error string if there is none to use.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("only a software adapter ({})", info.name));
        }
        let timed = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let limits = adapter.limits();
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("evolve brains"),
            required_features: if timed { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("GPU device: {e}"))?;
        device.on_uncaptured_error(Arc::new(|e| eprintln!("evolve: GPU error: {e}")));

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brain bindings"),
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, true),
                storage(3, false),
                storage(4, true),
                storage(5, false),
                storage(6, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brain pipeline"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let hidden_pipe = pipe("hidden");
        let output_pipe = pipe("output");

        let binding_units = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) / UNIT;
        let page_max_units = binding_units.min((1u64 << 30) / UNIT);
        let aux = Arena::new(&device, "brain aux", 1, 1 << 16, binding_units);
        let hstate = Arena::new(&device, "brain hidden state", 2, 1 << 14, binding_units);
        let su = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let inputs = Grow::new(&device, "brain inputs", 1 << 16, su);
        let outputs = Grow::new(&device, "brain outputs", 1 << 14, su | wgpu::BufferUsages::COPY_SRC);
        let metas = Grow::new(&device, "brain metas", 1 << 14, su);
        let staging = Grow::new(
            &device,
            "brain readback",
            1 << 16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let timer = timed.then(|| {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("brain timer"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = make_buffer(
                &device,
                "brain timer resolve",
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            );
            (qs, resolve, queue.get_timestamp_period())
        });
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            hidden_pipe,
            output_pipe,
            layout,
            pages: Vec::new(),
            page_max_units,
            aux,
            hstate,
            inputs,
            outputs,
            metas,
            staging,
            params: Vec::new(),
            timer,
            bind_groups: Vec::new(),
            dirty: true,
            parity: 0,
            slots: HashMap::new(),
            live: 0,
            in_flight: None,
            held: None,
            budget_bytes: DEFAULT_BUDGET_BYTES,
        })
    }

    /// GPU memory held for brains, in bytes. It never shrinks: buffers stay as big as they grew.
    pub fn memory_bytes(&self) -> u64 {
        self.pages.iter().map(|p| p.bytes()).sum::<u64>() + self.aux.bytes() + self.hstate.bytes()
    }

    /// GPU memory the living brains take up, in bytes: what's held, less the gaps the dead left.
    pub fn live_bytes(&self) -> u64 {
        self.live
    }

    pub fn contains(&self, id: u64) -> bool {
        self.slots.contains_key(&id)
    }

    /// Would a brain with `h` neurons fit in the memory budget?
    pub fn fits(&self, h: usize) -> bool {
        let l = Layout::new(h);
        self.memory_bytes() + l.bytes() <= self.budget_bytes || self.would_fit_in_place(l)
    }

    fn would_fit_in_place(&self, l: Layout) -> bool {
        self.pages.iter().any(|p| p.alloc.free.iter().any(|&(_, n)| n >= l.words()))
    }

    /// Uploads a brain. `state` restores learned weights and activations (after loading a save).
    pub fn register(&mut self, id: u64, g: &BrainGenes, state: Option<&BrainState>) -> Result<(), String> {
        if self.slots.contains_key(&id) {
            self.unregister(id);
        }
        let l = Layout::new(g.h);
        if l.words() > self.page_max_units {
            return Err(format!("a {}-neuron brain is too big for one GPU buffer", g.h));
        }
        if !self.fits(g.h) {
            return Err("GPU brain memory budget is full".into());
        }
        let mut grew = false;
        let mut placed = None;
        for (pi, page) in self.pages.iter_mut().enumerate() {
            if let Some(o) = page.alloc(&self.device, &self.queue, l.words(), &mut grew) {
                placed = Some((pi, o));
                break;
            }
        }
        if placed.is_none() {
            let start = (l.words() * 2).max(8 << 20).min(self.page_max_units);
            let mut page = Arena::new(&self.device, "brain weights", 1, start, self.page_max_units);
            let Some(o) = page.alloc(&self.device, &self.queue, l.words(), &mut grew) else {
                self.dirty |= grew;
                return Err("brain does not fit in a fresh page".into());
            };
            self.pages.push(page);
            self.params.push(make_buffer(
                &self.device,
                "brain params",
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ));
            grew = true;
            placed = Some((self.pages.len() - 1, o));
        }
        let (page, w_off) = placed.expect("placed");
        let aux_off = match self.aux.alloc(&self.device, &self.queue, l.aux_len() as u64, &mut grew) {
            Some(o) => o,
            None => {
                // An arena may already have grown into new buffers; the bind groups must follow.
                self.dirty |= grew;
                self.pages[page].release(w_off, l.words());
                return Err("GPU aux arena full".into());
            }
        };
        let h_off = match self.hstate.alloc(&self.device, &self.queue, l.hp as u64, &mut grew) {
            Some(o) => o,
            None => {
                self.dirty |= grew;
                self.pages[page].release(w_off, l.words());
                self.aux.release(aux_off, l.aux_len() as u64);
                return Err("GPU hidden-state arena full".into());
            }
        };
        self.dirty |= grew;
        let w = pack_weights(g, l);
        self.queue
            .write_buffer(&self.pages[page].bufs[0], w_off * UNIT, bytemuck::cast_slice(&w));
        let a = pack_aux(g, l, state);
        self.queue
            .write_buffer(&self.aux.bufs[0], aux_off * UNIT, bytemuck::cast_slice(&a));
        let mut hv = vec![0.0f32; l.hp];
        if let Some(s) = state
            && s.hidden.len() == g.h
        {
            hv[..g.h].copy_from_slice(&s.hidden);
        }
        for b in &self.hstate.bufs {
            self.queue.write_buffer(b, h_off * UNIT, bytemuck::cast_slice(&hv));
        }
        self.live += l.bytes();
        self.slots.insert(
            id,
            Slot {
                page,
                w_off,
                aux_off,
                h_off,
                layout: l,
            },
        );
        Ok(())
    }

    pub fn unregister(&mut self, id: u64) {
        if let Some(s) = self.slots.remove(&id) {
            self.live -= s.layout.bytes();
            self.pages[s.page].release(s.w_off, s.layout.words());
            self.aux.release(s.aux_off, s.layout.aux_len() as u64);
            self.hstate.release(s.h_off, s.layout.hp as u64);
        }
    }

    pub fn clear(&mut self) {
        let ids: Vec<u64> = self.slots.keys().copied().collect();
        for id in ids {
            self.unregister(id);
        }
    }

    fn rebuild_bind_groups(&mut self) {
        self.bind_groups = (0..self.pages.len())
            .map(|p| {
                let make = |prev: usize| {
                    let e = entry;
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("brain bind group"),
                        layout: &self.layout,
                        entries: &[
                            e(0, &self.pages[p].bufs[0]),
                            e(1, &self.aux.bufs[0]),
                            e(2, &self.hstate.bufs[prev]),
                            e(3, &self.hstate.bufs[1 - prev]),
                            e(4, &self.inputs.buf),
                            e(5, &self.outputs.buf),
                            e(6, &self.metas.buf),
                            e(7, &self.params[p]),
                        ],
                    })
                };
                [make(0), make(1)]
            })
            .collect();
        self.dirty = false;
    }

    /// Starts one brain tick for `batch` (whose brains must all be registered). Results come from
    /// the next `collect`.
    pub fn submit(&mut self, batch: &[BatchItem], selected: Option<u64>) {
        self.finish_in_flight();
        let n = batch.len();
        let mut order: Vec<usize> = (0..n).filter(|&b| self.slots.contains_key(&batch[b].id)).collect();
        order.sort_by_key(|&b| self.slots[&batch[b].id].page);
        let mut metas = Vec::with_capacity(order.len());
        let mut page_params = vec![Params::zeroed(); self.pages.len()];
        for &b in &order {
            let it = &batch[b];
            let s = self.slots[&it.id];
            let pp = &mut page_params[s.page];
            if pp.meta_count == 0 {
                pp.meta_start = metas.len() as u32;
            }
            debug_assert!(it.learn.pipelined || it.learn.lr == 0.0, "the GPU always learns pipelined");
            metas.push(Meta {
                w_off: s.w_off as u32,
                aux_off: s.aux_off as u32,
                h_off: s.h_off as u32,
                h: s.layout.h as u32,
                hp: s.layout.hp as u32,
                row_start: pp.rows,
                batch: b as u32,
                lr: it.learn.lr,
                bias_eat: it.bias.eat,
                bias_attack: it.bias.attack,
                bias_mate: it.bias.mate,
                trace_decay: it.learn.trace_decay,
            });
            pp.meta_count += 1;
            pp.plastic_decay = it.learn.plastic_decay;
            pp.rows += s.layout.h as u32;
        }
        let mut inputs = vec![0.0f32; n.max(1) * IN_STRIDE];
        for (b, it) in batch.iter().enumerate() {
            let x = b * IN_STRIDE;
            inputs[x..x + N_IN].copy_from_slice(&it.inputs);
            inputs[x + IN_PAD..x + IN_STRIDE].copy_from_slice(&it.noise);
        }
        let out_bytes = (n.max(1) * N_OUT) as u64 * UNIT;
        let sel = selected.and_then(|id| self.slots.get(&id).map(|s| (id, *s)));
        let sel_hidden = sel.map(|(_, s)| s.layout.hp as u64 * UNIT).unwrap_or(0);
        let sel_bytes = sel_hidden + sel.map(|(_, s)| s.layout.traces() as u64 * UNIT).unwrap_or(0);
        let total = out_bytes + sel_bytes + 16;
        let mut grew = self.inputs.ensure(&self.device, inputs.len() as u64 * UNIT);
        grew |= self.outputs.ensure(&self.device, out_bytes);
        grew |= self.metas.ensure(&self.device, (metas.len().max(1) * std::mem::size_of::<Meta>()) as u64);
        self.staging.ensure(&self.device, total);
        if grew || self.dirty || self.bind_groups.len() != self.pages.len() {
            self.rebuild_bind_groups();
        }
        self.queue.write_buffer(&self.inputs.buf, 0, bytemuck::cast_slice(&inputs));
        if !metas.is_empty() {
            self.queue.write_buffer(&self.metas.buf, 0, bytemuck::cast_slice(&metas));
        }
        for (p, pp) in page_params.iter().enumerate() {
            self.queue.write_buffer(&self.params[p], 0, bytemuck::bytes_of(pp));
        }

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("brain tick") });
        {
            let timestamp_writes = self.timer.as_ref().map(|(qs, _, _)| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brains"),
                timestamp_writes,
            });
            let dims = |rows: u32| {
                let x = rows.min(65535);
                (x, rows.div_ceil(x.max(1)))
            };
            pass.set_pipeline(&self.hidden_pipe);
            for (p, pp) in page_params.iter().enumerate() {
                if pp.rows > 0 {
                    pass.set_bind_group(0, &self.bind_groups[p][self.parity], &[]);
                    let (x, y) = dims(pp.rows);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }
            pass.set_pipeline(&self.output_pipe);
            for (p, pp) in page_params.iter().enumerate() {
                if pp.meta_count > 0 {
                    pass.set_bind_group(0, &self.bind_groups[p][self.parity], &[]);
                    let (x, y) = dims(pp.meta_count * N_OUT as u32);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }
        }
        enc.copy_buffer_to_buffer(&self.outputs.buf, 0, &self.staging.buf, 0, out_bytes);
        if let Some((_, s)) = sel {
            let latest = &self.hstate.bufs[1 - self.parity];
            enc.copy_buffer_to_buffer(latest, s.h_off * UNIT, &self.staging.buf, out_bytes, sel_hidden);
            // P sits at the start of the aux block, N_OUT x hp floats.
            enc.copy_buffer_to_buffer(
                &self.aux.bufs[0],
                s.aux_off * UNIT,
                &self.staging.buf,
                out_bytes + sel_hidden,
                sel_bytes - sel_hidden,
            );
        }
        if let Some((qs, resolve, _)) = &self.timer {
            enc.resolve_query_set(qs, 0..2, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, &self.staging.buf, out_bytes + sel_bytes, 16);
        }
        self.queue.submit([enc.finish()]);
        self.parity = 1 - self.parity;
        let ready = Arc::new(AtomicBool::new(false));
        let r2 = ready.clone();
        self.staging
            .buf
            .slice(0..total)
            .map_async(wgpu::MapMode::Read, move |res| r2.store(res.is_ok(), Ordering::Release));
        self.in_flight = Some(InFlight {
            ids: batch.iter().map(|b| b.id).collect(),
            selected: sel.map(|(id, s)| (id, s.layout.h)),
            out_bytes,
            total_bytes: total,
            timed: self.timer.is_some(),
            ready,
        });
    }

    /// Waits for the tick in flight (if any) and keeps its results for the next `collect`.
    fn finish_in_flight(&mut self) {
        let Some(f) = self.in_flight.take() else {
            return;
        };
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if !f.ready.load(Ordering::Acquire) {
            return;
        }
        let collected = {
            let data = self.staging.buf.slice(0..f.total_bytes).get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data[..f.out_bytes as usize]);
            let outputs = (0..f.ids.len())
                .map(|b| {
                    let mut o = [0.0f32; N_OUT];
                    o.copy_from_slice(&floats[b * N_OUT..(b + 1) * N_OUT]);
                    o
                })
                .collect();
            let mut at = f.out_bytes as usize;
            let selected = f.selected.map(|(id, h)| {
                let hp = pad2(h);
                let v: &[f32] = bytemuck::cast_slice(&data[at..at + hp * 4]);
                at += hp * 4;
                let p: &[f32] = bytemuck::cast_slice(&data[at..at + N_OUT * hp * 4]);
                at += N_OUT * hp * 4;
                SelectedState {
                    id,
                    hidden: v[..h].to_vec(),
                    plastic: (0..N_OUT).flat_map(|k| p[k * hp..k * hp + h].iter().copied()).collect(),
                }
            });
            let gpu_ms = (f.timed).then(|| {
                // Only 4-byte aligned here, so read the u64 timestamps byte-wise.
                let t = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().expect("8 bytes"));
                let period = self.timer.as_ref().map(|t| t.2).unwrap_or(1.0);
                t(at + 8).saturating_sub(t(at)) as f32 * period / 1e6
            });
            Collected {
                ids: f.ids,
                outputs,
                selected,
                gpu_ms,
            }
        };
        self.staging.buf.unmap();
        self.held = Some(collected);
    }

    /// Results of the last submitted tick, waiting for the GPU if it isn't done yet.
    pub fn collect(&mut self) -> Option<Collected> {
        self.finish_in_flight();
        self.held.take()
    }

    /// Every registered brain's activations, learned weights and last outputs, for saving.
    pub fn read_states(&mut self) -> HashMap<u64, SavedState> {
        let mut out = HashMap::new();
        self.finish_in_flight();
        let aux_bytes = self.aux.alloc.cap * UNIT;
        let h_bytes = self.hstate.alloc.cap * UNIT;
        let rb = make_buffer(
            &self.device,
            "brain state readback",
            aux_bytes + h_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("brain state readback") });
        enc.copy_buffer_to_buffer(&self.aux.bufs[0], 0, &rb, 0, aux_bytes);
        enc.copy_buffer_to_buffer(&self.hstate.bufs[self.parity], 0, &rb, aux_bytes, h_bytes);
        self.queue.submit([enc.finish()]);
        let ok = Arc::new(AtomicBool::new(false));
        let ok2 = ok.clone();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| ok2.store(r.is_ok(), Ordering::Release));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if !ok.load(Ordering::Acquire) {
            return out;
        }
        let data = rb.slice(..).get_mapped_range();
        let aux: &[f32] = bytemuck::cast_slice(&data[..aux_bytes as usize]);
        let hs: &[f32] = bytemuck::cast_slice(&data[aux_bytes as usize..]);
        for (&id, s) in &self.slots {
            let (h, hp) = (s.layout.h, s.layout.hp);
            let l = s.layout;
            let a = &aux[s.aux_off as usize..s.aux_off as usize + l.aux_len()];
            let mut plastic = vec![0.0; N_OUT * h];
            let mut traces = vec![0.0; N_OUT * h];
            for k in 0..N_OUT {
                plastic[k * h..(k + 1) * h].copy_from_slice(&a[k * hp..k * hp + h]);
                let e = l.traces() + k * hp;
                traces[k * h..(k + 1) * h].copy_from_slice(&a[e..e + h]);
            }
            out.insert(
                id,
                SavedState {
                    hidden: hs[s.h_off as usize..s.h_off as usize + h].to_vec(),
                    plastic,
                    traces,
                    outputs: a[l.last_out()..l.last_out() + N_OUT].to_vec(),
                },
            );
        }
        drop(data);
        rb.unmap();
        out
    }
}

impl GpuBrains {
    /// The learned output-weight offsets (N_OUT x h) of a few brains, in one round trip. The
    /// tick in flight finishes first, so these are at most a tick old.
    pub fn read_plastic(&mut self, ids: &[u64]) -> HashMap<u64, Vec<f32>> {
        let mut out = HashMap::new();
        let wanted: Vec<(u64, Slot)> = ids
            .iter()
            .filter_map(|id| self.slots.get(id).map(|s| (*id, *s)))
            .collect();
        if wanted.is_empty() {
            return out;
        }
        self.finish_in_flight();
        let sizes: Vec<u64> = wanted.iter().map(|(_, s)| s.layout.traces() as u64 * UNIT).collect();
        let total: u64 = sizes.iter().sum();
        let rb = make_buffer(
            &self.device,
            "learned weights readback",
            total,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("learned weights readback") });
        let mut at = 0;
        for ((_, s), &n) in wanted.iter().zip(&sizes) {
            enc.copy_buffer_to_buffer(&self.aux.bufs[0], s.aux_off * UNIT, &rb, at, n);
            at += n;
        }
        self.queue.submit([enc.finish()]);
        let ok = Arc::new(AtomicBool::new(false));
        let ok2 = ok.clone();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| ok2.store(r.is_ok(), Ordering::Release));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if !ok.load(Ordering::Acquire) {
            return out;
        }
        let data = rb.slice(..).get_mapped_range();
        let all: &[f32] = bytemuck::cast_slice(&data);
        let mut at = 0;
        for (id, s) in &wanted {
            let (h, hp) = (s.layout.h, s.layout.hp);
            let p = &all[at..at + N_OUT * hp];
            out.insert(*id, (0..N_OUT).flat_map(|k| p[k * hp..k * hp + h].iter().copied()).collect());
            at += N_OUT * hp;
        }
        drop(data);
        rb.unmap();
        out
    }
}

/// The kernel source with the brain's dimensions filled in.
fn shader_source() -> String {
    use crate::brain::output;
    include_str!("brain.wgsl")
        .replace("{IN_PAD}", &IN_PAD.to_string())
        .replace("{IN_STRIDE}", &IN_STRIDE.to_string())
        .replace("{N_OUT}", &N_OUT.to_string())
        .replace("{EAT}", &output::EAT.to_string())
        .replace("{ATTACK}", &output::ATTACK.to_string())
        .replace("{MATE}", &output::MATE.to_string())
}
