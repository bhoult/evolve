//! The visual environment: the sim's time of day and weather, optionally overridden from the
//! command line and smoothed per frame, driving the sun and moon light, ambient light, fog, ocean
//! waves and one GPU buffer that every custom shader reads.

use std::str::FromStr;

use bevy::asset::embedded_asset;
use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder, DirectionalLightShadowMap};
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, ShaderType, encase};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::ColorGrading;
use bevy::shader::load_shader_library;
use evolve_sim::terrain::Terrain;
use evolve_sim::weather;

use crate::camera::{MainCamera, OrbitCam};
use crate::{Args, Snap};

pub struct EnvPlugin;

/// Command-line overrides for what the visuals show, for repeatable screenshots.
#[derive(Resource, Clone, Default)]
pub struct VisualOverride {
    /// Time of day 0..1 (0 midnight, 0.5 noon).
    pub time_of_day: Option<f32>,
    pub weather: Option<WeatherPreset>,
    /// Strike lightning just before the screenshot.
    pub lightning: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeatherPreset {
    Clear,
    Cloudy,
    Rain,
    Storm,
}

impl FromStr for WeatherPreset {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s {
            "clear" => Self::Clear,
            "cloudy" => Self::Cloudy,
            "rain" => Self::Rain,
            "storm" => Self::Storm,
            _ => return Err(()),
        })
    }
}

impl WeatherPreset {
    /// Cloud cover, rain, storm and wind speed.
    fn values(self) -> (f32, f32, f32, f32) {
        match self {
            Self::Clear => (0.12, 0.0, 0.0, 3.0),
            Self::Cloudy => (0.55, 0.0, 0.0, 6.0),
            Self::Rain => (0.82, 0.8, 0.15, 9.0),
            Self::Storm => (1.0, 1.0, 1.0, 18.0),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShadowQuality {
    Off,
    Low,
    High,
}

/// Graphics toggles the UI can expose. Read every frame.
#[derive(Resource, Clone, Debug)]
pub struct GraphicsSettings {
    /// Clouds in the sky and their shadows on the ground.
    pub clouds: bool,
    pub rain: bool,
    pub lightning: bool,
    pub bloom: bool,
    /// Plants bending in the wind.
    pub sway: bool,
    pub shadows: ShadowQuality,
    /// Shadows that blur with distance from what casts them, more under cloud (percentage-closer
    /// soft shadows: dearer for the GPU, and a little grainy).
    pub soft_shadows: bool,
    /// Flecks of foam riding the streams (flow.rs).
    pub flow: bool,
    /// Ground tinted by its level of detail (scene.rs): green, yellow, orange and red as the
    /// stride doubles. For checking what the camera gets at full detail.
    pub lod_tint: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            clouds: true,
            rain: true,
            lightning: true,
            bloom: true,
            sway: true,
            shadows: ShadowQuality::High,
            soft_shadows: true,
            flow: true,
            lod_tint: false,
        }
    }
}

/// Seconds between gradual changes to the sun or moon (brightness, colour, direction; see
/// `update_env`).
const LIGHT_EVERY: f64 = 3.0;

/// The size of the light for soft shadows (see `GraphicsSettings::soft_shadows`) in a clear sky
/// and under full cloud.
const SOFT_CLEAR: f32 = 4.0;
const SOFT_OVERCAST: f32 = 30.0;

/// Marks the directional light: the sun by day, the moon by night.
#[derive(Component)]
pub struct SkyLight;

/// What the visuals show right now, smoothed so fast sim speeds don't strobe.
#[derive(Resource)]
pub struct Env {
    ready: bool,
    epoch: u64,
    pub sun_dir: Vec3,
    pub cloud: f32,
    pub rain: f32,
    pub storm: f32,
    pub wind: Vec2,
    /// How wet the ground is; lags the rain.
    pub wetness: f32,
    pub cloud_offset: bevy::math::DVec2,
    /// Wall time in seconds, wrapped (see `TIME_WRAP`).
    pub time: f32,
    pub flash: f32,
    pub flash_dir: Vec3,
    /// How high the wave crests reach above the water level in open water, m.
    pub wave_crest: f32,
    /// The waves as the ocean shader gets them: direction, wavenumber, phase; and amplitudes.
    pub waves: [Vec4; 8],
    pub wave_amps: [f32; 8],
    /// How far the rain has fallen, wrapped to the box of rain around the camera, and seconds of
    /// rain for the streaks' own extra speed (see weather_fx.rs).
    rain_fall: bevy::math::DVec3,
    rain_time: f64,
    wave_phase: [f32; 8],
    shadow_key: (u32, u32),
}

/// The shaders' clock wraps here. Wrapping makes animations jump, so it also restarts with each
/// new world (when everything changes anyway); this is the backstop, long enough to be rare and
/// short enough that f32 keeps millisecond steps.
const TIME_WRAP: f32 = 16_384.0;
/// Cloud scroll wraps here, m: the clouds jump then, but only after hours of wind.
const CLOUD_WRAP: f64 = 4.0e6;

impl Default for Env {
    fn default() -> Self {
        Self {
            ready: false,
            epoch: u64::MAX,
            sun_dir: Vec3::from_array(weather::sun_direction(0.36)),
            cloud: 0.2,
            rain: 0.0,
            storm: 0.0,
            wind: Vec2::new(2.0, 0.5),
            wetness: 0.0,
            cloud_offset: bevy::math::DVec2::ZERO,
            time: 0.0,
            flash: 0.0,
            flash_dir: Vec3::Y,
            wave_crest: 0.0,
            waves: [Vec4::ZERO; 8],
            wave_amps: [0.0; 8],
            rain_fall: bevy::math::DVec3::ZERO,
            rain_time: 0.0,
            wave_phase: [0.0; 8],
            shadow_key: (0, 0),
        }
    }
}

impl Env {
    /// Height of the sea surface above the water level at a point in water `depth` m deep, as the
    /// ocean shader moves it (vertically; the sideways chop is left out).
    pub fn wave_height(&self, xz: Vec2, depth: f32) -> f32 {
        // Waves die down in the shallows, as in water.wgsl.
        let shore = 0.2 + 0.8 * smoothstep(0.0, 5.0, depth);
        let mut y = 0.0;
        for (w, a) in self.waves.iter().zip(&self.wave_amps) {
            y += a * shore * (w.z * Vec2::new(w.x, w.y).dot(xz) - w.w).sin();
        }
        y
    }
}

/// Mirrors `Env` in shaders/env.wgsl.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct EnvUniform {
    sun: Vec4,
    sun_color: Vec4,
    moon: Vec4,
    moon_color: Vec4,
    light: Vec4,
    zenith: Vec4,
    horizon: Vec4,
    ambient: Vec4,
    weather: Vec4,
    wind: Vec4,
    world: Vec4,
    sea: Vec4,
    stars: Vec4,
    flash_dir: Vec4,
    fog_sun: Vec4,
    misc: Vec4,
    rain: Vec4,
    waves: [Vec4; 8],
    wave_amp: [Vec4; 2],
}

/// The GPU side of `EnvUniform`, bound by every custom material at binding 100.
#[derive(Resource, Clone)]
pub struct EnvGpu {
    pub buffer: Buffer,
}

impl Plugin for EnvPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/env.wgsl");
        load_shader_library!(app, "shaders/heights.wgsl");
        embedded_asset!(app, "shaders/sky.wgsl");
        embedded_asset!(app, "shaders/water.wgsl");
        embedded_asset!(app, "shaders/terrain.wgsl");
        embedded_asset!(app, "shaders/tree.wgsl");
        embedded_asset!(app, "shaders/rain.wgsl");
        app.init_resource::<Env>()
            .init_resource::<VisualOverride>()
            .add_systems(PreStartup, (create_env_buffer, init_settings))
            .add_systems(Startup, spawn_light)
            .add_systems(
                Update,
                (update_env, apply_settings).chain().in_set(EnvSystems).after(crate::camera::apply_camera),
            );
    }
}

/// Runs after the camera has moved and before anything that reads `Env` this frame.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnvSystems;

pub fn shader(name: &'static str) -> bevy::shader::ShaderRef {
    bevy::shader::ShaderRef::Path(format!("embedded://evolve/shaders/{name}").into())
}

fn create_env_buffer(mut commands: Commands, device: Res<RenderDevice>) {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("evolve env"),
        size: EnvUniform::min_size().get(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(EnvGpu { buffer });
}

fn init_settings(mut commands: Commands, args: Res<Args>, existing: Option<Res<GraphicsSettings>>) {
    if existing.is_none() {
        let mut s = GraphicsSettings::default();
        if args.no_shadows {
            s.shadows = ShadowQuality::Off;
        }
        commands.insert_resource(s);
    }
}

/// A terrain's heights as the bytes of the shaders' heights buffer.
pub fn heights_bytes(t: &Terrain) -> Vec<u8> {
    if t.heights.is_empty() {
        vec![0; 16]
    } else {
        t.heights.iter().flat_map(|h| h.to_le_bytes()).collect()
    }
}

/// A storage buffer holding a terrain's heights, for the water and rain shaders. Rewritten in
/// place while the terrain keeps its size.
pub fn heights_buffer(device: &RenderDevice, t: &Terrain) -> Buffer {
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("evolve heights"),
        contents: &heights_bytes(t),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

fn spawn_light(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 9_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(70.0, 120.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        cascades(150.0, ShadowQuality::High),
        SkyLight,
    ));
}

/// Shadow cascades sized to what the camera can see, so close-ups stay crisp and overviews still
/// get shadows.
fn cascades(distance: f32, q: ShadowQuality) -> CascadeShadowConfig {
    // Far shadows cost CPU time per entity and cascade, and plants' shadows are sub-pixel from
    // far away, so the reach is capped.
    let (n, reach, cap) = match q {
        ShadowQuality::High => (4, 2.0, 320.0),
        _ => (3, 1.6, 220.0),
    };
    CascadeShadowConfigBuilder {
        num_cascades: n,
        minimum_distance: 0.1,
        first_cascade_far_bound: (distance * 0.45).clamp(5.0, 120.0),
        maximum_distance: (distance * reach).clamp(50.0, cap),
        overlap_proportion: 0.2,
    }
    .build()
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Sunlight color after passing through the air, normalized to its brightest channel, and its
/// relative brightness.
fn sun_tint(elev: f32) -> (Vec3, f32) {
    let alt = elev.clamp(-0.05, 1.0).asin().to_degrees();
    // Kasten and Young's relative air mass.
    let airmass = 1.0 / (elev.max(-0.02) + 0.50572 * (alt + 6.07995).max(0.5).powf(-1.6364)).max(0.03);
    let beta = Vec3::new(0.022, 0.055, 0.13);
    let t = Vec3::new(
        (-beta.x * airmass).exp(),
        (-beta.y * airmass).exp(),
        (-beta.z * airmass).exp(),
    );
    let noon = Vec3::new((-beta.x).exp(), (-beta.y).exp(), (-beta.z).exp());
    let rel = t / noon;
    let m = rel.max_element().max(1e-4);
    (rel / m, (rel.x * 0.3 + rel.y * 0.6 + rel.z * 0.1).min(1.0))
}

const WAVE_SPREAD: [f32; 8] = [0.0, 0.62, -0.51, 1.15, -1.05, 0.31, -1.5, 1.7];
/// Fixed wavelengths, m. Changing them with the wind would make distant waves slide.
const WAVE_LEN: [f32; 8] = [37.0, 28.5, 22.0, 16.5, 12.0, 8.8, 6.4, 4.6];
const WAVE_WEIGHT: [f32; 8] = [1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5];

#[allow(clippy::too_many_arguments)]
fn update_env(
    time: Res<Time>,
    snap: Res<Snap>,
    over: Res<VisualOverride>,
    settings: Res<GraphicsSettings>,
    cam: Res<OrbitCam>,
    mut env: ResMut<Env>,
    gpu: Res<EnvGpu>,
    queue: Res<RenderQueue>,
    light: Single<(&mut DirectionalLight, &mut Transform, &mut CascadeShadowConfig), With<SkyLight>>,
    mut last_light: Local<f64>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut clear: ResMut<ClearColor>,
    view: Single<(&mut DistanceFog, &mut ColorGrading), With<MainCamera>>,
) {
    let _cost = crate::FrameCost::start("update_env");
    let (mut fog, mut grading) = view.into_inner();
    let env = &mut *env;
    let dt = time.delta_secs().min(0.1);
    let w = &snap.0.weather;
    let p = &snap.0.params;
    let valid = Vec3::from_array(w.sun_dir).length_squared() > 0.5;

    // Targets: the sim's weather unless overridden.
    let tod = over.time_of_day.or(valid.then_some(w.time_of_day));
    let target_sun = tod.map(|t| Vec3::from_array(weather::sun_direction(t.rem_euclid(1.0))));
    let sim_wind = Vec2::from_array(w.wind);
    let (cloud, rain, storm, wind) = match over.weather {
        Some(preset) => {
            let (c, r, s, speed) = preset.values();
            let dir = if sim_wind.length() > 0.1 { sim_wind.normalize() } else { Vec2::new(0.9, 0.44) };
            (c, r, s, dir * speed)
        }
        None if valid => (w.cloud, w.rain, w.storm, sim_wind),
        None => (env.cloud, env.rain, env.storm, env.wind),
    };

    let new_world = snap.0.world_epoch != env.epoch && valid;
    if !env.ready || new_world {
        // Jump straight to the first real weather instead of fading in from defaults.
        if let Some(s) = target_sun {
            env.sun_dir = s;
            env.cloud = cloud;
            env.rain = rain;
            env.storm = storm;
            env.wind = wind;
            env.wetness = rain;
            env.time = 0.0;
            env.cloud_offset = bevy::math::DVec2::ZERO;
            env.ready = valid || over.time_of_day.is_some();
            if valid {
                env.epoch = snap.0.world_epoch;
            }
        }
    }
    let k = 1.0 - (-dt / 0.4).exp();
    if let Some(s) = target_sun {
        env.sun_dir = (env.sun_dir + (s - env.sun_dir) * k).normalize_or(s);
    }
    env.cloud += (cloud - env.cloud) * k;
    env.rain += (rain - env.rain) * k;
    env.storm += (storm - env.storm) * k;
    let kw = 1.0 - (-dt / 1.0).exp();
    env.wind += (wind - env.wind) * kw;
    let wet_tau = if env.rain > env.wetness { 6.0 } else { 25.0 };
    env.wetness += (env.rain - env.wetness) * (1.0 - (-dt / wet_tau).exp());
    env.time = (env.time + dt) % TIME_WRAP;
    let wind_now = env.wind;
    env.cloud_offset = (env.cloud_offset - (wind_now * dt * 4.0).as_dvec2()) % CLOUD_WRAP;
    // Rain falls by the integral of its speed, so a change of wind or storm changes how fast the
    // streaks move, not where they are.
    let fall = crate::weather_fx::rain_velocity(env.wind, env.storm).as_dvec3() * dt as f64;
    let rain_box = crate::weather_fx::RAIN_BOX.as_dvec3();
    env.rain_fall = (env.rain_fall + fall).rem_euclid(rain_box);
    env.rain_time = (env.rain_time + dt as f64) % crate::weather_fx::RAIN_EXTRA_PERIOD;

    let sun = env.sun_dir;
    let elev = sun.y;
    let daylight = weather::daylight(elev);
    let overcast = smoothstep(0.45, 1.0, env.cloud);
    let storm = env.storm;
    let flash = env.flash;
    let exposure = bevy::camera::Exposure::default().exposure();

    // Sun and moon.
    let (tint, sun_lum) = sun_tint(elev);
    let sun_vis = smoothstep(-0.03, 0.1, elev);
    let cloud_dim = (1.0 - 0.85 * env.cloud.powf(1.6)) * (1.0 - 0.5 * storm);
    let sun_lux = 9_000.0 * sun_vis * (0.3 + 0.7 * sun_lum) * cloud_dim;
    let moon = -sun;
    let moon_vis = smoothstep(0.03, 0.2, moon.y);
    let moon_tint = Vec3::new(0.62, 0.72, 1.0);
    let moon_lux = 1_400.0 * moon_vis * (1.0 - 0.7 * env.cloud.powf(1.5));
    let use_sun = elev > -0.03;
    let (ldir, lcolor, lux) = if use_sun { (sun, tint, sun_lux) } else { (moon, moon_tint, moon_lux) };
    let (mut dl, mut tf, mut casc) = light.into_inner();
    // Touch the light only when it really changes, and gradual changes at most every
    // `LIGHT_EVERY` seconds: a changed light makes Bevy redo its shadow cascades' visibility for
    // every caster, and with twenty thousand plants and bodies that's a hitch you can see (with
    // drifting clouds it came every couple of seconds).
    let since = time.elapsed_secs_f64() - *last_light;
    let old = dl.color.to_linear();
    let color_moved = (Vec3::new(old.red, old.green, old.blue) - lcolor).abs().max_element();
    let lux_moved = (dl.illuminance - lux).abs() / lux.max(1.0);
    let sudden = lux_moved > 0.25;
    if sudden || (since > LIGHT_EVERY && (lux_moved > 0.03 || color_moved > 0.01)) {
        dl.illuminance = lux;
        dl.color = Color::linear_rgb(lcolor.x, lcolor.y, lcolor.z);
        *last_light = time.elapsed_secs_f64();
    }
    if tf.forward().dot(-ldir) < 0.9999 && (since > LIGHT_EVERY || tf.forward().dot(-ldir) < 0.999) {
        let up = if ldir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        *tf = Transform::default().looking_to(-ldir, up);
    }
    let shadows_on = settings.shadows != ShadowQuality::Off;
    if dl.shadows_enabled != shadows_on {
        dl.shadows_enabled = shadows_on;
    }
    // Soft shadows: the penumbra grows with the light's size, a small bright disc in a clear sky
    // and the whole glowing cloud deck when it's overcast or stormy. Stepped, so the light (and
    // with it every caster's shadow visibility) changes only now and then.
    let soft = (settings.soft_shadows && shadows_on).then(|| {
        let cover = smoothstep(0.15, 1.0, env.cloud).max(storm);
        ((SOFT_CLEAR + (SOFT_OVERCAST - SOFT_CLEAR) * cover) * 2.0).round() / 2.0
    });
    if dl.soft_shadow_size != soft {
        dl.soft_shadow_size = soft;
    }
    // Rebuild cascades only when the zoom changes noticeably.
    let key = ((cam.distance.max(1.0).ln() * 12.0) as u32, settings.shadows as u32);
    if key != env.shadow_key {
        env.shadow_key = key;
        *casc = cascades(cam.distance, settings.shadows);
    }

    // Sky colors, in display units. The sky stays lit a while after sunset.
    let sky_day = smoothstep(-0.2, 0.12, elev);
    let warm = (1.0 - smoothstep(0.02, 0.4, elev)) * smoothstep(-0.2, -0.02, elev);
    let day_z = Vec3::new(0.11, 0.27, 0.72);
    let day_h = Vec3::new(0.58, 0.70, 0.86);
    let dusk_z = Vec3::new(0.19, 0.24, 0.52);
    let dusk_h = Vec3::new(0.98, 0.56, 0.38);
    let night_z = Vec3::new(0.004, 0.007, 0.018);
    let night_h = Vec3::new(0.016, 0.024, 0.045);
    let clear_z = mix3(night_z, mix3(day_z, dusk_z, warm), sky_day);
    let clear_h = mix3(night_h, mix3(day_h, dusk_h, warm), sky_day);
    let grey_day = Vec3::new(0.58, 0.61, 0.66) * (1.0 - 0.72 * storm);
    let grey = mix3(night_h * 0.8, grey_day, sky_day);
    let mut zenith = mix3(clear_z, grey * 0.8, overcast);
    let mut horizon = mix3(clear_h, grey, overcast);
    let flash_light = Vec3::new(0.55, 0.6, 0.8) * flash * 0.6;
    zenith += flash_light;
    horizon += flash_light;
    let twilight = smoothstep(-0.2, -0.04, elev) * (1.0 - smoothstep(-0.01, 0.12, elev)) * (1.0 - overcast);
    let night = (1.0 - smoothstep(-0.2, 0.0, elev)) * (1.0 - overcast);

    // Ambient: bright sky by day, dim blue by night, never black.
    let day_amb = 360.0 * (1.0 + 0.25 * env.cloud) * (1.0 - 0.5 * storm);
    let night_amb = 110.0;
    let amb = night_amb + (day_amb - night_amb) * daylight + flash * 2500.0;
    let amb_day_col = mix3(Vec3::new(0.80, 0.88, 1.0), Vec3::new(0.86, 0.88, 0.92), overcast);
    let amb_day_col = mix3(amb_day_col, Vec3::new(1.0, 0.85, 0.78), warm * (1.0 - overcast));
    let amb_col = mix3(Vec3::new(0.5, 0.6, 1.0), amb_day_col, daylight);
    ambient.color = Color::linear_rgb(amb_col.x, amb_col.y, amb_col.z);
    ambient.brightness = amb;

    // Fog: hazy far off, thick in rain and storms.
    let vis = 4500.0 * (1.0 - 0.5 * env.cloud) * (1.0 - 0.5 * env.rain) * (1.0 - 0.45 * storm);
    let density = 1.7308 / vis.max(80.0);
    let glow_k = (0.09 + 0.3 * warm) * (1.0 - overcast) * if use_sun { 1.0 } else { 0.5 };
    let glow_exp = 6.0 + 20.0 * smoothstep(0.0, 0.5, ldir.y);
    fog.color = Color::linear_rgb(horizon.x, horizon.y, horizon.z);
    fog.falloff = FogFalloff::ExponentialSquared { density };
    fog.directional_light_color = Color::linear_rgba(glow_k, glow_k, glow_k, 1.0);
    fog.directional_light_exponent = glow_exp;
    clear.0 = Color::linear_rgb(horizon.x, horizon.y, horizon.z);

    // Moonlight reads blue and colorless; storms look washed out.
    let dark = 1.0 - smoothstep(-0.25, -0.04, elev);
    grading.global.post_saturation = (1.0 - 0.45 * dark) * (1.0 - 0.2 * storm);
    grading.global.temperature = -0.15 * dark;

    // Waves: longer and steeper with the wind, phases advancing in wall time.
    let speed = env.wind.length();
    let wdir = if speed > 0.05 { env.wind / speed } else { Vec2::X };
    let sea = smoothstep(1.0, 17.0, speed).max(storm);
    // The wind sets the longest waves present; longer ones fade out instead of changing length.
    let peak = 7.0 + 30.0 * sea;
    let steep = 0.05 + 0.08 * sea;
    let mut waves = [Vec4::ZERO; 8];
    let mut amps = [0.0f32; 8];
    for i in 0..8 {
        let len = WAVE_LEN[i];
        let kk = std::f32::consts::TAU / len;
        let over_peak = (len / peak).ln().max(0.0);
        let present = (-over_peak * over_peak / (2.0 * 0.35 * 0.35)).exp();
        let d = Vec2::from_angle(WAVE_SPREAD[i]).rotate(wdir);
        let omega = (9.81 * kk).sqrt();
        env.wave_phase[i] = (env.wave_phase[i] + omega * dt) % std::f32::consts::TAU;
        waves[i] = Vec4::new(d.x, d.y, kk, env.wave_phase[i]);
        amps[i] = steep * len / std::f32::consts::TAU * WAVE_WEIGHT[i] * 0.8 * present;
    }
    env.wave_crest = amps.iter().sum();
    env.waves = waves;
    env.wave_amps = amps;

    // Stars turn with the sun around the celestial pole.
    let lat = 0.5f32;
    let axis = Vec3::new(0.0, -lat.sin(), lat.cos());
    let hour = (sun.y / lat.cos()).atan2(sun.x);

    let sun_disp = tint * sun_lux * exposure;
    let moon_disp = moon_tint * moon_lux * exposure;
    let light_disp = lcolor * lux * exposure;
    let amb_disp = amb_col * amb * exposure;
    let t = &snap.0.terrain;
    let u = EnvUniform {
        sun: sun.extend(sun_vis),
        sun_color: sun_disp.extend(daylight),
        moon: moon.extend(moon_vis),
        moon_color: moon_disp.extend(night),
        light: ldir.extend(twilight),
        zenith: zenith.extend(0.0),
        horizon: horizon.extend(density),
        ambient: amb_disp.extend(flash),
        weather: Vec4::new(env.cloud, env.rain, env.storm, env.wetness),
        wind: Vec4::new(env.wind.x, env.wind.y, env.cloud_offset.x as f32, env.cloud_offset.y as f32),
        world: Vec4::new(p.water_level, t.size, t.res as f32, p.terrain_height),
        sea: Vec4::new(0.35 + 0.5 * sea, sea, smoothstep(0.3, 0.8, sea), 0.0),
        stars: axis.extend(hour),
        flash_dir: env.flash_dir.extend(0.0),
        fog_sun: (light_disp * glow_k).extend(glow_exp),
        misc: Vec4::new(settings.clouds as u32 as f32, settings.sway as u32 as f32, 0.0, env.time),
        rain: env.rain_fall.as_vec3().extend(env.rain_time as f32),
        waves,
        wave_amp: [
            Vec4::new(amps[0], amps[1], amps[2], amps[3]),
            Vec4::new(amps[4], amps[5], amps[6], amps[7]),
        ],
    };
    let mut bytes = encase::StorageBuffer::new(Vec::<u8>::new());
    if bytes.write(&u).is_ok() {
        queue.write_buffer(&gpu.buffer, 0, &bytes.into_inner());
    }
}

fn apply_settings(
    mut commands: Commands,
    settings: Res<GraphicsSettings>,
    camera: Single<(Entity, Has<Bloom>), With<MainCamera>>,
    mut map: ResMut<DirectionalLightShadowMap>,
) {
    if !settings.is_changed() {
        return;
    }
    let (e, has_bloom) = *camera;
    if settings.bloom && !has_bloom {
        commands.entity(e).insert(crate::camera::bloom());
    } else if !settings.bloom && has_bloom {
        commands.entity(e).remove::<Bloom>();
    }
    let size = if settings.shadows == ShadowQuality::Low { 2048 } else { 4096 };
    if map.size != size {
        map.size = size;
    }
}
