// Shared environment for the custom shaders: sun, moon, sky colors, weather, waves, plus noise
// and the sky/cloud model used by the sky dome, water reflections and cloud shadows.
#define_import_path evolve::env

#import bevy_pbr::mesh_view_bindings::view

struct Env {
    // xyz toward the sun, w sun visibility.
    sun: vec4<f32>,
    // rgb sun radiance in display units (after cloud dimming), w daylight.
    sun_color: vec4<f32>,
    // xyz toward the moon, w moon visibility.
    moon: vec4<f32>,
    // rgb moonlight in display units, w night (how much the stars show).
    moon_color: vec4<f32>,
    // xyz toward the light that casts shadows (sun or moon), w twilight glow after sunset.
    light: vec4<f32>,
    zenith: vec4<f32>,
    // rgb horizon and fog color, w fog density (exponential squared).
    horizon: vec4<f32>,
    // rgb ambient light in display units, w lightning flash.
    ambient: vec4<f32>,
    // cloud cover, rain, storm, wetness of the ground.
    weather: vec4<f32>,
    // xy wind m/s, zw cloud scroll offset in meters.
    wind: vec4<f32>,
    // water level, terrain size, terrain resolution, terrain height.
    world: vec4<f32>,
    // wave choppiness, sea state 0..1, foam amount, unused.
    sea: vec4<f32>,
    // xyz star rotation axis, w angle.
    stars: vec4<f32>,
    // xyz toward the latest lightning bolt.
    flash_dir: vec4<f32>,
    // rgb fog glow toward the light, w its exponent.
    fog_sun: vec4<f32>,
    // clouds on, sway on, unused, wall time in seconds.
    misc: vec4<f32>,
    // per wave: direction xy, wavenumber, phase.
    waves: array<vec4<f32>, 8>,
    wave_amp: array<vec4<f32>, 2>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<storage, read> env: Env;

const CLOUD_Y: f32 = 1500.0;
const TAU: f32 = 6.2831853;

fn hash_u(x: u32) -> u32 {
    var h = x;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn hash2i(p: vec2<i32>) -> f32 {
    let h = hash_u(u32(p.x) * 1597334677u ^ hash_u(u32(p.y) + 0x9e3779b9u));
    return f32(h) * (1.0 / 4294967296.0);
}

fn hash3i(p: vec3<i32>) -> vec3<f32> {
    let a = hash_u(u32(p.x) * 1597334677u ^ hash_u(u32(p.y) * 3812015801u ^ hash_u(u32(p.z) + 0x9e3779b9u)));
    let b = hash_u(a);
    let c = hash_u(b);
    return vec3<f32>(f32(a), f32(b), f32(c)) * (1.0 / 4294967296.0);
}

fn hash12(p: vec2<f32>) -> f32 {
    return hash2i(vec2<i32>(floor(p)));
}

/// Smooth value noise in [0, 1].
fn vnoise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2i(i);
    let b = hash2i(i + vec2<i32>(1, 0));
    let c = hash2i(i + vec2<i32>(0, 1));
    let d = hash2i(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// Value noise with its gradient: (value, d/dx, d/dy).
fn vnoise_d(p: vec2<f32>) -> vec3<f32> {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let a = hash2i(i);
    let b = hash2i(i + vec2<i32>(1, 0));
    let c = hash2i(i + vec2<i32>(0, 1));
    let d = hash2i(i + vec2<i32>(1, 1));
    let k1 = b - a;
    let k2 = c - a;
    let k4 = a - b - c + d;
    return vec3<f32>(a + k1 * u.x + k2 * u.y + k4 * u.x * u.y, du * vec2<f32>(k1 + k4 * u.y, k2 + k4 * u.x));
}

fn grad2(i: vec2<i32>) -> vec2<f32> {
    let a = hash2i(i) * TAU;
    return vec2<f32>(cos(a), sin(a));
}

/// Gradient noise in about [0, 1]: smoother and less blocky than value noise.
fn gnoise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = dot(grad2(i), f);
    let b = dot(grad2(i + vec2<i32>(1, 0)), f - vec2<f32>(1.0, 0.0));
    let c = dot(grad2(i + vec2<i32>(0, 1)), f - vec2<f32>(0.0, 1.0));
    let d = dot(grad2(i + vec2<i32>(1, 1)), f - vec2<f32>(1.0, 1.0));
    return 0.5 + 0.75 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p0: vec2<f32>, octaves: i32) -> f32 {
    var p = p0;
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    for (var o = 0; o < octaves; o++) {
        sum += amp * vnoise(p);
        norm += amp;
        p = vec2<f32>(p.x * 1.6 - p.y * 1.2, p.x * 1.2 + p.y * 1.6) + vec2<f32>(17.3, -9.1);
        amp *= 0.5;
    }
    return sum / norm;
}

fn fog_factor(dist: f32) -> f32 {
    let d = dist * env.horizon.w;
    return 1.0 - exp(-d * d);
}

/// Fog color seen along `dir`, including the glow toward the sun.
fn fog_color(dir: vec3<f32>) -> vec3<f32> {
    let s = pow(max(dot(dir, env.light.xyz), 0.0), env.fog_sun.w);
    return env.horizon.rgb + env.fog_sun.rgb * s;
}

/// Cloud cover at a point on the cloud layer (x, 0..1) and how thick it is there (y, 0..1).
fn cloud_field(p: vec2<f32>, octaves: i32) -> vec2<f32> {
    let q = (p + env.wind.zw) * (1.0 / 1100.0);
    // Warp the domain a little so the shapes look less like blobs.
    let w = vec2<f32>(vnoise(q * 0.7 + vec2<f32>(3.1, 7.7)), vnoise(q * 0.7 + vec2<f32>(11.4, 2.3)));
    let f = fbm(q + (w - 0.5) * 0.9, octaves);
    let cover = env.weather.x;
    let th = 0.70 - 0.58 * cover;
    return vec2<f32>(smoothstep(th, th + 0.22 - 0.1 * cover, f), clamp((f - th) * 3.0, 0.0, 1.0));
}

fn cloud_density(p: vec2<f32>, octaves: i32) -> f32 {
    return cloud_field(p, octaves).x;
}

const TWILIGHT: vec3<f32> = vec3<f32>(1.0, 0.42, 0.14);

/// Fraction of direct light reaching a world point through the clouds.
fn cloud_shadow(p: vec3<f32>) -> f32 {
    if (env.misc.x < 0.5 || env.light.y < 0.05) {
        return 1.0;
    }
    let l = env.light.xyz;
    let q = p.xz + l.xz / l.y * (CLOUD_Y - p.y);
    let d = cloud_density(q, 3);
    return 1.0 - d * 0.75 * (1.0 - 0.6 * env.weather.x);
}

fn rotate_axis(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

fn stars(dir: vec3<f32>) -> vec3<f32> {
    let d = rotate_axis(dir, env.stars.xyz, -env.stars.w);
    let scale = 180.0;
    let p = d * scale;
    let cell = vec3<i32>(floor(p));
    let r = hash3i(cell);
    let center = vec3<f32>(cell) + 0.25 + 0.5 * r;
    let dist = length(p - center);
    let bright = select(0.0, pow(r.x, 6.0), r.y > 0.85);
    let star = bright * (1.0 - smoothstep(0.0, 0.16, dist));
    let tint = mix(vec3<f32>(0.75, 0.85, 1.0), vec3<f32>(1.0, 0.9, 0.75), r.z);
    // A faint band of milky way.
    let band = exp(-pow(dot(d, normalize(vec3<f32>(0.3, 0.2, 0.93))) * 4.0, 2.0));
    let milky = band * (0.5 + fbm(vec2<f32>(atan2(d.z, d.x), d.y) * 6.0, 3)) * 0.012;
    return tint * star * 2.5 + vec3<f32>(0.75, 0.8, 1.0) * milky;
}

/// The clear sky along `dir` without clouds: gradient, glow around the sun, sun and moon disks,
/// stars at night.
fn sky_clear(dir: vec3<f32>, disks: bool) -> vec3<f32> {
    let up = max(dir.y, 0.0);
    let h = pow(1.0 - up, 4.0);
    var col = mix(env.zenith.rgb, env.horizon.rgb, h);
    let cs = dot(dir, env.sun.xyz);
    // Glow around the sun, broad near the horizon like the fog's.
    col += env.fog_sun.rgb * pow(max(dot(dir, env.light.xyz), 0.0), env.fog_sun.w) * (0.35 + 0.65 * h);
    col += env.sun_color.rgb * (0.025 * pow(max(cs, 0.0), 64.0) + 0.12 * pow(max(cs, 0.0), 900.0));
    // Afterglow low over where the sun went down.
    if (env.light.w > 0.0) {
        let az = normalize(vec3<f32>(env.sun.x, 0.0, env.sun.z) + vec3<f32>(1e-4, 0.0, 0.0));
        let toward = max(dot(normalize(vec3<f32>(dir.x, 0.0, dir.z) + vec3<f32>(0.0, 0.0, 1e-4)), az), 0.0);
        col += TWILIGHT * env.light.w * (0.15 + 0.85 * pow(toward, 3.0)) * pow(1.0 - up, 6.0) * 0.6;
    }
    let night = env.moon_color.w;
    if (night > 0.01) {
        col += stars(dir) * night * smoothstep(0.0, 0.15, dir.y);
    }
    if (disks) {
        // A low sun stays orange instead of washing out to white.
        let sun_r = 0.99993;
        let k = mix(1.4, 9.0, smoothstep(0.0, 0.35, env.sun.y));
        col += env.sun_color.rgb * k * smoothstep(sun_r, sun_r + 0.00003, cs) * smoothstep(-0.01, 0.0, dir.y);
        let cm = dot(dir, env.moon.xyz);
        if (env.moon.w > 0.0 && cm > 0.99) {
            let moon_r = 0.99965;
            // Rough maria on the disk.
            let t = normalize(cross(env.moon.xyz, vec3<f32>(0.0, 1.0, 0.0)));
            let b = cross(t, env.moon.xyz);
            let uv = vec2<f32>(dot(dir, t), dot(dir, b)) * 1200.0;
            let maria = 0.75 + 0.25 * fbm(uv * 0.12, 3);
            let disk = smoothstep(moon_r, moon_r + 0.00006, cm);
            col += vec3<f32>(1.0, 0.97, 0.9) * 1.4 * maria * disk * env.moon.w;
            col += env.moon_color.rgb * 0.05 * pow(max(cm, 0.0), 2500.0) * env.moon.w;
        }
    }
    return col;
}

/// Clouds along `dir` from `origin`: premultiplied rgb and coverage.
fn clouds(origin: vec3<f32>, dir: vec3<f32>, octaves: i32, lit: bool) -> vec4<f32> {
    if (env.misc.x < 0.5 || dir.y < 0.003 || octaves <= 0) {
        return vec4<f32>(0.0);
    }
    let t = (CLOUD_Y - origin.y) / dir.y;
    let p = origin.xz + dir.xz * t;
    let field = cloud_field(p, octaves);
    var d = field.x;
    // Fade into the haze far away.
    let haze = 1.0 - exp(-t / 14000.0);
    d *= 1.0 - smoothstep(0.55, 1.0, haze) * 0.85;
    if (d < 0.002) {
        return vec4<f32>(0.0);
    }
    let storm = env.weather.z;
    let cover = env.weather.x;
    var light = 0.6;
    if (lit) {
        // Light reaching this point from the sun: thinner toward the sun means brighter.
        let l = env.light.xyz;
        let off = l.xz / max(l.y, 0.12) * 120.0;
        let ds = cloud_density(p + off, max(octaves - 2, 2));
        light = exp(-ds * (1.6 + 2.5 * cover));
    }
    // Thick parts are darker underneath, so even a full overcast has shape.
    let base = mix(0.78, 0.32, storm) * (1.0 - field.y * mix(0.25, 0.6, cover));
    let cs = max(dot(dir, env.sun.xyz), 0.0);
    let silver = pow(cs, 8.0) * 1.6 * (1.0 - d) + pow(cs, 60.0) * 1.5;
    let sun_part = env.sun_color.rgb * (0.07 + 0.22 * light + silver * 0.12) * base;
    let moon_part = env.moon_color.rgb * (0.06 + 0.12 * light) * base;
    let amb = (env.ambient.rgb * 1.1 + env.zenith.rgb * 0.35) * mix(1.0, 0.55, storm * d);
    var col = sun_part + moon_part + amb * base * 1.3 + TWILIGHT * env.light.w * 0.35 * base;
    // Lightning lights the clouds from inside.
    let flash = env.ambient.w;
    if (flash > 0.0) {
        let fd = max(dot(dir, env.flash_dir.xyz), 0.0);
        col += vec3<f32>(0.75, 0.8, 1.0) * flash * (0.6 + 4.0 * pow(fd, 6.0)) * d;
    }
    col = mix(col, fog_color(dir), haze);
    return vec4<f32>(col * d, d);
}

/// The full sky along `dir` as seen from `origin`.
fn sky_color(origin: vec3<f32>, dir: vec3<f32>, octaves: i32, lit: bool, disks: bool) -> vec3<f32> {
    var col = sky_clear(dir, disks);
    let c = clouds(origin, dir, octaves, lit);
    col = col * (1.0 - c.a) + c.rgb;
    let flash = env.ambient.w;
    col += vec3<f32>(0.5, 0.55, 0.7) * flash * 0.25;
    // Below the horizon the sky is fog.
    let below = smoothstep(0.0, -0.03, dir.y);
    return mix(col, fog_color(dir), below);
}
