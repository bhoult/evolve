// Island ground: beaches, wet sand, sea floor with caustics, grass and rock, with noise detail,
// rain wetness and moving cloud shadows. The vertex color's red channel carries fertility, the
// green channel how much creatures have dug or piled the ground there (0 untouched, 1 a lot),
// the blue channel is 1 where it was dug out rather than piled up, and alpha is how trampled it
// is. Water standing or running on the ground (m) and soil moisture come from the surface
// buffer, interpolated smoothly per pixel. The surface of deep water is a second mesh whose UV is
// (depth, -1). Drawn deferred (for ray-traced lighting, raytracing.rs) it only writes what the
// ground is; the lighting is worked out later.
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    lighting::F_AB,
    mesh_view_bindings::{lights, view},
}
#endif
#import evolve::env::{env, vnoise, gnoise, fbm, cloud_shadow, sky_clear}

// Water depth and soil moisture at each grid point, two half floats per u32 (scene.rs).
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage, read> surface: array<u32>;

// Where creatures stir the water; written each frame by splash.rs (the same list the sea uses).
struct Ripple {
    // x, z, start time (env clock), strength
    a: vec4<f32>,
    // size of what made it (m), foam, how far it reaches right now squared, unused
    b: vec4<f32>,
}
struct Ripples {
    count: vec4<f32>,
    // The box (min x, min z, max x, max z) they all fit in.
    bounds: vec4<f32>,
    list: array<Ripple, 64>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<storage, read> ripples: Ripples;
// Must match splash.rs.
const RIPPLE_LIFE: f32 = 2.6;

/// Rings spreading from where creatures move through a lake, pond or stream: the slope they add,
/// and foam where the surface was broken hard. `fw` is meters per pixel. As water.wgsl's.
fn creature_ripples(p: vec2<f32>, t: f32, fw: f32) -> vec3<f32> {
    let rb = ripples.bounds;
    if (ripples.count.x < 0.5 || fw > 0.6 || any(p < rb.xy) || any(p > rb.zw)) {
        return vec3<f32>(0.0);
    }
    var g = vec2<f32>(0.0);
    var foam = 0.0;
    let n = min(u32(ripples.count.x), 64u);
    for (var i = 0u; i < n; i++) {
        let r = ripples.list[i];
        let to = p - r.a.xy;
        let d2 = dot(to, to);
        let age = t - r.a.z;
        if (d2 > r.b.z || age < 0.0 || age > RIPPLE_LIFE) {
            continue;
        }
        let size = r.b.x;
        let lambda = 0.22 + 0.2 * size;
        let front = size * 0.5 + (0.45 + 0.35 * size) * age;
        let d = sqrt(d2);
        let x = d - front;
        let width = select(lambda * (0.9 + 1.3 * age), lambda * 0.45, x > 0.0);
        let shape = exp(-x * x / (width * width));
        let fade = (1.0 - age / RIPPLE_LIFE) * (1.0 - age / RIPPLE_LIFE);
        let spread = inverseSqrt(1.0 + 2.0 * front);
        let resolved = 1.0 - smoothstep(lambda * 0.12, lambda * 0.5, fw);
        let slope = r.a.w * 0.35 * fade * spread * resolved * shape * sin(6.2831853 / lambda * x);
        g += to / max(d, 1e-3) * slope;
        let core = 1.0 - smoothstep(size * 0.2, size * 0.6 + 0.1, d);
        foam = max(foam, r.b.y * (core * exp(-age * 2.5) + 0.5 * exp(-age * 1.6) * exp(-x * x / 0.02)));
    }
    return vec3<f32>(g, foam);
}

fn surface_point(i: i32, j: i32, res: i32) -> vec2<f32> {
    let ii = clamp(i, 0, res - 1);
    let jj = clamp(j, 0, res - 1);
    return unpack2x16float(surface[ii * res + jj]);
}

/// Catmull-Rom weights for the four points around a fraction t of the way between the middle two.
fn catmull_rom(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

/// Water depth (m) and soil moisture at a spot, interpolated smoothly over the 4x4 grid points
/// around it (bicubic), so stream banks curve instead of following the triangles. Kept within
/// the four nearest points' range, so a stream's edge can't ring below zero or above its depth.
fn surface_at(xz: vec2<f32>) -> vec2<f32> {
    let size = env.world.y;
    let res = i32(env.world.z);
    if (res < 3 || arrayLength(&surface) < u32(res * res)) {
        return vec2<f32>(0.0);
    }
    let n = f32(res - 1);
    let g = clamp((xz / size + 0.5) * n, vec2<f32>(0.0), vec2<f32>(n));
    let j = min(i32(floor(g.x)), res - 2);
    let i = min(i32(floor(g.y)), res - 2);
    let f = g - vec2<f32>(f32(j), f32(i));
    let wx = catmull_rom(f.x);
    let wz = catmull_rom(f.y);
    var acc = vec2<f32>(0.0);
    var lo = vec2<f32>(1e9);
    var hi = vec2<f32>(-1e9);
    for (var a = 0; a < 4; a++) {
        var row = vec2<f32>(0.0);
        for (var b = 0; b < 4; b++) {
            let v = surface_point(i + a - 1, j + b - 1, res);
            row += v * wx[b];
            if (a == 1 || a == 2) && (b == 1 || b == 2) {
                lo = min(lo, v);
                hi = max(hi, v);
            }
        }
        acc += row * wz[a];
    }
    return clamp(acc, lo, hi);
}

fn lin(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

fn caustics(p: vec2<f32>, t: f32) -> f32 {
    let a = vnoise(p * 1.4 + vec2<f32>(t * 0.35, t * 0.2));
    let b = vnoise(p * 1.4 * 1.31 + vec2<f32>(-t * 0.27, t * 0.31) + 7.3);
    let line = 1.0 - abs(a - b) * 3.0;
    return pow(clamp(line, 0.0, 1.0), 5.0);
}

/// 1 while noise of frequency `f` (per meter) is well resolved, fading to 0 before it aliases.
fn band(px: f32, f: f32) -> f32 {
    return 1.0 - smoothstep(0.25 / f, 0.6 / f, px);
}

/// Triplanar noise: projected along whichever axes the ground faces, so cliffs don't smear it.
fn ground_noise(p: vec3<f32>, w: vec3<f32>, f: f32) -> f32 {
    var v = 0.0;
    if (w.y > 0.01) {
        v += gnoise(p.xz * f) * w.y;
    }
    if (w.x > 0.01) {
        v += gnoise(p.zy * f + 13.7) * w.x;
    }
    if (w.z > 0.01) {
        v += gnoise(p.xy * f + 31.1) * w.z;
    }
    return v;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, is_front);
    let p = in.world_position.xyz;
    let n = normalize(in.world_normal);
    let wl = env.world.x;
    let th = max(env.world.w, 1.0);
    let above = p.y - wl;
    let fert = in.color.r;
    let t = env.misc.w;
    let slope = 1.0 - n.y;

    // Meters per pixel: each noise octave fades out before its cells shrink below a few pixels.
    let px = length(fwidth(p));
    let mid = band(px, 0.8);
    let fine = band(px, 4.3);

    // Triplanar weights, sharpened so mostly one projection is sampled.
    var tw = pow(abs(n), vec3<f32>(4.0));
    tw = tw / (tw.x + tw.y + tw.z);

    let big = fbm(p.xz * 0.035, 3);
    let n1 = mix(0.5, ground_noise(p, tw, 0.8), mid);
    let n2 = mix(0.5, ground_noise(p, tw, 4.3), fine);
    let n3 = mix(0.5, ground_noise(p, tw, 13.0), band(px, 13.0));

    let sand_dry = lin(vec3<f32>(0.84, 0.77, 0.60));
    let sand_wet = lin(vec3<f32>(0.52, 0.45, 0.34));
    let seabed = lin(vec3<f32>(0.66, 0.62, 0.48));
    let seabed_deep = lin(vec3<f32>(0.30, 0.34, 0.30));
    let grass_lush = lin(vec3<f32>(0.26, 0.42, 0.14));
    let grass_dark = lin(vec3<f32>(0.17, 0.30, 0.11));
    let grass_dry = lin(vec3<f32>(0.55, 0.54, 0.29));
    let rock_lt = lin(vec3<f32>(0.52, 0.50, 0.46));
    let rock_dk = lin(vec3<f32>(0.36, 0.35, 0.33));

    // Land: grass by fertility with patches and clumps, rock on steep slopes and high ground.
    let lushness = clamp(fert * 1.1 + (big - 0.5) * 1.0 + (n1 - 0.5) * 0.35, 0.0, 1.0);
    var grass = mix(grass_dry, mix(grass_lush, grass_dark, smoothstep(0.55, 0.8, n1) * 0.45), lushness);
    grass *= 0.8 + 0.28 * n2 + 0.2 * (n3 - 0.5);
    // Close up, a fine grain of blades.
    let closeup = band(px, 40.0);
    if (closeup > 0.0) {
        let blades = vnoise(p.xz * vec2<f32>(48.0, 31.0)) * 0.6 + vnoise(p.xz * vec2<f32>(23.0, 61.0)) * 0.4;
        grass *= 1.0 + (blades - 0.5) * 0.45 * closeup;
    }
    let strata = vnoise(vec2<f32>(p.y * 2.2 + n1 * 1.4, dot(p.xz, vec2<f32>(0.04, 0.05))));
    let rock = mix(rock_dk, rock_lt, clamp(strata * 0.6 + n2 * 0.4 + (n3 - 0.5) * 0.25, 0.0, 1.0));
    let height01 = above / th;
    let rockiness = clamp(
        smoothstep(0.26, 0.42, slope + (n1 - 0.5) * 0.16) + smoothstep(0.78, 1.0, height01 + (big - 0.5) * 0.4),
        0.0, 1.0);
    var col = mix(grass, rock, rockiness);
    var rough = mix(0.95, 0.78, rockiness);

    // Beach: a band above the waterline that varies along the coast.
    let beach_top = 0.6 + 1.0 * big + 0.35 * n1;
    let sandy = (1.0 - smoothstep(beach_top - 0.3, beach_top + 0.3, above)) * (1.0 - smoothstep(0.3, 0.55, slope));
    let ripples = 0.94 + 0.06 * sin(dot(p.xz, vec2<f32>(2.3, 1.1)) * 3.0 + n1 * 6.0) * band(px, 1.5);
    col = mix(col, sand_dry * (0.88 + 0.16 * n2 + 0.08 * (n3 - 0.5)) * ripples, sandy);
    rough = mix(rough, 0.88, sandy);

    // Wet sand at the waterline, wider when the sea is rough.
    let wet_top = 0.25 + env.sea.y * 0.7;
    let wet = 1.0 - smoothstep(wet_top * 0.35, wet_top, above);
    col = mix(col, sand_wet * (0.9 + 0.2 * n2), wet * max(sandy, 0.5));
    rough = mix(rough, 0.45, wet);

    // Sea floor: sand turning darker and greener with depth, caustics in the shallows.
    if (above < 0.0) {
        let d = -above;
        let floor_col = mix(seabed, seabed_deep, smoothstep(1.5, 9.0, d + (big - 0.5) * 3.0)) * (0.9 + 0.2 * n2);
        col = mix(col, floor_col, smoothstep(0.0, 0.8, d));
        let c = caustics(p.xz, t) * exp(-d * 0.35) * smoothstep(0.0, 0.5, d) * env.sun.w
            * (1.0 - env.weather.x * 0.7) * smoothstep(0.35, 0.1, slope);
        col *= 1.0 + 1.6 * c * band(px, 2.0);
        rough = 0.9;
    }

    // Earth creatures have dug out or piled up: bare soil, darker in pits than on heaps.
    let moved = in.color.g;
    if (moved > 0.01 && above > -0.2) {
        let pit = 1.0 - 0.3 * in.color.b;
        let soil = lin(vec3<f32>(0.40, 0.30, 0.20)) * (0.85 + 0.25 * n2 + 0.1 * (n3 - 0.5)) * pit;
        col = mix(col, soil, smoothstep(0.0, 0.35, moved) * (1.0 - wet * 0.5));
        rough = mix(rough, 0.97, smoothstep(0.0, 0.35, moved));
    }

    // Feet wear the grass to bare earth along well-used paths, and wet paths turn to mud.
    let worn = in.color.a;
    let wet_here = surface_at(p.xz);
    let water = wet_here.x;
    let soil_wet = wet_here.y;
    var lake = false;
#ifdef VERTEX_UVS_A
    lake = in.uv.y < -0.5;
    // Debugging the terrain's detail levels (scene.rs, `lod_tint`): ground drawn at a coarser
    // stride carries its level in UV x, tinted green, yellow, orange, red as it doubles.
    if (!lake && in.uv.x > 0.5) {
        let level = i32(in.uv.x + 0.5);
        var tints = array<vec3<f32>, 4>(vec3<f32>(0.55, 1.0, 0.55), vec3<f32>(1.0, 1.0, 0.4), vec3<f32>(1.0, 0.6, 0.25), vec3<f32>(1.0, 0.3, 0.3));
        col *= tints[clamp(level - 1, 0, 3)];
    }
#endif
    let on_land = smoothstep(0.0, 0.25, above);
    let path = smoothstep(0.08, 0.7, worn + (n1 - 0.5) * 0.15) * on_land * (1.0 - 0.7 * rockiness);
    if (path > 0.001) {
        let dirt = lin(vec3<f32>(0.45, 0.37, 0.27)) * (0.85 + 0.25 * n2 + 0.1 * (n3 - 0.5));
        col = mix(col, dirt, path * (1.0 - 0.5 * sandy));
        rough = mix(rough, 0.92, path);
        let mud = path * smoothstep(0.3, 0.75, max(soil_wet, env.weather.w));
        let mud_col = lin(vec3<f32>(0.20, 0.14, 0.09)) * (0.85 + 0.3 * n2);
        col = mix(col, mud_col, mud);
        // Slick, with puddles standing in its hollows.
        let puddle = smoothstep(0.55, 0.7, n1 + (n2 - 0.5) * 0.3);
        rough = mix(rough, mix(0.35, 0.08, puddle), mud);
    }

    // Rain darkens dry ground and gives it a soft sheen; soaked soil stays dark after it stops.
    let wetness = max(env.weather.w, soil_wet * 0.8) * smoothstep(-0.2, 0.3, above);
    col *= 1.0 - 0.35 * wetness;
    rough = mix(rough, 0.55, wetness);

    // Water on the ground. Its surface (ground plus depth, both linear across a triangle) has
    // its own slope, found from screen-space derivatives: about level in a pond, falling along a
    // stream. It's constant per triangle, so it only tells ponds from streams; the shading itself
    // follows the smooth ground. (Derivatives need uniform control flow: no branch before.)
    let surf = p.y + water;
    let ax = dpdx(p.xz);
    let ay = dpdy(p.xz);
    let det = ax.x * ay.y - ax.y * ay.x;
    let sx = dpdx(surf);
    let sy = dpdy(surf);
    var grad = vec2<f32>(0.0);
    if (abs(det) > 1e-8) {
        grad = vec2<f32>(ay.y * sx - ax.y * sy, ax.x * sy - ay.x * sx) / det;
    }

    // The surface of deep water: a second mesh over lakes, ponds and deep streams (scene.rs),
    // flagged by a UV of -1. Sky reflections and sun glints over a dark body, see-through
    // where it thins out at the shore. Lakes lie level (their normal is up); streams follow their
    // bed and carry their ripples downstream.
    if (lake) {
        let fall = length(n.xz);
        let downhill = n.xz / max(fall, 1e-4);
        let drift = downhill * t * (0.3 + 5.0 * fall) + vec2<f32>(t * 0.05, 0.0);
        let q1 = p.xz * 2.3 - drift;
        let q2 = p.xz * 0.9 - drift * 0.6;
        let ripple = (vec2<f32>(vnoise(q1), vnoise(q1 + 17.3)) - 0.5) * 0.12 * band(px, 2.3)
            + (vec2<f32>(vnoise(q2 + 5.1), vnoise(q2 + 23.7)) - 0.5) * 0.08 * band(px, 0.9);
        let stirred = creature_ripples(p.xz, t, px);
        let wn_ = normalize(n + vec3<f32>(-ripple.x - stirred.x, 0.0, -ripple.y - stirred.y));
        let v = pbr.V;
        let ndv_ = clamp(dot(wn_, v), 0.0, 1.0);
        let fres = 0.02 + 0.98 * pow(1.0 - ndv_, 5.0);
        var rr = reflect(-v, wn_);
        rr.y = abs(rr.y);
        let incoming = env.ambient.rgb * 1.2 + env.sun_color.rgb * max(env.sun.y, 0.0) * 0.35
            + env.moon_color.rgb * max(env.moon.y, 0.0) * 0.35;
        let body = mix(vec3<f32>(0.03, 0.17, 0.18), vec3<f32>(0.006, 0.035, 0.055), smoothstep(0.5, 4.0, water)) * incoming;
        let hs = normalize(v + env.sun.xyz);
        let glint = pow(max(dot(wn_, hs), 0.0), 400.0) * 18.0 * step(0.0, env.sun.y) * cloud_shadow(p);
        let foam = clamp(stirred.z, 0.0, 1.0);
        let lake_col = mix(body * (1.0 - fres) + sky_clear(rr, false) * fres + env.sun_color.rgb * glint, incoming * 0.8, foam * 0.8);
        let alpha = max(smoothstep(0.1, 0.35, water) * clamp(0.55 + 0.35 * smoothstep(0.3, 3.0, water) + fres, 0.0, 0.97), foam * 0.9);
        var surface_out: FragmentOutput;
        surface_out.color = main_pass_post_lighting_processing(pbr, vec4<f32>(lake_col, alpha));
        return surface_out;
    }

    let cover = smoothstep(0.005, 0.05, water) * on_land;
    if (cover > 0.001) {
        // Shallows show their bed through tinted water; a lake metres deep is dark blue-green.
        let deep = smoothstep(0.03, 0.5, water);
        let abyss = smoothstep(0.5, 4.0, water);
        let shallows = mix(col * 0.5, lin(vec3<f32>(0.12, 0.25, 0.27)), deep * 0.7);
        let body = mix(shallows, lin(vec3<f32>(0.03, 0.10, 0.13)), abyss * 0.85);
        col = mix(col, body, cover);
        rough = mix(rough, 0.05, cover);
        // A stream lies along its bed and runs down it; a pond lies level.
        let running = smoothstep(0.015, 0.06, length(grad));
        let fall = length(n.xz) * running;
        let downhill = n.xz / max(length(n.xz), 1e-4);
        // Ripples carried downstream, faster where it falls faster; a pond only shivers. Each
        // octave fades out before it's fine enough to alias.
        let drift = downhill * t * (0.3 + 5.0 * fall) + vec2<f32>(t * 0.05, 0.0);
        let q1 = p.xz * 2.3 - drift;
        let q2 = p.xz * 0.9 - drift * 0.6;
        let ripple = (vec2<f32>(vnoise(q1), vnoise(q1 + 17.3)) - 0.5) * 0.16 * band(px, 2.3)
            + (vec2<f32>(vnoise(q2 + 5.1), vnoise(q2 + 23.7)) - 0.5) * 0.1 * band(px, 0.9);
        let surface = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), n, running));
        let stirred = creature_ripples(p.xz, t, px);
        pbr.N = normalize(mix(pbr.N, normalize(surface + vec3<f32>(-ripple.x - stirred.x, 0.0, -ripple.y - stirred.y)), cover));
        col = mix(col, vec3<f32>(0.8), clamp(stirred.z, 0.0, 1.0) * cover * 0.7);
    }

    // Bumps from two octaves of noise, much stronger on rock than on grass or sand. Skipped far
    // away where they would only shimmer.
    // Faces turned from the light keep flat normals, or bumps would catch spots of sun.
    let toward = smoothstep(-0.05, 0.3, dot(n, env.light.xyz));
    let bump = (0.1 + 0.3 * rockiness * (1.0 - sandy)) * band(px, 2.1) * toward * (1.0 - cover);
    if (bump > 0.01) {
        let e = 0.08;
        var grad = vec3<f32>(0.0);
        let fine_bump = band(px, 5.3);
        let octaves = select(1, 2, fine_bump > 0.0);
        for (var o = 0; o < octaves; o++) {
            let f = select(2.1, 5.3, o == 1);
            let amp = select(1.0, 0.4 * fine_bump, o == 1);
            let b0 = ground_noise(p, tw, f);
            let bx = ground_noise(p + vec3<f32>(e, 0.0, 0.0), tw, f);
            let by = ground_noise(p + vec3<f32>(0.0, e, 0.0), tw, f);
            let bz = ground_noise(p + vec3<f32>(0.0, 0.0, e), tw, f);
            grad += vec3<f32>(bx - b0, by - b0, bz - b0) / e * amp;
        }
        pbr.N = normalize(pbr.N - (grad - n * dot(grad, n)) * bump * 0.3);
    }

    pbr.material.perceptual_roughness = rough;
    // Cloud shadows dim the direct light only. Darkening the base color dims the ambient diffuse
    // too, so that part is added back (Bevy's ambient diffuse term with the albedo lost).
    let shade = cloud_shadow(p);
    pbr.material.base_color = vec4<f32>(col * shade, 1.0);

#ifdef PREPASS_PIPELINE
    // Without the ambient put back under cloud shadows or the sky mirrored in wet ground.
    return deferred_output(in, pbr);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    let ndv = max(dot(pbr.N, pbr.V), 0.0001);
    let lost = col * (1.0 - shade) * F_AB(1.0, ndv).x * lights.ambient_color.rgb * pbr.diffuse_occlusion;
    out.color = vec4<f32>(out.color.rgb + lost * view.exposure, out.color.a);
    // Water on the ground mirrors the sky, unless it's deep enough to have a surface of its own
    // above (then this is its bed).
    let open = cover * (1.0 - smoothstep(0.1, 0.25, water));
    if (open > 0.001) {
        let wn = clamp(dot(pbr.N, pbr.V), 0.0, 1.0);
        let fresnel = 0.02 + 0.98 * pow(1.0 - wn, 5.0);
        var r = reflect(-pbr.V, pbr.N);
        r.y = abs(r.y);
        out.color = vec4<f32>(mix(out.color.rgb, sky_clear(r, false), fresnel * open), out.color.a);
    }
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
#endif
}
