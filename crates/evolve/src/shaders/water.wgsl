// Ocean surface: Gerstner waves on a camera-centered polar grid, sky reflection, sun glints,
// depth-tinted transparency and foam. Outputs premultiplied alpha.
#import bevy_pbr::{
    mesh_functions,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}
#import evolve::env::{env, sky_clear, sky_over, fog_factor, fog_color, vnoise, fbm, hash3i}
#import evolve::heights::terrain_height

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    // Where this point rests when the sea is flat.
    @location(1) rest: vec2<f32>,
    // Wave height relative to the sea state, for crest foam.
    @location(2) crest: f32,
}

// Must match the ring layout in water.rs.
const RING_A: f32 = 12.0;
const RING_B: f32 = 0.033;

fn amp(i: u32) -> f32 {
    return env.wave_amp[i / 4u][i % 4u];
}

/// Waves die down in the shallows, but some swell still reaches the beach.
fn shore_factor(depth: f32) -> f32 {
    return 0.2 + 0.8 * smoothstep(0.0, 5.0, depth);
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let model = mesh_functions::get_world_from_local(v.instance_index);
    let base = (model * vec4<f32>(v.position, 1.0)).xyz;
    let xz = base.xz;
    let depth = env.world.x - terrain_height(xz);
    let shore = shore_factor(depth);
    // Rings get coarser away from the camera; fade waves too short for the spacing.
    let spacing = RING_B * (length(v.position.xz) + RING_A);
    let chop = env.sea.x;
    var disp = vec3<f32>(0.0);
    var total = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let w = env.waves[i];
        let len = 6.2831853 / w.z;
        let a = amp(i) * shore * (1.0 - smoothstep(len * 0.1, len * 0.22, spacing));
        let th = w.z * dot(w.xy, xz) - w.w;
        disp.y += a * sin(th);
        disp.x += w.x * a * chop * cos(th);
        disp.z += w.y * a * chop * cos(th);
        total += amp(i);
    }
    let world = vec3<f32>(xz.x + disp.x, env.world.x + disp.y, xz.y + disp.z);
    out.world = world;
    out.clip = position_world_to_clip(world);
    out.rest = xz;
    out.crest = disp.y / max(total, 1e-4);
    return out;
}

fn rain_ripples(p: vec2<f32>, t: f32) -> vec2<f32> {
    var g = vec2<f32>(0.0);
    for (var layer = 0; layer < 2; layer++) {
        let q = p * 1.7 + vec2<f32>(3.7, 1.3) * f32(layer);
        let cell = floor(q);
        let r = hash3i(vec3<i32>(i32(cell.x), i32(cell.y), layer));
        let center = cell + 0.25 + 0.5 * r.xy;
        let phase = fract(t * 0.8 + r.z);
        let to = q - center;
        let d = length(to);
        let x = d - phase * 0.45;
        let ring = sin(x * 45.0) * exp(-x * x * 150.0) * (1.0 - phase) * (1.0 - phase);
        g += to / max(d, 1e-3) * ring;
    }
    return g * 0.35;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let xz = in.rest;
    let cam = view.world_position;
    let to_cam = cam - in.world;
    let dist = length(to_cam);
    let V = to_cam / dist;
    let t = env.misc.w;
    // Meters per pixel, to fade detail finer than a pixel.
    let fw = length(fwidth(xz));
    // Per pixel: far out the grid is too coarse to follow the shoreline.
    let water_depth = env.world.x - terrain_height(xz);
    let shore = shore_factor(water_depth);

    // Slope of the wave field.
    var g = vec2<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        let w = env.waves[i];
        let len = 6.2831853 / w.z;
        let a = amp(i) * shore * (1.0 - smoothstep(len * 0.08, len * 0.4, fw));
        let th = w.z * dot(w.xy, xz) - w.w;
        g += w.xy * (a * w.z * cos(th) * 0.75);
    }
    // Short waves for detail, per pixel only, spread around the wind.
    let wind = env.wind.xy;
    let sea = env.sea.y;
    let wa = atan2(wind.y, wind.x);
    // Gusts roughen the water in drifting patches.
    let gusts = vnoise(xz * 0.025 - wind * t * 0.004) * 0.7 + vnoise(xz * 0.07 + 3.3) * 0.3;
    let detail = (0.035 + 0.06 * sea) * (1.0 - smoothstep(0.15, 1.5, fw)) * (0.35 + 1.1 * gusts);
    for (var i = 0; i < 12; i++) {
        let fi = f32(i);
        let ang = wa + (fract(fi * 0.618034) - 0.5) * 2.8;
        let d = vec2<f32>(cos(ang), sin(ang));
        let len = 2.8 * pow(0.79, fi) * (0.85 + 0.3 * fract(fi * 0.3819));
        let k = 6.2831853 / len;
        let omega = sqrt(9.81 * k);
        let th = k * dot(d, xz) - omega * t + fi * 1.7;
        let fade = 1.0 - smoothstep(len * 0.1, len * 0.45, fw);
        g += d * (detail * cos(th) * fade * (1.0 - 0.04 * fi));
    }
    // Far off the sea reads as a sheen, not as rows of waves.
    g *= 1.0 - 0.85 * smoothstep(0.15, 3.0, fw);
    let rain = env.weather.y;
    if (rain > 0.01 && fw < 0.2) {
        g += rain_ripples(xz, t) * rain * (1.0 - smoothstep(0.03, 0.2, fw));
    }
    let N = normalize(vec3<f32>(-g.x, 1.0, -g.y));

    // Reflection of the sky.
    let ndv = clamp(dot(N, V), 0.0, 1.0);
    let F = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    var R = reflect(-V, N);
    R.y = abs(R.y);
    // Waves blur what the sea reflects: clouds show soft and faint.
    let clear = sky_clear(R, false);
    let clear_refl = sky_over(clear, in.world, R, 0, false);
    let refl = mix(sky_over(clear, in.world, R, 2, false), clear_refl, 0.45);

    // Glints of the sun and moon.
    let Hs = normalize(V + env.sun.xyz);
    let Hm = normalize(V + env.moon.xyz);
    let rough_k = mix(900.0, 250.0, sea);
    let spec_s = (pow(max(dot(N, Hs), 0.0), rough_k) * 22.0 + pow(max(dot(N, Hs), 0.0), 60.0) * 0.12)
        * step(0.0, env.sun.y);
    let spec_m = pow(max(dot(N, Hm), 0.0), rough_k) * 22.0 * step(0.0, env.moon.y);
    let spec = env.sun_color.rgb * spec_s + env.moon_color.rgb * spec_m;

    // Water body: turquoise over sand in the shallows, deep blue offshore.
    let depth = max(water_depth, 0.0);
    let shallow = exp(-depth * 0.35);
    let deep_col = vec3<f32>(0.004, 0.025, 0.055);
    let shallow_col = vec3<f32>(0.03, 0.22, 0.24);
    let body_albedo = mix(deep_col, shallow_col, shallow) * mix(1.0, 0.55, env.weather.z);
    let incoming = env.ambient.rgb * 1.2 + env.sun_color.rgb * max(env.sun.y, 0.0) * 0.35
        + env.moon_color.rgb * max(env.moon.y, 0.0) * 0.35;
    let body = body_albedo * incoming;
    // How much of the ground under the water shows through.
    let path = depth / max(V.y, 0.1);
    let trans = exp(-path * mix(0.45, 0.9, env.weather.y * 0.5 + env.weather.z * 0.5));

    // Foam: surf along the shore, whitecaps in rough seas.
    let fnoise = fbm(xz * 1.3 + vec2<f32>(t * 0.15, -t * 0.11), 3);
    let surf_w = 0.35 + 0.9 * sea;
    // Bands of surf rolling in toward the beach.
    let surge = 0.5 + 0.5 * sin(depth * 4.0 + t * 1.3 + fnoise * 5.0);
    var foam = (1.0 - smoothstep(0.0, surf_w, depth)) * smoothstep(0.38, 0.78, fnoise * 0.7 + surge * 0.5);
    foam = max(foam, (1.0 - smoothstep(0.0, 0.15, depth)) * (0.35 + 0.55 * smoothstep(0.3, 0.65, fnoise)));
    // Whitecaps: only some crests break, in streaks along the wind.
    let along = vec2<f32>(dot(xz, wind), dot(xz, vec2<f32>(-wind.y, wind.x))) / max(length(wind), 0.1);
    let streak = vnoise(along * vec2<f32>(0.5, 2.2) + t * 0.2);
    let breaking = smoothstep(0.55, 0.75, vnoise(xz * 0.04 + t * 0.01));
    let caps = smoothstep(0.5, 0.9, in.crest) * smoothstep(0.3, 0.9, sea) * breaking * smoothstep(0.45, 0.7, streak * 0.6 + fnoise * 0.5);
    foam = max(foam * (1.0 - smoothstep(0.3, 2.5, fw)), caps * env.sea.z * (1.0 - smoothstep(0.1, 0.8, fw)));
    foam = clamp(foam, 0.0, 1.0);
    let foam_col = (env.ambient.rgb * 1.6 + env.sun_color.rgb * max(env.sun.y, 0.0) * 0.55
        + env.moon_color.rgb * max(env.moon.y, 0.0) * 0.5) * 0.9;

    // Premultiplied: what the water adds, and how much of the background it hides.
    var col = body * (1.0 - trans) * (1.0 - F) + refl * F + spec * (1.0 - foam);
    var a = 1.0 - trans * (1.0 - F);
    col = mix(col, foam_col, foam);
    a = mix(a, 1.0, foam);

    // The ground beneath was already fogged; fog only this layer's own contribution.
    let f = fog_factor(dist);
    col = col * (1.0 - f) + fog_color(-V) * f * a;
    return vec4<f32>(col, a);
}
