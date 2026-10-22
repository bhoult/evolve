// Rain streaks falling through a box that wraps around the camera. Each streak is one quad:
// position holds its random seed, uv.x the side (-1, 1) and uv.y the end (0 top, 1 bottom).
#import bevy_pbr::mesh_view_bindings::view
#import evolve::env::{env, fog_factor, fog_color, hash_u}
#import evolve::heights::terrain_height

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
    @location(2) color: vec3<f32>,
}

// Must match RAIN_BOX in weather_fx.rs.
const BOX: vec3<f32> = vec3<f32>(44.0, 26.0, 44.0);

fn hidden() -> VertexOutput {
    var out: VertexOutput;
    out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.alpha = 0.0;
    out.color = vec3<f32>(0.0);
    return out;
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    let seed = v.position;
    let pick = f32(hash_u(u32(seed.x * 65535.0) ^ (u32(seed.z * 65535.0) << 16u))) * (1.0 / 4294967296.0);
    let rain = env.weather.y;
    if (pick > rain) {
        return hidden();
    }
    let storm = env.weather.z;
    // One of eight extra speeds; see RAIN_EXTRA_PERIOD in weather_fx.rs.
    let extra = floor(pick * 8.0) * 0.25;
    let vel = vec3<f32>(env.wind.x * 0.7, -(8.5 + 3.0 * storm + extra), env.wind.y * 0.7);
    let cam = view.world_position;
    // Fall in world space by what the CPU has added up, then wrap into a box around the camera.
    var p = seed * BOX + env.rain.xyz - vec3<f32>(0.0, extra * env.rain.w, 0.0);
    let center = cam + vec3<f32>(0.0, 4.0, 0.0);
    p = center + (fract((p - center) / BOX + 0.5) - 0.5) * BOX;
    let ground = max(terrain_height(p.xz), env.world.x);
    if (p.y < ground) {
        return hidden();
    }
    let speed = length(vel);
    let dir = vel / speed;
    let len = speed * 0.035;
    let end = p + dir * len * v.uv.y;
    let to_cam = cam - end;
    let d = length(to_cam);
    let side = normalize(cross(dir, to_cam / max(d, 1e-3)));
    // At least about a pixel wide; thinner streaks fade instead of shimmering.
    let width = max(0.005, d * 0.0011);
    let world = end + side * width * v.uv.x;

    var out: VertexOutput;
    out.clip = view.clip_from_world * vec4<f32>(world, 1.0);
    out.uv = v.uv;
    let near_fade = smoothstep(1.0, 4.0, d);
    let far_fade = 1.0 - smoothstep(BOX.x * 0.25, BOX.x * 0.5, d);
    out.alpha = (0.07 + 0.05 * storm) * (0.005 / width * 0.7 + 0.3) * near_fade * far_fade;
    let flash = env.ambient.w;
    out.color = env.ambient.rgb * 1.2 + env.horizon.rgb * 0.6 + env.sun_color.rgb * 0.03
        + vec3<f32>(0.8, 0.85, 1.0) * flash * 1.5;
    out.color = mix(out.color, fog_color(-to_cam / max(d, 1e-3)), fog_factor(d));
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let edge = 1.0 - abs(in.uv.x);
    let along = sin(in.uv.y * 3.14159);
    let a = clamp(in.alpha * edge * along, 0.0, 1.0);
    return vec4<f32>(in.color * a, a);
}
