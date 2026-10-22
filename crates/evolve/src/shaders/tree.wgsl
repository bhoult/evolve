// Plants: bend with the wind above the base, with a little color variety per plant. Vertex color
// alpha marks foliage (1) versus trunk (0). Also the prepass vertex shader, so shadows sway too.
#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput}
#else
#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#endif
#import evolve::env::{env, hash3i}

/// Per-plant random numbers, from where the plant stands.
fn plant_hash(root: vec3<f32>) -> vec3<f32> {
    return hash3i(vec3<i32>(vec2<i32>(floor(root.xz * 16.0)), 7));
}

/// Where a vertex of the plant at `model` ends up, bent by the wind.
fn swayed(model: mat4x4<f32>, position: vec3<f32>) -> vec3<f32> {
    let root = model[3].xyz;
    let scale = length(model[1].xyz);
    let r = plant_hash(root);
    var wp = (model * vec4<f32>(position, 1.0)).xyz;

    // Height above the base, 0..1; the mesh spans y = -0.45..0.45.
    let h = clamp((position.y + 0.45) / 0.9, 0.0, 1.0);
    let wind = env.wind.xy;
    let speed = length(wind);
    let dir = select(vec2<f32>(1.0, 0.0), wind / max(speed, 1e-3), speed > 0.01);
    let t = env.misc.w;
    let phase = r.x * 40.0 + dot(root.xz, dir) * 0.35;
    let gust = 0.55 + 0.3 * sin(t * (1.1 + 0.3 * r.y) - phase) + 0.15 * sin(t * 3.3 + phase * 1.7);
    let bend = h * h * scale * (0.01 + 0.011 * speed) * gust * env.misc.y;
    wp.x += dir.x * bend;
    wp.z += dir.y * bend;
    // Leaves flutter.
    let flutter = sin(t * 7.0 + position.x * 9.0 + position.z * 7.0 + phase) * 0.004 * speed * h * scale * env.misc.y;
    wp.y += flutter;
    return wp;
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let model = mesh_functions::get_world_from_local(vertex.instance_index);
    let wp = swayed(model, vertex.position);

    out.world_position = vec4<f32>(wp, 1.0);
    out.position = position_world_to_clip(wp);
#ifdef PREPASS_PIPELINE
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
#endif
#endif
#ifdef MOTION_VECTOR_PREPASS
    out.previous_world_position = out.world_position;
#endif
#else
#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
#endif
#endif
#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_COLORS
    let r = plant_hash(model[3].xyz);
    var c = vertex.color;
    let tint = vec3<f32>(0.85 + 0.3 * r.y, 0.88 + 0.24 * r.z, 0.8 + 0.25 * r.x);
    c = vec4<f32>(c.rgb * mix(vec3<f32>(1.0), tint, c.a), 1.0);
    out.color = c;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(vertex.instance_index, model[3]);
#endif
    return out;
}
