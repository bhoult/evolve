// Sky dome: drawn around the camera at infinite depth.
#import bevy_pbr::mesh_view_bindings::view
#import evolve::env::{env, sky_color}

struct Vertex {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world = view.world_position + v.position * 100.0;
    var clip = view.clip_from_world * vec4<f32>(world, 1.0);
    // Reverse-z: just in front of the far plane at infinity, behind everything else.
    clip.z = clip.w * 1e-7;
    out.clip = clip;
    out.dir = v.position;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.dir);
    return vec4<f32>(sky_color(view.world_position, dir, 5, true, true), 1.0);
}
