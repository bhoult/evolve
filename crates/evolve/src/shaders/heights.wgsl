// The terrain heightfield, for shaders that need to know how deep the water is.
#define_import_path evolve::heights

#import evolve::env::env

@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage, read> heights: array<f32>;

/// Height of the triangulated terrain, matching the physics surface. Past the edge of the
/// heightfield it is deep ocean.
fn terrain_height(xz: vec2<f32>) -> f32 {
    let size = env.world.y;
    let res = i32(env.world.z);
    let n = f32(res - 1);
    let g = (xz / size + 0.5) * n;
    if (res < 3 || g.x < 0.0 || g.y < 0.0 || g.x > n || g.y > n) {
        return env.world.x - 60.0;
    }
    let gc = clamp(g, vec2<f32>(0.0), vec2<f32>(n - 0.001));
    let j = i32(floor(gc.x));
    let i = i32(floor(gc.y));
    let fx = gc.x - f32(j);
    let fz = gc.y - f32(i);
    let h00 = heights[i * res + j];
    let h01 = heights[i * res + j + 1];
    let h10 = heights[(i + 1) * res + j];
    let h11 = heights[(i + 1) * res + j + 1];
    if (fx + fz <= 1.0) {
        return h00 + fx * (h01 - h00) + fz * (h10 - h00);
    }
    return h11 + (1.0 - fx) * (h10 - h11) + (1.0 - fz) * (h01 - h11);
}
