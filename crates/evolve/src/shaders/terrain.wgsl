// Island ground: beaches, wet sand, sea floor with caustics, grass and rock, with noise detail,
// rain wetness and moving cloud shadows. The vertex color's red channel carries fertility.
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#import evolve::env::{env, vnoise, gnoise, fbm, cloud_shadow}

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

    // Rain darkens dry ground and gives it a soft sheen.
    let wetness = env.weather.w * smoothstep(-0.2, 0.3, above);
    col *= 1.0 - 0.35 * wetness;
    rough = mix(rough, 0.55, wetness);

    // Bumps from the noise, stronger on rock.
    // Bumps from two octaves of noise, much stronger on rock than on grass or sand. Skipped far
    // away where they would only shimmer.
    // Faces turned from the light keep flat normals, or bumps would catch spots of sun.
    let toward = smoothstep(-0.05, 0.3, dot(n, env.light.xyz));
    let bump = (0.1 + 0.3 * rockiness * (1.0 - sandy)) * band(px, 2.1) * toward;
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
    // Cloud shadows dim the direct light only; keep the ambient as it was.
    let shade = cloud_shadow(p);
    pbr.material.base_color = vec4<f32>(col * shade, 1.0);
    pbr.diffuse_occlusion = pbr.diffuse_occlusion / max(shade, 0.05);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
}
