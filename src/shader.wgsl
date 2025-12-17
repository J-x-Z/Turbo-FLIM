// shader.wgsl (Hybrid Architecture)
// GPU Task: Heavy Convolution & Reduction of Matrix Elements
// CPU Task: Solving the small 4x4 matrix (Hybrid)

struct Uniforms {
    dt: f32,
    num_time_points: u32,
    num_pixels: u32,
    _padding: u32,
}
@group(0) @binding(4) var<uniform> params: Uniforms;

@group(0) @binding(0) var<storage, read> input_time: array<f32>;
@group(0) @binding(1) var<storage, read> input_irf: array<f32>;
@group(0) @binding(2) var<storage, read> input_signal: array<f32>;

// Output: 14 scalars per pixel (10 for A symmetric, 4 for B)
// Map: 
// 0..9: ff11, ff12, ff22, fg11, fg12, fg21, fg22, gg11, gg12, gg22
// 10..13: bf1, bf2, bg1, bg2
@group(0) @binding(3) var<storage, read_write> output_coeffs: array<f32>; 

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> s_signal: array<f32, 1024>;
var<workgroup> s_irf: array<f32, 1024>;
// Intermediate
var<workgroup> s_f1: array<f32, 1024>;
var<workgroup> s_f2: array<f32, 1024>;
var<workgroup> s_g1: array<f32, 1024>;
var<workgroup> s_g2: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>, 
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let pixel_idx = group_id.x;
    if (pixel_idx >= params.num_pixels) { return; }

    let tid = local_id.x;
    let L = params.num_time_points;
    let dt = params.dt;

    // 1. Data load & Normalization (Simpler f32 version)
    for (var i = tid; i < L; i += WORKGROUP_SIZE) {
        s_signal[i] = input_signal[pixel_idx * L + i];
        s_irf[i] = input_irf[i];
    }
    workgroupBarrier();

    if (tid == 0u) {
        var max_val = 0.0;
        for (var i = 0u; i < L; i++) {
            let val = abs(s_signal[i]);
            if (val > max_val) { max_val = val; }
        }
        if (max_val > 1e-9) {
            for (var i = 0u; i < L; i++) { s_signal[i] /= max_val; }
        }
    }
    workgroupBarrier();

    // 3. Convolution with Kahan Summation
    // Compensated summation to preserve precision in f32
    for (var i = tid; i < L; i += WORKGROUP_SIZE) {
        var sum_f1 = 0.0; var c_f1 = 0.0;
        var sum_f2 = 0.0; var c_f2 = 0.0;
        var sum_g1 = 0.0; var c_g1 = 0.0;
        var sum_g2 = 0.0; var c_g2 = 0.0;

        for (var j = 0u; j <= i; j++) {
            let t_val = input_time[i - j];
            let phi1 = 1.0;
            let phi2 = t_val; // t
            
            let sig = s_signal[j];
            let irf = s_irf[j];
            
            // Term calculation
            let term_f1 = sig * phi1;
            let term_f2 = sig * phi2;
            let term_g1 = irf * phi1;
            let term_g2 = irf * phi2;
            
            // Kahan Sum for F1
            let y_f1 = term_f1 - c_f1;
            let t_f1 = sum_f1 + y_f1;
            c_f1 = (t_f1 - sum_f1) - y_f1;
            sum_f1 = t_f1;
            
            // Kahan Sum for F2
            let y_f2 = term_f2 - c_f2;
            let t_f2 = sum_f2 + y_f2;
            c_f2 = (t_f2 - sum_f2) - y_f2;
            sum_f2 = t_f2;
            
            // Kahan Sum for G1
            let y_g1 = term_g1 - c_g1;
            let t_g1 = sum_g1 + y_g1;
            c_g1 = (t_g1 - sum_g1) - y_g1;
            sum_g1 = t_g1;
            
            // Kahan Sum for G2
            let y_g2 = term_g2 - c_g2;
            let t_g2 = sum_g2 + y_g2;
            c_g2 = (t_g2 - sum_g2) - y_g2;
            sum_g2 = t_g2;
        }
        
        s_f1[i] = sum_f1 * dt;
        s_f2[i] = sum_f2 * dt;
        s_g1[i] = sum_g1 * dt;
        s_g2[i] = sum_g2 * dt;
    }
    workgroupBarrier();

    // 4. Matrix Coefficients (Reduction) with Kahan Summation
    if (tid == 0u) {
        // Accumulators for 14 coefficients
        var ff_11 = 0.0; var c_ff11 = 0.0;
        var ff_12 = 0.0; var c_ff12 = 0.0;
        var ff_22 = 0.0; var c_ff22 = 0.0;
        
        var fg_11 = 0.0; var c_fg11 = 0.0;
        var fg_12 = 0.0; var c_fg12 = 0.0;
        var fg_21 = 0.0; var c_fg21 = 0.0;
        var fg_22 = 0.0; var c_fg22 = 0.0;
        
        var gg_11 = 0.0; var c_gg11 = 0.0;
        var gg_12 = 0.0; var c_gg12 = 0.0;
        var gg_22 = 0.0; var c_gg22 = 0.0;
        
        var bf_1 = 0.0; var c_bf1 = 0.0;
        var bf_2 = 0.0; var c_bf2 = 0.0;
        var bg_1 = 0.0; var c_bg1 = 0.0;
        var bg_2 = 0.0; var c_bg2 = 0.0;
        
        for (var k = 0u; k < L; k++) {
            let f1 = s_f1[k]; let f2 = s_f2[k];
            let g1 = s_g1[k]; let g2 = s_g2[k];
            let y = s_signal[k];
            
            // Apply Kahan to all 14 sums (Tedious but necessary for "Perfect" solution)
            // Macro-like pattern not available in WGSL, so unfolding manually
            
            // FF11
            let v_ff11 = f1 * f1;
            let y_ff11 = v_ff11 - c_ff11; let t_ff11 = ff_11 + y_ff11; c_ff11 = (t_ff11 - ff_11) - y_ff11; ff_11 = t_ff11;
            
            // FF12
            let v_ff12 = f1 * f2;
            let y_ff12 = v_ff12 - c_ff12; let t_ff12 = ff_12 + y_ff12; c_ff12 = (t_ff12 - ff_12) - y_ff12; ff_12 = t_ff12;
            
            // FF22
            let v_ff22 = f2 * f2;
            let y_ff22 = v_ff22 - c_ff22; let t_ff22 = ff_22 + y_ff22; c_ff22 = (t_ff22 - ff_22) - y_ff22; ff_22 = t_ff22;

            // FG terms
            let v_fg11 = f1 * g1; let y_fg11 = v_fg11 - c_fg11; let t_fg11 = fg_11 + y_fg11; c_fg11 = (t_fg11 - fg_11) - y_fg11; fg_11 = t_fg11;
            let v_fg12 = f1 * g2; let y_fg12 = v_fg12 - c_fg12; let t_fg12 = fg_12 + y_fg12; c_fg12 = (t_fg12 - fg_12) - y_fg12; fg_12 = t_fg12;
            let v_fg21 = f2 * g1; let y_fg21 = v_fg21 - c_fg21; let t_fg21 = fg_21 + y_fg21; c_fg21 = (t_fg21 - fg_21) - y_fg21; fg_21 = t_fg21;
            let v_fg22 = f2 * g2; let y_fg22 = v_fg22 - c_fg22; let t_fg22 = fg_22 + y_fg22; c_fg22 = (t_fg22 - fg_22) - y_fg22; fg_22 = t_fg22;
            
            // GG terms
            let v_gg11 = g1 * g1; let y_gg11 = v_gg11 - c_gg11; let t_gg11 = gg_11 + y_gg11; c_gg11 = (t_gg11 - gg_11) - y_gg11; gg_11 = t_gg11;
            let v_gg12 = g1 * g2; let y_gg12 = v_gg12 - c_gg12; let t_gg12 = gg_12 + y_gg12; c_gg12 = (t_gg12 - gg_12) - y_gg12; gg_12 = t_gg12;
            let v_gg22 = g2 * g2; let y_gg22 = v_gg22 - c_gg22; let t_gg22 = gg_22 + y_gg22; c_gg22 = (t_gg22 - gg_22) - y_gg22; gg_22 = t_gg22;
            
            // B terms
            let v_bf1 = f1 * y; let y_bf1 = v_bf1 - c_bf1; let t_bf1 = bf_1 + y_bf1; c_bf1 = (t_bf1 - bf_1) - y_bf1; bf_1 = t_bf1;
            let v_bf2 = f2 * y; let y_bf2 = v_bf2 - c_bf2; let t_bf2 = bf_2 + y_bf2; c_bf2 = (t_bf2 - bf_2) - y_bf2; bf_2 = t_bf2;
            let v_bg1 = g1 * y; let y_bg1 = v_bg1 - c_bg1; let t_bg1 = bg_1 + y_bg1; c_bg1 = (t_bg1 - bg_1) - y_bg1; bg_1 = t_bg1;
            let v_bg2 = g2 * y; let y_bg2 = v_bg2 - c_bg2; let t_bg2 = bg_2 + y_bg2; c_bg2 = (t_bg2 - bg_2) - y_bg2; bg_2 = t_bg2;
        }
        
        // Write outputs mapped to linear array
        let base = pixel_idx * 14u;
        output_coeffs[base + 0u] = ff_11;
        output_coeffs[base + 1u] = ff_12;
        output_coeffs[base + 2u] = ff_22;
        output_coeffs[base + 3u] = fg_11;
        output_coeffs[base + 4u] = fg_12;
        output_coeffs[base + 5u] = fg_21;
        output_coeffs[base + 6u] = fg_22;
        output_coeffs[base + 7u] = gg_11;
        output_coeffs[base + 8u] = gg_12;
        output_coeffs[base + 9u] = gg_22;
        
        output_coeffs[base + 10u] = bf_1;
        output_coeffs[base + 11u] = bf_2;
        output_coeffs[base + 12u] = bg_1;
        output_coeffs[base + 13u] = bg_2;
    }
}
