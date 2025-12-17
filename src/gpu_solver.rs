use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor, Features, Instance, Limits,
    MapMode, PowerPreference, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};
use ndarray::Array1;
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use rayon::prelude::*;
use nalgebra::{DMatrix, DVector};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    dt: f32,
    num_time_points: u32,
    num_pixels: u32,
    _padding: u32, // align to 16
}

pub struct MatrixCoeffs {
    // 14 floats: 
    // FF(3), FG(4), GG(3)
    // BF(2), BG(2)
    pub data: [f32; 14],
}

pub async fn run_hybrid_solver(
    time: &Array1<f64>,
    irf: &Array1<f64>,
    signals: &Vec<Array1<f64>>,
    dt: f64,
) -> Vec<(f64, f64)> {
    // 1. GPU Phase: Compute Matrices
    let instance = Instance::default();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find adapter");

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                required_features: Features::empty(), // Back to standard f32
                required_limits: Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let num_pixels = signals.len() as u32;
    let num_time = time.len() as u32;
    
    // Cast to f32 for GPU
    let time_f32: Vec<f32> = time.iter().map(|&x| x as f32).collect();
    let irf_f32: Vec<f32> = irf.iter().map(|&x| x as f32).collect();
    let mut signals_flat: Vec<f32> = Vec::with_capacity((num_pixels * num_time) as usize);
    for sig in signals {
        signals_flat.extend(sig.iter().map(|&x| x as f32));
    }
    
    // Output size: 14 floats * 4 bytes * num_pixels
    let output_size = (num_pixels * 14 * 4) as u64; 
    
    let time_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Time Buffer"),
        contents: bytemuck::cast_slice(&time_f32),
        usage: BufferUsages::STORAGE,
    });
    
    let irf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("IRF Buffer"),
        contents: bytemuck::cast_slice(&irf_f32),
        usage: BufferUsages::STORAGE,
    });
    
    let signal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Signal Buffer"),
        contents: bytemuck::cast_slice(&signals_flat),
        usage: BufferUsages::STORAGE,
    });
    
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let uniforms = Uniforms {
        dt: dt as f32,
        num_time_points: num_time,
        num_pixels,
        _padding: 0,
    };
    
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Hybrid Shader"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    
    // Bind Group Layout & Pipeline (Same structure, just buffer sizes differ)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: time_buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: irf_buffer.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: signal_buffer.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: result_buffer.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: uniform_buffer.as_entire_binding() },
        ],
    });
    
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_pixels, 1, 1);
    }
    
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));
    
    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    slice.map_async(MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    
    let data = slice.get_mapped_range();
    let raw_floats: &[f32] = bytemuck::cast_slice(&data);
    
    // Copy data out to avoid holding GPU lock
    let mut coeffs_list = Vec::with_capacity(num_pixels as usize);
    for i in 0..num_pixels {
        let base = (i * 14) as usize;
        let mut c = [0.0; 14];
        c.copy_from_slice(&raw_floats[base..base+14]);
        coeffs_list.push(c);
    }
    drop(data);
    staging_buffer.unmap();
    
    // 2. CPU Phase: Parallel Solve (Rayon)
    // Convert f32 coeffs to f64 for high precision solve
    let results: Vec<(f64, f64)> = coeffs_list.par_iter().map(|c| {
        solve_4x4_f64(c)
    }).collect();
    
    results
}

fn solve_4x4_f64(c: &[f32; 14]) -> (f64, f64) {
    // Unpack to f64
    let ff_11 = c[0] as f64; let ff_12 = c[1] as f64; let ff_22 = c[2] as f64;
    let fg_11 = c[3] as f64; let fg_12 = c[4] as f64; let fg_21 = c[5] as f64; let fg_22 = c[6] as f64;
    let gg_11 = c[7] as f64; let gg_12 = c[8] as f64; let gg_22 = c[9] as f64;
    
    let bf_1 = c[10] as f64; let bf_2 = c[11] as f64;
    let bg_1 = c[12] as f64; let bg_2 = c[13] as f64;
    
    // A Matrix (4x4)
    // A = [ F1F1  F1F2  -F1G1 -F1G2 ]
    //     [ F1F2  F2F2  -F2G1 -F2G2 ]
    //     [ -G1F1 -G2F1  G1G1  G1G2 ]  <-- Note: G*F' terms, symmetric to F*G'
    //     [ -G1F2 -G2F2  G1G2  G2G2 ]
    
    // Correction: A is constructed as [F*F', -F*G'; -G*F', G*G']
    // Top-Right block (-F*G') is:
    // [-fg_11, -fg_12]
    // [-fg_21, -fg_22]
    // Bottom-Left block (-G*F') is transpose of Top-Right?
    // (F*G')^T = G*F'. So yes.
    // -fg_11, -fg_21
    // -fg_12, -fg_22
    
    let mut a_mat = DMatrix::zeros(4, 4);
    a_mat[(0,0)] = ff_11; a_mat[(0,1)] = ff_12; a_mat[(0,2)] = -fg_11; a_mat[(0,3)] = -fg_12;
    a_mat[(1,0)] = ff_12; a_mat[(1,1)] = ff_22; a_mat[(1,2)] = -fg_21; a_mat[(1,3)] = -fg_22;
    a_mat[(2,0)] = -fg_11; a_mat[(2,1)] = -fg_21; a_mat[(2,2)] = gg_11; a_mat[(2,3)] = gg_12;
    a_mat[(3,0)] = -fg_12; a_mat[(3,1)] = -fg_22; a_mat[(3,2)] = gg_12; a_mat[(3,3)] = gg_22;
    
    let b_vec = DVector::from_vec(vec![-bf_1, -bf_2, bg_1, bg_2]);
    
    // Solve A * X = B
    match a_mat.try_inverse() {
        Some(inv) => {
            let x = inv * b_vec;
            // X = [x0, x1, x2, x3]. First 2 are for poly.
            let x0 = x[0];
            let x1 = x[1];
            
            // x1 * t^2 - x0 * t + 1 = 0
            if x1.abs() < 1e-12 { return (0.0, 0.0); } // Degenerate
            
            let delta = x0*x0 - 4.0*x1;
            if delta < 0.0 { return (0.0, 0.0); }
            
            let sqrt_d = delta.sqrt();
            let mut t1 = (x0 - sqrt_d) / (2.0 * x1);
            let mut t2 = (x0 + sqrt_d) / (2.0 * x1);
            
            if t1 > t2 { let tmp = t1; t1 = t2; t2 = tmp; }
            (t1, t2)
        },
        None => (0.0, 0.0)
    }
}
