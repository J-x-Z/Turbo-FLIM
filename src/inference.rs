use tract_onnx::prelude::*;
use ndarray::Array1;

pub struct DeepFlimEngine {
    // Tract Model: SimplePlan is optimized for execution
    plan: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl DeepFlimEngine {
    pub fn new(model_path: &str) -> Self {
        // Load ONNX model using Tract
        println!("Loading ONNX model using pure Rust (Tract)...");
        let model = tract_onnx::onnx()
            // Load the model
            .model_for_path(model_path).unwrap()
            // Fix Input Shape: [1, 1003] (1001 Time + 2 Phasor)
            .with_input_fact(0, f32::fact(&[1, 1003]).into()).unwrap()
            // Optimize
            .into_optimized().unwrap()
            // Make runnable
            .into_runnable().unwrap();

        Self { plan: model }
    }

    pub fn predict(&self, signal: &Array1<f64>) -> (f64, f64) {
        // Normalize
        let max_val = signal.fold(0.0f64, |a, &b| a.max(b));
        let normalized = if max_val > 0.0 { signal / max_val } else { signal.clone() };
        
        // --- Phasor Transform (Rust Implementation) ---
        // Consistent with Python: Omega = 2*pi / 20.0 (Period=20ns)
        let n_points = normalized.len(); // 1001
        let dt = 0.02; // Fixed dt
        let mut g_sum = 0.0;
        let mut s_sum = 0.0;
        let mut total_sum = 0.0;
        let omega = 2.0 * std::f64::consts::PI / 20.0;
        
        // Manual loop for pre-calculation efficiency?
        // Actually simple loop is vectorizable by Rust compiler.
        for i in 0..n_points {
            let t = i as f64 * dt;
            let val = normalized[i];
            
            g_sum += val * (omega * t).cos();
            s_sum += val * (omega * t).sin();
            total_sum += val;
        }
        
        let g = if total_sum > 0.0 { g_sum / total_sum } else { 0.0 };
        let s = if total_sum > 0.0 { s_sum / total_sum } else { 0.0 };
        
        // Construct Feature Vector: [Signal... , G, S]
        let mut input_vec: Vec<f32> = normalized.iter().map(|&x| x as f32).collect();
        input_vec.push(g as f32);
        input_vec.push(s as f32);
        
        // Tract expects Tensor input shape [1, 1003]
        let input: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, input_vec.len()),
            input_vec
        ).unwrap().into();

        // Run Inference
        let result = self.plan.run(tvec!(input.into())).unwrap();
        
        // Extract Output
        // Output[0] is prediction
        let output_tensor = result[0].to_array_view::<f32>().unwrap();
        // println!("DEBUG: Output Shape: {:?}", output_tensor.shape());
        
        // Handle different output shapes safely (Flatten to 1D)
        let flat_iter: Vec<f32> = output_tensor.iter().cloned().collect();
        
        if flat_iter.len() >= 2 {
             (flat_iter[0] as f64, flat_iter[1] as f64)
        } else {
             (0.0, 0.0) // Fallback
        }
    }
}
