mod simulation;
mod solver;
mod utils;
mod gpu_solver;
mod cpu_parallel_solver;
mod clustering;
pub mod inference;

use ndarray::Array1;
use rand::Rng;
use simulation::{generate_time_axis, generate_irf, generate_decay, add_poisson_noise};
use cpu_parallel_solver::run_cpu_parallel;
use clustering::run_clustering;
use rayon::prelude::*;

fn main() {
    println!("Sci-Rust FLIM: Intelligent AI Segmentation Demo");
    
    // Setup
    let dt = 0.02;
    let t = generate_time_axis(dt, 20.0);
    // Note: generate_irf and t cloning might be costly to do inside parallel loop if not careful, 
    // but here we pass reference.
    let irf = generate_irf(&t, 0.25, 0.6);
    
    // Simulating a "Tissue Boundary"
    println!("Simulating biological tissue with 2 distinct populations...");
    let num_pixels_a = 5000;
    let num_pixels_b = 5000;
    
    // Use Rayon for parallel simulation
    let signals: Vec<Array1<f64>> = (0..(num_pixels_a + num_pixels_b))
        .into_par_iter()
        .map(|i| {
            // Re-create local clojures or logic
            let taus = if i < num_pixels_a { vec![2.0, 3.0] } else { vec![0.8, 4.0] };
            let d = generate_decay(&t, &taus, &vec![0.5, 0.5]);
            let c = utils::convolution(&d, &irf, dt);
            add_poisson_noise(&c, 5000.0)
        })
        .collect();

    // 1. High-Speed Analysis
    println!("Running Phase Plane Analysis (Parallel CPU)...");
    let start_calc = std::time::Instant::now();
    let results = run_cpu_parallel(&t, &irf, &signals, dt);
    let dur_calc = start_calc.elapsed();
    println!("Analysis Complete: 10,000 pixels analyzed in {:.0} ms", dur_calc.as_millis());
    
    // 2. Unsupervised AI Clustering (GMM - Probabilistic)
    println!("Running Probabilistic AI Clustering (Gaussian Mixture Model)...");
    let start_ai = std::time::Instant::now();
    let (assignments, probs) = run_clustering(&results); 
    let dur_ai = start_ai.elapsed();
    
    // 3. Evaluate AI Performance
    // We expect probs > 0.5 to be Class 1, < 0.5 to be Class 0
    let mut c0_count_a = 0;
    let mut c1_count_a = 0;
    let mut certainty_a = 0.0;
    
    for i in 0..num_pixels_a {
        if assignments[i] == 0 { c0_count_a += 1; } else { c1_count_a += 1; }
        // Certainty is abs(p - 0.5) * 2.  (0.5 -> 0, 1.0 -> 1)
        certainty_a += (probs[i] - 0.5).abs() * 2.0;
    }
    certainty_a /= num_pixels_a as f64;
    
    let mut c0_count_b = 0;
    let mut c1_count_b = 0;
    let mut certainty_b = 0.0;
    
    for i in num_pixels_a..(num_pixels_a+num_pixels_b) {
         if assignments[i] == 0 { c0_count_b += 1; } else { c1_count_b += 1; }
         certainty_b += (probs[i] - 0.5).abs() * 2.0;
    }
    certainty_b /= num_pixels_b as f64;
    
    // Identify dominant clusters
    let (normal_id, tumor_id) = if c0_count_a > c1_count_a { (0, 1) } else { (1, 0) };
    let accuracy_a = if normal_id == 0 { c0_count_a } else { c1_count_a } as f64 / num_pixels_a as f64 * 100.0;
    let accuracy_b = if tumor_id == 0 { c0_count_b } else { c1_count_b } as f64 / num_pixels_b as f64 * 100.0;
    
    println!("AI Segmentation Results (Bayesian GMM):");
    println!("  Clustering Time: {:.0} ms", dur_ai.as_millis());
    println!("  'Normal' Tissue Accuracy: {:.2}% (Confidence: {:.2}%)", accuracy_a, certainty_a * 100.0);
    println!("  'Tumor' Tissue Accuracy: {:.2}% (Confidence: {:.2}%)", accuracy_b, certainty_b * 100.0);
    
    // 4. Export Data for Visualization
    println!("Exporting data for scientific visualization...");
    use std::fs::File;
    use std::io::Write;
    
    // Clustering Data (T1, T2, ClusterID)
    let mut file = File::create("results.csv").expect("Unable to create file");
    writeln!(file, "t1,t2,cluster_id").expect("Unable to write header");
    for i in 0..(num_pixels_a + num_pixels_b) {
        let (t1, t2) = results[i];
        let c = assignments[i];
        if t1 > 0.0 && t2 > 0.0 && t1 < 10.0 && t2 < 10.0 {
            writeln!(file, "{:.6},{:.6},{}", t1, t2, c).expect("Unable to write line");
        }
    }
    
    // Signal Data (First pixel of each cluster)
    let mut file_sig = File::create("signal_examples.csv").expect("Unable to create signal file");
    writeln!(file_sig, "t,sig_normal,sig_tumor").expect("Unable to write header");
    for k in 0..t.len() {
        writeln!(file_sig, "{:.6},{:.6},{:.6}", t[k], signals[0][k], signals[num_pixels_a][k]).expect("Unable to write line");
    }
    
    // --- Phase 6: Deep Learning Data Factory ---
    println!("Step 6: Generating 50,000 synthetic curves (Universal Physics Mode)...");
    
    // --- Phase 8: Universal Data Factory ---
    // Generate data covering a wide range of instrument conditions.
    // This is the "Moonshot" dataset.
    
    let dataset: Vec<String> = (0..50000)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            
            // 1. Randomized Instrument Physics (The "Universal" aspect)
            // - IRF Width: 0.1ns to 0.6ns (simulating different lasers)
            // - Time Shift: -0.5ns to 0.5ns (simulating trigger Jitter)
            let sigma = rng.gen_range(0.1..0.6);
            let center = 0.25 + rng.gen_range(-0.1..0.1); // Jitter around 0.25
            
            // 2. Randomized Biological Parameters
            let t1 = rng.gen_range(0.4..2.5);
            let t2 = rng.gen_range(2.0..5.0);
            let a1 = rng.gen_range(0.1..0.9);
            let a2 = 1.0 - a1;
            
            // 3. Simulation
            let d = generate_decay(&t, &vec![t1, t2], &vec![a1, a2]);
            
            // Generate dynamic IRF for *this specific pixel*
            let dynamic_irf = generate_irf(&t, sigma, center);
            let c = utils::convolution(&d, &dynamic_irf, dt);
            
            // 4. Randomized Noise Environment
            // - Photon Count: 100 to 5000 (Very Low to High)
            let photons = rng.gen_range(100.0..5000.0);
            let noisy = add_poisson_noise(&c, photons);
            
            // Normalize for AI input (Scale invalidates photon count absolute value, but shape remains)
            let max_val = noisy.fold(0.0f64, |a, &b| a.max(b));
            let final_signal = if max_val > 0.0 { &noisy / max_val } else { noisy.clone() };
            
            // Format: t1, t2, bin0, bin1, ...
            let mut line = format!("{:.4},{:.4}", t1, t2);
            for val in final_signal.iter() {
                line.push_str(&format!(",{:.4}", val));
            }
            line
        })
        .collect();
        
    // Write training data
    println!("Writing Universal Training Data to disk...");
    let mut file_train = File::create("training_data.csv").expect("Unable to create file");
    write!(file_train, "t1,t2").unwrap();
    for k in 0..t.len() {
        write!(file_train, ",bin{}", k).unwrap();
    }
    writeln!(file_train).unwrap();
    
    for line in dataset {
        writeln!(file_train, "{}", line).unwrap();
    }
    
    // --- Step 11.1: Validation Dataset (Stratified) ---
    println!("Step 11.1: Generating Validation Dataset for Robustness Curve...");
    let strat_photons = vec![50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0];
    let samples_per_level = 2000;
    
    // We want to write: t1, t2, photon_count, bin0, ... binN.
    // Photon count column is new.
    let t_ref = &t; // Shared reference for closure
    
    let mut val_csv: Vec<String> = strat_photons.into_iter().flat_map(|p_count| {
        (0..samples_per_level).into_iter().map(move |_| {
            let mut rng = rand::thread_rng();
            // Physics
            let t1 = rng.gen_range(0.4..2.5);
            let t2 = rng.gen_range(2.0..5.0);
            let a1 = rng.gen_range(0.1..0.9);
            let a2 = 1.0 - a1;
            
            // Fixed IRF for validation (fair comparison)
            // Actually let's use the local dynamic one but keep it reasonable?
            // No, for the curve, we want to isolate NOISE effect, so keep IRF constant or random?
            // Let's keep it random to prove robustness, or constant to prove noise limit?
            // "Universal Physics" means we should test on random IRF too.
            let sigma = rng.gen_range(0.1..0.6);
            let center = 0.25 + rng.gen_range(-0.1..0.1);
            let dyn_irf = generate_irf(t_ref, sigma, center);
            
            let d = generate_decay(t_ref, &vec![t1, t2], &vec![a1, a2]);
            let c = utils::convolution(&d, &dyn_irf, dt);
            let noisy = add_poisson_noise(&c, p_count);
            
            // Normalize
            let max_val = noisy.fold(0.0f64, |a, &b| a.max(b));
            let final_signal = if max_val > 0.0 { &noisy / max_val } else { noisy.clone() };
            
            let mut line = format!("{:.4},{:.4},{:.0}", t1, t2, p_count);
            for val in final_signal.iter() {
                line.push_str(&format!(",{:.4}", val));
            }
            line
        }).collect::<Vec<String>>()
    }).collect();
    
    // Write validation file
    let mut file_val = File::create("validation_data.csv").expect("Unable to create validation file");
    write!(file_val, "t1,t2,photons").unwrap();
    for k in 0..t.len() {
        write!(file_val, ",bin{}", k).unwrap();
    }
    writeln!(file_val).unwrap();
    
    for line in val_csv {
        writeln!(file_val, "{}", line).unwrap();
    }
    println!("Validation Data (Stratified) Generated: validation_data.csv");
    
    println!("Deep Learning Training Set Generated: training_data.csv");
    
    // --- Phase 7: Pure Rust AI Integration ---
    println!("Step 7: Proving 'Pure Rust' AI Capabilities...");
    
    if std::path::Path::new("flim_model.onnx").exists() {
        println!("Found 'flim_model.onnx'. Initializing Rust Inference Engine...");
        use inference::DeepFlimEngine;
        
        let engine = DeepFlimEngine::new("flim_model.onnx");
        
        println!("Running Native Rust Inference on 5 random low-photon samples...");
        let mut rng = rand::thread_rng();
        
        for i in 0..5 {
            // Generate a test sample (Low Photon)
            let t1 = rng.gen_range(0.5..2.5);
            let t2 = rng.gen_range(2.5..5.0);
            let a1 = rng.gen_range(0.3..0.7);
            let a2 = 1.0 - a1;
            let photons = rng.gen_range(50.0..150.0); // Extremely low!
            
            let decay = generate_decay(&t, &vec![t1, t2], &vec![a1, a2]);
            let conv = utils::convolution(&decay, &irf, dt);
            let noisy = add_poisson_noise(&conv, photons);
            
            // Rust Predicts!
            let (p1, p2) = engine.predict(&noisy);
            
            println!("  [Sample {}] True: ({:.2}, {:.2}) | AI Predicted: ({:.2}, {:.2}) | Photon Count: {:.0}", 
                     i+1, t1, t2, p1, p2, photons);
        }
        println!("SUCCESS: Neural Network is running entirely inside Rust binary. No Python required at runtime.");
    } else {
        println!("Warning: 'flim_model.onnx' not found. Run 'run_full_demo.sh' first to train and export the model.");
    }
}


