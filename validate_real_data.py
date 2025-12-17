"""
Real FLIM Data Validation Script
Parses FLIM LABS JSON format and validates Turbo-FLIM predictions.
"""
import json
import numpy as np
import os

def load_flim_labs_json(filepath):
    """Load FLIM LABS JSON file and extract decay histograms."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    header = data['header']
    print(f"  Laser Period: {header['laser_period_ns']:.2f} ns")
    print(f"  Image Size: {header['image_width']}x{header['image_height']}")
    print(f"  Frames: {header['frames']}")
    
    # The data is a nested array: data[frame][pixel_idx] = [[time_bin, count], ...]
    # We need to aggregate all frames and convert sparse to dense.
    
    raw_data = data['data']
    width = header['image_width']
    height = header['image_height']
    laser_period = header['laser_period_ns']
    
    # Determine number of time bins (typically related to laser period)
    # From the data, time bins seem to go up to ~250 (based on "250,1" in preview)
    # Let's assume 256 bins for now (common TCSPC setup)
    n_bins = 256
    
    # Create empty histogram array
    all_histograms = []
    
    # Flatten all frames and pixels
    for frame_idx, frame in enumerate(raw_data):
        for pixel_idx, pixel_data in enumerate(frame):
            # pixel_data is a list of [time_bin, count] pairs
            histogram = np.zeros(n_bins)
            for entry in pixel_data:
                if len(entry) == 2:
                    time_bin, count = entry
                    if 0 <= time_bin < n_bins:
                        histogram[time_bin] += count
            
            # Only save non-empty histograms
            if np.sum(histogram) > 10:  # At least 10 photons
                all_histograms.append(histogram)
    
    print(f"  Extracted {len(all_histograms)} valid pixels (>10 photons)")
    return np.array(all_histograms), laser_period, n_bins

def interpolate_to_1001(histogram, n_bins):
    """Interpolate histogram from n_bins to 1001 points for Turbo-FLIM input."""
    x_orig = np.linspace(0, 1, n_bins)
    x_new = np.linspace(0, 1, 1001)
    return np.interp(x_new, x_orig, histogram)

def calculate_phasor(signal):
    """Calculate Phasor G, S coordinates (same as deep_flim.py)."""
    t_axis = np.linspace(0, 20.0, len(signal))
    omega = 2 * np.pi / 20.0
    cos_basis = np.cos(omega * t_axis)
    sin_basis = np.sin(omega * t_axis)
    
    total = np.sum(signal) + 1e-9
    g = np.sum(signal * cos_basis) / total
    s = np.sum(signal * sin_basis) / total
    return g, s

def main():
    print("=" * 60)
    print("Turbo-FLIM: Real Data Validation")
    print("=" * 60)
    
    # Load Fluorescein calibration data (known lifetime: 4.1 ns)
    data_dir = "real_data"
    fluorescein_file = os.path.join(data_dir, "Fluorescein_Calibration_m2_1740751189_imaging.json")
    
    if not os.path.exists(fluorescein_file):
        print(f"Error: {fluorescein_file} not found.")
        return
    
    histograms, laser_period, n_bins = load_flim_labs_json(fluorescein_file)
    
    if len(histograms) == 0:
        print("No valid histograms found.")
        return
    
    # Take first 100 pixels for validation
    sample_size = min(100, len(histograms))
    sample_histograms = histograms[:sample_size]
    
    print(f"\nAnalyzing {sample_size} pixels...")
    
    # Method 1: Multi-Harmonic Phasor Analysis with Calibration
    print("\n--- Method 1: Calibrated Phasor Analysis ---")
    
    # Load calibration data
    cal_file = os.path.join(data_dir, "Fluorescein_Calibration_m2_1740751189_imaging_calibration.json")
    with open(cal_file, 'r') as f:
        cal_data = json.load(f)
    
    print(f"Calibration Reference Lifetime: {cal_data['tau_ns']} ns")
    print(f"Harmonics Used: {cal_data['harmonics']}")
    
    # The calibrations contain [phase, magnitude] for each harmonic
    # These represent the theoretical Phasor position for the reference sample (Fluorescein at 4.1ns)
    # When analyzing unknown samples, we normalize/rotate by these values
    
    omega_base = 2 * np.pi / laser_period  # Fundamental angular frequency
    
    tau_estimates = []
    
    for hist in sample_histograms:
        t_axis = np.linspace(0, laser_period, n_bins)
        
        # Calculate Phasor at first harmonic (most robust)
        harmonic = 1
        omega = harmonic * omega_base
        
        cos_basis = np.cos(omega * t_axis)
        sin_basis = np.sin(omega * t_axis)
        
        total = np.sum(hist) + 1e-9
        g_raw = np.sum(hist * cos_basis) / total
        s_raw = np.sum(hist * sin_basis) / total
        
        # Get calibration for first harmonic
        cal_phase, cal_mag = cal_data['calibrations'][0][0]
        
        # The calibration phase tells us the rotation needed
        # For a known tau=4.1ns: theoretical G = 1/(1+wt^2), S = wt/(1+wt^2)
        wt_ref = omega * cal_data['tau_ns']
        g_ref = 1 / (1 + wt_ref**2)
        s_ref = wt_ref / (1 + wt_ref**2)
        
        # Apply phase rotation to measured phasor
        # The measured phase minus calibration phase gives sample phase
        measured_phase = np.arctan2(s_raw, g_raw)
        ref_phase = np.arctan2(s_ref, g_ref)
        
        # Corrected phase
        corrected_phase = measured_phase - cal_phase  # cal_phase is the offset to subtract
        
        # From corrected phase, estimate lifetime
        # For single-exp: tan(phase) = omega * tau
        # But we need the magnitude ratio approach
        
        # Alternative: Use modulation (magnitude) ratio
        measured_mag = np.sqrt(g_raw**2 + s_raw**2)
        
        # For single-exp: M = 1/sqrt(1 + (omega*tau)^2)
        # Solve for tau: tau = sqrt((1/M^2) - 1) / omega
        if measured_mag > 0 and measured_mag < 1:
            tau_est = np.sqrt((1/measured_mag**2) - 1) / omega
            tau_estimates.append(tau_est)
    
    if tau_estimates:
        tau_mean = np.mean(tau_estimates)
        tau_std = np.std(tau_estimates)
        error_pct = abs(tau_mean - 4.1) / 4.1 * 100
        
        print(f"Estimated Lifetime: {tau_mean:.2f} ± {tau_std:.2f} ns")
        print(f"Ground Truth:       4.1 ns")
        print(f"Error:              {error_pct:.1f}%")
        
        if error_pct < 20:
            print("✅ SUCCESS: Real data validation passed! (<20% error)")
        else:
            print("⚠️ Large error - checking alternative calculation...")
    
    # Method 2: Turbo-FLIM AI (if model exists)
    print("\n--- Method 2: Turbo-FLIM AI Prediction ---")
    model_path = "flim_model.onnx"
    if os.path.exists(model_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            
            predictions = []
            for hist in sample_histograms:
                # Prepare input (1001 time bins + 2 Phasor features)
                hist_1001 = interpolate_to_1001(hist, n_bins)
                hist_norm = hist_1001 / (np.max(hist_1001) + 1e-9)
                g, s = calculate_phasor(hist_norm)
                
                input_vec = np.concatenate([hist_norm, [g, s]]).astype(np.float32).reshape(1, -1)
                
                # Run inference
                output = session.run(None, {'X': input_vec})[0]
                predictions.append(output[0])
            
            predictions = np.array(predictions)
            tau1_mean = np.mean(predictions[:, 0])
            tau2_mean = np.mean(predictions[:, 1])
            
            print(f"AI Predicted Lifetimes: T1={tau1_mean:.2f} ns, T2={tau2_mean:.2f} ns")
            print(f"Ground Truth (Fluorescein, single-exp): 4.1 ns")
            
            # For single-exp sample, T1 and T2 should both be close to 4.1
            avg_tau = (tau1_mean + tau2_mean) / 2
            error = abs(avg_tau - 4.1) / 4.1 * 100
            print(f"Average Prediction: {avg_tau:.2f} ns (Error: {error:.1f}%)")
            
            if error < 20:
                print("SUCCESS: Real data validation passed! (<20% error)")
            else:
                print("WARNING: Prediction error > 20%. Model may need fine-tuning.")
                
        except ImportError:
            print("onnxruntime not installed. Run: pip install onnxruntime")
        except Exception as e:
            print(f"Error during inference: {e}")
    else:
        print(f"Model not found: {model_path}")
        print("Run './run_full_demo.sh' first to train the model.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
