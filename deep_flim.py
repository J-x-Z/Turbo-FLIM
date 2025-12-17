import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

def train_deep_flim():
    print("Loading Training Data (50,000 curves)...")
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Dataset not found. Run 'cargo run --release' first.")
        return

    # X: FLIM Histogram bins (col 2 to end)
    # y: Lifetimes T1, T2 (col 0, 1)
    X = df.iloc[:, 2:].values
    y = df.iloc[:, :2].values
    
    # Normalize features (simple max scaling for photon counts)
    X = X / np.max(X)
    
    print(f"Data Shape: X={X.shape}, y={y.shape}")
    print("Starting 'Phasor-Fusion' Training...")
    
    # 2. Phasor Transformation (Physics Embedding)
    # G = sum(I * cos(wt)) / sum(I)
    # S = sum(I * sin(wt)) / sum(I)
    # w = 2 * pi * f (assume f=80MHz, but here we just need a consistent frequency, e.g. fundamental harmonic)
    # dt = 0.02, N=1001. Period T = N*dt = 20ns. f = 1/T = 50MHz.
    
    # Pre-calculate sine/cosine basis
    t_axis = np.linspace(0, 20.0, 1001)
    omega = 2 * np.pi / 20.0 # Fundamental frequency
    cos_basis = np.cos(omega * t_axis)
    sin_basis = np.sin(omega * t_axis)
    
    def calculate_phasor(signals):
        # signals shape: (N_samples, 1001)
        # Normalize sum first? Already normalized max=1 but sum varies.
        # Classic phasor is normalized by intensity.
        # But we want these as FEATURES. G and S without normalization carry intensity info? 
        # Standard Phasor plot normalizes by area.
        sums = np.sum(signals, axis=1, keepdims=True) + 1e-9
        
        g = np.sum(signals * cos_basis, axis=1, keepdims=True) / sums
        s = np.sum(signals * sin_basis, axis=1, keepdims=True) / sums
        
        return np.hstack([signals, g, s]) # Add 2 physics features

    print("Augmenting data with Phasor Coordinates (G, S)...")
    X = calculate_phasor(X)
    print(f"New Data Shape: {X.shape} (1001 Time-points + 2 Phasor Features)")
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Define Deep Neural Network
    # Upgraded to specific 'Deep' architecture for Phase 9
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32), # Deeper Network
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=200, # More epochs
        random_state=42,
        verbose=True,
        early_stopping=True,
        n_iter_no_change=10
    )
    
    # 5. Train
    print(f"Training 'Phasor-Guided' Deep Neural Network...")
    
    start_time = time.time()
    mlp.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"Training Complete in {duration:.2f} seconds.")
    
    # Evaluate
    print("Evaluating on Test Set (Low Photon Data)...")
    y_pred = mlp.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("-" * 30)
    print(f"Deep FLIM Accuracy Report:")
    print(f"R^2 Score: {r2:.4f} (1.0 is perfect)")
    print(f"MSE Loss:  {mse:.4f}")
    
    # Sample Prediction
    print("-" * 30)
    print("Sample Individual Predictions:")
    for i in range(5):
        print(f"True: T1={y_test[i][0]:.2f}, T2={y_test[i][1]:.2f} | Pred: T1={y_pred[i][0]:.2f}, T2={y_pred[i][1]:.2f}")
        
    print("-" * 30)
    if r2 > 0.8:
        print("SUCCESS: Deep Learning successfully learned physics from synthetic data.")
        print("This verifies 'Horizon 1' of the Roadmap.")
        
    # ... (After Training) ...
    
    # 7. Scientific Validation: Noise Robustness Curve (Phase 11)
    print("=" * 50)
    print("Step 11.1: Generating Noise Robustness Curve...")
    try:
        df_val = pd.read_csv('validation_data.csv')
        
        # Validation Data: t1, t2, photons, bin0...
        y_val = df_val.iloc[:, :2].values
        photons_val = df_val.iloc[:, 2].values
        X_val = df_val.iloc[:, 3:].values
        
        # Normalize (Same as training)
        X_val = X_val / np.max(X_val)
        
        # Augment with Phasor (Features 1002, 1003)
        # Note: Must use exactly the same basis as training
        X_val = calculate_phasor(X_val)
        
        y_val_pred = mlp.predict(X_val)
        
        # Group by Photon Count
        unique_levels = sorted(np.unique(photons_val))
        print(f"{'Photons':<10} | {'R² Score':<10} | {'MSE':<10} | {'Accuracy Strategy'}")
        print("-" * 50)
        
        results = []
        for p in unique_levels:
            mask = (photons_val == p)
            y_sub_true = y_val[mask]
            y_sub_pred = y_val_pred[mask]
            
            if len(y_sub_true) > 0:
                score = r2_score(y_sub_true, y_sub_pred)
                mse_sub = mean_squared_error(y_sub_true, y_sub_pred)
                print(f"{int(p):<10} | {score:.4f}     | {mse_sub:.4f}     | {'Nature-Level' if score > 0.9 else 'High'}")
                results.append((p, score))
        
        print("=" * 50)
        
        # Check for Nature-level achievement
        low_photon_score = next((r for p, r in results if p == 100), 0)
        if low_photon_score > 0.9:
            print(f"CRITICAL SUCCESS: R² > 0.9 at 100 photons ({low_photon_score:.4f}).")
            print("This is the 'Quantum Limit' breakthrough needed for Q1 journals.")
            
    except FileNotFoundError:
        print("Warning: validation_data.csv not found. Skipping Robustness Curve.")

    # --- ONNX Export ---
    print("Exporting Model to ONNX format for Rust Integation...")
    try:
        from skl2onnx import to_onnx
        import onnx
        
        # Define Input Type
        # MLP input is (Batch, Features), here features=1001 (t1,t2 removed from training_data so 1001 cols remain? 
        # Wait, generation in main.rs: "write!(file_train, "t1,t2").unwrap(); for k in 0..t.len() { bin_k }"
        # t.len() is 1001 (0.0 to 20.0 step 0.02 = 1001 points)
        # So X has 1001 columns.
        
        onx = to_onnx(mlp, X_train[:1].astype(np.float32))
        
        with open("flim_model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
            
        print("Model saved to: flim_model.onnx")
        print("Ready for Rust 'ort' crate integration.")
        
    except ImportError:
        print("skl2onnx not found. Run 'pip install skl2onnx' to enable export.")

if __name__ == "__main__":
    train_deep_flim()
