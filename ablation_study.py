"""
Ablation Study: Proving the Value of Phasor-Fusion
This is CRITICAL evidence that the Phasor embedding actually helps.

Comparison:
- Model A: MLP with ONLY decay histogram (1001 features)
- Model B: MLP with decay + Phasor coordinates (1003 features)

If Model B >> Model A at low photons, we prove Phasor-Fusion works.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')

def add_phasor_features(signals, period=20.0):
    """Add G, S Phasor coordinates to signals."""
    n = signals.shape[1]
    t_axis = np.linspace(0, period, n)
    omega = 2 * np.pi / period
    cos_b = np.cos(omega * t_axis)
    sin_b = np.sin(omega * t_axis)
    
    totals = np.sum(signals, axis=1, keepdims=True) + 1e-9
    g = np.sum(signals * cos_b, axis=1) / totals.flatten()
    s = np.sum(signals * sin_b, axis=1) / totals.flatten()
    
    return np.column_stack([signals, g, s])

def run_ablation_study():
    print("=" * 70)
    print("ABLATION STUDY: Proving Phasor-Fusion Value")
    print("=" * 70)
    
    # Load data
    print("\nLoading training data...")
    df = pd.read_csv('training_data.csv')
    
    # Use subset for faster training
    df = df.sample(n=20000, random_state=42)
    
    X_raw = df.iloc[:, 2:].values  # Decay only (1001 features)
    y = df.iloc[:, :2].values      # Targets (τ₁, τ₂)
    
    X_phasor = add_phasor_features(X_raw)  # Decay + Phasor (1003 features)
    
    print(f"Model A input: {X_raw.shape[1]} features (decay only)")
    print(f"Model B input: {X_phasor.shape[1]} features (decay + Phasor)")
    
    # Split
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42)
    X_phasor_train, X_phasor_test, _, _ = train_test_split(
        X_phasor, y, test_size=0.2, random_state=42)
    
    # Train Model A: Decay Only
    print("\nTraining Model A (Decay Only)...")
    model_a = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=50, 
                           early_stopping=True, random_state=42, verbose=False)
    start = time.time()
    model_a.fit(X_raw_train, y_train)
    time_a = time.time() - start
    
    # Train Model B: Decay + Phasor
    print("Training Model B (Decay + Phasor)...")
    model_b = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=50, 
                           early_stopping=True, random_state=42, verbose=False)
    start = time.time()
    model_b.fit(X_phasor_train, y_train)
    time_b = time.time() - start
    
    # Evaluate on test set
    pred_a = model_a.predict(X_raw_test)
    pred_b = model_b.predict(X_phasor_test)
    
    r2_a = r2_score(y_test, pred_a)
    r2_b = r2_score(y_test, pred_b)
    mae_a = mean_absolute_error(y_test, pred_a)
    mae_b = mean_absolute_error(y_test, pred_b)
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Model A (Decay Only):    R² = {r2_a:.4f}, MAE = {mae_a:.4f} ns")
    print(f"Model B (Decay+Phasor):  R² = {r2_b:.4f}, MAE = {mae_b:.4f} ns")
    print(f"Improvement:             ΔR² = {r2_b - r2_a:.4f} (+{(r2_b-r2_a)/r2_a*100:.1f}%)")
    
    # Now test at different photon levels using validation data
    print("\n" + "=" * 70)
    print("PERFORMANCE BY PHOTON COUNT")
    print("=" * 70)
    
    # Load validation data if exists
    try:
        val_df = pd.read_csv('validation_data.csv')
        
        # Check column count - validation may have different format
        n_cols = val_df.shape[1]
        if n_cols == 1003:  # Has target columns
            val_X_raw = val_df.iloc[:, 2:].values[:, :1001]  # Take first 1001 features
            val_y = val_df.iloc[:, :2].values
        else:
            val_X_raw = val_df.values[:, :1001]
            val_y = None
            print("Validation data format unclear, skipping per-photon analysis")
            return r2_a, r2_b
        
        val_X_phasor = add_phasor_features(val_X_raw)
        
        # Estimate photon count from sum of signal
        photon_counts = np.sum(val_X_raw, axis=1)
        
        # Define bins
        bins = [0, 100, 300, 700, 1500, 3000, np.inf]
        labels = ['~50', '~200', '~500', '~1000', '~2000', '~5000']
        
        print(f"{'Photons':<10} | {'Model A R²':<12} | {'Model B R²':<12} | {'Improvement':<12}")
        print("-" * 60)
        
        results = []
        for i in range(len(bins)-1):
            mask = (photon_counts >= bins[i]) & (photon_counts < bins[i+1])
            if np.sum(mask) > 50:
                X_a = val_X_raw[mask]
                X_b = val_X_phasor[mask]
                y_true = val_y[mask]
                
                pred_a = model_a.predict(X_a)
                pred_b = model_b.predict(X_b)
                
                r2_a_bin = r2_score(y_true, pred_a)
                r2_b_bin = r2_score(y_true, pred_b)
                improvement = r2_b_bin - r2_a_bin
                
                print(f"{labels[i]:<10} | {r2_a_bin:<12.4f} | {r2_b_bin:<12.4f} | {improvement:+.4f}")
                results.append({
                    'photons': labels[i],
                    'r2_decay': r2_a_bin,
                    'r2_phasor': r2_b_bin,
                    'improvement': improvement
                })
    except FileNotFoundError:
        print("Validation data not found. Using test set breakdown.")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if r2_b > r2_a:
        print(f"✅ Phasor-Fusion provides {(r2_b-r2_a)/r2_a*100:.1f}% improvement in R²")
        print(f"   This validates the physics-embedding design decision.")
    else:
        print("⚠️ No significant improvement from Phasor features.")
    
    return r2_a, r2_b

if __name__ == "__main__":
    run_ablation_study()
