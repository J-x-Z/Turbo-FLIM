#!/usr/bin/env python3
"""
PyTurboFLIM Basic Usage Example

This script demonstrates the core functionality of PyTurboFLIM:
1. Generate synthetic FLIM data
2. Train a TurboFLIM model
3. Analyze decay curves
4. Visualize results
"""

import numpy as np
from pyturboflim import TurboFLIM, analyze_batch
from pyturboflim.io import generate_synthetic_data
from pyturboflim.phasor import compute_phasor_batch
from pyturboflim.visualization import plot_phasor, plot_lifetime_map

print("=" * 60)
print(" PyTurboFLIM - Basic Usage Example")
print("=" * 60)

# Step 1: Generate synthetic training data
print("\n[1/4] Generating synthetic FLIM data...")
X_train, y_train = generate_synthetic_data(
    n_samples=5000, 
    n_bins=256,
    tau_range=(0.5, 5.0),
    photon_range=(100, 2000),
    random_state=42
)
print(f"  Training data: {X_train.shape[0]} curves, {X_train.shape[1]} time bins")

# Generate test data
X_test, y_test = generate_synthetic_data(
    n_samples=1000,
    n_bins=256,
    tau_range=(0.5, 5.0),
    photon_range=(100, 2000),
    random_state=123
)
print(f"  Test data: {X_test.shape[0]} curves")

# Step 2: Train TurboFLIM model
print("\n[2/4] Training TurboFLIM model...")
model = TurboFLIM(
    laser_period=12.5,
    hidden_layers=(128, 64, 32),  # Smaller for demo
    max_iter=200
)
model.fit(X_train, y_train, verbose=True)

# Step 3: Evaluate on test data
print("\n[3/4] Evaluating model...")
r2_score = model.score(X_test, y_test)
print(f"  R² Score: {r2_score:.4f}")

predictions = model.predict(X_test[:5])
print("\n  Sample predictions:")
for i in range(5):
    true_tau1, true_tau2 = y_test[i]
    pred_tau1, pred_tau2 = predictions[i]
    print(f"    True: τ₁={true_tau1:.2f}ns, τ₂={true_tau2:.2f}ns | "
          f"Pred: τ₁={pred_tau1:.2f}ns, τ₂={pred_tau2:.2f}ns")

# Step 4: Phasor analysis visualization
print("\n[4/4] Creating phasor plot...")
G, S = compute_phasor_batch(X_test[:500], laser_period=12.5)
fig, ax = plot_phasor(G, S, alpha=0.3, save_path='phasor_example.png')
print("  Saved: phasor_example.png")

print("\n" + "=" * 60)
print(" Example complete!")
print("=" * 60)
print("\nFor more information, see: https://github.com/J-x-Z/Turbo-FLIM")
