#!/bin/bash
set -e

echo "=== Turbo-FLIM: Full Pipeline Demo ==="

# 1. Run Rust Engine (Generates Data)
echo "[1/3] Running High-Performance Rust Engine..."
cargo run --release

# 2. Setup Python Environment (Always check dependencies)
echo "[Check] Verifying Python Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
./venv/bin/pip install --upgrade pip
./venv/bin/pip install pandas scikit-learn matplotlib

# 3. Traing Deep Learning Model
echo "[2/3] Training Deep FLIM Neural Network..."
./venv/bin/python3 deep_flim.py

# 4. Generate Figures
echo "[3/4] Generating Scientific Figures..."
./venv/bin/python3 paper_figure.py

# 5. Run Pure Rust ONNX Inference
echo "[4/4] Activating Pure Rust AI Engine (ONNX)..."
if [ -f "flim_model.onnx" ]; then
    echo "Model found. Running Rust binary in Inference Mode..."
    cargo run --release
else
    echo "Error: flim_model.onnx missing. Training failed."
    exit 1
fi

echo "=== Demo Complete! ==="
echo "Artifacts generated:"
echo " - flim_model.onnx (Exported AI Model)"
echo " - results.csv (Analysis Results)"
echo " - training_data.csv (Synthetic Data)"
echo " - figure1_nature_style.png (Publication Figure)"
