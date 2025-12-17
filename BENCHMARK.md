# Benchmark Documentation

This document describes the benchmark methodology and test environment for reproducing the performance results reported in the Turbo-FLIM paper.

## Test Environment

| Component | Specification |
|-----------|---------------|
| **OS** | macOS 14.x / Linux (Ubuntu 22.04) |
| **CPU** | Apple M1/M2 or Intel/AMD x86-64 |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **Python** | 3.10+ |
| **Rust** | 1.70+ (for full pipeline) |

### Python Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
flimlib>=1.1.0
skl2onnx>=1.14.0
```

## Benchmark Scripts

### 1. `benchmark_flimlib.py` - Speed and Accuracy Comparison

Compares Turbo-FLIM against FLIMlib (FLIMfit/FLIMJ core algorithms).

**Run:**
```bash
python3 benchmark_flimlib.py
```

**Outputs:**
- Speed comparison table (μs/pixel)
- Accuracy comparison at various photon levels
- CSV file with raw results

### 2. `ablation_study.py` - Phasor Embedding Validation

Compares MLP with/without Phasor features to quantify the benefit of physics embedding.

**Run:**
```bash
python3 ablation_study.py
```

**Outputs:**
- R² comparison: Decay-only vs Phasor-Fusion
- Per-photon-level breakdown

### 3. `crlb_figure.py` - Theoretical Limit Analysis

Generates CRLB comparison figure.

**Run:**
```bash
python3 crlb_figure.py
```

**Outputs:**
- `crlb_comparison.png` - Publication-quality figure

## Reproducibility

All benchmark scripts use fixed random seeds for reproducibility:

```python
np.random.seed(42)
random_state=42
```

## Expected Results

### Speed Benchmark (typical M1 Mac)

| Method | Speed (μs/pixel) |
|--------|------------------|
| Turbo-FLIM Phasor | 15-20 |
| FLIMlib RLD | 45-55 |
| FLIMlib LMA | 90-110 |

### Accuracy Benchmark

| Photons | R² (Phasor-Fusion) |
|---------|-------------------|
| 500 | 0.90 ± 0.02 |
| 1000 | 0.94 ± 0.01 |
| 5000 | 0.96 ± 0.01 |

## Citation

Results generated using these scripts should cite:

```
Turbo-FLIM v1.0.0
DOI: [To be assigned via Zenodo]
```
