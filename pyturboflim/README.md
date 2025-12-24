# PyTurboFLIM

[![CI](https://github.com/J-x-Z/Turbo-FLIM/actions/workflows/ci.yml/badge.svg)](https://github.com/J-x-Z/Turbo-FLIM/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![PyPI](https://img.shields.io/badge/PyPI-pyturboflim-orange.svg)](https://pypi.org/project/pyturboflim/)

**Fast physics-guided fluorescence lifetime analysis using Phasor-Fusion neural networks.**

PyTurboFLIM combines the speed of phasor analysis with the accuracy of deep learning, achieving **5.5× faster** analysis than traditional LMA fitting while maintaining **R² > 0.9** accuracy even in low-photon regimes.

## Installation

```bash
pip install pyturboflim
```

Or install from source:
```bash
git clone https://github.com/J-x-Z/Turbo-FLIM.git
cd Turbo-FLIM
pip install -e .
```

## Quick Start

```python
from pyturboflim import TurboFLIM
from pyturboflim.io import generate_synthetic_data

# Generate training data
X_train, y_train = generate_synthetic_data(n_samples=10000)

# Train model
model = TurboFLIM()
model.fit(X_train, y_train)

# Predict lifetimes
lifetimes = model.predict(my_decay_curves)
print(f"τ₁ = {lifetimes[0, 0]:.2f} ns, τ₂ = {lifetimes[0, 1]:.2f} ns")
```

## Features

- **Phasor-Fusion Architecture**: Combines time-domain decay histograms with frequency-domain phasor coordinates
- **Low-Photon Robustness**: Maintains R² > 0.9 at 500 photons where traditional methods fail
- **Fast Inference**: 18 μs/pixel on standard CPU (5.5× faster than LMA)
- **Easy Integration**: Simple Python API compatible with NumPy arrays
- **Visualization Tools**: Built-in lifetime map and phasor plot generation

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `TurboFLIM()` | Main model class with `fit()` and `predict()` |
| `analyze(decay)` | Quick single-curve analysis |
| `analyze_batch(image)` | Batch processing for full images |

### Phasor Functions

| Function | Description |
|----------|-------------|
| `compute_phasor(decay)` | Compute (G, S) coordinates |
| `compute_phasor_batch(decays)` | Vectorized phasor computation |
| `phasor_to_lifetime(G, S)` | Convert phasor to lifetime |

### I/O Functions

| Function | Description |
|----------|-------------|
| `load_flim_data(path)` | Load data from JSON/NPY/CSV |
| `export_results(data, path)` | Export analysis results |
| `generate_synthetic_data()` | Create training data |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_lifetime_map(tau)` | 2D lifetime heatmap |
| `plot_phasor(G, S)` | Phasor scatter plot |

## Performance

| Metric | Value |
|--------|-------|
| Speed | 18 μs/pixel (CPU) |
| Accuracy @ 500 photons | R² = 0.907 |
| Accuracy @ 1000 photons | R² = 0.943 |
| Model Size | < 1 MB |

## Citation

If you use PyTurboFLIM in your research, please cite:

```bibtex
@software{turboflim2024,
  title={PyTurboFLIM: Fast Physics-Guided Fluorescence Lifetime Analysis},
  author={Zhang, Jiaxi},
  year={2024},
  url={https://github.com/J-x-Z/Turbo-FLIM}
}
```

## License

Apache License 2.0
