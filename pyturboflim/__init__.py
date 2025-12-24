"""
PyTurboFLIM: Fast Physics-Guided Fluorescence Lifetime Analysis

A Python package for rapid, accurate fluorescence lifetime imaging analysis
using physics-embedded neural networks with Phasor-Fusion architecture.

Usage:
    from pyturboflim import TurboFLIM, analyze
    
    # Single decay curve
    lifetime = analyze(decay_curve)
    
    # Batch processing
    model = TurboFLIM()
    model.fit(training_data, labels)
    lifetimes = model.predict(image_stack)
"""

__version__ = "1.0.0"
__author__ = "Jiaxi Zhang"
__email__ = "z1529105815@outlook.com"

from .core import TurboFLIM, analyze, analyze_batch
from .phasor import compute_phasor, compute_phasor_batch
from .io import load_flim_data, export_results, load_sdt, load_json
from .visualization import plot_lifetime_map, plot_phasor

__all__ = [
    "TurboFLIM",
    "analyze",
    "analyze_batch",
    "compute_phasor",
    "compute_phasor_batch",
    "load_flim_data",
    "export_results",
    "load_sdt",
    "load_json",
    "plot_lifetime_map",
    "plot_phasor",
]
