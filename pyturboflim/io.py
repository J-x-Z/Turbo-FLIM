"""
I/O module for PyTurboFLIM.

Provides utilities for loading FLIM data from various formats
and exporting analysis results.
"""

import json
import numpy as np
from pathlib import Path


def load_flim_data(filepath, format='auto'):
    """
    Load FLIM data from various file formats.
    
    Args:
        filepath: Path to the data file
        format: File format ('auto', 'json', 'sdt', 'npy', 'csv')
    
    Returns:
        dict: Dictionary containing:
            - 'data': Decay histogram data (n_pixels, n_bins) or (h, w, n_bins)
            - 'metadata': Acquisition metadata (if available)
    
    Example:
        >>> result = load_flim_data('measurement.json')
        >>> decay_data = result['data']
    """
    filepath = Path(filepath)
    
    if format == 'auto':
        suffix = filepath.suffix.lower()
        format = suffix[1:]  # Remove leading dot
    
    if format == 'json':
        return load_json(filepath)
    elif format == 'npy':
        return {'data': np.load(filepath), 'metadata': {}}
    elif format == 'csv':
        return {'data': np.loadtxt(filepath, delimiter=','), 'metadata': {}}
    elif format == 'sdt':
        return load_sdt(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_json(filepath):
    """
    Load FLIM data from JSON format (common in open-source FLIM systems).
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict: Contains 'data' array and 'metadata'
    """
    with open(filepath, 'r') as f:
        raw = json.load(f)
    
    # Handle different JSON structures
    if 'histogram' in raw:
        data = np.array(raw['histogram'])
    elif 'data' in raw:
        data = np.array(raw['data'])
    elif 'decays' in raw:
        data = np.array(raw['decays'])
    else:
        # Assume the entire file is a list of lists
        data = np.array(raw)
    
    metadata = {k: v for k, v in raw.items() if k not in ['histogram', 'data', 'decays']}
    
    return {'data': data, 'metadata': metadata}


def load_sdt(filepath):
    """
    Load Becker & Hickl .sdt files (placeholder).
    
    Note: Full SDT support requires the 'sdt-python' package.
    This is a placeholder that returns a NotImplementedError until
    the optional dependency is installed.
    
    Args:
        filepath: Path to .sdt file
    
    Returns:
        dict: Contains 'data' array and 'metadata'
    """
    try:
        import sdtfile
        sdt = sdtfile.SdtFile(str(filepath))
        data = sdt.data[0]  # First measurement block
        metadata = {
            'times': sdt.times[0] if sdt.times else None,
            'info': str(sdt.info) if hasattr(sdt, 'info') else None
        }
        return {'data': data, 'metadata': metadata}
    except ImportError:
        raise ImportError(
            "SDT file support requires 'sdtfile' package. "
            "Install with: pip install sdtfile"
        )


def export_results(lifetimes, filepath, format='csv', metadata=None):
    """
    Export lifetime analysis results to file.
    
    Args:
        lifetimes: Array of lifetime values
        filepath: Output file path
        format: Output format ('csv', 'npy', 'json')
        metadata: Optional metadata dictionary to include
    
    Example:
        >>> export_results(tau_map, 'results.csv')
    """
    filepath = Path(filepath)
    
    if format == 'csv':
        if lifetimes.ndim == 1:
            np.savetxt(filepath, lifetimes, delimiter=',', header='tau')
        elif lifetimes.ndim == 2:
            np.savetxt(filepath, lifetimes, delimiter=',', header='tau1,tau2')
        else:
            # Flatten multi-dimensional data
            flat = lifetimes.reshape(-1, lifetimes.shape[-1])
            np.savetxt(filepath, flat, delimiter=',')
    
    elif format == 'npy':
        np.save(filepath, lifetimes)
    
    elif format == 'json':
        result = {'lifetimes': lifetimes.tolist()}
        if metadata:
            result['metadata'] = metadata
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to: {filepath}")


def generate_synthetic_data(n_samples=1000, n_bins=1001, 
                           tau_range=(0.5, 5.0), photon_range=(100, 5000),
                           laser_period=12.5, random_state=None):
    """
    Generate synthetic FLIM data for testing and training.
    
    Args:
        n_samples: Number of decay curves to generate
        n_bins: Number of time bins per decay
        tau_range: Tuple of (min_tau, max_tau) in nanoseconds
        photon_range: Tuple of (min_photons, max_photons)
        laser_period: Laser repetition period in nanoseconds
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (X, y) where X is decay data and y is lifetime labels
    
    Example:
        >>> X_train, y_train = generate_synthetic_data(n_samples=10000)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = []
    y = []
    
    time = np.linspace(0, laser_period, n_bins)
    
    for _ in range(n_samples):
        # Random lifetimes
        tau1 = np.random.uniform(tau_range[0], tau_range[1] * 0.5)
        tau2 = np.random.uniform(tau_range[1] * 0.4, tau_range[1])
        if tau2 < tau1:
            tau1, tau2 = tau2, tau1
        
        # Random amplitude ratio
        alpha = np.random.uniform(0.3, 0.7)
        
        # Generate decay
        decay = alpha * np.exp(-time / tau1) + (1 - alpha) * np.exp(-time / tau2)
        
        # Add Poisson noise
        n_photons = np.random.randint(photon_range[0], photon_range[1])
        decay = decay / decay.sum() * n_photons
        decay = np.random.poisson(np.maximum(decay, 0)).astype(float)
        
        X.append(decay)
        y.append([tau1, tau2])
    
    return np.array(X), np.array(y)
