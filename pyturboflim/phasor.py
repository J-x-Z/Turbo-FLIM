"""
Phasor transform module for PyTurboFLIM.

Implements the Fourier-domain phasor analysis that is the core physics
embedding of the Turbo-FLIM architecture.
"""

import numpy as np


def compute_phasor(decay_curve, laser_period=12.5, harmonic=1):
    """
    Compute phasor coordinates (G, S) for a single decay curve.
    
    The phasor transform projects the time-domain fluorescence decay onto
    the Fourier domain, providing noise-robust features for lifetime estimation.
    
    Args:
        decay_curve: 1D array of photon counts per time bin
        laser_period: Laser repetition period in nanoseconds (default: 12.5)
        harmonic: Harmonic number for phasor calculation (default: 1)
    
    Returns:
        tuple: (G, S) phasor coordinates
    
    Example:
        >>> G, S = compute_phasor(my_decay)
        >>> print(f"Phasor: G={G:.3f}, S={S:.3f}")
    """
    decay = np.asarray(decay_curve, dtype=np.float64)
    n_bins = len(decay)
    
    # Time axis normalized to [0, 2π]
    t = np.linspace(0, 2 * np.pi * harmonic, n_bins, endpoint=False)
    
    # Compute phasor coordinates
    total = np.sum(decay)
    if total == 0:
        return 0.0, 0.0
    
    G = np.sum(decay * np.cos(t)) / total
    S = np.sum(decay * np.sin(t)) / total
    
    return float(G), float(S)


def compute_phasor_batch(decay_curves, laser_period=12.5, harmonic=1):
    """
    Compute phasor coordinates for multiple decay curves (vectorized).
    
    Args:
        decay_curves: 2D array of shape (n_samples, n_bins)
        laser_period: Laser repetition period in nanoseconds (default: 12.5)
        harmonic: Harmonic number for phasor calculation (default: 1)
    
    Returns:
        tuple: (G_array, S_array) arrays of phasor coordinates
    
    Example:
        >>> G, S = compute_phasor_batch(all_decays)
        >>> phasor_plot(G, S)
    """
    decay_curves = np.asarray(decay_curves, dtype=np.float64)
    n_samples, n_bins = decay_curves.shape
    
    # Time axis normalized to [0, 2π]
    t = np.linspace(0, 2 * np.pi * harmonic, n_bins, endpoint=False)
    
    # Compute totals (avoid division by zero)
    totals = np.sum(decay_curves, axis=1, keepdims=True)
    totals = np.maximum(totals, 1e-10)  # Prevent division by zero
    
    # Vectorized phasor computation
    G = np.sum(decay_curves * np.cos(t), axis=1) / totals.ravel()
    S = np.sum(decay_curves * np.sin(t), axis=1) / totals.ravel()
    
    return G, S


def phasor_to_lifetime(G, S, laser_period=12.5):
    """
    Convert phasor coordinates to an estimated single-exponential lifetime.
    
    This is a geometric inversion assuming single-exponential decay.
    For multi-exponential decays, the result represents an "apparent" lifetime.
    
    Args:
        G: Phasor G coordinate (cosine component)
        S: Phasor S coordinate (sine component)
        laser_period: Laser repetition period in nanoseconds (default: 12.5)
    
    Returns:
        Estimated lifetime in nanoseconds
    
    Note:
        For bi-exponential analysis, use the TurboFLIM neural network model
        instead of this geometric inversion.
    """
    omega = 2 * np.pi / laser_period
    
    # Geometric inversion: tau = S / (omega * G)
    if abs(G) < 1e-10:
        return 0.0
    
    tau = S / (omega * G)
    return max(0.0, float(tau))


def phasor_distance(G1, S1, G2, S2):
    """
    Compute Euclidean distance between two phasor points.
    
    Useful for clustering or similarity analysis in phasor space.
    
    Args:
        G1, S1: First phasor coordinates
        G2, S2: Second phasor coordinates
    
    Returns:
        Euclidean distance in phasor space
    """
    return np.sqrt((G1 - G2)**2 + (S1 - S2)**2)


def universal_semicircle(n_points=100):
    """
    Generate the universal semicircle for phasor plots.
    
    Single-exponential decays lie exactly on this semicircle in phasor space.
    Multi-exponential decays lie inside the semicircle.
    
    Args:
        n_points: Number of points to generate (default: 100)
    
    Returns:
        tuple: (G_circle, S_circle) coordinates of the semicircle
    """
    theta = np.linspace(0, np.pi, n_points)
    G = 0.5 * (1 + np.cos(theta))
    S = 0.5 * np.sin(theta)
    return G, S
