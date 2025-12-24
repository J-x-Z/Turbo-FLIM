"""
Visualization module for PyTurboFLIM.

Provides plotting functions for lifetime maps and phasor analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# FLIM-standard colormap (similar to FLIMfit)
FLIM_CMAP = LinearSegmentedColormap.from_list(
    'flim', ['blue', 'cyan', 'green', 'yellow', 'red']
)


def plot_lifetime_map(lifetimes, tau_range=(0, 5), cmap='viridis',
                      title='Fluorescence Lifetime Map', colorbar_label='τ (ns)',
                      figsize=(8, 6), save_path=None):
    """
    Plot a 2D lifetime map from FLIM analysis.
    
    Args:
        lifetimes: 2D array of lifetime values (height, width) or
                   3D array (height, width, 2) for bi-exponential
        tau_range: Tuple of (min, max) for colorbar range
        cmap: Colormap name or colormap object
        title: Plot title
        colorbar_label: Label for colorbar
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        Figure and Axes objects
    
    Example:
        >>> fig, ax = plot_lifetime_map(tau_map, tau_range=(1, 4))
        >>> plt.show()
    """
    if lifetimes.ndim == 3:
        # For bi-exponential, plot mean lifetime
        lifetimes = np.mean(lifetimes, axis=2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(lifetimes, cmap=cmap, vmin=tau_range[0], vmax=tau_range[1])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_phasor(G, S, color='blue', alpha=0.5, s=1, 
                show_semicircle=True, title='Phasor Plot',
                figsize=(8, 8), save_path=None):
    """
    Create a phasor plot showing the distribution of decay signatures.
    
    The phasor plot is a powerful visualization where single-exponential
    decays lie on a universal semicircle, and multi-exponential decays
    fall inside the semicircle.
    
    Args:
        G: Array of G (cosine) phasor coordinates
        S: Array of S (sine) phasor coordinates
        color: Point color
        alpha: Point transparency
        s: Point size
        show_semicircle: Whether to draw the universal semicircle
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        Figure and Axes objects
    
    Example:
        >>> G, S = compute_phasor_batch(all_decays)
        >>> fig, ax = plot_phasor(G, S)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data points
    ax.scatter(G, S, c=color, alpha=alpha, s=s, edgecolors='none')
    
    # Draw universal semicircle
    if show_semicircle:
        theta = np.linspace(0, np.pi, 100)
        G_circle = 0.5 * (1 + np.cos(theta))
        S_circle = 0.5 * np.sin(theta)
        ax.plot(G_circle, S_circle, 'k--', linewidth=2, label='Universal Semicircle')
    
    # Formatting
    ax.set_xlabel('G (cosine)', fontsize=12)
    ax.set_ylabel('S (sine)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if show_semicircle:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_decay_curve(decay, time=None, laser_period=12.5, 
                     log_scale=True, title='Decay Curve',
                     figsize=(10, 6), save_path=None):
    """
    Plot a fluorescence decay curve.
    
    Args:
        decay: 1D array of photon counts
        time: Optional time axis (if None, generated from laser_period)
        laser_period: Laser repetition period in ns
        log_scale: Whether to use logarithmic y-axis
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        Figure and Axes objects
    """
    if time is None:
        time = np.linspace(0, laser_period, len(decay))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time, decay, 'b-', linewidth=1.5)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Photon Counts', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_comparison(turboflim_result, reference_result, 
                    tau_range=(0, 5), figsize=(12, 5), save_path=None):
    """
    Create side-by-side comparison of TurboFLIM vs reference method.
    
    Args:
        turboflim_result: Lifetime map from TurboFLIM
        reference_result: Lifetime map from reference method (e.g., FLIMlib)
        tau_range: Colorbar range
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # TurboFLIM
    im1 = axes[0].imshow(turboflim_result, cmap='viridis', 
                         vmin=tau_range[0], vmax=tau_range[1])
    axes[0].set_title('TurboFLIM', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Reference
    im2 = axes[1].imshow(reference_result, cmap='viridis',
                         vmin=tau_range[0], vmax=tau_range[1])
    axes[1].set_title('Reference (LMA)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Difference
    diff = turboflim_result - reference_result
    max_diff = np.max(np.abs(diff))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title('Difference', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Colorbars
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='τ (ns)')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='τ (ns)')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Δτ (ns)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig
