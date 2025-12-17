"""
Generate Publication-Quality Composite Figure for Convallaria Tissue
This creates a high-quality figure suitable for Bioinformatics submission.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

def load_convallaria_data():
    """Load Convallaria phasor data."""
    filepath = "real_data/Convallaria_m2_1740751781_phasor_ch1.json"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    header = data['header']
    phasors = data['phasors_data']
    
    h1 = phasors[0]
    g_image = np.array(h1['g_data'])
    s_image = np.array(h1['s_data'])
    
    return g_image, s_image, header

def calculate_lifetime(g, s, laser_period=25.0):
    """Calculate lifetime from Phasor coordinates."""
    omega = 2 * np.pi / laser_period
    m = np.sqrt(g**2 + s**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.sqrt(1.0/m**2 - 1.0) / omega
        tau = np.where((m > 0.1) & (m < 0.99) & np.isfinite(tau), tau, np.nan)
    
    return tau

def create_publication_figure():
    """Create high-quality composite figure."""
    print("Loading Convallaria data...")
    g_image, s_image, header = load_convallaria_data()
    laser_period = header['laser_period_ns']
    
    # Calculate derived quantities
    intensity = np.sqrt(g_image**2 + s_image**2)
    tau_map = calculate_lifetime(g_image, s_image, laser_period)
    
    # Valid data mask
    valid_mask = ~np.isnan(tau_map) & (tau_map > 0) & (tau_map < 15)
    
    # Create custom colormap for lifetime
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
    lifetime_cmap = LinearSegmentedColormap.from_list('lifetime', colors, N=256)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)
    
    # Panel A: Phasor Modulation (pseudo-intensity)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(intensity, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('(A) Phasor Modulation', fontsize=12, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Modulation (a.u.)', fontsize=10)
    
    # Panel B: Lifetime Map
    ax2 = fig.add_subplot(gs[0, 1])
    tau_display = np.where(valid_mask, tau_map, np.nan)
    im2 = ax2.imshow(tau_display, cmap=lifetime_cmap, vmin=1, vmax=8)
    ax2.set_title('(B) Fluorescence Lifetime Map', fontsize=12, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Lifetime τ (ns)', fontsize=10)
    
    # Panel C: Phasor Plot
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Universal semicircle
    theta = np.linspace(0, np.pi, 100)
    g_circle = 0.5 + 0.5 * np.cos(theta)
    s_circle = 0.5 * np.sin(theta)
    ax3.plot(g_circle, s_circle, 'k-', linewidth=2, label='Universal Semicircle')
    
    # Subsample for scatter
    g_flat = g_image[valid_mask].flatten()[::20]
    s_flat = s_image[valid_mask].flatten()[::20]
    tau_flat = tau_map[valid_mask].flatten()[::20]
    
    scatter = ax3.scatter(g_flat, s_flat, c=tau_flat, cmap=lifetime_cmap, 
                          alpha=0.4, s=3, vmin=1, vmax=8)
    cbar3 = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Lifetime τ (ns)', fontsize=10)
    
    ax3.set_xlabel('G', fontsize=11)
    ax3.set_ylabel('S', fontsize=11)
    ax3.set_title('(C) Phasor Diagram', fontsize=12, fontweight='bold')
    ax3.set_xlim([-0.1, 1.1])
    ax3.set_ylim([-0.1, 0.6])
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Lifetime Histogram
    ax4 = fig.add_subplot(gs[1, 0])
    valid_tau = tau_map[valid_mask]
    
    ax4.hist(valid_tau, bins=50, color='steelblue', edgecolor='black', 
             alpha=0.7, density=True)
    ax4.axvline(np.mean(valid_tau), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(valid_tau):.2f} ns')
    ax4.axvline(np.median(valid_tau), color='orange', linestyle=':', linewidth=2,
                label=f'Median: {np.median(valid_tau):.2f} ns')
    
    ax4.set_xlabel('Lifetime (ns)', fontsize=11)
    ax4.set_ylabel('Probability Density', fontsize=11)
    ax4.set_title('(D) Lifetime Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.set_xlim([0, 12])
    
    # Panel E: Simple Segmentation (based on lifetime threshold)
    ax5 = fig.add_subplot(gs[1, 1])
    
    tau_thresh = np.nanmedian(valid_tau)
    segmentation = np.zeros_like(tau_map)
    segmentation[valid_mask & (tau_map < tau_thresh)] = 1  # Low lifetime
    segmentation[valid_mask & (tau_map >= tau_thresh)] = 2  # High lifetime
    segmentation[~valid_mask] = 0  # Invalid
    
    cmap_seg = plt.cm.colors.ListedColormap(['black', '#3498db', '#e74c3c'])
    im5 = ax5.imshow(segmentation, cmap=cmap_seg, vmin=0, vmax=2)
    ax5.set_title(f'(E) Lifetime Segmentation (τ_thresh = {tau_thresh:.1f} ns)', 
                  fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Add legend for segmentation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label=f'τ < {tau_thresh:.1f} ns (low)'),
        Patch(facecolor='#e74c3c', label=f'τ ≥ {tau_thresh:.1f} ns (high)')
    ]
    ax5.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Panel F: Summary Statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
    Summary Statistics
    ══════════════════════════════
    
    Sample: Convallaria majalis
    Source: Zenodo DOI:10.5281/zenodo.15007900
    
    Image Size: {header['image_width']}×{header['image_height']} pixels
    Valid Pixels: {np.sum(valid_mask):,} / {valid_mask.size:,} 
                  ({100*np.sum(valid_mask)/valid_mask.size:.1f}%)
    
    Lifetime Statistics:
      Mean:   {np.mean(valid_tau):.2f} ± {np.std(valid_tau):.2f} ns
      Median: {np.median(valid_tau):.2f} ns
      Range:  {np.nanmin(tau_map[valid_mask]):.2f} – {np.nanmax(tau_map[valid_mask]):.2f} ns
      CV:     {100*np.std(valid_tau)/np.mean(valid_tau):.1f}%
    
    Segmentation:
      Low τ:  {100*np.sum(segmentation==1)/np.sum(valid_mask):.1f}%
      High τ: {100*np.sum(segmentation==2)/np.sum(valid_mask):.1f}%
    """
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('(F) Analysis Summary', fontsize=12, fontweight='bold')
    
    # Main title
    fig.suptitle('Turbo-FLIM: Biological Tissue Analysis\nConvallaria majalis (Lily of the Valley)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('figure_convallaria_composite.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ High-quality figure saved: figure_convallaria_composite.png")
    print(f"   Resolution: 300 DPI")
    print(f"   Size: ~14 × 10 inches")

if __name__ == "__main__":
    create_publication_figure()
