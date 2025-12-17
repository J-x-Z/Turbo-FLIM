"""
Generate Schematic Figures for IEEE Manuscript
Creates:
1. figure_1_system_block.png (Sensor System Workflow)
2. figure_2_network_arch.png (Phasor-Fusion MLP)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_system_block_diagram():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Styles
    arrow_props = dict(facecolor='black', width=1, headwidth=8)
    
    # 1. SPAD Sensor Node
    ax.text(1.5, 3.0, "SPAD Sensor\nArray", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#e1f5fe", ec="#01579b"), fontsize=11, fontweight='bold')
    ax.text(1.5, 5.0, "INPUT", ha='center', va='center', fontsize=12, fontweight='bold', color='gray')
    
    # Arrow to TCSPC
    ax.annotate('', xy=(3.0, 3.0), xytext=(2.2, 3.0), arrowprops=arrow_props)
    
    # 2. TCSPC Hardware
    ax.text(3.8, 3.0, "TCSPC\nModule", ha='center', va='center', bbox=dict(boxstyle="square,pad=0.4", fc="#fff9c4", ec="#fbc02d"), fontsize=10)
    ax.text(3.8, 2.2, "(Histogramming)", ha='center', va='center', fontsize=8, color='#fbc02d')
    
    # Arrow to Turbo-FLIM
    ax.annotate('', xy=(5.2, 3.0), xytext=(4.6, 3.0), arrowprops=arrow_props)
    ax.text(4.9, 3.2, "Raw Histograms\n(Gb/s)", ha='center', va='center', fontsize=8)
    
    # 3. TURBO-FLIM ENGINE (The Core)
    rect = patches.Rectangle((5.2, 1.5), 3.6, 3.0, linewidth=2, edgecolor='#2e7d32', facecolor='#e8f5e9', linestyle='--')
    ax.add_patch(rect)
    ax.text(7.0, 4.2, "Turbo-FLIM Engine (Rust)", ha='center', va='center', fontsize=11, fontweight='bold', color='#2e7d32')
    
    # Inside Engine: Parallelism -> MLP
    ax.text(6.0, 3.0, "Parallel\nPreproc\n(SIMD/Rayon)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2e7d32"), fontsize=8)
    ax.annotate('', xy=(7.0, 3.0), xytext=(6.5, 3.0), arrowprops=dict(facecolor='#2e7d32', width=0.5, headwidth=5))
    ax.text(7.8, 3.0, "Phasor-Fusion\nMLP\n(Tract/ONNX)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2e7d32"), fontsize=8)
    
    # Key Metrics attached to Engine
    ax.text(7.0, 1.2, "18 µs/pixel | ~0.6 MB Model", ha='center', va='center', fontsize=9, fontweight='bold', color='#2e7d32', bbox=dict(fc='white', ec='none'))

    # Arrow to Output
    ax.annotate('', xy=(9.5, 3.0), xytext=(8.8, 3.0), arrowprops=arrow_props)
    
    # 4. Real-Time Output
    ax.text(10.5, 3.0, "Lifetime Map\n(Real-Time)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#c62828"), fontsize=11, fontweight='bold')
    ax.text(10.5, 5.0, "OUTPUT", ha='center', va='center', fontsize=12, fontweight='bold', color='gray')
    
    # Feedback Control Loop (Optional hint)
    ax.annotate('', xy=(1.5, 2.0), xytext=(10.5, 2.0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="gray", linestyle="dashed"))
    ax.text(6.0, 0.8, "Closed-Loop Control / Feedback", ha='center', va='center', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('figure_1_system_block.png', dpi=300, bbox_inches='tight')
    print("Figure 1 generated.")

def create_network_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.3', facecolor='#fff', edgecolor='#333')
    layer_style = dict(boxstyle='round,pad=0.4', facecolor='#eeeeee', edgecolor='#333')

    # Inputs with Vector Notation
    ax.text(1.5, 4.5, "Decay Vector\n$\\mathbf{h} \\in \\mathbb{R}^{1001}$", ha='center', va='center', bbox=box_style)
    ax.text(1.5, 1.5, "Phasor Vector\n$\\mathbf{p} = [G, S]$", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', edgecolor='red', linewidth=2))
    ax.text(1.5, 0.8, "Physics Prior", ha='center', fontsize=10, color='red', fontweight='bold')

    # Concatenation Hub
    ax.text(3.5, 3.0, "Concatenation\n$\\mathbf{x} = [\\mathbf{h}, \\mathbf{p}]$\n(1003 features)", ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.5', facecolor='#ddd', edgecolor='#333'))
    
    # Arrows to Concat
    ax.annotate("", xy=(2.9, 3.2), xytext=(2.2, 4.2), arrowprops=dict(arrowstyle='->'))
    ax.annotate("", xy=(2.9, 2.8), xytext=(2.2, 1.8), arrowprops=dict(arrowstyle='->'))

    # MLP Layers
    x_start = 5.0
    layers = [256, 128, 64, 32]
    for i, nodes in enumerate(layers):
        x = x_start + i * 1.2
        ax.text(x, 3.0, f"FC\n{nodes}\nReLU", ha='center', va='center', bbox=layer_style, fontsize=9)
        if i > 0:
             ax.annotate("", xy=(x-0.4, 3.0), xytext=(x-1.2+0.4, 3.0), arrowprops=dict(arrowstyle='->'))
    
    # Arrow to First Layer
    ax.annotate("", xy=(4.6, 3.0), xytext=(4.0, 3.0), arrowprops=dict(arrowstyle='->'))

    # Output
    ax.text(9.5, 3.0, "Output\n$\\tau_1, \\tau_2$", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#ccffcc', edgecolor='green', linewidth=2))
    ax.annotate("", xy=(9.0, 3.0), xytext=(8.6, 3.0), arrowprops=dict(arrowstyle='->'))

    ax.set_title("Figure 2: Phasor-Fusion Network Architecture", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figure_2_network_arch.png', dpi=300, bbox_inches='tight')
    print("figure_2_network_arch.png generated.")

if __name__ == "__main__":
    create_system_block_diagram()
    create_network_diagram()
