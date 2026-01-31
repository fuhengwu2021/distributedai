import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import os

# Save figure setup
script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
output_path = os.path.join(script_dir, 'cpu_gpu_interaction.png')

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_aspect('equal')
ax.axis('off')

# Configuration
SPACING = 3.5
CENTER_Y = 2.5
LABEL_OFFSET = 0.7

# Positions
cpu_x = 1.5
gpu_x = cpu_x + SPACING

def draw_cpu_vector(x, y, ax):
    """Draws a CPU using vector shapes."""
    # CPU substrate
    rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor='#E8F5E9', 
                             edgecolor='#2E7D32', linewidth=2, zorder=3)
    ax.add_patch(rect)
    # CPU Integrated Heat Spreader (IHS)
    ihs = patches.Rectangle((x-0.25, y-0.25), 0.5, 0.5, facecolor='#CFD8DC', 
                            edgecolor='#455A64', linewidth=1, zorder=4)
    ax.add_patch(ihs)
    # Labels
    ax.text(x, y - LABEL_OFFSET, 'CPU', fontsize=12, ha='center', fontweight='bold', color='#2E7D32')

def draw_gpu_vector(x, y, ax):
    """Draws a GPU using vector shapes."""
    # GPU PCB
    pcb = patches.Rectangle((x-0.5, y-0.3), 1.0, 0.6, facecolor='#E3F2FD', 
                            edgecolor='#1E64AC', linewidth=2, zorder=3)
    ax.add_patch(pcb)
    # GPU Fan/Cooler
    fan = plt.Circle((x+0.1, y), 0.2, facecolor='#90CAF9', edgecolor='#1E64AC', zorder=4)
    ax.add_artist(fan)
    # Labels
    ax.text(x, y - LABEL_OFFSET, 'GPU', fontsize=12, ha='center', fontweight='bold', color='#1E64AC')

# Draw Components
draw_cpu_vector(cpu_x, CENTER_Y, ax)
draw_gpu_vector(gpu_x, CENTER_Y, ax)

# PCIe connection (Main Pipe)
pcie_line = ConnectionPatch(
    (cpu_x + 0.45, CENTER_Y), (gpu_x - 0.55, CENTER_Y),
    "data", "data", arrowstyle='<->', mutation_scale=20, 
    linewidth=4, color='#E67E22', zorder=1
)
ax.add_patch(pcie_line)

# PCIe label
ax.text((cpu_x + gpu_x) / 2, CENTER_Y + 0.2, 'PCIe Gen 4/5',
        fontsize=10, ha='center', va='bottom', fontweight='bold', color='#E67E22',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E67E22', alpha=0.9))

# Data flow: CUDA Kernels
kernel_y = CENTER_Y - 0.9
ax.annotate('', xy=(gpu_x - 0.5, kernel_y), xytext=(cpu_x + 0.4, kernel_y),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2, ls='--'))
ax.text((cpu_x + gpu_x) / 2, kernel_y - 0.1, 'Control: CUDA Kernels',
        fontsize=9, ha='center', va='top', color='#27AE60', style='italic')

# Data flow: Memory Transfers
memory_y = CENTER_Y + 0.9
ax.annotate('', xy=(gpu_x - 0.5, memory_y), xytext=(cpu_x + 0.4, memory_y),
            arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=2, ls='--'))
ax.text((cpu_x + gpu_x) / 2, memory_y + 0.1, 'Data: Memory Transfers',
        fontsize=9, ha='center', va='bottom', color='#8E44AD', style='italic')

# Bottleneck indicator
ax.text((cpu_x + gpu_x) / 2, 0.8, 'Bottleneck: PCIe bandwidth << NVLink',
        fontsize=9, ha='center', style='italic', color='#C0392B',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FADBD8', edgecolor='#C0392B', alpha=0.9))

# Plot limits and save
ax.set_xlim(0, 7)
ax.set_ylim(0, 5)
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Success! Diagram saved to: {output_path}")