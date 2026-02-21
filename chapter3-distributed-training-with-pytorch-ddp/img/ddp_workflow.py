import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

def plot_ddp_workflow():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Helper function to draw labeled rectangles
    def draw_box(x, y, w, h, text, color, text_color='black', fontsize=14):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, color=text_color, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', wrap=True)

    # 1. Dataloader (The Source)
    draw_box(2, 25, 12, 10, "Dataloader", "#ffdb99")

    # 2. GPU Rows (Sharding and Processing)
    gpu_colors = ["#f2f2f2", "#e6f0ff", "#ffcccc"] # GPU, Pass, Update colors
    # Different colors for each data shard - distinct colors for easy differentiation
    shard_colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24"]  # Red, Teal, Blue, Yellow
    y_offsets = [45, 32, 19, 6]
    
    for i, y in enumerate(y_offsets):
        # Data Shard
        draw_box(18, y+2, 6, 6, "", "white")
        ax.add_patch(patches.Rectangle((18 + (i*1.5), y+2), 1.5, 6, color=shard_colors[i]))
        
        # GPU / Model
        # Draw box without text first
        draw_box(28, y-0.4, 10, 10, "", "#f2f2f2")
        # Add GPU text separately, positioned lower in the box
        ax.text(33, y-0.4 + 3, f"GPU{3-i}", color='black', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        ax.add_patch(plt.Circle((33, y+8), 2.8, color="#b19cd9", ec='black'))
        ax.text(33, y+8, "Model", fontsize=12, ha='center', va='center', fontweight='bold', color='black')

        # Forward/Backward Pass
        draw_box(42, y, 15, 10, "Forward/\nBackward pass", "#99bcff", text_color="black")
        
        # Connections: Dataloader -> Shard -> GPU -> Pass
        ax.annotate('', xy=(18, y+5), xytext=(14, 30), arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'))
        ax.annotate('', xy=(28, y+5), xytext=(24, y+5), arrowprops=dict(arrowstyle='->', lw=0.5))
        ax.annotate('', xy=(42, y+5), xytext=(38, y+5), arrowprops=dict(arrowstyle='->', lw=0.5))

    # 3. Synchronize Gradients (The Bottleneck/Communication)
    ax.text(70, 55, "Synchronize", fontsize=18, fontweight='bold', ha='center', color='black')
    draw_box(65, 23, 14, 14, "Synchronize\ngradients", "#b19cd9", text_color="black", fontsize=14)

    # Connections: Passes -> Synchronize
    for y in y_offsets:
        ax.annotate('', xy=(65, 30), xytext=(57, y+5), arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))

    # 4. Update Model (Final Step)
    for y in y_offsets:
        draw_box(85, y, 12, 10, "Update\nModel", "#e69183", text_color="black")
        # Connection: Synchronize -> Update
        ax.annotate('', xy=(85, y+5), xytext=(79, 30), arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))

    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    fig, ax = plot_ddp_workflow()
    save_figure(__file__)
    