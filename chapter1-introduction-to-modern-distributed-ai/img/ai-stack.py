import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_layer_diagram():
    # Configuration
    fig_width = 12
    fig_height = 16
    dpi = 300  # High resolution
    box_width = 0.8
    box_height = 0.1
    vertical_gap = 0.1
    
    # Data for the layers (Top to Bottom)
    layers = [
        {
            "title": "Framework Layer",
            "tech": "PyTorch, JAX, TensorFlow",
            "desc": "High-level APIs for models, optimizers, data loaders",
            "color": "#6fa8dc"  # Light Blue
        },
        {
            "title": "Messaging Layer",
            "tech": "Tensor, Bucket",
            "desc": "Organizes data into chunks for efficient communication",
            "color": "#93c47d"  # Light Green
        },
        {
            "title": "Collective Operations Layer",
            "tech": "AllReduce, AllGather, Broadcast, Scatter, etc.",
            "desc": "Defines communication patterns between processes",
            "color": "#ffd966"  # Light Yellow
        },
        {
            "title": "Data Transfer Layer",
            "tech": "NCCL (GPU), GLOO (CPU), MPI",
            "desc": "Implements collective operations efficiently",
            "color": "#f6b26b"  # Light Orange
        },
        {
            "title": "Topology Layer",
            "tech": "Ring, Fat-Tree, Mesh, Torus",
            "desc": "Determines communication paths between devices",
            "color": "#e06666"  # Light Red
        },
        {
            "title": "Link Layer",
            "tech": "NVLink, InfiniBand (RDMA), PCIe, Ethernet",
            "desc": "Physical interconnects between devices",
            "color": "#b4a7d6"  # Light Purple
        },
        {
            "title": "Physical Layer",
            "tech": "GPU, TPU, NPU, CPU",
            "desc": "Actual compute and memory hardware",
            "color": "#cccccc"  # Light Grey
        }
    ]

    # Initialize Figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(layers) * (box_height + vertical_gap) + 0.1)
    
    # Remove axes
    ax.axis('off')

    # Draw Layers
    start_y = len(layers) * (box_height + vertical_gap) - vertical_gap
    
    for i, layer in enumerate(layers):
        current_y = start_y - (i * (box_height + vertical_gap))
        
        # 1. Draw the Box (Rectangle)
        # We center it horizontally: x = (1 - box_width) / 2
        x_pos = (1 - box_width) / 2
        
        rect = patches.FancyBboxPatch(
            (x_pos, current_y), 
            box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            edgecolor="black",
            facecolor=layer["color"],
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(rect)

        # 2. Add Text
        # Center of the box
        center_x = 0.5
        center_y = current_y + (box_height / 2)
        
        # Title (Bold)
        ax.text(center_x, center_y + 0.025, layer["title"], 
                ha='center', va='center', fontsize=8, fontweight='bold', color='black', zorder=3)
        
        # Technologies (Normal)
        ax.text(center_x, center_y, layer["tech"], 
                ha='center', va='center', fontsize=7, color='#222222', zorder=3)
        
        # Description (Italic or smaller)
        ax.text(center_x, center_y - 0.025, layer["desc"], 
                ha='center', va='center', fontsize=6, style='italic', color='#333333', zorder=3)

        # 3. Draw Arrow to next layer (if not the last layer)
        if i < len(layers) - 1:
            arrow_start_x = 0.5
            arrow_start_y = current_y 
            arrow_end_y = current_y - vertical_gap
            
            ax.annotate(
                '', 
                xy=(arrow_start_x, arrow_end_y), 
                xytext=(arrow_start_x, arrow_start_y),
                arrowprops=dict(facecolor='black', edgecolor='black', width=3, headwidth=10, shrink=0.05),
                zorder=1
            )

    # Save and Show
    output_filename = 'ai_infrastructure_stack.png'
    plt.tight_layout()
    plt.savefig(output_filename, facecolor='white', bbox_inches='tight')
    print(f"Successfully created {output_filename} with high resolution.")
    plt.show()

if __name__ == "__main__":
    create_layer_diagram()