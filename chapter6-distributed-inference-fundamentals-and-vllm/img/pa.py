import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_kv_cache_plot():
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Configuration: (Label, Type, Color, Hatch)
    # Types: prompt, current, reserved, internal, external, eos
    data = [
        ("Coffee", "prompt", "#e8f5e9", ""), 
        ("solves", "prompt", "#e8f5e9", ""), 
        ("everything", "prompt", "#e8f5e9", ""),
        ("trust", "current", "#fff3e0", ""),
        ("me", "reserved", "#f3e5f5", "xx"),
        ("<EOS>", "eos", "#f3e5f5", ""),
        ("<RESV>", "internal", "#ffebee", "xx"),
        ("..", "internal", "#ffebee", ""),
        ("<RESV>", "internal", "#ffebee", "xx"),
        ("", "external", "white", "xx"), # External fragmentation gap
        ("New", "prompt", "#e8f5e9", ""),
        ("York", "prompt", "#e8f5e9", ""),
        ("City", "current", "#fff3e0", ""),
        ("<EOS>", "eos", "#f3e5f5", "xx"),
        ("<RESV>", "internal", "#ffebee", "xx"),
        ("..", "internal", "#ffebee", ""),
        ("<RESV>", "internal", "#ffebee", "xx"),
    ]

    x_start = 0
    slot_width = 1.0
    slot_height = 0.6
    
    for i, (text, dtype, color, hatch) in enumerate(data):
        # Draw the rectangle
        rect = patches.FancyBboxPatch(
            (x_start + 0.1, 0.2), 0.8, slot_height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            linewidth=1, edgecolor='black', facecolor=color, hatch=hatch
        )
        ax.add_patch(rect)
        
        # Add text inside
        ax.text(x_start + 0.5, 0.5, text, ha='center', va='center', fontsize=12, family='sans-serif')
        x_start += 1

    # --- Brackets and Annotations ---
    def draw_bracket(ax, x_start, x_end, y, label, is_top=True):
        direction = 1 if is_top else -1
        # Draw bracket line
        ax.annotate('', xy=(x_start + 0.1, y), xytext=(x_end - 0.1, y),
                    arrowprops=dict(arrowstyle='<->', connectionstyle=f"bar,fraction={0.2 * direction}"))
        # Add Label
        label_y = y + (0.3 * direction)
        ax.text((x_start + x_end)/2, label_y, label, ha='center', va='center', fontsize=11)

    # Top Annotations
    draw_bracket(ax, 3, 4, 0.9, "Request A current iteration", True)
    draw_bracket(ax, 6, 9, 0.9, "Slots never used\n(Internal\nFragmentation)", True)
    ax.text(7.5, 1.1, "Slots never used", color='red', ha='center') # Color red for emphasis
    draw_bracket(ax, 10, 12, 0.9, "2 KV cache\nstates for\nrequest A's\nprompt", True)
    draw_bracket(ax, 13, 14, 0.9, "1 slot reserved\nfor future\ngenerations", True)

    # Bottom Annotations
    draw_bracket(ax, 0, 3, 0.1, "3 KV cache states for\nrequest A's prompt", False)
    draw_bracket(ax, 4, 6, 0.1, "2 slots reserved\nfor future generations", False)
    
    # External Fragmentation Label
    ax.text(9.5, 0.05, "External\nfragmentation", color='red', ha='center', va='top')
    ax.annotate('', xy=(9.1, 0.2), xytext=(9.9, 0.2), arrowprops=dict(arrowstyle='<->', connectionstyle="bar,fraction=-0.4"))

    draw_bracket(ax, 12, 13, 0.1, "Request B\ncurrent iteration", False)
    draw_bracket(ax, 14, 17, 0.1, "Slots never used\n(Internal\nFragmentation)", False)

    # Final styling
    ax.set_xlim(-0.5, len(data) + 0.5)
    ax.set_ylim(-0.8, 1.8)
    ax.axis('off')
    plt.tight_layout()
    #plt.show()

draw_kv_cache_plot()