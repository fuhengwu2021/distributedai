import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart_optimized_v2():
    # Figure size
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # --- Style Parameters (Updated) ---
    # Smaller boxes
    box_width = 0.18  # Reduced from 0.20
    box_height = 0.09 # Reduced from 0.10
    
    # Larger fonts
    font_box = 18     # Increased from 16
    font_edge = 15    # Increased from 14
    font_root = 20    # Increased from 18
    font_title = 24   # Increased from 22

    # Colors
    c_root = '#FFE0B2'   # Orange
    c_cat = '#B3E5FC'    # Blue
    c_cond = '#E1BEE7'   # Purple
    c_res = '#C8E6C9'    # Green

    def draw_box(x, y, text, color, fontsize=font_box):
        # x, y is center of box
        rect = patches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2), 
            box_width, box_height,
            boxstyle="round,pad=0.01", # Slightly reduced padding
            linewidth=1.5,
            edgecolor='gray',
            facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, wrap=True, fontweight='medium')
        return (x, y)

    def draw_edge(p1, p2, label=None):
        # Draw arrow from p1 to p2. 
        x1, y1 = p1
        x2, y2 = p2
        
        # Adjustment for new box size
        x1 += box_width/2
        x2 -= box_width/2
        
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=2))
        
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            # Larger edge label font
            ax.text(mid_x, mid_y + 0.01, label, ha='center', va='bottom', fontsize=font_edge, color='darkblue', fontweight='bold')

    # --- Layout ---
    
    # Column X positions (kept from previous optimized version)
    x_root = 0.10
    x_cat = 0.36
    x_cond = 0.62
    x_res = 0.88

    # Root
    root_pos = draw_box(x_root, 0.5, "Start:\nWhat is your\nuse case?", c_root, fontsize=font_root)

    # Categories (Level 1)
    pos_train = draw_box(x_cat, 0.85, "Training from\nscratch?", c_cat)
    pos_ft = draw_box(x_cat, 0.50, "Fine-tuning?", c_cat)
    pos_inf = draw_box(x_cat, 0.15, "Inference/Serving?", c_cat)

    draw_edge(root_pos, pos_train)
    draw_edge(root_pos, pos_ft)
    draw_edge(root_pos, pos_inf)

    # Branches
    
    def draw_branch(parent_pos, y_start, conditions, results):
        y_gap = 0.11 # Kept from previous version
        
        for i, (cond, res) in enumerate(zip(conditions, results)):
            y = y_start - (i * y_gap)
            
            # Condition Node
            p_cond = draw_box(x_cond, y, cond, c_cond)
            # Result Node
            p_res = draw_box(x_res, y, res, c_res)
            
            # Edges
            draw_edge(parent_pos, p_cond)
            draw_edge(p_cond, p_res, label="Yes")

    # 1. Training Branch (Top)
    t_conds = ["Model > 60GB?", "Training time\ntoo long?", "Both OK?"]
    t_ress = ["Distributed Training\n(FSDP/Model Parallel)", "Distributed Training\n(Data Parallel)", "Single GPU"]
    draw_branch(pos_train, 0.95, t_conds, t_ress)

    # 2. Fine-tuning Branch (Middle)
    f_conds = ["Base model >\nGPU memory?", "Using\nLoRA/QLoRA?", "Large dataset?"]
    f_ress = ["Distributed\nFine-tuning", "Usually single\nGPU OK", "Consider\ndistributed"]
    draw_branch(pos_ft, 0.61, f_conds, f_ress)

    # 3. Inference Branch (Bottom)
    i_conds = ["Model >\nGPU memory?", "High throughput\nneeded?", "Both OK?"]
    i_ress = ["Distributed Inference\n(Model Parallel)", "Distributed Inference\n(Multiple GPUs)", "Single GPU with opt\n(vLLM/SGLang)"]
    draw_branch(pos_inf, 0.26, i_conds, i_ress)

    # Title
    plt.title("GPU Strategy Decision Tree", fontsize=font_title, fontweight='bold', y=1.02)
    
    # Save
    filename = 'gpu_decision_tree_optimized_v2.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

file_path = draw_flowchart_optimized_v2()
print(file_path)