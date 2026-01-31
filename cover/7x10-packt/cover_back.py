import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def generate_packt_style_back_cover():
    width, height = 7.0, 10.0
    dpi = 300
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # 1. Background (Dark Charcoal - same as front)
    ax.add_patch(patches.Rectangle((0, 0), width, height, color='#343a40'))

    # 2. Header Tagline (same style as front)
    ax.text(0.5, 9.6, "EXPERT INSIGHT", color='#e67e22', fontsize=12, weight='bold', ha='left')
    ax.add_patch(patches.Rectangle((0.5, 9.55), 2.5, 0.02, color='#e67e22'))

    # 3. Abstract Wave (same as front) - full width, more abstract
    x_wave = np.linspace(0, width, 200)
    for i in range(15):
        y_wave = 3.5 + 0.4 * np.sin(x_wave * (0.5 + i*0.1) + i*0.3) + \
                 0.3 * np.cos(x_wave * (0.3 + i*0.05) + i*0.5) + \
                 0.2 * np.sin(x_wave * (0.8 + i*0.15))
        alpha_val = 0.15 + (i % 3) * 0.1
        line_width = 0.8 + (i % 2) * 0.4
        ax.plot(x_wave, y_wave, color='#e67e22', alpha=alpha_val, lw=line_width)

    # 4. Main Title
    title_x = 0.5
    ax.text(title_x, 8.5, "DISTRIBUTED AI SYSTEMS", color='white', fontsize=24, weight='bold', ha='left')

    # 5. Book Description
    desc_text = (
        "With large AI models now reaching billions or even trillions of parameters,\n"
        "distributed systems have become essential for training and serving AI.\n"
        "This comprehensive guide bridges the gap between theory and practice\n"
        "with hands-on code examples demonstrating production-grade techniques\n"
        "used in real-world systems, from distributed training through inference\n"
        "to production serving."
    )
    ax.text(title_x, 7.5, desc_text, color='white', fontsize=10, weight='normal', 
            ha='left', va='top', linespacing=1.4)

    # 6. Key Features
    features_header = "Key Features"
    features_content = (
        "• Master distributed training with DDP, FSDP, DeepSpeed, and Megatron\n"
        "• Build high-performance inference systems with vLLM and SGLang\n"
        "• Understand GPU hardware, interconnects, and parallelism strategies\n"
        "• Deploy production serving stacks with orchestration and observability\n"
        "• Hands-on code examples tested on real infrastructure throughout"
    )
    
    features_y = 5.8
    ax.text(title_x, features_y, features_header, color='#e67e22', fontsize=12, 
            weight='bold', ha='left')
    ax.text(title_x, features_y - 0.3, features_content, color='white', fontsize=10, 
            weight='normal', ha='left', va='top', linespacing=1.4)

    # 7. Intended Audience
    audience_header = "Intended Audience"
    audience_content = (
        "Designed for ML engineers, AI researchers, and DevOps engineers who\n"
        "need to train or serve large AI models at scale. Platform engineers,\n"
        "HPC cluster administrators, and cloud architects will advance their\n"
        "skills with this guide."
    )
    
    audience_y = 3.5
    ax.text(title_x, audience_y, audience_header, color='#e67e22', fontsize=12, 
            weight='bold', ha='left')
    ax.text(title_x, audience_y - 0.3, audience_content, color='white', fontsize=10, 
            weight='normal', ha='left', va='top', linespacing=1.4)

    # 8. Footer - Orange accent line (same style as front)
    footer_y = 1.5
    ax.axhline(y=footer_y - 0.065, xmin=0, xmax=1, color='#e67e22', linewidth=4, alpha=0.52)
    
    # Footer text
    ax.text(0.5, footer_y + 0.25, "First Edition", color='#d35400', fontsize=14, weight='bold')
    ax.text(0.5, footer_y - 0.6, "<packt>", color='#d35400', fontsize=18, weight='bold', ha='left')

    # Save to cover/ directory (same directory as this script)
    script_dir = Path(__file__).parent
    output_path = script_dir / "cover_back.pdf"
    plt.savefig(str(output_path), dpi=dpi, format='pdf')
    print(f"Packt-style back cover generated: {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_packt_style_back_cover()
