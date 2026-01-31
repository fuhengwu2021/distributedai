import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from pathlib import Path

def generate_packt_style_cover():
    width, height = 7.0, 10.0
    dpi = 300
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # 1. Background (Dark Charcoal)
    ax.add_patch(patches.Rectangle((0, 0), width, height, color='#343a40'))

    # 2. Packt Footer (Orange Accent) - positioned above author image
    footer_height = 1.2
    footer_y = 1.5  # Position footer above author image (which is at y=1.8)
    #ax.add_patch(patches.Rectangle((0, footer_y), width, footer_height, color='#e67e22'))
    # Decorative line - using axhline is simpler than Rectangle for a horizontal line
    ax.axhline(y=footer_y - 0.065, xmin=0, xmax=1, color='#e67e22', linewidth=4, alpha=0.52)

    # 3. Header Tagline
    ax.text(0.5, 9.6, "EXPERT INSIGHT", color='#e67e22', fontsize=12, weight='bold', ha='left')
    ax.add_patch(patches.Rectangle((0.5, 9.55), 2.5, 0.02, color='#e67e22'))

    # 4. "New Release" Badge (Top Right)
    circle = patches.Circle((6.2, 9.2), 0.6, color='#d35400', alpha=0.8)
    ax.add_patch(circle)
    ax.text(6.2, 9.3, "New", color='white', fontsize=10, weight='bold', ha='center')
    ax.text(6.2, 9.1, "Release", color='white', fontsize=10, weight='bold', ha='center')

    # 5. Main Typography (Left Aligned)
    title_x = 0.5
    ax.text(title_x, 8.5, "Distributed", color='white', fontsize=38, weight='bold', ha='left')
    ax.text(title_x, 7.8, "AI Systems", color='white', fontsize=38, weight='bold', ha='left')
    
    subtitle = "A practical guide to building scalable training,\ninference, and serving systems for production AI"
    ax.text(title_x, 7.0, subtitle, color='white', fontsize=14, weight='medium', ha='left', linespacing=1.5)

    # 6. Abstract Wave (The "Packt" look) - full width, more abstract
    x_wave = np.linspace(0, width, 200)  # Full width, more points for smoother curves
    for i in range(15):  # More layers for depth
        # More abstract: combine multiple frequencies and add randomness
        y_wave = 3.5 + 0.4 * np.sin(x_wave * (0.5 + i*0.1) + i*0.3) + \
                 0.3 * np.cos(x_wave * (0.3 + i*0.05) + i*0.5) + \
                 0.2 * np.sin(x_wave * (0.8 + i*0.15))
        # Vary alpha and line width for more depth
        alpha_val = 0.15 + (i % 3) * 0.1
        line_width = 0.8 + (i % 2) * 0.4
        ax.plot(x_wave, y_wave, color='#e67e22', alpha=alpha_val, lw=line_width)

    # 7. Author Image (me.jpg)
    script_dir = Path(__file__).parent
    me_img_path = script_dir / "me_bkg_rm.png"
    try:
        img = mpimg.imread(str(me_img_path))
        # Simple crop/zoom simulation
        imagebox = OffsetImage(img, zoom=0.15) # Adjust zoom to fit
        ab = AnnotationBbox(imagebox, (5.5, 1.55), frameon=False)  # Moved upward from 1.8 to 2.2 3.06
        ax.add_artist(ab)
    except:
        ax.text(5.5, 2.2, "[Author Photo]", color='white', ha='center')

    # 8. Footer Text (positioned within the footer band)
    footer_y = 1.5
    ax.text(0.5, footer_y + 0.25, "First Edition", color='#d35400', fontsize=14, weight='bold')
    ax.text(0.5, footer_y + 5., "Henry Fuheng Wu", color='white', fontsize=16, weight='bold')
    
    # Packt Logo Simulation
    ax.text(0.5, footer_y - 0.6, "<packt>", color='#d35400', fontsize=18, weight='bold', ha='left')

    # Save to cover/ directory (same directory as this script)
    output_path = script_dir / "cover_front.pdf"
    plt.savefig(str(output_path), dpi=dpi, format='pdf')
    print(f"Packt-style cover generated: {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_packt_style_cover()