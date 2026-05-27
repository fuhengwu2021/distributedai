import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib for math expressions
configure_math_fonts()



def draw_extended_simple_dnn_diagram():
    # Define color scheme
    bg_color = 'white'
    text_color = '#1b5e20' # Dark green
    weight_color = 'black'
    loss_color = '#8d6e63' # Brown
    activation_color = 'red'

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    
    # Ensure circles are perfect rounds
    ax.set_aspect('equal')

    # Component layout coordinates
    x_coords = {
        'x_in': 0.5,
        'linear1_edge': 1.6,
        'linear1_center': 2.0,
        'linear1_out': 2.4,
        'z': 3.5,
        'sigmoid_edge': 4.6,
        'sigmoid_center': 5.0,
        'sigmoid_out': 5.4,
        'h': 6.5,
        'linear2_edge': 7.6,
        'linear2_center': 8.0,
        'linear2_out': 8.4,
        'y_hat_in': 10.,   # centered in gap between W2→ŷ and ŷ→L arrows
        'y_hat_out': 10.2,
        'L_loss': 11.5
    }
    
    y_center = 0.5
    circle_radius = 0.4

    # Draw Nodes (Perfect Circles)
    for center_x, color in [(x_coords['linear1_center'], weight_color), 
                           (x_coords['sigmoid_center'], activation_color), 
                           (x_coords['linear2_center'], weight_color)]:
        # White background fill for clarity within nodes
        fill_circle = patches.Circle((center_x, y_center), circle_radius, color='white', zorder=1)
        ax.add_patch(fill_circle)
        # Outline
        edge_circle = patches.Circle((center_x, y_center), circle_radius, fill=False, lw=2.5, color=color, zorder=2)
        ax.add_patch(edge_circle)

    # Internal details for nodes (matching the sketch style)
    # Linear layers (/ mark)
    mark_offset = circle_radius * 0.5
    ax.plot([x_coords['linear1_center'] - mark_offset, x_coords['linear1_center'] + mark_offset], 
            [y_center - mark_offset, y_center + mark_offset], color=weight_color, lw=2.5, zorder=3)
    ax.plot([x_coords['linear2_center'] - mark_offset, x_coords['linear2_center'] + mark_offset], 
            [y_center - mark_offset, y_center + mark_offset], color=weight_color, lw=2.5, zorder=3)
    
    # Sigmoid node S-curve (in red)
    t = np.linspace(-0.25, 0.25, 50)
    sigmoid_curve = t * circle_radius / 0.25
    sigmoid_offset_y = (circle_radius * 0.4) * np.sin(4 * np.pi * t) # Create an S shape
    ax.plot(x_coords['sigmoid_center'] + sigmoid_curve, y_center + sigmoid_offset_y, color=activation_color, lw=2.5, zorder=3)

    # Text Labels
    text_kwargs = {'va': 'center', 'ha': 'center', 'fontweight': 'bold'}
    
    plt.text(x_coords['x_in'], y_center, '$x$', fontsize=32, color=text_color, **text_kwargs)
    plt.text(x_coords['linear1_center'], y_center - 0.7, '$W_1$', fontsize=26, color=weight_color, **text_kwargs)
    
    plt.text(x_coords['z'], y_center + 0.35, '$z$', fontsize=32, color=text_color, **text_kwargs)
    
    plt.text(x_coords['h'], y_center + 0.35, '$h$', fontsize=32, color=text_color, **text_kwargs)
    plt.text(x_coords['linear2_center'], y_center - 0.7, '$W_2$', fontsize=26, color=weight_color, **text_kwargs)
    
    # Render \hat{y} carefully
    plt.text(x_coords['y_hat_in'], y_center, '$\hat{y}$', fontsize=32, color=text_color, **text_kwargs)
    plt.text(x_coords['L_loss'], y_center, '$\mathcal{L}$', fontsize=32, color=loss_color, **text_kwargs)

    # Draw Long Arrows (Maximized Length)
    arrow_width = 3
    arrow_color = '#263238' # Dark grey/black for arrows

    # x -> W1 (circle input)
    ax.annotate('', xy=(x_coords['linear1_edge'], y_center), xytext=(x_coords['x_in'] + 0.2, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    # W1 -> z -> sigmoid (extended from W1 out, through z, to sigmoid edge)
    # The arrow should pass near z, but have a continuous path. We'll start it after W1 and point to Sigmoid.
    ax.annotate('', xy=(x_coords['sigmoid_edge'], y_center), xytext=(x_coords['linear1_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    # sigmoid -> h -> W2 (extended from sigmoid out, through h, to W2 edge)
    ax.annotate('', xy=(x_coords['linear2_edge'], y_center), xytext=(x_coords['sigmoid_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    # W2 -> y_hat (extended from W2 out to y_hat label)
    ax.annotate('', xy=(x_coords['y_hat_in'] - 0.2, y_center), xytext=(x_coords['linear2_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    # y_hat -> L (from y_hat label out to L label)
    ax.annotate('', xy=(x_coords['L_loss'] - 0.3, y_center), xytext=(x_coords['y_hat_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))

    # Final plot adjustments
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 1.3) # Expand bounds vertically for labels and nodes
    ax.axis('off')
    
    plt.tight_layout(pad=0.5)
    save_figure(__file__, facecolor=bg_color)


if __name__ == '__main__':
    draw_extended_simple_dnn_diagram()