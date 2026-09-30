#!/usr/bin/env python3
"""
Simple Deep Neural Network (DNN) computational graph.
Shows input x, weights W1 and W2, hidden activations z and h, prediction y_hat, and loss L.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "x": "$x$",
        "w1": "$W_1$",
        "z": "$z$",
        "h": "$h$",
        "w2": "$W_2$",
        "y_hat": r"$\hat{y}$",
        "loss": r"$\mathcal{L}$",
    },
    "zh": {
        "x": "$x$",
        "w1": "$W_1$",
        "z": "$z$",
        "h": "$h$",
        "w2": "$W_2$",
        "y_hat": r"$\hat{y}$",
        "loss": r"$\mathcal{L}$",
    }
}


def draw(text: dict) -> plt.Figure:
    bg_color = 'white'
    text_color = '#1b5e20'
    weight_color = 'black'
    loss_color = '#8d6e63'
    activation_color = 'red'

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.set_aspect('equal')

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
        'y_hat_in': 10.,
        'y_hat_out': 10.2,
        'L_loss': 11.5
    }
    
    y_center = 0.5
    circle_radius = 0.4

    for center_x, color in [(x_coords['linear1_center'], weight_color), 
                           (x_coords['sigmoid_center'], activation_color), 
                           (x_coords['linear2_center'], weight_color)]:
        fill_circle = patches.Circle((center_x, y_center), circle_radius, color='white', zorder=1)
        ax.add_patch(fill_circle)
        edge_circle = patches.Circle((center_x, y_center), circle_radius, fill=False, lw=2.5, color=color, zorder=2)
        ax.add_patch(edge_circle)

    mark_offset = circle_radius * 0.5
    ax.plot([x_coords['linear1_center'] - mark_offset, x_coords['linear1_center'] + mark_offset], 
            [y_center - mark_offset, y_center + mark_offset], color=weight_color, lw=2.5, zorder=3)
    ax.plot([x_coords['linear2_center'] - mark_offset, x_coords['linear2_center'] + mark_offset], 
            [y_center - mark_offset, y_center + mark_offset], color=weight_color, lw=2.5, zorder=3)
    
    t = np.linspace(-0.25, 0.25, 50)
    sigmoid_curve = t * circle_radius / 0.25
    sigmoid_offset_y = (circle_radius * 0.4) * np.sin(4 * np.pi * t)
    ax.plot(x_coords['sigmoid_center'] + sigmoid_curve, y_center + sigmoid_offset_y, color=activation_color, lw=2.5, zorder=3)

    text_kwargs = {'va': 'center', 'ha': 'center', 'fontweight': 'bold'}
    
    ax.text(x_coords['x_in'], y_center, text["x"], fontsize=32, color=text_color, **text_kwargs)
    ax.text(x_coords['linear1_center'], y_center - 0.7, text["w1"], fontsize=26, color=weight_color, **text_kwargs)
    
    ax.text(x_coords['z'], y_center + 0.35, text["z"], fontsize=32, color=text_color, **text_kwargs)
    
    ax.text(x_coords['h'], y_center + 0.35, text["h"], fontsize=32, color=text_color, **text_kwargs)
    ax.text(x_coords['linear2_center'], y_center - 0.7, text["w2"], fontsize=26, color=weight_color, **text_kwargs)
    
    ax.text(x_coords['y_hat_in'], y_center, text["y_hat"], fontsize=32, color=text_color, **text_kwargs)
    ax.text(x_coords['L_loss'], y_center, text["loss"], fontsize=32, color=loss_color, **text_kwargs)

    arrow_width = 3
    arrow_color = '#263238'

    ax.annotate('', xy=(x_coords['linear1_edge'], y_center), xytext=(x_coords['x_in'] + 0.2, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    ax.annotate('', xy=(x_coords['sigmoid_edge'], y_center), xytext=(x_coords['linear1_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    ax.annotate('', xy=(x_coords['linear2_edge'], y_center), xytext=(x_coords['sigmoid_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    ax.annotate('', xy=(x_coords['y_hat_in'] - 0.2, y_center), xytext=(x_coords['linear2_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))
    
    ax.annotate('', xy=(x_coords['L_loss'] - 0.3, y_center), xytext=(x_coords['y_hat_out'] + 0.1, y_center),
                arrowprops=dict(arrowstyle="-|>", lw=arrow_width, color=arrow_color, mutation_scale=30))

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 1.3)
    ax.axis('off')
    
    plt.tight_layout(pad=0.5)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "simplednn", LABELS, __file__)