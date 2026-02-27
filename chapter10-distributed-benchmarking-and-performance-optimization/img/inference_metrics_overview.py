"""
Generate LLM Inference Metrics Overview diagram.
Shows TTFT, ITL, and Generation time between Inference service and User.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up figure with white background
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
blue = '#2563EB'
gray_line = '#6B7280'
text_color = '#374151'
box_border = '#1F2937'

# Y positions
y_service = 5.0
y_user = 2.0
box_height = 0.6

# Draw Inference service box
service_box = FancyBboxPatch(
    (1, y_service - box_height/2), 12, box_height,
    boxstyle="square,pad=0",
    facecolor='white',
    edgecolor=box_border,
    linewidth=2
)
ax.add_patch(service_box)
ax.text(7, y_service, 'Inference service', ha='center', va='center', 
        fontsize=12, color=text_color, fontweight='bold')

# Draw User box
user_box = FancyBboxPatch(
    (1, y_user - box_height/2), 12, box_height,
    boxstyle="square,pad=0",
    facecolor='white',
    edgecolor=box_border,
    linewidth=2
)
ax.add_patch(user_box)
ax.text(7, y_user, 'User', ha='center', va='center', 
        fontsize=12, color=text_color, fontweight='bold')

def draw_arrow_up(ax, x, y_start, y_end, label, label_side='left'):
    """Draw an upward arrow with rotated label."""
    # Arrow body
    arrow_width = 0.3
    arrow_head_height = 0.3
    body_height = y_end - y_start - arrow_head_height
    
    # Arrow body (rectangle)
    body = Rectangle((x - arrow_width/2, y_start), arrow_width, body_height,
                     facecolor=blue, edgecolor=blue)
    ax.add_patch(body)
    
    # Arrow head (triangle)
    head_y = y_start + body_height
    triangle = plt.Polygon([
        (x - arrow_width, head_y),
        (x + arrow_width, head_y),
        (x, head_y + arrow_head_height)
    ], facecolor=blue, edgecolor=blue)
    ax.add_patch(triangle)
    
    # Label (rotated 90 degrees)
    if label_side == 'left':
        ax.text(x - 0.05, (y_start + y_end) / 2, label, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold', rotation=90)
    else:
        ax.text(x + 0.05, (y_start + y_end) / 2, label, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold', rotation=90)

def draw_arrow_down(ax, x, y_start, y_end, label):
    """Draw a downward arrow with rotated label."""
    arrow_width = 0.3
    arrow_head_height = 0.3
    body_height = y_start - y_end - arrow_head_height
    
    # Arrow body (rectangle)
    body = Rectangle((x - arrow_width/2, y_end + arrow_head_height), arrow_width, body_height,
                     facecolor=blue, edgecolor=blue)
    ax.add_patch(body)
    
    # Arrow head (triangle pointing down)
    head_y = y_end + arrow_head_height
    triangle = plt.Polygon([
        (x - arrow_width, head_y),
        (x + arrow_width, head_y),
        (x, y_end)
    ], facecolor=blue, edgecolor=blue)
    ax.add_patch(triangle)
    
    # Label (rotated 90 degrees)
    ax.text(x, (y_start + y_end) / 2 + 0.1, label, ha='center', va='center',
            fontsize=10, color='white', fontweight='bold', rotation=90)

def draw_double_arrow_horizontal(ax, x_start, x_end, y, label):
    """Draw a horizontal double-headed arrow with label."""
    arrow_head_width = 0.2
    arrow_height = 0.25
    
    # Line
    ax.plot([x_start + arrow_head_width, x_end - arrow_head_width], [y, y], 
            color=box_border, linewidth=2)
    
    # Left arrow head
    left_triangle = plt.Polygon([
        (x_start, y),
        (x_start + arrow_head_width, y + arrow_height/2),
        (x_start + arrow_head_width, y - arrow_height/2)
    ], facecolor=box_border, edgecolor=box_border)
    ax.add_patch(left_triangle)
    
    # Right arrow head
    right_triangle = plt.Polygon([
        (x_end, y),
        (x_end - arrow_head_width, y + arrow_height/2),
        (x_end - arrow_head_width, y - arrow_height/2)
    ], facecolor=box_border, edgecolor=box_border)
    ax.add_patch(right_triangle)
    
    # Label
    ax.text((x_start + x_end) / 2, y + 0.35, label, ha='center', va='bottom',
            fontsize=10, color=text_color, fontweight='bold')

# Query arrow (up)
draw_arrow_up(ax, 1.8, y_user + box_height/2, y_service - box_height/2, 'Query')

# TTFT double arrow
draw_double_arrow_horizontal(ax, 2.2, 6.5, 3.8, 'Time to First Token (TTFT)')

# Token arrows (down)
token_positions = [7.0, 9.0, 11.0, 13.0]
token_labels = ['Token 1', 'Token 2', 'Token 3', 'Token 4']

for i, (x, label) in enumerate(zip(token_positions, token_labels)):
    draw_arrow_down(ax, x, y_service - box_height/2, y_user + box_height/2, label)

# ITL double arrows between tokens
itl_positions = [(7.5, 8.5), (9.5, 10.5), (11.5, 12.5)]
for x_start, x_end in itl_positions:
    draw_double_arrow_horizontal(ax, x_start, x_end, 3.8, 'ITL')

# Generation time bracket
bracket_y = 1.2
bracket_x_start = 7.0
bracket_x_end = 13.0

# Dashed vertical lines
ax.plot([bracket_x_start, bracket_x_start], [y_user - box_height/2, bracket_y + 0.3], 
        color=gray_line, linewidth=1, linestyle='--')
ax.plot([bracket_x_end, bracket_x_end], [y_user - box_height/2, bracket_y + 0.3], 
        color=gray_line, linewidth=1, linestyle='--')

# Horizontal bracket line
ax.plot([bracket_x_start, bracket_x_end], [bracket_y, bracket_y], 
        color=gray_line, linewidth=1.5)

# Curly bracket effect (simplified with lines)
ax.plot([bracket_x_start, bracket_x_start], [bracket_y, bracket_y + 0.15], 
        color=gray_line, linewidth=1.5)
ax.plot([bracket_x_end, bracket_x_end], [bracket_y, bracket_y + 0.15], 
        color=gray_line, linewidth=1.5)

# Center point of bracket
mid_x = (bracket_x_start + bracket_x_end) / 2
ax.plot([mid_x - 0.3, mid_x], [bracket_y, bracket_y - 0.2], color=gray_line, linewidth=1.5)
ax.plot([mid_x + 0.3, mid_x], [bracket_y, bracket_y - 0.2], color=gray_line, linewidth=1.5)

# Generation time label
ax.text(mid_x, bracket_y - 0.5, 'Generation time', ha='center', va='top',
        fontsize=11, color=text_color, fontweight='normal', style='italic')

plt.tight_layout()
plt.savefig('img/inference_metrics_overview.png', dpi=150, facecolor='white', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.2)
plt.savefig('img/inference_metrics_overview.pdf', facecolor='white', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.2)
plt.close()

print("Generated: img/inference_metrics_overview.png and img/inference_metrics_overview.pdf")
