"""
Multi-Agent Coordination Patterns

This diagram illustrates three common patterns for multi-agent systems:
1. Sequential: Agents process in order, passing output to the next
2. Parallel: Multiple agents process the same task concurrently
3. Hierarchical: Coordinator delegates subtasks to worker agents
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

configure_math_fonts()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Colors
agent_color = '#E3F2FD'
agent_border = '#1976D2'
coord_color = '#FFF3E0'
coord_border = '#F57C00'
task_color = '#E8F5E9'
task_border = '#388E3C'
arrow_color = '#546E7A'

def draw_agent(ax, x, y, label, color=agent_color, border=agent_border, w=1.4, h=0.6):
    """Draw an agent box"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor=border,
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold')

# === Panel 1: Sequential Processing ===
ax1 = axes[0]
ax1.set_title('Sequential', fontsize=12, fontweight='bold', pad=10)

# Task input
ax1.text(1.5, 4.5, 'Task', fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor=task_color, edgecolor=task_border))

# Agents in sequence (more vertical spacing)
draw_agent(ax1, 1.5, 3.5, 'Agent 1')
draw_agent(ax1, 1.5, 2.3, 'Agent 2')
draw_agent(ax1, 1.5, 1.1, 'Agent 3')

# Arrows (straight down)
ax1.annotate('', xy=(1.5, 3.8), xytext=(1.5, 4.2),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax1.annotate('', xy=(1.5, 2.6), xytext=(1.5, 3.2),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax1.annotate('', xy=(1.5, 1.4), xytext=(1.5, 2.0),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Output
ax1.text(1.5, 0.2, 'Output', fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='#C62828'))
ax1.annotate('', xy=(1.5, 0.5), xytext=(1.5, 0.8),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

ax1.set_xlim(0, 3)
ax1.set_ylim(-0.3, 5)
ax1.set_aspect('equal')
ax1.axis('off')

# === Panel 2: Parallel Processing ===
ax2 = axes[1]
ax2.set_title('Parallel', fontsize=12, fontweight='bold', pad=10)

# Task input
ax2.text(2.5, 4.5, 'Task', fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor=task_color, edgecolor=task_border))

# Agents in parallel (spread out horizontally)
draw_agent(ax2, 0.8, 2.8, 'Agent 1')
draw_agent(ax2, 2.5, 2.8, 'Agent 2')
draw_agent(ax2, 4.2, 2.8, 'Agent 3')

# Arrows from task to agents (fan out)
ax2.annotate('', xy=(0.8, 3.1), xytext=(2.2, 4.2),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax2.annotate('', xy=(2.5, 3.1), xytext=(2.5, 4.2),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax2.annotate('', xy=(4.2, 3.1), xytext=(2.8, 4.2),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Aggregator
ax2.text(2.5, 1.3, 'Aggregate', fontsize=9, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5', edgecolor='#7B1FA2'))

# Arrows from agents to aggregator (fan in)
ax2.annotate('', xy=(2.2, 1.5), xytext=(0.8, 2.5),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax2.annotate('', xy=(2.5, 1.5), xytext=(2.5, 2.5),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax2.annotate('', xy=(2.8, 1.5), xytext=(4.2, 2.5),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Output
ax2.text(2.5, 0.2, 'Output', fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='#C62828'))
ax2.annotate('', xy=(2.5, 0.5), xytext=(2.5, 1.0),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

ax2.set_xlim(-0.2, 5.2)
ax2.set_ylim(-0.3, 5)
ax2.set_aspect('equal')
ax2.axis('off')

# === Panel 3: Hierarchical Processing ===
ax3 = axes[2]
ax3.set_title('Hierarchical', fontsize=12, fontweight='bold', pad=10)

# Task input (left side, top)
ax3.text(1.0, 4.5, 'Task', fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor=task_color, edgecolor=task_border))

# Coordinator (left side, middle)
draw_agent(ax3, 1.0, 3.0, 'Coordinator', color=coord_color, border=coord_border, w=1.6)

# Worker agents (right side, stacked vertically)
draw_agent(ax3, 3.8, 4.0, 'Worker 1')
draw_agent(ax3, 3.8, 3.0, 'Worker 2')
draw_agent(ax3, 3.8, 2.0, 'Worker 3')

# Arrow from task to coordinator
ax3.annotate('', xy=(1.0, 3.3), xytext=(1.0, 4.2),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Arrows from coordinator to workers (delegate - solid)
ax3.annotate('', xy=(3.1, 4.0), xytext=(1.8, 3.2),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax3.annotate('', xy=(3.1, 3.0), xytext=(1.8, 3.0),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax3.annotate('', xy=(3.1, 2.0), xytext=(1.8, 2.8),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Arrows from workers back to coordinator (report - dashed green)
ax3.annotate('', xy=(1.8, 3.15), xytext=(3.1, 3.85),
             arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.2, linestyle='--'))
ax3.annotate('', xy=(1.8, 2.95), xytext=(3.1, 2.95),
             arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.2, linestyle='--'))
ax3.annotate('', xy=(1.8, 2.75), xytext=(3.1, 2.15),
             arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.2, linestyle='--'))

# Output from coordinator (left side, bottom)
ax3.text(1.0, 1.2, 'Output', fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='#C62828'))

# Arrow from coordinator to output (straight down)
ax3.annotate('', xy=(1.0, 1.5), xytext=(1.0, 2.7),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

ax3.set_xlim(-0.2, 5.2)
ax3.set_ylim(0.5, 5)
ax3.set_aspect('equal')
ax3.axis('off')

plt.tight_layout(pad=0.5)
save_figure(__file__)
