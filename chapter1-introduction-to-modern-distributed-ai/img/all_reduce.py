"""
Visualize the AllReduce operation in distributed training.
Shows input tensors from different ranks being reduced and distributed
to all ranks, resulting in the same output tensor on each rank.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import os

# ---------------------------
# Setup
# ---------------------------
# Figure size suitable for 7x10 inch page
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")

num_ranks = 4
block_width = 0.8
block_height = 2.0
y0 = 0.5

x_left = [i * 1.2 for i in range(num_ranks)]
x_right = [i * 1.2 + 7 for i in range(num_ranks)]

colors = ["#0096d6", "#d62728", "#6aa84f", "#ffd11a"]

# ---------------------------
# Left: input tensors
# ---------------------------
for i in range(num_ranks):
    ax.text(x_left[i] + block_width / 2, y0 + block_height + 0.3,
            f"rank {i}", ha="center", fontsize=11)

    rect = Rectangle(
        (x_left[i], y0),
        block_width,
        block_height,
        facecolor=colors[i],
        edgecolor="black"
    )
    ax.add_patch(rect)

    ax.text(x_left[i] + block_width / 2,
            y0 + block_height / 2,
            f"in{i}",
            ha="center", va="center", fontsize=12, weight="bold")

# ---------------------------
# Arrow (AllReduce)
# ---------------------------
arrow = FancyArrow(
    5.0, y0 + block_height / 2,
    1.2, 0,
    width=0.15,
    length_includes_head=True,
    head_width=0.4,
    head_length=0.3,
    color="gray"
)
ax.add_patch(arrow)

# ---------------------------
# Right: output tensors
# ---------------------------
for i in range(num_ranks):
    ax.text(x_right[i] + block_width / 2, y0 + block_height + 0.3,
            f"rank {i}", ha="center", fontsize=11)

    rect = Rectangle(
        (x_right[i], y0),
        block_width,
        block_height,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(x_right[i] + block_width / 2,
            y0 + block_height / 2,
            "out",
            ha="center", va="center", fontsize=12)

# ---------------------------
# Equation annotation
# ---------------------------
ax.text(7.0, y0 - 0.4,
        r"$\mathrm{out}[j] = \sum_i \mathrm{in}_i[j]$",
        fontsize=11, ha="center")

# ---------------------------
# Limits
# ---------------------------
ax.set_xlim(-0.5, 12)
ax.set_ylim(0, 3.5)

plt.tight_layout()

# Save figure (standard pattern: same name as script)
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, f'{script_name}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.03)
print(f"Saved figure to: {output_path}")
plt.close()  # Close to free memory


