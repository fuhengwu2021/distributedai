import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12,6))

# ---- Node 0 ----
node0 = patches.Rectangle((0.1, 0.3), 0.3, 0.4, fill=False)
ax.add_patch(node0)
ax.text(0.25, 0.65, "Node 0 (Flower)", ha='center', fontsize=12)
#ax.text(0.25, 0.58, "Ubuntu 20.04", ha='center')
ax.text(0.25, 0.53, "Rank 0", ha='center')

gpu0 = patches.Rectangle((0.17, 0.4), 0.16, 0.12, fill=False)
ax.add_patch(gpu0)
ax.text(0.25, 0.42, "RTX 4090\n(sm_89)", ha='center')

# ---- Node 1 ----
node1 = patches.Rectangle((0.6, 0.3), 0.3, 0.4, fill=False)
ax.add_patch(node1)
ax.text(0.75, 0.65, "Node 1 (Potato)", ha='center', fontsize=12)
#ax.text(0.75, 0.58, "Ubuntu 24.04", ha='center')
ax.text(0.75, 0.53, "Rank 1", ha='center')

gpu1 = patches.Rectangle((0.67, 0.4), 0.16, 0.12, fill=False)
ax.add_patch(gpu1)
ax.text(0.75, 0.42, "RTX 5070\n(sm_120)", ha='center')

# ---- Communication Arrow ----
ax.annotate(
    "NCCL AllReduce", # \n(TCP over Wi-Fi)
    xy=(0.4, 0.5),
    xytext=(0.6, 0.5),
    arrowprops=dict(arrowstyle="<->"),
    ha='center',
    va='center'
)

# ---- Training Flow ----
#ax.text(0.25, 0.37, "Forward\nBackward\nGradients", ha='center')
#ax.text(0.75, 0.37, "Forward\nBackward\nGradients", ha='center')

# ---- Output ----
ax.text(0.5, 0.15, "Output: Rank 0 & Rank 1 print all_reduce_sum=1.0", ha='center')

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis('off')

plt.title("2-Node Heterogeneous GPU DDP Architecture")
plt.show()