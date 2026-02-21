import matplotlib.pyplot as plt
import numpy as np

param_sizes = np.array([5, 10, 8, 6, 12, 4, 9])
cumulative = np.cumsum(param_sizes)

bucket_cap = 25

fig, ax = plt.subplots(figsize=(10,2))

start = 0
for i, size in enumerate(param_sizes):
    ax.barh(0, size, left=start)
    ax.text(start + size/2, 0, f"P{i}", ha='center', va='center')
    start += size

# Bucket boundaries
for boundary in range(bucket_cap, int(sum(param_sizes)), bucket_cap):
    ax.axvline(boundary, linestyle="--")

ax.set_yticks([])
ax.set_xlabel("Parameter Size (MB)")
ax.set_title("Gradient Bucketing in DDP")
plt.show()

