"""
PCIe-only Multi-GPU Server Topology Diagram

This diagram shows a server topology where GPUs are connected only via PCIe,
without NVLink or NVSwitch. All GPU communication must go through the CPU.
"""

import os
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.compute import Server

# Logical GPU node using Server icon (stable for book repos)
# 
# Alternative: For NVIDIA-branded icons, use:
#   from diagrams.nvidia.compute import GPU
# 
# However, this is riskier for long-lived book repos because:
# - NVIDIA icon sets have changed in the past
# - May break with diagrams library updates
# - Generic icons are more stable and semantic clarity > branding
class GPU(Server):
    """Logical GPU node (icon reuse)"""
    pass

# Save figure (standard pattern: same name as script)
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
# diagrams library automatically appends .png, so don't include extension here
output_path_base = os.path.join(script_dir, script_name)
# Full path with extension for print statement
output_path = os.path.join(script_dir, f'{script_name}.png')

with Diagram(
    "PCIe-only Multi-GPU Server Topology",
    show=False,
    filename=output_path_base,  # No extension - diagrams will add .png
    direction="TB",
):
    cpu = Server("CPU\n(Host + NUMA)")

    with Cluster("PCIe Root Complex"):
        gpus = [
            GPU("GPU 0"),
            GPU("GPU 1"),
            GPU("GPU 2"),
            GPU("GPU 3"),
        ]

    for gpu in gpus:
        cpu >> Edge(label="PCIe Gen4/5") >> gpu

print(f"Saved figure to: {output_path}")
