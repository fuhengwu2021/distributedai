#!/usr/bin/env python3
"""
k3d GPU Cluster Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the architecture of a k3d cluster with GPU support:
- Host machine with Docker Engine
- k3d network containing server and agent nodes
- GPU passthrough from host to containers
- NVIDIA Device Plugin for Kubernetes GPU scheduling
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib for math expressions
configure_math_fonts()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

# Colors
host_color = '#E8EAF6'       # Light indigo
docker_color = '#E3F2FD'     # Light blue
node_color = '#E8F5E9'       # Light green
gpu_color = '#FFF3E0'        # Light orange
k8s_color = '#F3E5F5'        # Light purple
border_color = '#424242'     # Dark gray

# Host Machine (outermost box)
host_box = FancyBboxPatch(
    (0.3, 0.3), 11.4, 8.4,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=host_color,
    edgecolor=border_color,
    linewidth=2
)
ax.add_patch(host_box)
ax.text(6, 8.5, 'Host Machine (Linux)', ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Docker Engine box
docker_box = FancyBboxPatch(
    (0.6, 2.8), 10.8, 5.2,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    facecolor=docker_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(docker_box)
ax.text(6, 7.8, 'Docker Engine', ha='center', va='center', 
        fontsize=14, fontweight='bold')

# k3d Network box
k3d_box = FancyBboxPatch(
    (0.9, 3.0), 10.2, 4.5,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor='white',
    edgecolor='#1976D2',
    linewidth=1.5,
    linestyle='--'
)
ax.add_patch(k3d_box)
ax.text(6, 7.3, 'k3d Network: k3d-mycluster-gpu', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#1976D2')

# Control Plane Node (Server-0)
server_box = FancyBboxPatch(
    (1.2, 4.8), 4.3, 2.3,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=node_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(server_box)
ax.text(3.35, 6.9, 'Control Plane (server-0)', ha='center', va='center', 
        fontsize=14, fontweight='bold')
ax.text(3.35, 6.55, 'k3s-cuda:v1.35.1-cuda-13.0.0', ha='center', va='center', 
        fontsize=14, style='italic', color='black')

# Control plane services
services = ['kube-apiserver', 'kube-scheduler', 'etcd', 'kubelet']
for i, svc in enumerate(services):
    y_pos = 6.1 - i * 0.35
    ax.text(2.0, y_pos, f'• {svc}', ha='left', va='center', fontsize=14, color='#424242')

# NVIDIA device plugin in server
ax.text(4.42, 5., 'nvidia-device-plugin', ha='center', va='center', 
        fontsize=14, color='#E65100', fontweight='bold')

# Worker Node (Agent-0)
agent_box = FancyBboxPatch(
    (6.5, 4.8), 4.3, 2.3,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=node_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(agent_box)
ax.text(8.65, 6.9, 'Worker Node (agent-0)', ha='center', va='center', 
        fontsize=14, fontweight='bold')
ax.text(8.65, 6.55, 'k3s-cuda:v1.35.1-cuda-13.0.0', ha='center', va='center', 
        fontsize=14, style='italic', color='black')

# Worker services
worker_services = ['kubelet', 'kube-proxy', 'containerd']
for i, svc in enumerate(worker_services):
    y_pos = 6.1 - i * 0.35
    ax.text(7.3, y_pos, f'• {svc}', ha='left', va='center', fontsize=14, color='#424242')

# Workload pods box
workload_box = FancyBboxPatch(
    (8.8, 5.0), 1.7, 1.0,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    facecolor=k8s_color,
    edgecolor='#7B1FA2',
    linewidth=1
)
ax.add_patch(workload_box)
ax.text(9.65, 5.75, 'Workloads', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#7B1FA2')
ax.text(9.65, 5.45, 'vLLM pods', ha='center', va='center', fontsize=14, color='black')
ax.text(9.65, 5.2, 'SGLang pods', ha='center', va='center', fontsize=14, color='black')

# GPU passthrough indicator
ax.text(3.58, 4.2, '--gpus=all', ha='center', va='center', 
        fontsize=14, color='#E65100', style='italic')
ax.text(8.85, 4.2, '--gpus=all', ha='center', va='center', 
        fontsize=14, color='#E65100', style='italic')

# Volume mount indicator
ax.text(3.58, 3.2, '/models', ha='center', va='center', 
        fontsize=14, color='#1565C0', style='italic')
ax.text(7.8, 3.2, '/models', ha='center', va='center', 
        fontsize=14, color='#1565C0', style='italic')

# Physical GPU Resources box
gpu_box = FancyBboxPatch(
    (0.6, 0.6), 7.0, 1.8,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor=gpu_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(gpu_box)
ax.text(4.1, 2.2, 'Physical GPU Resources', ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Individual GPUs
gpu_width = 0.8
gpu_height = 0.5
gpu_y = 1.2
gpus = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3', '...', 'GPU N']
gpu_positions = [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]

for i, (gpu, x_pos) in enumerate(zip(gpus, gpu_positions)):
    if gpu == '...':
        ax.text(x_pos, gpu_y, '...', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='black')
    else:
        gpu_rect = FancyBboxPatch(
            (x_pos - gpu_width/2, gpu_y - gpu_height/2), gpu_width, gpu_height,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor='#FFE0B2',
            edgecolor='#E65100',
            linewidth=1
        )
        ax.add_patch(gpu_rect)
        ax.text(x_pos, gpu_y, gpu, ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#E65100')

# Host Filesystem / kubectl box
fs_box = FancyBboxPatch(
    (8.0, 0.6), 3.5, 1.8,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor='#ECEFF1',
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(fs_box)
ax.text(9.75, 2.2, 'Host Access', ha='center', va='center', 
        fontsize=14, fontweight='bold')
ax.text(9.75, 1.7, '~/.kube/config', ha='center', va='center', 
        fontsize=14, color='#424242')
ax.text(9.75, 1.35, '/path/to/models', ha='center', va='center', 
        fontsize=14, color='#424242')
ax.text(9.75, 1.0, 'kubectl, docker', ha='center', va='center', 
        fontsize=14, color='#424242')

# Arrows for GPU passthrough
ax.annotate('', xy=(3., 4.8), xytext=(3., 2.5),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5,
                           connectionstyle='arc3,rad=0'))
ax.annotate('', xy=(8.65, 4.8), xytext=(5.5, 2.5),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5,
                           connectionstyle='arc3,rad=0.2'))

# Arrow between nodes (API communication)
ax.annotate('', xy=(6.5, 5.9), xytext=(5.5, 5.9),
            arrowprops=dict(arrowstyle='<->', color='#1976D2', lw=1.5))
ax.text(6.0, 6.15, 'API', ha='center', va='center', 
        fontsize=14, color='#1976D2')

# Legend
legend_y = 3.6
legend_items = [
    (1.5, legend_y, node_color, 'k3d Node'),
    (3.5, legend_y, gpu_color, 'GPU Resources'),
    (5.5, legend_y, k8s_color, 'K8s Workloads'),
]

for x, y, color, label in legend_items:
    box = FancyBboxPatch(
        (x - 0.2, y - 0.12), 0.25, 0.25,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1
    )
    ax.add_patch(box)
    ax.text(x + 0.1, y, label, ha='left', va='center', fontsize=14)

plt.tight_layout(pad=0.1)
save_figure(__file__)
