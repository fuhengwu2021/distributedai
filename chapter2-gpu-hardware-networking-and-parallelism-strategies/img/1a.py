"""
PCIe-only Multi-GPU Server Topology Diagram

This diagram shows a server topology where GPUs are connected only via PCIe,
without NVLink or NVSwitch. All GPU communication must go through the CPU.
"""

import os
import sys
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.compute import Server

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure


class GPU(Server):
    """Logical GPU node (icon reuse)"""
    pass


script_dir = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    "en": {
        "title": "PCIe-only Multi-GPU Server Topology",
        "cpu": "CPU\n(Host + NUMA)",
        "cluster": "PCIe Root Complex",
        "gpu_prefix": "GPU ",
        "edge_label": "PCIe Gen4/5",
        "out_base": os.path.join(script_dir, "1a"),
    },
    "zh": {
        "title": "仅依赖 PCIe 的多 GPU 服务器拓扑",
        "cpu": "CPU\n(Host + NUMA)",
        "cluster": "PCIe 根复合体 (Root Complex)",
        "gpu_prefix": "GPU ",
        "edge_label": "PCIe Gen4/5",
        "out_base": os.path.join(script_dir, "1a_zh"),
    }
}


def draw(text: dict):
    with Diagram(
        text["title"],
        show=False,
        filename=text["out_base"],
        direction="TB",
    ):
        cpu = Server(text["cpu"])

        with Cluster(text["cluster"]):
            gpus = [
                GPU(f"{text['gpu_prefix']}{i}")
                for i in range(4)
            ]

        for gpu in gpus:
            cpu >> Edge(label=text["edge_label"]) >> gpu
    return None


if __name__ == '__main__':
    localized_figure(draw, "1a", LABELS, __file__)
