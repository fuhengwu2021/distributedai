import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "tokens": [
            "Coffee", "solves", "everything", "trust", "me", "<EOS>", "<RESV>", "..", "<RESV>",
            "", "New", "York", "City", "<EOS>", "<RESV>", "..", "<RESV>"
        ],
        "req_a_current": "Request A current iteration",
        "slots_never_used_internal": "Slots never used\n(Internal\nFragmentation)",
        "slots_never_used_red": "Slots never used",
        "req_b_prompt": "2 KV cache\nstates for\nrequest A's\nprompt",
        "req_b_reserved": "1 slot reserved\nfor future\ngenerations",
        "req_a_prompt": "3 KV cache states for\nrequest A's prompt",
        "req_a_reserved": "2 slots reserved\nfor future generations",
        "external_frag": "External\nfragmentation",
        "req_b_current": "Request B\ncurrent iteration",
        "slots_never_used_internal_b": "Slots never used\n(Internal\nFragmentation)",
    },
    "zh": {
        "tokens": [
            "Coffee", "solves", "everything", "trust", "me", "<EOS>", "<RESV>", "..", "<RESV>",
            "", "New", "York", "City", "<EOS>", "<RESV>", "..", "<RESV>"
        ],
        "req_a_current": "请求 A 当前迭代步",
        "slots_never_used_internal": "从未使用插槽\n(内部碎片)",
        "slots_never_used_red": "从未使用插槽",
        "req_b_prompt": "请求 B 提示词的\n2 个 KV Cache\n状态",
        "req_b_reserved": "为未来生成\n预留的 1 个插槽",
        "req_a_prompt": "请求 A 提示词的\n3 个 KV Cache 状态",
        "req_a_reserved": "为未来生成预留\n的 2 个插槽",
        "external_frag": "外部碎片",
        "req_b_current": "请求 B\n当前迭代步",
        "slots_never_used_internal_b": "从未使用插槽\n(内部碎片)",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Types: prompt, current, reserved, internal, external, eos
    types_and_styles = [
        ("prompt", "#e8f5e9", ""), 
        ("prompt", "#e8f5e9", ""), 
        ("prompt", "#e8f5e9", ""),
        ("current", "#fff3e0", ""),
        ("reserved", "#f3e5f5", "xx"),
        ("eos", "#f3e5f5", ""),
        ("internal", "#ffebee", "xx"),
        ("internal", "#ffebee", ""),
        ("internal", "#ffebee", "xx"),
        ("external", "white", "xx"),  # External fragmentation gap
        ("prompt", "#e8f5e9", ""),
        ("prompt", "#e8f5e9", ""),
        ("current", "#fff3e0", ""),
        ("eos", "#f3e5f5", "xx"),
        ("internal", "#ffebee", "xx"),
        ("internal", "#ffebee", ""),
        ("internal", "#ffebee", "xx"),
    ]

    tokens = text["tokens"]
    x_start = 0
    slot_width = 1.0
    slot_height = 0.6
    
    for i, (tok_str, (dtype, color, hatch)) in enumerate(zip(tokens, types_and_styles)):
        # Draw the rectangle
        rect = patches.FancyBboxPatch(
            (x_start + 0.1, 0.2), 0.8, slot_height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            linewidth=1, edgecolor='black', facecolor=color, hatch=hatch
        )
        ax.add_patch(rect)
        
        # Add text inside
        ax.text(x_start + 0.5, 0.5, tok_str, ha='center', va='center', fontsize=12)
        x_start += 1

    # --- Brackets and Annotations ---
    def draw_bracket(ax, x_start, x_end, y, label, is_top=True):
        direction = 1 if is_top else -1
        # Draw bracket line
        ax.annotate('', xy=(x_start + 0.1, y), xytext=(x_end - 0.1, y),
                    arrowprops=dict(arrowstyle='<->', connectionstyle=f"bar,fraction={0.2 * direction}"))
        # Add Label
        label_y = y + (0.3 * direction)
        ax.text((x_start + x_end)/2, label_y, label, ha='center', va='center', fontsize=11)

    # Top Annotations
    draw_bracket(ax, 3, 4, 0.9, text["req_a_current"], True)
    draw_bracket(ax, 6, 9, 0.9, text["slots_never_used_internal"], True)
    ax.text(7.5, 1.1, text["slots_never_used_red"], color='red', ha='center')  # Color red for emphasis
    draw_bracket(ax, 10, 12, 0.9, text["req_b_prompt"], True)
    draw_bracket(ax, 13, 14, 0.9, text["req_b_reserved"], True)

    # Bottom Annotations
    draw_bracket(ax, 0, 3, 0.1, text["req_a_prompt"], False)
    draw_bracket(ax, 4, 6, 0.1, text["req_a_reserved"], False)
    
    # External Fragmentation Label
    ax.text(9.5, 0.05, text["external_frag"], color='red', ha='center', va='top')
    ax.annotate('', xy=(9.1, 0.2), xytext=(9.9, 0.2), arrowprops=dict(arrowstyle='<->', connectionstyle="bar,fraction=-0.4"))

    draw_bracket(ax, 12, 13, 0.1, text["req_b_current"], False)
    draw_bracket(ax, 14, 17, 0.1, text["slots_never_used_internal_b"], False)

    # Final styling
    ax.set_xlim(-0.5, len(tokens) + 0.5)
    ax.set_ylim(-0.8, 1.8)
    ax.axis('off')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "pa", LABELS, __file__)