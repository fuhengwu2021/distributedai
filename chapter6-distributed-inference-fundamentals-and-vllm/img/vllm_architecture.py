import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math4ai import save_figure


def draw_vllm_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    box_width = 2.5
    box_height = 0.9
    center_x = 5

    # Colors
    scheduler_color = '#e3f2fd'
    executor_color = '#fff3e0'
    worker_color = '#e8f5e9'
    arrow_color = '#546e7a'

    # Scheduler box
    scheduler_y = 6.5
    scheduler = patches.FancyBboxPatch(
        (center_x - box_width / 2, scheduler_y), box_width, box_height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2, edgecolor='#1976d2', facecolor=scheduler_color
    )
    ax.add_patch(scheduler)
    ax.text(center_x, scheduler_y + box_height / 2, 'Scheduler',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Executor box
    executor_y = 4.5
    executor = patches.FancyBboxPatch(
        (center_x - box_width / 2, executor_y), box_width, box_height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2, edgecolor='#f57c00', facecolor=executor_color
    )
    ax.add_patch(executor)
    ax.text(center_x, executor_y + box_height / 2, 'Executor',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Worker boxes (3 workers)
    worker_y = 2.0
    worker_width = 2.0
    worker_positions = [2.0, 5.0, 8.0]
    worker_labels = ['Worker 0\n(GPU 0)', 'Worker 1\n(GPU 1)', 'Worker N\n(GPU N)']

    for i, (wx, label) in enumerate(zip(worker_positions, worker_labels)):
        worker = patches.FancyBboxPatch(
            (wx - worker_width / 2, worker_y), worker_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=2, edgecolor='#388e3c', facecolor=worker_color
        )
        ax.add_patch(worker)
        ax.text(wx, worker_y + box_height / 2, label,
                ha='center', va='center', fontsize=11, fontweight='bold')

    # Ellipsis between workers
    ax.text(6.5, worker_y + box_height / 2, '...', ha='center', va='center',
            fontsize=20, fontweight='bold', color='#666')

    # Arrows: Scheduler -> Executor
    ax.annotate('', xy=(center_x, executor_y + box_height),
                xytext=(center_x, scheduler_y),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    # Arrows: Executor -> Workers (fan out)
    for wx in worker_positions:
        ax.annotate('', xy=(wx, worker_y + box_height),
                    xytext=(center_x, executor_y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5,
                                    connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    save_figure(__file__)


if __name__ == '__main__':
    draw_vllm_architecture()
    print('Saved vllm_architecture.png')
