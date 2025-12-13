"""
Visualize the mismatch between exponential model growth and linear GPU growth.
Two subplots: (1) Model parameters over time (exponential), (2) GPU memory over time (linear).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Model data: [Year, Parameters (in billions), Model Name]
# Using representative models from the comparison table
model_data = [
    [2020, 175, 'GPT-3'],      # GPT-3 (175B)
    [2022, 540, 'PaLM'],       # PaLM (540B)
    [2023.25, 22, 'ViT-22B'],  # ViT-22B (22B)
    [2023.54, 700, 'LLaMA-2'], # LLaMA-2 (700B)
    [2023.93, 1600, 'Gemini-1'], # Gemini-1 (1.6T = 1600B)
    [2024.33, 1800, 'GPT-4o'], # GPT-4o (1.8T = 1800B)
    [2024.98, 671, 'DeepSeek-V3'], # DeepSeek-V3 (671B)
    [2025.59, 4000, 'GPT-5*'], # GPT-5* (4T = 4000B)
    [2025.88, 7500, 'Gemini-3-Pro*'], # Gemini-3-Pro* (7.5T = 7500B)
]

# GPU memory data: [Year, Memory (GB), GPU Name]
gpu_data = [
    [2016, 16, 'P100'],   # P100
    [2017, 16, 'V100'],   # V100
    [2020, 40, 'A100-40'], # A100 (40GB variant)
    [2020, 80, 'A100-80'], # A100 (80GB variant)
    [2022, 80, 'H100'],   # H100
    [2024, 141, 'H200'],  # H200
    [2025, 192, 'B200'],  # B200 (estimated)
]

def create_growth_mismatch_plot():
    """Create a two-subplot figure showing exponential vs linear growth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Model Parameters (Exponential Growth)
    years_models = [row[0] for row in model_data]
    params_b = [row[1] for row in model_data]  # Already in billions
    model_names = [row[2] for row in model_data]
    
    ax1.scatter(years_models, params_b, s=100, alpha=0.7, color='#d62728', 
                edgecolors='black', linewidths=1.5, zorder=3)
    
    # Add text labels for each model
    for i, (year, param, name) in enumerate(zip(years_models, params_b, model_names)):
        # Alternate label position to reduce overlap
        offset_x = 8 if i % 2 == 0 else -8
        offset_y = 8 if i % 3 == 0 else -8
        
        ax1.annotate(name, (year, param),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=8, ha='left' if offset_x > 0 else 'right',
                    va='bottom' if offset_y > 0 else 'top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor='#d62728', linewidth=1),
                    zorder=4)
    
    # Fit exponential trend line to actual data
    # Exponential fit: log(y) = a + b*x, so y = exp(a) * exp(b*x)
    log_params = np.log(params_b)
    coeffs = np.polyfit(years_models, log_params, 1)
    x_fit = np.linspace(2020, 2026, 100)
    y_fit = np.exp(coeffs[1] + coeffs[0] * x_fit)
    ax1.plot(x_fit, y_fit, '--', color='#d62728', alpha=0.5, linewidth=2, 
             label='Exponential trend')
    
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Model Parameters (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Size Growth (Exponential)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(2019.5, 2026.2)
    
    # Format Y-axis (values are in billions)
    def format_params(value, pos):
        if value >= 1000:
            return f'{value/1000:.1f}T'  # Trillions
        elif value >= 1:
            return f'{value:.0f}B'  # Billions
        else:
            return f'{value:.1f}B'
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_params))
    
    # Subplot 2: GPU Memory (Linear Growth)
    years_gpu = [row[0] for row in gpu_data]
    memory_gb = [row[1] for row in gpu_data]
    gpu_names = [row[2] for row in gpu_data]
    
    ax2.scatter(years_gpu, memory_gb, s=100, alpha=0.7, color='#2ca02c', 
                edgecolors='black', linewidths=1.5, zorder=3)
    
    # Add text labels for each GPU
    for i, (year, memory, name) in enumerate(zip(years_gpu, memory_gb, gpu_names)):
        # Alternate label position to reduce overlap
        offset_x = 8 if i % 2 == 0 else -8
        offset_y = 8 if i % 3 == 0 else -8
        
        # Create label with memory info
        label = f'{name}\n{memory}GB'
        
        ax2.annotate(label, (year, memory),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=8, ha='left' if offset_x > 0 else 'right',
                    va='bottom' if offset_y > 0 else 'top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor='#2ca02c', linewidth=1),
                    zorder=4)
    
    # Fit linear trend line using actual data points
    x_fit_gpu = np.linspace(2016, 2026, 100)
    # Linear fit: y = a + b * x
    # Using numpy polyfit for accurate linear regression
    coeffs = np.polyfit(years_gpu, memory_gb, 1)
    y_fit_gpu = np.polyval(coeffs, x_fit_gpu)
    y_fit_gpu = np.maximum(y_fit_gpu, 0)  # Ensure non-negative
    ax2.plot(x_fit_gpu, y_fit_gpu, '--', color='#2ca02c', alpha=0.5, linewidth=2,
             label='Linear trend')
    
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
    ax2.set_title('Single-GPU Memory Growth (Linear)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(2015.5, 2026.2)
    ax2.set_ylim(0, 220)
    
    plt.tight_layout()
    
    # Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'growth_mismatch.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the figure
    plt.show()

if __name__ == '__main__':
    create_growth_mismatch_plot()
