"""
Visualize the comparison of large AI models using matplotlib.
This script creates a scatter plot with Year on X-axis and Parameters on Y-axis,
with model names as text labels for each point.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
import os

# Table data: [Model Name, Parameters, Company, Year, Release Date (decimal year)]
# Release dates converted to decimal years for precise plotting
# Format: year + (month-1)/12 + (day-1)/365.25
data = [
    ['ViT-22B', '22B', 'Google', 2023.25, '2023-03-31'],  # March 31, 2023
    ['Sora', '30B', 'OpenAI', 2024.08, '2024-02'],  # February 2024 (preview), Dec 2024 (public)
    ['Grok-1', '314B', 'xAI', 2023.84, '2023-11-03'],  # November 3, 2023
    ['Gemini-1', '1.6T', 'Google', 2023.93, '2023-12-06'],  # December 6, 2023
    ['LLaMA-2', '700B', 'Meta', 2023.54, '2023-07-18'],  # July 18, 2023
    ['PanGu-Σ', '1.085T', 'Huawei', 2023.25, '2023-04'],  # April 2023
    ['DeepSeek-V1', '6.7B', 'DeepSeek', 2023.83, '2023-11'],  # November 2023
    ['Claude-3', 'undisclosed (~>1T MoE est.)', 'Anthropic', 2024.17, '2024-03-04'],  # March 4, 2024
    ['GPT-4o', '1.8T', 'OpenAI', 2024.33, '2024-05'],  # May 2024 (GPT-4V in table, but GPT-4o is the actual model)
    ['DeepSeek-V2', '236B MoE (16 experts, 2 active)', 'DeepSeek', 2024.34, '2024-05-06'],  # May 6, 2024
    ['Grok-4*', '1.7T (MoE)', 'xAI', 2025.52, '2025-07-09'],  # July 9, 2025
    ['Qwen-Max*', '1.2T', 'Alibaba', 2025.67, '2025-09-05'],  # September 5, 2025 (Qwen3-Max)
    ['GPT-4.5', 'undisclosed', 'OpenAI', 2025.16, '2025-02-27'],  # February 27, 2025
    ['GPT-5*', '4T', 'OpenAI', 2025.59, '2025-08-07'],  # August 7, 2025
    ['DeepSeek-V3', '671B MoE (64 experts, 8 active)', 'DeepSeek', 2024.98, '2024-12-26'],  # December 26, 2024
    ['Gemini-3-Pro*', '7.5T', 'Google', 2025.88, '2025-11-18'],  # November 18, 2025
]

def parse_parameters(param_str):
    """
    Parse parameter string to numeric value.
    Handles formats like: '22B', '1.6T', '~1.7T', '>1T', '~2–5T', 'undisclosed'
    Returns None if cannot parse.
    """
    # First, try to extract any number with unit from the string
    # This handles cases like "undisclosed (~>1T MoE est.)"
    
    # Handle ranges like "~2–5T" or "2-5T"
    range_match = re.search(r'~?(\d+\.?\d*)\s*[–-]\s*(\d+\.?\d*)\s*T', param_str, re.IGNORECASE)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return ((low + high) / 2) * 1e12
    
    # Remove extra text in parentheses, but keep the main number
    # Split by '(' to remove MoE details, etc.
    main_part = param_str.split('(')[0].strip()
    
    # Try to find number with unit (B, T, etc.)
    # Pattern: optional ~ or >, number, optional unit
    match = re.search(r'[~>]?\s*(\d+\.?\d*)\s*([BMKT])', main_part, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        
        multipliers = {
            'B': 1e9,
            'M': 1e6,
            'K': 1e3,
            'T': 1e12
        }
        
        return value * multipliers[unit]
    
    # If no match found, return None
    return None

def format_parameter_label(param_value):
    """Format parameter value as a string label (e.g., 30e9 -> '30B', 1.6e12 -> '1.6T')."""
    if param_value >= 1e12:
        return f'{param_value/1e12:.1f}T'.rstrip('0').rstrip('.')
    elif param_value >= 1e9:
        return f'{param_value/1e9:.0f}B'
    elif param_value >= 1e6:
        return f'{param_value/1e6:.0f}M'
    else:
        return f'{param_value/1e3:.0f}K'

def create_model_comparison_plot():
    """Create a scatter plot with Year on X-axis and Parameters on Y-axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = []
    parameters = []
    model_names = []
    companies = []
    param_labels = []  # Store formatted parameter labels
    
    # Parse data
    for row in data:
        model_name = row[0]
        param_str = row[1]
        company = row[2]
        year_decimal = row[3]  # Use decimal year for precise plotting
        release_date = row[4]  # Keep original date string for reference
        
        param_value = parse_parameters(param_str)
        
        if param_value is not None:
            years.append(year_decimal)
            parameters.append(param_value)
            model_names.append(model_name)
            companies.append(company)
            param_labels.append(format_parameter_label(param_value))
    
    # Create scatter plot with different colors per year
    year_colors = {2023: '#1f77b4', 2024: '#ff7f0e', 2025: '#2ca02c'}
    colors = [year_colors[int(year)] for year in years]
    
    scatter = ax.scatter(years, parameters, s=150, alpha=0.7, c=colors, 
                        edgecolors='black', linewidths=1.5, zorder=3)
    
    # Add text labels for each point with smart positioning
    for i, (year, param, name, param_label) in enumerate(zip(years, parameters, model_names, param_labels)):
        # Check if name ends with * (indicating estimation)
        is_estimate = name.endswith('*')
        display_name = name.rstrip('*')  # Remove asterisk for display
        
        # Check if parameter number is already in the model name (without asterisk)
        # Look for common patterns: number followed by B/T/M/K
        param_in_name = re.search(r'\d+\.?\d*\s*[BMKT]', display_name, re.IGNORECASE) is not None
        
        # Create label: add parameter if not already in name
        if param_in_name:
            label = display_name
        else:
            # Add "~" prefix for estimates, or "(est.)" suffix
            if is_estimate:
                label = f'{display_name} (~{param_label})'
            else:
                label = f'{display_name} ({param_label})'
        
        # Alternate label position to reduce overlap
        offset_x = 8 if i % 2 == 0 else -8
        offset_y = 8 if i % 3 == 0 else -8
        
        ax.annotate(label, (year, param), 
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, ha='left' if offset_x > 0 else 'right',
                   va='bottom' if offset_y > 0 else 'top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.85, edgecolor=colors[i], linewidth=1),
                   zorder=4)
    
    # Set labels and title
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameters', fontsize=12, fontweight='bold')
    ax.set_title('Model Parameters v.s. Year', fontsize=15, pad=10)
    
    # Format Y-axis to show in billions/trillions (log scale)
    ax.set_yscale('log')
    
    def format_y_axis(value, pos):
        if value >= 1e12:
            return f'{value/1e12:.1f}T'
        elif value >= 1e9:
            return f'{value/1e9:.0f}B'
        else:
            return f'{value/1e6:.0f}M'
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))
    
    # Set X-axis ticks with year labels
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xlim(2022.8, 2026.0)  # Extended to include late 2025 models
    
    # Add minor ticks for months (optional, for better granularity)
    ax.xaxis.set_minor_locator(MultipleLocator(1/12))  # Monthly minor ticks
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add note about precise dates
    #ax.text(0.02, 0.98, 'Release dates shown with month precision', 
    #        transform=ax.transAxes, fontsize=8, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'model_comparison_plot.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the figure
    plt.show()

if __name__ == '__main__':
    create_model_comparison_plot()
