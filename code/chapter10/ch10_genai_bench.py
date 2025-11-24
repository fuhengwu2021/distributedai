"""
Comprehensive inference benchmarking using genai-bench CLI.
Note: genai-bench is a CLI tool, not a Python API.
"""
import subprocess
import json
import os
from pathlib import Path

def run_genai_benchmark(
    api_base: str,
    api_key: str,
    model_name: str,
    tokenizer_path: str,
    max_requests: int = 1000,
    max_time_minutes: int = 15,
    concurrency: int = 100,
    traffic_scenario: str = "D(100,100)",
    server_engine: str = "vLLM",
    server_gpu_type: str = "H100"
):
    """Run genai-bench benchmark via CLI"""
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "openai",
        "--api-base", api_base,
        "--api-key", api_key,
        "--api-model-name", model_name,
        "--model-tokenizer", tokenizer_path,
        "--task", "text-to-text",
        "--max-time-per-run", str(max_time_minutes),
        "--max-requests-per-run", str(max_requests),
        "--num-concurrency", str(concurrency),
        "--traffic-scenario", traffic_scenario,
        "--server-engine", server_engine,
        "--server-gpu-type", server_gpu_type
    ]
    
    print("Running genai-bench benchmark...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    print("Benchmark completed successfully!")
    print(result.stdout)
    
    # Results are saved in experiments/ folder
    # You can parse the results from the experiment folder
    return result

def analyze_experiment_results(experiment_folder: str):
    """Analyze results from genai-bench experiment"""
    # Generate Excel report
    excel_cmd = [
        "genai-bench", "excel",
        "--experiment-folder", experiment_folder,
        "--excel-name", "benchmark_results",
        "--metric-percentile", "mean"
    ]
    
    subprocess.run(excel_cmd)
    
    # Generate plots
    plot_cmd = [
        "genai-bench", "plot",
        "--experiments-folder", experiment_folder,
        "--group-key", "traffic_scenario",
        "--preset", "2x4_default"
    ]
    
    subprocess.run(plot_cmd)

if __name__ == "__main__":
    # Example usage
    result = run_genai_benchmark(
        api_base="http://localhost:8000",
        api_key="your-api-key",
        model_name="llama-2-7b-chat",
        tokenizer_path="/path/to/tokenizer",
        max_requests=1000,
        max_time_minutes=15,
        concurrency=100,
        traffic_scenario="D(100,100)"
    )
    
    # Analyze results (experiment folder path from genai-bench output)
    # analyze_experiment_results("./experiments/your_experiment_folder")

