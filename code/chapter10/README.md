# Chapter 10: Distributed Benchmarking and Performance Optimization - Code Examples

This directory contains code examples for benchmarking distributed training and inference systems.

## Files

- `ch10_genai_bench.py` - Inference benchmarking using genai-bench
- `ch10_scaling_efficiency.py` - Measure scaling efficiency across GPU counts
- `ch10_network_diagnostics.py` - Network bandwidth and health diagnostics

## Prerequisites

```bash
pip install genai-bench torch torchvision numpy
```

## Running Examples

### genai-bench Benchmarking
```bash
# Requires inference server running on localhost:8000
python ch10_genai_bench.py
```

### Scaling Efficiency
```bash
# Requires model and dataloader to be provided
# Adapt the script with your model before running
python ch10_scaling_efficiency.py
```

### Network Diagnostics
```bash
# Run with torchrun for distributed testing
torchrun --nproc_per_node=2 ch10_network_diagnostics.py
```

