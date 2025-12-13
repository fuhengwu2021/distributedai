# Quick Start Guide

## The Problem You Encountered

The job failed with: `ModuleNotFoundError: No module named 'torch'`

This is because PyTorch is not in the default Python environment when Slurm runs the job.

## Solution

Use the script that activates a conda environment with PyTorch:

```bash
cd /home/fuhwu/workspace/distributedai/code/chapter8

# Edit the script to use your conda environment
# Available environments with PyTorch: bm, calli, hunyuan, wan22
nano train_distributed_example_with_env.sh
# Change: conda activate wan22  (to your preferred environment)

# Submit the job
sbatch train_distributed_example_with_env.sh

# Monitor
squeue -u $USER
tail -f train_distributed_*.out
```

## Available Conda Environments with PyTorch

- `bm`: PyTorch 2.8.0+cu128
- `calli`: PyTorch 2.9.0+cu128  
- `hunyuan`: PyTorch 2.9.1+cu128
- `wan22`: PyTorch 2.9.1+cu128

## Alternative: Install PyTorch in Base Environment

If you prefer to use the base environment:

```bash
conda activate base
pip install torch
```

Then use `train_distributed_example.sh` (without _with_env).
