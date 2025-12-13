#!/usr/bin/env bash
# Launch Chapter 1 multi-GPU example
# Run from the book root directory

# Set OMP_NUM_THREADS to control OpenMP threads per process
# This prevents oversubscription of CPU cores when using multiple GPUs
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_ddp.py
