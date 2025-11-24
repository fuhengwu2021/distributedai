#!/usr/bin/env bash
# Launch Chapter 1 multi-GPU example
# Run from the book root directory
torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_ddp.py
