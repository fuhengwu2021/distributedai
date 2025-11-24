#!/usr/bin/env bash
# Single-node, 4 GPUs
torchrun --nproc_per_node=4 train.py --epochs 10 --batch-size 32
