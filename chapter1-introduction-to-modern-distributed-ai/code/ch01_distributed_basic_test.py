"""
Basic distributed test to verify process group initialization works.

This script tests the fundamental distributed setup: process group initialization,
rank identification, and basic communication. It doesn't use DDP - it's just
testing that multiple processes can communicate.

Usage:
    OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 code/chapter1/ch01_distributed_basic_test.py

Expected output:
    Rank 0 says hello.
    Rank 1 says hello.
"""

import torch
import torch.distributed as dist

def test_distributed_setup():
    """Test basic distributed process group initialization and communication"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Rank {rank} says hello.")
    dist.destroy_process_group()

if __name__ == "__main__":
    test_distributed_setup()



