"""
Single-GPU simulation of multi-GPU distributed training.

This script allows you to test distributed training code on a single GPU by simulating
multiple processes. It's useful for:
- Testing distributed code logic without multiple GPUs
- Debugging distributed training issues on a single-GPU machine
- Learning how distributed training works

The script can optionally use CUDA Multi-Process Service (MPS) to efficiently share
a single GPU among multiple processes, reducing context switching overhead. MPS is
optional - the script works without it, but performance may be better with MPS enabled.

Usage (run from the book root directory):
    # Option 1: With MPS (recommended for better performance)
    # Step 1: Start MPS daemon (only needed once, requires sudo)
    sudo nvidia-cuda-mps-control -d
    
    # Step 2: Run the simulation (2 processes on GPU 0)
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_simulation.py
    
    # Option 2: Without MPS (works but may have higher context switching overhead)
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_simulation.py

Expected output:
    Rank 0 says hello.
    Rank 1 says hello.

Note: MPS (Multi-Process Service) is optional but recommended. It allows multiple CUDA
      processes to share a single GPU more efficiently by reducing context switching overhead.
      Without MPS, the script will still work correctly, but you may experience:
      - Higher GPU context switching overhead
      - Slightly slower performance
      - More GPU memory fragmentation
"""
import torch
import torch.distributed as dist

def simulate_multi_gpu():
    """Simulate multi-GPU distributed training on a single GPU"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Rank {rank} says hello.")
    dist.destroy_process_group()

if __name__ == "__main__":
    simulate_multi_gpu()
