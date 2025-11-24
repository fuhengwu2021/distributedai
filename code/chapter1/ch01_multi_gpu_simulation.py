"""
$ sudo nvidia-cuda-mps-control -d
$ CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 ddp_test.py
Rank 0 says hello.
Rank 1 says hello.
"""
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Rank {rank} says hello.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
