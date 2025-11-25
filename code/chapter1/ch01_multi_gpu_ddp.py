"""
First multi-GPU distributed training using PyTorch DDP.

Usage:
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_ddp.py

Or use the launch script:
    bash distributedai/code/chapter1/ch01_launch_torchrun.sh
"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import TensorDataset
import os

def setup():
    """Initialize the process group using torchrun environment variables"""
    # torchrun sets these environment variables automatically
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), local_rank

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def train_ddp():
    """Run distributed training using PyTorch DDP"""
    rank, world_size, local_rank = setup()
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).cuda()
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Create dataset with distributed sampler
    dataset = TensorDataset(
        torch.randn(1000, 1000),
        torch.randint(0, 10, (1000,))
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Important for shuffling
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(dataloader):.4f}", flush=True)
    
    cleanup()

if __name__ == "__main__":
    train_ddp()
