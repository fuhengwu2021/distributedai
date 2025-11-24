"""
First multi-GPU distributed training using PyTorch DDP.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import TensorDataset
import os

def setup(rank, world_size):
    """Initialize the process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """Training function for DDP"""
    setup(rank, world_size)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
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
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(dataloader):.4f}")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Warning: This script requires at least 2 GPUs")
        print("Running on single GPU for demonstration...")
        world_size = 1
    
    torch.multiprocessing.spawn(
        train_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
