#!/usr/bin/env python3
"""
Distributed Training Example with PyTorch DDP and Slurm

This script demonstrates distributed training across 2 nodes (node6 and node7)
using PyTorch's DistributedDataParallel (DDP).

Usage:
    sbatch train_distributed_example.sh
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import argparse


class SimpleDataset(Dataset):
    """Simple synthetic dataset for demonstration"""
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_distributed():
    """Initialize distributed training environment"""
    # Get environment variables set by Slurm
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    # Get master address and port
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    # Initialize process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device


def train_one_epoch(model, dataloader, criterion, optimizer, device, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress (only on rank 0)
        if rank == 0 and batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Distributed Training Example')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        print(f"Starting distributed training")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Create model and move to device
    model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=1).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Create dataset and dataloader
    dataset = SimpleDataset(size=args.data_size, input_dim=10)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for sampler (important for shuffling)
        sampler.set_epoch(epoch)
        
        # Train one epoch
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device, rank)
        
        # Print epoch summary (only on rank 0)
        if rank == 0:
            print(f'Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # Synchronize all processes
        dist.barrier()
    
    # Save model (only on rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), 'model_distributed.pt')
        print(f"Model saved to model_distributed.pt")
    
    # Cleanup
    dist.destroy_process_group()
    
    if rank == 0:
        print("Training completed successfully!")


if __name__ == '__main__':
    main()
