"""
FSDP2 training example - can run on 2 GPUs to demonstrate sharding

Usage:
    torchrun --nproc_per_node=2 code/chapter4/train_fsdp2.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data.distributed import DistributedSampler
import os


class LargeModel(nn.Module):
    """A model that's too large for a single GPU"""
    def __init__(self, hidden_dim=4096, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class SimpleDataset(Dataset):
    def __init__(self, size=10000, dim=4096):
        self.x = torch.randn(size, dim)
        self.y = (self.x.sum(dim=1, keepdim=True) > 0).float()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Create device mesh
    mesh = init_device_mesh("cuda", (world_size,))
    
    if rank == 0:
        print(f"Training with {world_size} GPUs using FSDP2")
        print(f"Model will be sharded across {world_size} ranks")
    
    # Create model - this is intentionally large
    model = LargeModel(hidden_dim=4096, num_layers=8).to(device)
    
    # Count parameters before FSDP
    total_params_before = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {total_params_before:,} ({total_params_before/1e6:.2f}M)")
    
    # Apply FSDP2
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )
    
    # Create dataset and dataloader
    dataset = SimpleDataset(size=10000, dim=4096)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, dtype=torch.bfloat16)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    if rank == 0:
        print("Training completed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

