"""
DDP training script that supports torchrun launch and can simulate multi-GPU training on a single GPU.

Usage (run from project root directory):
    # Simulate 2 GPUs on a single GPU
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/chapter3/train_ddp_torchrun.py
    
    # Simulate 4 GPUs on a single GPU
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=4 code/chapter3/train_ddp_torchrun.py
    
    # If you have multiple GPUs, use all GPUs
    torchrun --nproc_per_node=4 code/chapter3/train_ddp_torchrun.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import time


class SimpleDataset(Dataset):
    def __init__(self, size=10000):
        """Create a simple linear dataset y = 2x + 3 + noise"""
        self.x = torch.randn(size, 1)
        self.y = 2 * self.x + 3 + 0.1 * torch.randn(size, 1)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)


def main():
    # torchrun automatically sets these environment variables
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Single GPU simulation: if only 1 GPU is visible, special setup is needed
    # Key findings:
    # - NCCL backend: DDP does not allow multiple processes to share the same GPU (will error)
    # - Gloo backend + GPU: Works! All processes can use the same GPU (cuda:0)
    # - Gloo backend + CPU: Also works, but slower
    # 
    # Recommendation: Use Gloo + GPU for single GPU simulation (faster than Gloo + CPU, and actually uses GPU)
    available_gpus = torch.cuda.device_count()
    
    # Choose single GPU simulation mode
    # Can force CPU usage by setting environment variable USE_GPU_IN_SINGLE_GPU_MODE=0
    use_gpu_in_single_gpu_mode = os.environ.get('USE_GPU_IN_SINGLE_GPU_MODE', '0') == '1'
    
    if available_gpus == 1:
        # Single GPU simulation mode: use Gloo backend (supports single GPU multi-process)
        backend = 'gloo'
        
        if use_gpu_in_single_gpu_mode:
            # Use Gloo + GPU
            # Note: For small models, GPU mode may be slower than CPU mode because:
            # 1. Multiple processes sharing GPU causes frequent context switching overhead
            # 2. Gloo communication efficiency on GPU is lower than NCCL
            # 3. Small model computation benefit may be less than data transfer and kernel launch overhead
            # For large models, GPU mode is usually faster
            actual_device_id = 0
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            mode_info = f"[Single GPU Simulation Mode (GPU+Gloo): {world_size} processes sharing GPU 0]"
            if rank == 0:
                print("Single GPU simulation: Using Gloo backend + GPU mode")
                print("                      Note: For small models, may be slower than CPU mode (context switching overhead)")
                print("                      For large models, GPU mode is usually faster")
        else:
            # Use Gloo + CPU
            # For small models, CPU mode may be faster because:
            # 1. No GPU context switching overhead
            # 2. CPU process switching is more efficient than GPU context switching
            # 3. Small model computation is insufficient to show GPU advantage
            device = torch.device('cpu')
            actual_device_id = None
            mode_info = f"[Single GPU Simulation Mode (CPU+Gloo): {world_size} processes simulating multi-GPU training on CPU]"
            if rank == 0:
                print("Single GPU simulation: Using Gloo backend + CPU mode")
                print("                      For small models, CPU mode may be faster (no GPU context switching overhead)")
                print("                      To use GPU, set environment variable: USE_GPU_IN_SINGLE_GPU_MODE=1")
    else:
        # Multi-GPU mode, use NCCL backend
        backend = 'nccl'
        actual_device_id = local_rank
        device = torch.device(f'cuda:{actual_device_id}')
        mode_info = f"[Multi-GPU Mode: {available_gpus} GPUs]"
        torch.cuda.set_device(actual_device_id)
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    # Set random seed (ensures same initial weights for each process, but different data)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    epochs = 10
    batch_size_per_gpu = 32
    learning_rate = 0.01
    
    # Create dataset
    dataset = SimpleDataset(size=10000)
    
    # Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=0  # Usually set to 0 in notebooks
    )
    
    # Create model and move to device
    model = SimpleModel().to(device)
    
    # Wrap model with DDP
    # Key: Gloo backend supports single GPU multi-process, all processes can use the same device
    if actual_device_id is not None:
        # GPU mode (Gloo + GPU or NCCL + multi-GPU)
        # For Gloo + GPU single GPU simulation, all processes use device_ids=[0]
        # For NCCL + multi-GPU, each process uses its corresponding device_id
        ddp_model = DDP(model, device_ids=[actual_device_id])
    else:
        # CPU mode (Gloo), don't use device_ids
        ddp_model = DDP(model)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate)
    
    # Record training start time (only in main process)
    if rank == 0:
        print(f"Starting DDP training: world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device} {mode_info}")
        print(f"Batch size per GPU: {batch_size_per_gpu}, Total batch size: {batch_size_per_gpu * world_size}")
        start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        # Set sampler epoch to ensure consistent shuffle across epochs
        sampler.set_epoch(epoch)
        
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to current device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            running_loss += loss.item()
            
            # Only print in main process
            if rank == 0 and i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.5f}')
                running_loss = 0.0
    
    # Calculate total training time
    if rank == 0:
        total_time = time.time() - start_time
        print(f'\nDDP training completed, total time: {total_time:.2f} seconds')
        
        # Print learned parameters
        for name, param in model.named_parameters():
            print(f'{name}: {param.item():.4f}')
    
    # Cleanup process group
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

