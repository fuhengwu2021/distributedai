"""
DDP Training Script v2 - Using a Larger Model to Test GPU Performance Advantages

This version uses a larger multi-layer neural network to better demonstrate GPU performance advantages in single-GPU multi-process simulation.

Usage (run from project root directory):
    # Simulate 2 GPUs on a single GPU, using GPU mode
    USE_GPU_IN_SINGLE_GPU_MODE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/chapter3/train_ddp_torchrun_v2.py
    
    # Simulate 2 GPUs on a single GPU, using CPU mode (for comparison)
    USE_GPU_IN_SINGLE_GPU_MODE=0 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/chapter3/train_ddp_torchrun_v2.py
    
    # If you have multiple GPUs, use all GPUs
    torchrun --nproc_per_node=4 code/chapter3/train_ddp_torchrun_v2.py
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


class LargeDataset(Dataset):
    def __init__(self, size=50000, input_dim=512):
        """Create a larger dataset for testing"""
        self.x = torch.randn(size, input_dim)
        # Create a non-linear target: y = sum(x^2) + noise
        self.y = (self.x ** 2).sum(dim=1, keepdim=True) + 0.1 * torch.randn(size, 1)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LargeModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024, 2048, 1024, 512], output_dim=1):
        """
        Create a larger multi-layer neural network
        Default parameters:
        - input_dim: 512
        - hidden layers: 1024 -> 2048 -> 1024 -> 512
        - output_dim: 1
        Total parameters approximately: 512*1024 + 1024*2048 + 2048*1024 + 1024*512 + 512*1 ≈ 6.5M parameters
        """
        super(LargeModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


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
    # For large models, GPU mode is usually much faster than CPU mode
    available_gpus = torch.cuda.device_count()
    
    # Choose single GPU simulation mode
    # Can force CPU usage by setting environment variable USE_GPU_IN_SINGLE_GPU_MODE=0
    use_gpu_in_single_gpu_mode = os.environ.get('USE_GPU_IN_SINGLE_GPU_MODE', '1') == '1'
    
    if available_gpus == 1:
        # Single GPU simulation mode: use Gloo backend (supports single GPU multi-process)
        backend = 'gloo'
        
        if use_gpu_in_single_gpu_mode:
            # Use Gloo + GPU (recommended for large models)
            actual_device_id = 0
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            mode_info = f"[Single GPU Simulation Mode (GPU+Gloo): {world_size} processes sharing GPU 0]"
            if rank == 0:
                print("Single GPU simulation: Using Gloo backend + GPU mode (recommended for large models)")
                print("                      All processes share GPU 0, GPU advantage is significant for large models")
        else:
            # Use Gloo + CPU (for comparison)
            device = torch.device('cpu')
            actual_device_id = None
            mode_info = f"[Single GPU Simulation Mode (CPU+Gloo): {world_size} processes simulating multi-GPU training on CPU]"
            if rank == 0:
                print("Single GPU simulation: Using Gloo backend + CPU mode (for performance comparison)")
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
    epochs = 5  # Reduced epochs since model is larger
    batch_size_per_gpu = 64  # Increased batch size to fully utilize GPU
    learning_rate = 0.001
    input_dim = 512
    dataset_size = 50000
    
    # Create dataset
    dataset = LargeDataset(size=dataset_size, input_dim=input_dim)
    
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
    model = LargeModel(input_dim=input_dim).to(device)
    
    # Calculate model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        print(f"\nModel Information:")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Input dimension: {input_dim}")
        print(f"  Dataset size: {dataset_size:,}")
    
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
    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)
    
    # Record training start time (only in main process)
    if rank == 0:
        print(f"\nStarting DDP training: world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device} {mode_info}")
        print(f"Batch size per GPU: {batch_size_per_gpu}, Total batch size: {batch_size_per_gpu * world_size}")
        print(f"Number of epochs: {epochs}")
        start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        # Set sampler epoch to ensure consistent shuffle across epochs
        sampler.set_epoch(epoch)
        
        running_loss = 0.0
        num_batches = 0
        
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
            num_batches += 1
            
            # Only print in main process
            if rank == 0 and (i + 1) % 50 == 0:
                avg_loss = running_loss / num_batches
                print(f'[{epoch + 1}, {i + 1}/{len(dataloader)}] loss: {avg_loss:.5f}')
    
    # Calculate total training time
    if rank == 0:
        total_time = time.time() - start_time
        print(f'\nDDP training completed, total time: {total_time:.2f} seconds')
        print(f'Average time per epoch: {total_time/epochs:.2f} seconds')
        
        # Print some model parameters (only first layer's first 5 values as example)
        print(f'\nModel parameter example (first 5 values of first layer weights):')
        first_layer_weight = list(model.named_parameters())[0][1]
        print(f'  {first_layer_weight.data[0, :5].cpu().numpy()}')
    
    # Cleanup process group
    dist.destroy_process_group()


if __name__ == '__main__':
    main()


"""
Example output:

$ USE_GPU_IN_SINGLE_GPU_MODE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/chapter3/train_ddp_torchrun_v2.py
Single GPU simulation: Using Gloo backend + GPU mode (recommended for large models)
                      All processes share GPU 0, GPU advantage is significant for large models
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1

Model Information:
  Total parameters: 5,248,001 (5.25M)
  Trainable parameters: 5,248,001
  Input dimension: 512
  Dataset size: 50,000

Starting DDP training: world_size=2, rank=0, local_rank=0, device=cuda:0 [Single GPU Simulation Mode (GPU+Gloo): 2 processes sharing GPU 0]
Batch size per GPU: 64, Total batch size: 128
Number of epochs: 5
[1, 50/391] loss: 28772.11514
[1, 100/391] loss: 14956.01543
[1, 150/391] loss: 10315.12192
[1, 200/391] loss: 7982.63838
[1, 250/391] loss: 6588.49506
[1, 300/391] loss: 5668.09185
[1, 350/391] loss: 5001.30517
[2, 50/391] loss: 986.53420
[2, 100/391] loss: 963.81267
[2, 150/391] loss: 960.80586
[2, 200/391] loss: 956.29564
[2, 250/391] loss: 959.67079
[2, 300/391] loss: 961.18708
[2, 350/391] loss: 970.91166
[3, 50/391] loss: 1038.79444
[3, 100/391] loss: 1005.54960
[3, 150/391] loss: 1009.57969
[3, 200/391] loss: 1050.07297
[3, 250/391] loss: 1038.98511
[3, 300/391] loss: 1017.42196
[3, 350/391] loss: 1007.66771
[4, 50/391] loss: 831.31287
[4, 100/391] loss: 995.74391
[4, 150/391] loss: 1054.04187
[4, 200/391] loss: 1072.20759
[4, 250/391] loss: 1025.21441
[4, 300/391] loss: 1001.62857
[4, 350/391] loss: 1047.37880
[5, 50/391] loss: 855.93906
[5, 100/391] loss: 830.19203
[5, 150/391] loss: 826.84724
[5, 200/391] loss: 835.06436
[5, 250/391] loss: 835.67413
[5, 300/391] loss: 858.37035
[5, 350/391] loss: 873.60756

DDP training completed, total time: 17.33 seconds
Average time per epoch: 3.47 seconds

Model parameter example (first 5 values of first layer weights):
  [ 0.05920285  0.06992833 -0.06980816  0.05822118 -0.01640381]

$ USE_GPU_IN_SINGLE_GPU_MODE=0 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/chapter3/train_ddp_torchrun_v2.py
Single GPU simulation: Using Gloo backend + CPU mode (for performance comparison)
                      To use GPU, set environment variable: USE_GPU_IN_SINGLE_GPU_MODE=1
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1

Model Information:
  Total parameters: 5,248,001 (5.25M)
  Trainable parameters: 5,248,001
  Input dimension: 512
  Dataset size: 50,000

Starting DDP training: world_size=2, rank=0, local_rank=0, device=cpu [Single GPU Simulation Mode (CPU+Gloo): 2 processes simulating multi-GPU training on CPU]
Batch size per GPU: 64, Total batch size: 128
Number of epochs: 5
[1, 50/391] loss: 28932.17227
[1, 100/391] loss: 15065.98116
[1, 150/391] loss: 10372.54478
[1, 200/391] loss: 8033.65060
[1, 250/391] loss: 6641.03007
[1, 300/391] loss: 5729.89617
[1, 350/391] loss: 5069.65996
[2, 50/391] loss: 1035.36540
[2, 100/391] loss: 1038.12470
[2, 150/391] loss: 1000.15533
[2, 200/391] loss: 1015.24815
[2, 250/391] loss: 1013.81194
[2, 300/391] loss: 1017.76764
[2, 350/391] loss: 1032.75030
[3, 50/391] loss: 1000.18230
[3, 100/391] loss: 943.71390
[3, 150/391] loss: 1016.36460
[3, 200/391] loss: 1022.63624
[3, 250/391] loss: 1011.57026
[3, 300/391] loss: 1020.26165
[3, 350/391] loss: 1016.26472
[4, 50/391] loss: 955.69782
[4, 100/391] loss: 892.73379
[4, 150/391] loss: 873.28538
[4, 200/391] loss: 914.33904
[4, 250/391] loss: 919.88503
[4, 300/391] loss: 912.24177
[4, 350/391] loss: 938.85826
[5, 50/391] loss: 913.90956
[5, 100/391] loss: 1012.92773
[5, 150/391] loss: 1009.03721
[5, 200/391] loss: 966.03360
[5, 250/391] loss: 986.41318
[5, 300/391] loss: 962.41697
[5, 350/391] loss: 946.46640

DDP training completed, total time: 63.83 seconds
Average time per epoch: 12.77 seconds

Model parameter example (first 5 values of first layer weights):
  [ 0.00797506  0.04807181 -0.03788038  0.08893362 -0.03416257]

Performance comparison:
- GPU mode: 17.33 seconds (3.7x faster)
- CPU mode: 63.83 seconds
"""
