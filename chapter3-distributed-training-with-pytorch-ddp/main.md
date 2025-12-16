# Chapter 3 — Distributed Training with PyTorch DDP

When you're training large models, a single GPU isn't enough. You need to distribute the work across multiple GPUs, and that's where PyTorch's DistributedDataParallel (DDP) comes in. DDP is the workhorse of distributed training—it's what most production training pipelines use, and understanding how it works is essential for building scalable AI systems.

This chapter is a hands-on guide to using DDP for multi-GPU and multi-node training. We'll cover DDP internals, how to initialize process groups, common failure modes and debugging techniques, and practical optimization strategies. Every section includes runnable examples you can adapt for your own workloads.

## 1. How DDP Works Internally

Before diving into code, let's understand what DDP actually does. When you wrap a model with `DistributedDataParallel`, you're telling PyTorch to replicate the model across multiple processes (typically one per GPU) and synchronize gradients during training. The magic happens during the backward pass—DDP automatically aggregates gradients from all processes and ensures every process has the same updated parameters.

### The Basic Flow

Here's what happens during a single training step with DDP:

1. **Forward pass**: Each process runs forward on its own data shard. The model is identical on all processes, but each process sees different data thanks to `DistributedSampler`.

2. **Backward pass**: Each process computes gradients locally. This is where DDP kicks in—instead of each process updating its model independently, DDP collects all gradients.

3. **Gradient synchronization**: DDP uses AllReduce (via NCCL on GPUs) to sum gradients across all processes. After AllReduce completes, every process has the same averaged gradients.

4. **Parameter update**: Each process applies the optimizer step using the synchronized gradients. Since all processes started with the same parameters and applied the same gradients, they end up with identical parameters.

This is **data parallelism**: the model is replicated, but data is sharded. Each GPU processes a different batch, and gradients are averaged. Compare this to **model parallelism** (covered in later chapters) where the model itself is split across GPUs.

### Data Parallel (DP) vs Distributed Data Parallel (DDP)

Before DDP, PyTorch had `DataParallel` (DP), which is still available but largely superseded by DDP. Understanding the differences helps explain why DDP is the preferred choice for distributed training.

**DataParallel (DP)** uses a single-process, multi-threaded approach. It runs on a single machine and can only use GPUs on that machine. Here's how it works:

1. **Forward pass**: The model is replicated on each GPU, and the mini-batch is split across GPUs. Each GPU processes its portion of the data.

2. **Gradient collection**: After backward pass, gradients from all GPUs are collected on GPU 0 (the main GPU).

3. **Parameter update**: GPU 0 updates model parameters, then broadcasts updated parameters to all other GPUs.

DP has several limitations:

- **Python GIL bottleneck**: DP uses multi-threading, which is limited by Python's Global Interpreter Lock (GIL). This prevents true parallelism and limits CPU utilization.

- **Single GPU bottleneck**: All gradient accumulation and parameter updates happen on GPU 0, creating an imbalance where GPU 0 is heavily utilized while other GPUs sit idle.

- **Single-machine only**: DP can't scale across multiple machines, limiting the maximum number of GPUs you can use.

- **Communication overhead**: Gradients must be transferred to GPU 0 and parameters broadcast back, creating communication bottlenecks.

**DistributedDataParallel (DDP)** addresses these limitations:

- **Multi-process architecture**: DDP uses separate processes (one per GPU), avoiding Python GIL limitations and enabling true parallelism.

- **Multi-machine support**: DDP can scale across multiple machines connected via network, enabling training on hundreds or thousands of GPUs.

- **Efficient communication**: DDP uses optimized collective communication (Ring AllReduce, tree algorithms) that distribute work across all GPUs, not just GPU 0.

- **Computation-communication overlap**: DDP overlaps gradient synchronization with computation, hiding communication latency.

- **Balanced workload**: All GPUs participate equally in gradient synchronization, eliminating the single-GPU bottleneck.

For these reasons, DDP is the standard for distributed training. DP is mainly useful for simple single-machine multi-GPU scenarios, but even then, DDP usually performs better.

### Gradient Bucketing: Why It Matters

If DDP synchronized every gradient tensor individually, you'd have thousands of small AllReduce operations. Each AllReduce has overhead—network latency, kernel launch overhead, synchronization costs. The solution is **gradient bucketing**: DDP groups small gradient tensors into buckets and performs AllReduce on entire buckets.

Here's how it works: DDP analyzes your model's parameter order (the order they appear in `model.parameters()`). It groups consecutive parameters into buckets based on size. When backward pass reaches a bucket boundary, DDP triggers an AllReduce for that bucket. The default bucket size is 25 MB, but you can tune it with `bucket_cap_mb`.

Bucketing reduces communication overhead, but there's a tradeoff: larger buckets mean fewer AllReduce calls (less overhead) but later synchronization (gradients aren't available until the bucket is ready). Smaller buckets mean earlier synchronization but more overhead. For most models, the default 25 MB works well, but you might tune it for very large or very small models.

### Communication-Computation Overlap

The real performance win comes from **overlapping communication and computation**. While DDP is doing AllReduce for one bucket, your backward pass can continue computing gradients for the next bucket. This hides communication latency behind computation.

DDP achieves overlap by:

- Launching AllReduce operations asynchronously
- Using CUDA streams to overlap communication kernels with compute kernels
- Processing buckets in the order they're ready (not necessarily parameter order)

For overlap to work, you need enough computation between bucket boundaries. If your model has very few parameters or very small layers, there might not be enough work to overlap. In that case, you'll see communication time dominate, and overlap won't help much.

You can check if overlap is working by profiling. If you see AllReduce operations happening concurrently with backward compute, overlap is working. If AllReduce happens sequentially after all gradients are computed, overlap isn't happening (maybe your model is too small, or there's a synchronization point blocking it).

### The AllReduce Operation

AllReduce is the core collective operation that makes DDP work. It takes gradients from all processes, sums them, and distributes the result back to all processes. On GPUs, DDP uses NCCL (NVIDIA Collective Communications Library) to implement AllReduce efficiently.

NCCL uses different algorithms depending on the number of GPUs and topology:

- **Ring AllReduce**: For small numbers of GPUs or when topology is a ring. Each GPU sends data to its neighbor in a ring, and after multiple steps, all GPUs have the sum.
- **Tree AllReduce**: For larger numbers of GPUs. Data flows up a tree to a root, then back down. Fewer steps than ring, but more complex.
- **NVLink-optimized**: When GPUs are connected via NVLink, NCCL uses topology-aware algorithms that minimize cross-node communication.

You don't need to choose the algorithm—NCCL picks it automatically based on your hardware topology. But understanding that different algorithms exist helps when debugging performance. If you're seeing slow AllReduce, it might be because NCCL picked a suboptimal algorithm for your topology, or because network bandwidth is saturated.

### Mixed Precision and Gradient Scaling

When using mixed precision training (FP16/BF16), gradients can underflow (become zero) because FP16 has limited range. The solution is **gradient scaling**: multiply loss by a scale factor before backward, then unscale gradients before optimizer step.

DDP works with PyTorch's Automatic Mixed Precision (AMP). The flow is:
1. Scale loss: `loss = loss * scale`
2. Backward: `loss.backward()` (gradients are also scaled)
3. DDP AllReduce: Synchronizes scaled gradients
4. Unscale: Divide gradients by scale before optimizer step
5. Update scale: Adjust scale factor based on gradient overflow detection

The key point: DDP synchronizes gradients **after** they're scaled. Each process scales its own gradients, then DDP sums the scaled gradients. After AllReduce, all processes have the same scaled gradients, which are then unscaled before the optimizer step.

If you're using AMP with DDP, make sure to use `GradScaler` correctly. The scaler must be created before wrapping the model with DDP, and you must call `scaler.step()` and `scaler.update()` on all processes (not just rank 0).

### Buffer Synchronization

DDP doesn't just synchronize gradients—it also synchronizes **buffers**. Buffers are model parameters that don't require gradients, like BatchNorm running mean and variance. During forward pass, DDP broadcasts buffers from rank 0 to all other ranks to ensure consistency.

This happens automatically, but there's a performance consideration: buffer synchronization adds communication overhead. If your model has many buffers or large buffers, this can slow down training. You can disable it with `broadcast_buffers=False`, but only if you're sure buffers don't need synchronization (e.g., you're not using BatchNorm, or you're manually synchronizing buffers).

For most models, keeping `broadcast_buffers=True` (the default) is the right choice. BatchNorm and similar layers need synchronized statistics to work correctly in distributed training.

### DDP Forward Pass Implementation Details

Understanding how DDP maintains model consistency during forward pass helps when debugging. In PyTorch, all models inherit from `torch.nn.Module`, which maintains two key dictionaries:

- **`_parameters`**: Network parameters that require gradients
- **`_buffers`**: Non-parameter data that persists (e.g., BatchNorm's running mean and variance)

DDP ensures model consistency through `_sync_module_states`, which synchronizes both `_parameters` and `_buffers` across all processes. This happens in two places:

1. **During DDP initialization**: When you create a DDP model, it synchronizes initial parameters and buffers from rank 0 to all other ranks.

2. **Before each forward pass**: If `broadcast_buffers=True` (the default), DDP synchronizes buffers before forward pass to ensure all processes have the same buffer values.

This synchronization ensures that all processes start with identical model states, which is crucial for maintaining consistency during training.

### DDP Computation-Communication Overlap Implementation

The overlap mechanism in DDP is implemented using autograd hooks, parameter bucketing, and a reducer component. Here's how it works:

**Autograd Hooks**: DDP registers hooks on model parameters. These hooks are triggered when gradients are computed during backward pass. The hook function marks the parameter gradient as "ready" for reduction.

**Parameter Bucketing**: The reducer organizes parameter gradients into buckets based on the `bucket_cap_mb` setting. Parameters are assigned to buckets in roughly reverse order of `model.parameters()` (reverse order because gradients are computed in reverse during backward pass). This ensures gradients in the same bucket become ready around the same time.

**Reducer**: When all gradients in a bucket are ready, the reducer launches an asynchronous AllReduce operation for that bucket. While AllReduce is in progress, backward pass continues computing gradients for the next bucket, achieving overlap.

**Unused Parameters**: If a parameter isn't used in forward pass (e.g., in conditional models), its gradient never becomes ready, causing the bucket to wait forever. Setting `find_unused_parameters=True` tells DDP to analyze the computation graph to identify unused parameters and mark them as ready without waiting for their gradients. This adds overhead but prevents hangs.

The key insight: DDP doesn't wait for all gradients before starting communication. Instead, it communicates gradients as soon as buckets are ready, overlapping communication with ongoing computation.

## 2. Setting Up Single-Node DDP

The simplest DDP setup is single-node multi-GPU: one machine with multiple GPUs. This is where most people start, and it's what you'll use for development and smaller-scale training.

### Using torchrun (Recommended)

The modern way to launch DDP training is with `torchrun` (or `torch.distributed.run` in older PyTorch versions). `torchrun` handles process creation, environment variable setup, and error handling. It's the recommended launcher for most cases.

Here's a minimal example. First, create a training script `train.py`:

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup():
    """Initialize process group and set device."""
    # torchrun sets these environment variables automatically
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    return rank, local_rank, world_size, device

def cleanup():
    """Clean up process group."""
    dist.destroy_process_group()

def main():
    rank, local_rank, world_size, device = setup()
    
    # Create model and move to device
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Create dummy data
    data = torch.randn(64, 10).to(device)
    target = torch.randn(64, 1).to(device)
    
    # Training step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    for epoch in range(10):
        # DistributedSampler would go here for real data
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    cleanup()

if __name__ == '__main__':
    main()
```

Launch it with:

```bash
torchrun --nproc_per_node=4 train.py
```

This launches 4 processes, one per GPU (assuming you have 4 GPUs). `torchrun` automatically sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` environment variables.

### Understanding the Environment Variables

When using `torchrun`, these environment variables are set automatically:

- **RANK**: Global rank of this process (0 to WORLD_SIZE-1)
- **LOCAL_RANK**: Local rank within this node (0 to number of GPUs per node - 1)
- **WORLD_SIZE**: Total number of processes
- **MASTER_ADDR**: IP address of the master node (for single-node, this is localhost)
- **MASTER_PORT**: Port for process group initialization (torchrun picks a free port)

For single-node training, you typically only care about `LOCAL_RANK` (to set which GPU this process uses) and `RANK` (to identify the main process for logging/checkpointing).

### Using DistributedSampler

The key to data parallelism is ensuring each process sees different data. `DistributedSampler` does this by sharding the dataset across processes.

```python
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloader(rank, world_size, batch_size=32):
    dataset = MyDataset(size=1000)
    
    # Create DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # Shuffle data each epoch
        drop_last=False  # Don't drop last incomplete batch
    )
    
    # Create DataLoader with sampler
    # Important: don't set shuffle=True when using DistributedSampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True  # Faster CPU->GPU transfer
    )
    
    return dataloader, sampler

def train():
    rank, local_rank, world_size, device = setup()
    
    dataloader, sampler = get_dataloader(rank, world_size)
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    for epoch in range(10):
        # CRITICAL: Set epoch for DistributedSampler
        # This ensures different shuffling each epoch
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            
            # Training step...
            pass
    
    cleanup()
```

Key points about `DistributedSampler`:

- **Sharding**: Each process gets a different subset of data. With 4 processes and 1000 samples, each process sees 250 samples.
- **Shuffling**: Set `shuffle=True` in the sampler, not in DataLoader. The sampler handles shuffling per-process.
- **set_epoch()**: Call this at the start of each epoch to ensure different shuffling each epoch. Without this, all epochs see data in the same order.
- **drop_last**: If `True`, drops the last incomplete batch. This ensures all processes have the same number of batches, which simplifies synchronization. If `False`, some processes might have one extra batch.

### DataLoader Internals for Distributed Training

Understanding how `DataLoader` works internally helps optimize data loading performance. When you create a `DataLoader` with `num_workers > 0`, PyTorch uses multi-process data loading.

**Single-process vs Multi-process**: The `DataLoader` chooses between `_SingleProcessDataLoaderIter` (for `num_workers=0`) and `_MultiProcessDataLoaderIter` (for `num_workers > 0`) based on the `num_workers` parameter.

**Multi-process data loading** works as follows:

1. **Main process**: Creates an index queue and a result queue. It also spawns worker processes.

2. **Worker processes**: Each worker process:
   - Reads indices from the index queue
   - Fetches corresponding data from the dataset
   - Applies transforms/preprocessing
   - Puts processed data into the result queue

3. **Prefetching**: While the main process is using the current batch for training, workers are already loading the next batch. This overlaps data loading with computation.

4. **Pin memory**: If `pin_memory=True`, a separate thread copies data from CPU to GPU memory asynchronously, further overlapping data transfer with computation.

**DistributedSampler integration**: When using `DistributedSampler`, each process's `DataLoader` only sees the indices assigned to that process. The sampler ensures no data overlap between processes.

**Performance tips**:
- Set `num_workers` to 2-4x the number of GPUs (but not more than CPU cores)
- Use `pin_memory=True` for faster CPU-to-GPU transfer
- Set `prefetch_factor=2` (default) to prefetch batches ahead
- Use `persistent_workers=True` to keep workers alive between epochs (reduces startup overhead)

### A Complete Single-Node Example

Here's a complete example that trains a ResNet on CIFAR-10 with DDP:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_dataloader(rank, world_size, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    sampler = DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    
    return trainloader, sampler

def train(rank, world_size):
    setup(rank, world_size)
    
    model = Net().to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    trainloader, sampler = get_dataloader(rank, world_size)
    
    for epoch in range(10):
        sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    if rank == 0:
        print('Training finished')
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train, args=(world_size,), nprocs=world_size, join=True
    )

if __name__ == '__main__':
    main()
```

This example uses `torch.multiprocessing.spawn` instead of `torchrun`. Both work, but `torchrun` is preferred because it handles errors better and provides more control.

### Using torch.multiprocessing.spawn

If you can't use `torchrun` (e.g., older PyTorch version or custom launcher), you can use `torch.multiprocessing.spawn`:

```python
import torch.multiprocessing as mp

def main():
    world_size = 4  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

The `spawn` method creates `world_size` processes, each calling `train(rank, world_size)`. You're responsible for setting up the process group in each process.

### Device Selection Best Practices

When setting up DDP, you need to assign each process to a GPU. The standard approach:

```python
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
```

This ensures process 0 uses GPU 0, process 1 uses GPU 1, etc. Always use `LOCAL_RANK` for device selection—don't use `RANK` (which is global across all nodes).

You can also set `CUDA_VISIBLE_DEVICES` before launching to restrict which GPUs are visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py
```

This makes only GPUs 0-3 visible, and `LOCAL_RANK` will map to these GPUs (LOCAL_RANK 0 → GPU 0, LOCAL_RANK 1 → GPU 1, etc.).

## 3. Setting Up Multi-Node DDP

Multi-node DDP scales training across multiple machines. This is where you'll see the real benefits of distributed training—training models that don't fit on a single node, or training faster by using hundreds of GPUs.

### Multi-Node Architecture

In multi-node DDP, you have:

- **Nodes**: Physical machines, each with multiple GPUs
- **Processes**: One process per GPU across all nodes
- **World size**: Total number of processes (nodes × GPUs per node)

For example, with 4 nodes and 8 GPUs per node:

- Total GPUs: 32
- World size: 32
- Each node runs 8 processes

Communication happens at two levels:

- **Intra-node**: GPUs on the same node communicate via NVLink (fast, 300-900 GB/s)
- **Inter-node**: GPUs on different nodes communicate via InfiniBand or Ethernet (slower, 25-50 GB/s per link, but aggregated across multiple links)

NCCL automatically optimizes communication patterns to minimize inter-node communication. For AllReduce, NCCL uses a hierarchical approach: first reduce within each node, then reduce across nodes, then broadcast results back.

### Launching Multi-Node Training

The simplest way to launch multi-node training is with `torchrun` on each node. You need to:

1. Set `MASTER_ADDR` to the IP of the master node (node 0)
2. Set `MASTER_PORT` to a free port (same on all nodes)
3. Set `WORLD_SIZE` to total number of processes
4. Set `NODE_RANK` to the node's rank (0 for master, 1 for first worker, etc.)
5. Set `NNODES` to number of nodes

On the master node (node 0):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_ip> --master_port=29500 train.py
```

On worker node (node 1):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_ip> --master_port=29500 train.py
```

Replace `<master_ip>` with the actual IP address of the master node. You can find it with:

```bash
hostname -I
```

Or if you have multiple interfaces:

```bash
ip addr show | grep inet
```

### Using SLURM for Multi-Node Launch

Most HPC clusters use SLURM for job scheduling. Here's a SLURM script that launches multi-node DDP:

```bash
#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Get node list
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Launch training
srun python train.py
```

Or using `torchrun` with SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py
```

### Network Configuration

For multi-node training, network bandwidth and latency matter. InfiniBand is preferred over Ethernet because:

- Higher bandwidth: 200-400 Gb/s per link vs 10-100 Gb/s for Ethernet
- Lower latency: Sub-microsecond vs microseconds
- RDMA support: Direct GPU-to-GPU memory access

If you're using InfiniBand, make sure:

- All nodes are on the same InfiniBand subnet
- NCCL can detect InfiniBand interfaces (set `NCCL_IB_DISABLE=0` if needed)
- Firewall allows the master port (or disable firewall for cluster network)

You can test network connectivity:

```bash
# On node 0
ib_write_bw

# On node 1
ib_write_bw <node0_ip>
```

### Environment Variables for Multi-Node

When launching multi-node, these environment variables are critical:

- **MASTER_ADDR**: IP address of master node (rank 0). All nodes must use the same value.
- **MASTER_PORT**: Port for rendezvous. Must be the same on all nodes and not in use.
- **WORLD_SIZE**: Total number of processes. Must be the same on all nodes.
- **RANK**: Global rank of this process (0 to WORLD_SIZE-1). Set automatically by launcher.
- **LOCAL_RANK**: Local rank within this node (0 to GPUs per node - 1). Set automatically.
- **NODE_RANK**: Rank of this node (0 to num_nodes - 1). Needed for torchrun.

### A Complete Multi-Node Example

Here's a training script that works for both single-node and multi-node:

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    """Initialize process group. Works for both single-node and multi-node."""
    # torchrun sets these automatically
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        # Fallback for manual launch (not recommended)
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = setup()
    
    if rank == 0:
        print(f'Initialized process group: world_size={world_size}')
        print(f'Master: {os.environ.get("MASTER_ADDR")}:{os.environ.get("MASTER_PORT")}')
    
    # Create model
    model = nn.Linear(10, 1).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Training loop...
    # (same as single-node example)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

Launch with:

```bash
# Single-node
torchrun --nproc_per_node=8 train.py

# Multi-node (run on each node)
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR train.py
```

## 4. Debugging and Troubleshooting Common DDP Failures

DDP training can fail in many ways. Most failures fall into a few categories: hangs, wrong results, out-of-memory errors, or performance issues. Let's go through each category with common causes and fixes.

### Hangs: The Most Common Problem

DDP hangs are usually caused by mismatched collective operations. Every process must call the same collectives in the same order. If one process calls `all_reduce` while another is waiting for something else, everything hangs.

**Symptom**: Training starts but hangs at a specific point, often during `init_process_group` or during the first backward pass.

**Common causes**:

1. **Mismatched WORLD_SIZE**: If processes have different `WORLD_SIZE` values, initialization hangs.

```python
# WRONG: Different processes see different world_size
world_size = torch.cuda.device_count()  # Might differ per node

# RIGHT: Use environment variable set by launcher
world_size = int(os.environ['WORLD_SIZE'])
```

2. **Conditional collectives**: If some processes skip a collective call, others hang waiting.

```python
# WRONG: Only rank 0 calls all_reduce
if rank == 0:
    dist.all_reduce(tensor)

# RIGHT: All processes call all_reduce
dist.all_reduce(tensor)
```

3. **Firewall blocking ports**: If `MASTER_PORT` is blocked, processes can't communicate.

```bash
# Test if port is accessible
telnet <master_ip> <master_port>

# Or use a different port
export MASTER_PORT=29501
```

4. **NCCL initialization timeout**: If NCCL can't initialize within the timeout, it hangs.

```bash
# Increase NCCL timeout (default is 10 minutes)
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

**Debugging hangs**:

Add logging to see where each process gets stuck:

```python
import logging
logging.basicConfig(level=logging.INFO)

def setup():
    rank = int(os.environ['RANK'])
    logging.info(f'Rank {rank}: Starting setup')
    
    torch.cuda.set_device(rank)
    logging.info(f'Rank {rank}: Set device')
    
    dist.init_process_group(backend='nccl')
    logging.info(f'Rank {rank}: Initialized process group')
    
    return rank
```

Run with one process first to test basic correctness:

```bash
# Test single-process first
CUDA_VISIBLE_DEVICES=0 python train.py  # Should work without DDP

# Then test with torchrun
torchrun --nproc_per_node=1 train.py  # Single process with DDP

# Then scale up
torchrun --nproc_per_node=2 train.py
```

### Wrong Results or Inconsistent Gradients

If training runs but produces wrong results or doesn't converge, the problem is usually data sharding or non-deterministic operations.

**Symptom**: Loss doesn't decrease, or different runs produce different results.

**Common causes**:

1. **Missing DistributedSampler.set_epoch()**: Without this, all epochs see data in the same order.

```python
# WRONG: Same data order every epoch
for epoch in range(10):
    for data, target in dataloader:
        # Training...

# RIGHT: Shuffle data each epoch
for epoch in range(10):
    sampler.set_epoch(epoch)  # CRITICAL
    for data, target in dataloader:
        # Training...
```

2. **Different random seeds**: If processes have different random seeds, they'll produce different results.

```python
# Set seeds on all processes
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Call this after setup(), before creating model
set_seed(42)
```

3. **Non-deterministic operations**: Some operations (e.g., `torch.bmm`, `torch.baddbmm`) are non-deterministic by default.

```python
# Enable deterministic mode (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

4. **Data leakage**: If `DistributedSampler` isn't used correctly, processes might see overlapping data.

```python
# WRONG: Using shuffle=True in DataLoader with DistributedSampler
dataloader = DataLoader(dataset, shuffle=True, sampler=sampler)  # Conflict!

# RIGHT: Shuffle in sampler, not DataLoader
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, sampler=sampler)  # No shuffle=True here
```

**Verifying correctness**:

Compare single-GPU vs multi-GPU results. They should match (within numerical precision):

```python
# Run with 1 GPU
torchrun --nproc_per_node=1 train.py --seed 42
# Save checkpoint

# Run with 4 GPUs
torchrun --nproc_per_node=4 train.py --seed 42
# Compare checkpoints - should be identical
```

### CUDA Out of Memory

OOM errors are common when scaling to multiple GPUs. The issue is that DDP replicates the model on each GPU, so memory usage scales with the number of GPUs.

**Symptom**: `RuntimeError: CUDA out of memory` during training.

**Common causes**:

1. **Batch size too large**: Even with DDP, per-GPU batch size matters.

```python
# If global batch size is 128 and you have 4 GPUs
# Per-GPU batch size should be 32, not 128
global_batch_size = 128
per_gpu_batch_size = global_batch_size // world_size
```

2. **Gradients accumulating**: If you're doing gradient accumulation, make sure to zero gradients.

```python
# WRONG: Gradients accumulate across accumulation steps
for i, (data, target) in enumerate(dataloader):
    loss = model(data, target)
    loss.backward()  # Gradients accumulate!
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()

# RIGHT: Zero gradients at the start of accumulation
optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    loss = model(data, target)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Large activations**: Some models (e.g., transformers with long sequences) have large activation memory.

**Solutions**:

- **Reduce batch size**: Lower per-GPU batch size
- **Gradient checkpointing**: Trade compute for memory by recomputing activations

```python
from torch.utils.checkpoint import checkpoint

# Replace
output = model(x)

# With
output = checkpoint(model, x)
```

- **Mixed precision**: Use FP16/BF16 to halve memory usage

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

- **Clear cache**: Sometimes PyTorch holds onto memory

```python
torch.cuda.empty_cache()
```

### Performance Issues

If training runs but is slow, the bottleneck is usually communication, data loading, or inefficient kernels.

**Symptom**: GPU utilization is low, or training is slower than expected.

**Common causes**:

1. **Data loading bottleneck**: If CPU can't keep up with GPU, GPUs sit idle.

```python
# Increase num_workers
dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)

# Or use prefetching
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, num_workers=8, prefetch_factor=2)
```

2. **Small batch size**: If batch size is too small, GPUs aren't fully utilized.

```python
# Increase batch size (if memory allows)
batch_size = 128  # Instead of 32
```

3. **Communication overhead**: If model is small or communication is slow, AllReduce dominates.

```python
# Profile to see where time is spent
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Training step
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

4. **Inefficient NCCL topology**: NCCL might pick a suboptimal algorithm.

```bash
# Set NCCL debug to see what algorithm is used
export NCCL_DEBUG=INFO

# Force specific algorithm (advanced, usually not needed)
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
```

**Debugging performance**:

Use PyTorch profiler to identify bottlenecks:

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("training_step"):
        # Your training step
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Check GPU utilization:

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Or use dstat
dstat -cdngy
```

### Network Debugging

For multi-node training, network issues are common. Use NCCL debugging:

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Test NCCL connectivity
python -c "import torch; torch.distributed.init_process_group('nccl'); print('OK')"
```

Check InfiniBand connectivity:

```bash
# List InfiniBand devices
ibdev2netdev

# Test bandwidth
ib_write_bw  # On one node
ib_write_bw <other_node_ip>  # On another node
```

## 5. Profiling DDP Performance

Before optimizing DDP, you need to understand where time is spent. PyTorch's profiler provides detailed insights into DDP's computation-communication overlap, gradient synchronization overhead, and data loading bottlenecks.

### Using torch.profiler.profile for DDP Analysis

The `torch.profiler.profile` context manager captures detailed timing information for CPU and CUDA operations. For DDP, you want to profile:
- Forward pass time
- Backward pass time (gradient computation)
- AllReduce communication time
- Overlap between computation and communication
- Data loading time

Here's a complete example of profiling DDP training:

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist

def train_with_profiling(model, dataloader, optimizer, criterion, num_iterations=10):
    """Train with profiling to analyze DDP performance."""
    rank = dist.get_rank()
    
    # Create profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,  # Enable stack traces for deeper analysis
    ) as prof:
        with record_function("training_loop"):
            for i, (data, target) in enumerate(dataloader):
                if i >= num_iterations:
                    break
                
                data = data.cuda(rank, non_blocking=True)
                target = target.cuda(rank, non_blocking=True)
                
                # Forward pass
                with record_function("forward_pass"):
                    output = model(data)
                    loss = criterion(output, target)
                
                # Backward pass (where DDP communication happens)
                with record_function("backward_pass"):
                    loss.backward()
                
                # Optimizer step
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()
    
    # Print profiling results (only on rank 0 to avoid duplicate output)
    if rank == 0:
        # Print key averages sorted by CUDA time
        print("=" * 80)
        print("DDP Performance Profile - Key Averages")
        print("=" * 80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30
        ))
        
        # Print events sorted by self CUDA time (excludes child operations)
        print("\n" + "=" * 80)
        print("Top Operations by Self CUDA Time")
        print("=" * 80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        ))
        
        # Export to Chrome trace format for visualization
        prof.export_chrome_trace("ddp_trace.json")
        print("\nChrome trace exported to ddp_trace.json")
        print("Open chrome://tracing in Chrome browser to visualize")
    
    return prof
```

### Analyzing Computation-Communication Overlap

The key metric for DDP performance is whether communication overlaps with computation. You can verify this by looking for concurrent AllReduce and backward operations in the profiler output.

Here's a focused example that profiles just the backward pass to analyze overlap:

```python
def analyze_ddp_overlap(model, loss):
    """Analyze computation-communication overlap in DDP backward pass."""
    rank = dist.get_rank()
    
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("backward_with_ddp"):
            loss.backward()
    
    if rank == 0:
        # Look for AllReduce operations
        events = prof.key_averages()
        
        # Filter for NCCL AllReduce operations
        allreduce_ops = [e for e in events if 'nccl' in e.key.lower() and 'allreduce' in e.key.lower()]
        backward_ops = [e for e in events if 'backward' in e.key.lower() or 'gradient' in e.key.lower()]
        
        print("=" * 80)
        print("DDP Overlap Analysis")
        print("=" * 80)
        print(f"AllReduce operations found: {len(allreduce_ops)}")
        print(f"Backward operations found: {len(backward_ops)}")
        
        # Check if AllReduce overlaps with backward compute
        total_allreduce_time = sum(e.cuda_time_total for e in allreduce_ops)
        total_backward_time = sum(e.cuda_time_total for e in backward_ops)
        
        print(f"\nTotal AllReduce time: {total_allreduce_time / 1000:.2f} ms")
        print(f"Total backward compute time: {total_backward_time / 1000:.2f} ms")
        
        # If backward time >> AllReduce time, overlap is working
        if total_backward_time > total_allreduce_time * 1.5:
            print("✓ Good overlap: Computation time exceeds communication time")
            print("  This indicates AllReduce is happening concurrently with gradient computation")
        else:
            print("⚠ Limited overlap: Communication time is significant")
            print("  Consider: larger bucket size, faster interconnects, or larger models")
        
        # Export trace for detailed visualization
        prof.export_chrome_trace("ddp_overlap_trace.json")
```

### Profiling Multi-Node DDP

For multi-node training, you want to profile communication between nodes separately from intra-node communication:

```python
def profile_multi_node_ddp(model, dataloader, optimizer, criterion):
    """Profile DDP with focus on inter-node vs intra-node communication."""
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        # Training step
        for data, target in dataloader:
            data = data.cuda(local_rank)
            target = target.cuda(local_rank)
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break  # Profile just one iteration
    
    if rank == 0:
        events = prof.key_averages()
        
        # Analyze NCCL operations
        nccl_ops = [e for e in events if 'nccl' in e.key.lower()]
        
        print("=" * 80)
        print("Multi-Node DDP Communication Analysis")
        print("=" * 80)
        
        for op in nccl_ops[:10]:  # Top 10 NCCL operations
            print(f"{op.key}: {op.cuda_time_total / 1000:.2f} ms")
        
        # Export for detailed analysis
        prof.export_chrome_trace(f"ddp_multinode_rank{rank}.json")
```

### Interpreting Profiler Results

When analyzing DDP profiler output, look for:

1. **AllReduce operations**: Should see `nccl:all_reduce` or similar. These represent gradient synchronization.

2. **Overlap indicators**: If you see backward compute operations (e.g., `ConvolutionBackward0`, `LinearBackward`) happening concurrently with AllReduce, overlap is working.

3. **Communication time**: AllReduce time should be a small fraction of total backward time for good performance. If AllReduce time is >30% of backward time, you have a communication bottleneck.

4. **Bucket boundaries**: You might see multiple AllReduce operations during backward pass—these correspond to different gradient buckets.

5. **Data loading**: Look for `DataLoader` operations. If data loading time is significant, increase `num_workers` or optimize data preprocessing.

### Example: Profiling ResNet50 on CIFAR-10

Here's a complete example profiling ResNet50 training with DDP:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist

def profile_resnet_ddp():
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    
    # Create model
    model = torchvision.models.resnet50(num_classes=10)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=rank)
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Profile training
    model.train()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for i, (data, target) in enumerate(dataloader):
            if i >= 5:  # Profile 5 iterations
                break
            
            data = data.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Print results on rank 0
    if rank == 0:
        print("=" * 80)
        print("ResNet50 DDP Performance Profile")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        
        # Export trace
        prof.export_chrome_trace("resnet50_ddp_trace.json")
        print("\nTrace exported to resnet50_ddp_trace.json")
```

### Visualizing Profiler Traces

The profiler exports traces in Chrome trace format. To visualize:

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select the exported `.json` file
4. Use the timeline view to see:
   - When AllReduce operations occur
   - Whether they overlap with backward compute
   - Data loading timing
   - GPU utilization

In the trace view, you should see:
- **Forward pass**: Dense compute operations
- **Backward pass**: Mix of compute (gradient computation) and communication (AllReduce)
- **Overlap**: AllReduce operations happening concurrently with backward compute operations

If AllReduce operations appear as separate blocks after all backward compute, overlap isn't working. If AllReduce appears interleaved with backward compute, overlap is working.

### Performance Metrics from Profiling

From profiler results, calculate these metrics:

```python
def calculate_ddp_metrics(prof):
    """Calculate key DDP performance metrics from profiler output."""
    events = prof.key_averages()
    
    # Find key operations
    forward_ops = [e for e in events if 'forward' in e.key.lower()]
    backward_ops = [e for e in events if 'backward' in e.key.lower() or 'gradient' in e.key.lower()]
    allreduce_ops = [e for e in events if 'allreduce' in e.key.lower() or 'nccl' in e.key.lower()]
    dataloader_ops = [e for e in events if 'dataloader' in e.key.lower()]
    
    # Calculate times (in milliseconds)
    forward_time = sum(e.cuda_time_total for e in forward_ops) / 1000
    backward_time = sum(e.cuda_time_total for e in backward_ops) / 1000
    comm_time = sum(e.cuda_time_total for e in allreduce_ops) / 1000
    data_time = sum(e.cuda_time_total for e in dataloader_ops) / 1000
    
    total_time = forward_time + backward_time
    
    metrics = {
        'forward_time_ms': forward_time,
        'backward_time_ms': backward_time,
        'communication_time_ms': comm_time,
        'data_loading_time_ms': data_time,
        'total_time_ms': total_time,
        'comm_overhead_percent': (comm_time / total_time) * 100 if total_time > 0 else 0,
        'overlap_ratio': (backward_time - comm_time) / backward_time if backward_time > comm_time else 0,
    }
    
    return metrics

# Usage
prof = train_with_profiling(model, dataloader, optimizer, criterion)
if dist.get_rank() == 0:
    metrics = calculate_ddp_metrics(prof)
    print("\nDDP Performance Metrics:")
    print(f"Forward time: {metrics['forward_time_ms']:.2f} ms")
    print(f"Backward time: {metrics['backward_time_ms']:.2f} ms")
    print(f"Communication time: {metrics['communication_time_ms']:.2f} ms")
    print(f"Communication overhead: {metrics['comm_overhead_percent']:.2f}%")
    print(f"Overlap ratio: {metrics['overlap_ratio']:.2%}")
```

**Interpreting metrics**:
- **Communication overhead < 20%**: Good—communication is well hidden
- **Communication overhead 20-40%**: Acceptable—some optimization possible
- **Communication overhead > 40%**: Poor—communication is a bottleneck
- **Overlap ratio > 0.7**: Excellent overlap
- **Overlap ratio 0.5-0.7**: Good overlap
- **Overlap ratio < 0.5**: Limited overlap—consider tuning bucket size or model architecture

## 6. Optimizing DDP Performance

Once you've profiled and identified bottlenecks, the next step is optimization. There are several levers you can tune: bucket size, gradient accumulation, mixed precision, and communication overlap.

### Tuning Bucket Size

DDP groups gradients into buckets for AllReduce. The default bucket size is 25 MB, but you can tune it:

```python
model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=50  # Increase from default 25 MB
)
```

**When to increase bucket size**:

- Large models with many parameters: Larger buckets mean fewer AllReduce calls
- Fast interconnects (NVLink): Can handle larger buckets efficiently
- Communication-bound workloads: Larger buckets reduce communication overhead

**When to decrease bucket size**:

- Small models: Smaller buckets enable earlier synchronization
- Slow interconnects: Smaller buckets reduce latency
- Memory-constrained: Smaller buckets use less memory for communication buffers

**How to find optimal bucket size**:

Profile with different bucket sizes and measure throughput:

```python
bucket_sizes = [10, 25, 50, 100]  # MB
for bucket_size in bucket_sizes:
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=bucket_size)
    # Run training for a few iterations
    # Measure throughput
    # Record results
```

The optimal size depends on your model and hardware. Start with the default (25 MB) and tune if needed.

### Gradient Accumulation

Gradient accumulation lets you simulate larger batch sizes without increasing memory usage. Instead of updating parameters every step, you accumulate gradients over multiple steps:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    
    # Scale loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()
    
    # Update every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Handle remaining steps
if (i + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
```

**When to use gradient accumulation**:

- Memory constraints: Can't fit desired batch size in memory
- Small batch sizes: Want larger effective batch size for stability
- Uneven data: When dataset size isn't divisible by batch size

**Important**: With DDP, gradient accumulation works correctly because DDP synchronizes gradients during `backward()`, not during `step()`. Each accumulation step triggers AllReduce, so you get the same gradients as if you used a larger batch size.

### Mixed Precision Training

Mixed precision (FP16/BF16) halves memory usage and can double throughput. PyTorch's Automatic Mixed Precision (AMP) makes it easy:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(10):
    for data, target in dataloader:
        optimizer.zero_grad()
        
        # Forward pass in mixed precision
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        
        # Optimizer step with unscaling
        scaler.step(optimizer)
        scaler.update()
```

**GradScaler** handles gradient scaling to prevent underflow. It:
1. Scales loss before backward (gradients are also scaled)
2. Detects overflow (inf/NaN gradients)
3. Skips optimizer step if overflow detected
4. Adjusts scale factor dynamically

**BF16 vs FP16**:

- **FP16**: 5-bit exponent, 10-bit mantissa. Can underflow easily.
- **BF16**: 8-bit exponent (same as FP32), 7-bit mantissa. More stable, less precision loss.

For training, BF16 is often preferred because it's more stable. For inference, FP16 is fine and sometimes faster.

```python
# Use BF16 (if supported)
with autocast(dtype=torch.bfloat16):
    output = model(data)
```

**DDP + AMP best practices**:

- Create scaler before wrapping model with DDP
- Call `scaler.step()` and `scaler.update()` on all processes
- Monitor for overflow (scaler will skip steps if overflow detected)

### Communication Overlap Optimization

DDP automatically overlaps communication with computation, but you can help it:

1. **Avoid blocking operations in backward**: Don't call `synchronize()` or blocking operations during backward pass.

```python
# WRONG: Blocks and prevents overlap
loss.backward()
torch.cuda.synchronize()  # Blocks!
optimizer.step()

# RIGHT: Let DDP handle synchronization
loss.backward()
optimizer.step()  # DDP synchronizes automatically
```

2. **Use asynchronous data loading**: Keep data loading pipeline busy:

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster CPU->GPU transfer
    prefetch_factor=2  # Prefetch batches
)
```

3. **Profile to verify overlap**: Use profiler to see if AllReduce overlaps with compute:

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    loss.backward()

# Check if AllReduce overlaps with backward compute
print(prof.key_averages().table())
```

### Finding Unused Parameters

By default, DDP assumes all parameters receive gradients. If some parameters don't (e.g., in conditional models), DDP will hang waiting for gradients that never come.

Enable `find_unused_parameters=True`:

```python
model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True  # Slower but handles unused params
)
```

**Warning**: `find_unused_parameters=True` adds overhead because DDP must traverse the computation graph to find which parameters are used. Only enable it if you have unused parameters.

**Better solution**: If possible, ensure all parameters receive gradients. For conditional models, you might need to restructure code.

### Static Graph Optimization

If your model's computation graph doesn't change between iterations, you can enable `static_graph=True` for better performance:

```python
model = DDP(
    model,
    device_ids=[local_rank],
    static_graph=True  # Graph structure is static
)
```

This tells DDP that:

- The set of used/unused parameters doesn't change
- The graph structure is the same every iteration

DDP can then optimize communication patterns. This is especially useful for:

- Models without conditional logic
- Models where you've already verified the graph is static

Check if your model can use static graph:

```python
# After training for a few iterations
ddp_logging_data = model._get_ddp_logging_data()
can_set_static_graph = ddp_logging_data.get("can_set_static_graph", False)
if can_set_static_graph:
    print("Can enable static_graph=True")
```

## 7. Checkpointing and Resuming Distributed Jobs

After optimizing your DDP training, you'll want to save progress regularly. Long training jobs need checkpointing, and with DDP, you need to save and restore model state, optimizer state, and random number generator state correctly.

### Saving Checkpoints

Only rank 0 should write checkpoints to avoid race conditions:

```python
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    rank = dist.get_rank()
    
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Note: .module
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        # Save scaler state if using AMP
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')
    
    # All processes wait for rank 0 to finish
    dist.barrier()
```

**Important**: When saving DDP model state, use `model.module.state_dict()`, not `model.state_dict()`. The DDP wrapper adds a `.module` attribute containing the actual model.

### Loading Checkpoints

All processes should load the same checkpoint:

```python
def load_checkpoint(model, optimizer, filepath, scaler=None):
    rank = dist.get_rank()
    
    # All processes load from the same file
    checkpoint = torch.load(filepath, map_location=f'cuda:{rank}')
    
    # Load model state
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state if using AMP
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    
    if rank == 0:
        print(f'Checkpoint loaded: {filepath}')
        print(f'Resuming from epoch {start_epoch}')
    
    return start_epoch, best_loss
```

### Saving RNG State for Reproducibility

For fully reproducible resumes, save random number generator state:

```python
def save_checkpoint_with_rng(model, optimizer, epoch, filepath):
    rank = dist.get_rank()
    
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
        }
        
        # Also save Python and NumPy RNG if used
        import random
        import numpy as np
        checkpoint['python_rng_state'] = random.getstate()
        checkpoint['numpy_rng_state'] = np.random.get_state()
        
        torch.save(checkpoint, filepath)

def load_checkpoint_with_rng(model, optimizer, filepath):
    rank = dist.get_rank()
    checkpoint = torch.load(filepath, map_location=f'cuda:{rank}')
    
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore RNG states
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    
    import random
    import numpy as np
    random.setstate(checkpoint['python_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    
    return checkpoint['epoch']
```

### Atomic Checkpointing

To avoid corrupted checkpoints if training crashes during save, use atomic writes:

```python
import os
import tempfile

def save_checkpoint_atomic(model, optimizer, epoch, filepath):
    rank = dist.get_rank()
    
    if rank == 0:
        # Write to temporary file first
        temp_file = filepath + '.tmp'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, temp_file)
        
        # Atomic rename
        os.rename(temp_file, filepath)
        print(f'Checkpoint saved: {filepath}')
    
    dist.barrier()
```

### Checkpointing Best Practices

1. **Save regularly**: Save checkpoints every N epochs or every N iterations
2. **Keep multiple checkpoints**: Don't overwrite the only checkpoint—keep the last few
3. **Save on rank 0 only**: Avoid race conditions by having only one process write
4. **Use barriers**: After saving, use `dist.barrier()` to ensure all processes see the checkpoint
5. **Test loading**: Periodically test that checkpoints can be loaded correctly

Now that we've covered the essentials of DDP setup, debugging, profiling, optimization, and checkpointing, let's explore some advanced features that DDP provides for specialized use cases.

## 8. Advanced DDP Features

DDP has several advanced features for specialized use cases: gradient hooks, communication hooks, and join() for uneven inputs. These features give you fine-grained control over DDP's behavior when you need to customize gradient synchronization or handle edge cases.

### Gradient Hooks

You can register hooks to inspect or modify gradients during backward:

```python
def gradient_hook(grad):
    # Inspect or modify gradient
    print(f'Gradient norm: {grad.norm().item()}')
    return grad  # Must return gradient

# Register hook on a parameter
model.module.fc.weight.register_hook(gradient_hook)
```

Hooks are useful for:

- Gradient clipping
- Gradient logging/monitoring
- Custom gradient modifications

### Communication Hooks

For advanced use cases, you can customize how DDP synchronizes gradients using communication hooks:

```python
def allreduce_hook(state, bucket):
    """Custom hook that does AllReduce on gradient bucket."""
    tensor = bucket.buffer()
    
    # Custom AllReduce (e.g., with compression)
    dist.all_reduce(tensor, async_op=False)
    
    # Return future (DDP expects this)
    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut

# Register hook
model.register_comm_hook(state=None, hook=allreduce_hook)
```

Communication hooks let you implement:

- Gradient compression (quantization, sparsification)
- Custom reduction operations
- Gradient filtering

**Warning**: Communication hooks are advanced and can break DDP if implemented incorrectly. Only use if you know what you're doing.

### Handling Uneven Inputs with join()

If different processes have different amounts of data (uneven inputs), DDP will hang because some processes finish early. The `join()` context manager handles this:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])

# Wrap training loop with join()
with model.join():
    for data, target in dataloader:
        # Training step
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

`join()` automatically handles processes that finish early by having them participate in dummy AllReduce operations to match processes that are still training.

**When to use join()**:

- Dataset size isn't divisible by batch size × world_size
- Different processes have different dataset sizes
- You're using dynamic batching

With the advanced features covered, let's consolidate the key practices that will help you write robust, efficient DDP training code.

## 9. Best Practices and Common Patterns

Here are best practices distilled from production DDP training. Following these patterns will help you avoid common pitfalls and build reliable distributed training pipelines:

### Always Validate Single-Process First

Before scaling to multiple GPUs, make sure single-GPU training works:

```bash
# Test without DDP first
CUDA_VISIBLE_DEVICES=0 python train.py

# Then test with DDP (single process)
torchrun --nproc_per_node=1 train.py

# Then scale up
torchrun --nproc_per_node=4 train.py
```

### Use torchrun for Launching

`torchrun` is the recommended launcher because it:

- Handles process creation and cleanup
- Sets environment variables correctly
- Provides better error messages
- Supports elastic training (restarting failed processes)

Avoid manual process spawning unless you have a specific reason.

### Set Seeds for Reproducibility

Always set random seeds on all processes:

```python
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)  # After setup(), before creating model
```

### Use DistributedSampler Correctly

Remember to:

- Set `shuffle=True` in sampler, not DataLoader
- Call `sampler.set_epoch(epoch)` each epoch
- Don't use `shuffle=True` in DataLoader when using sampler

### Profile Before Optimizing

Don't guess what's slow—profile:

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Training step
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Monitor GPU Utilization

Keep an eye on GPU utilization:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use dstat
dstat -cdngy
```

If utilization is low, you're likely bottlenecked by data loading or communication.

### Use Mixed Precision

For most training, mixed precision (FP16/BF16) is a free win:

- Halves memory usage
- Often doubles throughput
- Minimal code changes

Always try it unless you have a specific reason not to.

### Keep Checkpoints

Long training jobs will fail. Save checkpoints regularly and test that they can be loaded.

### Test Multi-Node Early

If you plan to use multi-node training, test it early. Multi-node has different failure modes than single-node (network issues, different hardware, etc.).

So far, we've focused on **synchronous** DDP, where all GPUs wait for each other before synchronizing gradients. This is the standard approach and works well for most cases. However, there are alternative paradigms worth understanding: asynchronous and elastic data parallelism.

## 10. Asynchronous Data Parallelism

All the DDP implementations we've discussed so far are **synchronous**: all GPUs wait for each other to complete gradient computation before synchronizing. This ensures model consistency but can be inefficient when GPUs have different speeds or when communication overhead is high.

**Asynchronous Data Parallelism (ADP)** allows GPUs to update parameters independently without waiting for others. Fast GPUs can update parameters more frequently, while slow GPUs don't block the entire training process.

### How Asynchronous Data Parallel Works

In asynchronous data parallel:

1. **Forward pass**: Each GPU processes its data shard independently, computing gradients at its own pace.

2. **Gradient push**: When a GPU finishes computing gradients, it immediately sends them to a parameter server (or master process) without waiting for other GPUs.

3. **Parameter update**: The parameter server accumulates gradients and updates model parameters as soon as it receives gradients from any GPU.

4. **Parameter pull**: GPUs pull the latest parameters from the parameter server before the next iteration.

The key difference from synchronous DDP: GPUs don't wait for each other. A fast GPU might update parameters multiple times while a slow GPU is still computing gradients.

### Advantages of Asynchronous Data Parallel

- **No straggler waiting**: Fast GPUs don't wait for slow GPUs, improving overall throughput when GPUs have different speeds.

- **Better GPU utilization**: GPUs stay busy computing instead of waiting for synchronization.

- **Scalability**: Can handle large numbers of GPUs without communication bottlenecks (each GPU communicates independently with the parameter server).

### Challenges of Asynchronous Data Parallel

- **Stale gradients**: A GPU might compute gradients using old parameters while parameters are being updated by other GPUs. This creates gradient staleness, which can hurt convergence.

- **Convergence issues**: The lack of synchronization can cause training instability. Models might converge slower or not converge at all, especially with high staleness.

- **Race conditions**: Multiple GPUs updating parameters simultaneously can cause race conditions, requiring careful synchronization at the parameter server.

- **Parameter server bottleneck**: All GPUs communicate with a central parameter server, which can become a bottleneck at scale.

### When to Use Asynchronous Data Parallel

Asynchronous data parallel is rarely used in modern training because:

1. **DDP is fast enough**: With efficient communication (NVLink, InfiniBand) and overlap, synchronous DDP achieves high efficiency without the convergence risks.

2. **Convergence is critical**: For most models, training stability and convergence are more important than marginal speed improvements.

3. **Hardware is homogeneous**: Modern clusters have uniform GPU speeds, so straggler issues are less common.

However, asynchronous data parallel can be useful for:
- **Heterogeneous clusters**: When GPUs have significantly different speeds
- **Fault tolerance**: When you want training to continue even if some GPUs fail
- **Research**: When exploring trade-offs between speed and convergence

### Implementing Asynchronous Data Parallel

PyTorch doesn't provide built-in asynchronous data parallel support (DDP is synchronous). You'd need to implement it manually using parameter servers or custom communication patterns. This is complex and error-prone, which is why most practitioners stick with DDP.

If you need asynchronous behavior, consider:
- **Gradient accumulation**: Simulate larger batches without synchronization overhead
- **Pipeline parallelism**: Overlap computation across model layers (covered in later chapters)
- **Elastic training**: Handle node failures and dynamic scaling (covered next)

While asynchronous data parallel is rarely used in practice, **elastic data parallelism** is increasingly important for production training systems that need to handle failures and dynamic resource allocation.

## 11. Elastic Data Parallelism

Elastic training is a distributed training approach that handles dynamic environments: node failures, resource changes, and membership changes. Instead of failing when a node crashes, elastic training automatically adjusts and continues training. This is crucial for long-running training jobs where node failures are inevitable.

PyTorch provides **TorchElastic** (now part of `torchrun`) for elastic distributed training. It enables:
- **Fault tolerance**: Automatically recover from node failures
- **Dynamic scaling**: Add or remove nodes during training
- **Checkpoint-based recovery**: Resume from the last checkpoint after failures

### How Elastic Training Works

Elastic training uses a **rendezvous** mechanism to coordinate nodes:

1. **Rendezvous**: Nodes join a rendezvous point, waiting until a minimum number of nodes are available.

2. **Barrier**: Once minimum nodes are reached, all nodes proceed together. If maximum nodes are specified, rendezvous completes immediately when maximum is reached.

3. **Rank assignment**: Each node receives a unique rank for the training job.

4. **Training**: Nodes run training with the assigned ranks.

5. **Failure handling**: If a node fails, remaining nodes detect the failure and trigger a new rendezvous, reassigning ranks and continuing training.

### Elastic Agent

The **Elastic Agent** is the control plane for elastic training. It:
- Launches and manages worker processes
- Monitors worker health and detects failures
- Handles rendezvous and rank assignment
- Restarts workers when failures occur

Each node runs an Elastic Agent that manages local workers. Agents coordinate with each other through the rendezvous backend.

### Rendezvous Backend

The rendezvous backend coordinates node discovery and synchronization. PyTorch provides two backends:

1. **C10d backend**: Uses TCPStore (default). No external dependencies required.

2. **etcd backend**: Uses etcd for coordination. More robust for large-scale deployments.

The rendezvous process has several states:

- **Non-existent**: No active rendezvous
- **Joinable**: Nodes can join (waiting for minimum nodes)
- **Frozen**: Minimum nodes reached, finalizing participant list
- **Final**: Rendezvous complete, ranks assigned

### Launching Elastic Training

Use `torchrun` with elastic parameters:

```bash
torchrun \
    --nnodes=2:4 \
    --nproc-per-node=8 \
    --max-restarts=3 \
    --rdzv-id=my_job \
    --rdzv-backend=c10d \
    --rdzv-endpoint=master_node:29500 \
    train.py
```

Parameters:
- `--nnodes=MIN:MAX`: Minimum and maximum number of nodes (2 to 4 in this example)
- `--nproc-per-node`: Number of processes (GPUs) per node
- `--max-restarts`: Maximum number of restart attempts
- `--rdzv-id`: Unique job identifier
- `--rdzv-backend`: Rendezvous backend (c10d or etcd)
- `--rdzv-endpoint`: Master node address and port

### Implementing Checkpointing for Elastic Training

Elastic training requires proper checkpointing because nodes can fail and restart. Your training script should:

1. **Load checkpoint at startup**: Always try to load the latest checkpoint before starting training.

2. **Save checkpoints regularly**: Save checkpoints frequently (every N epochs or iterations) so minimal progress is lost on failure.

3. **Handle checkpoint loading**: If checkpoint exists, resume from that point. Otherwise, start from scratch.

Example:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def load_checkpoint(checkpoint_path):
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    return None

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save checkpoint (only on rank 0)."""
    if dist.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

def train():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    # Load checkpoint
    checkpoint = load_checkpoint('checkpoint.pt')
    start_epoch = 0
    
    model = create_model().to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = create_optimizer(model.parameters())
    
    if checkpoint:
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f'Resuming from epoch {start_epoch}')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training...
        train_epoch(model, optimizer, dataloader)
        
        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, 'checkpoint.pt')
    
    dist.destroy_process_group()
```

### Elastic Training Best Practices

1. **Frequent checkpoints**: Save checkpoints often. In the worst case, you'll lose progress since the last checkpoint.

2. **Checkpoint on rank 0**: Only rank 0 should write checkpoints to avoid race conditions.

3. **Atomic checkpoint writes**: Write to a temporary file, then rename to avoid corrupted checkpoints.

4. **Test failure scenarios**: Intentionally kill nodes to verify recovery works correctly.

5. **Monitor rendezvous**: Use `NCCL_DEBUG=INFO` to monitor rendezvous and communication.

### When to Use Elastic Training

Use elastic training when:
- **Long-running jobs**: Training jobs that run for days or weeks benefit from fault tolerance
- **Unreliable infrastructure**: Clusters with frequent node failures
- **Dynamic resource allocation**: When you want to add/remove nodes based on availability
- **Cost optimization**: Scale down during low-priority periods, scale up when needed

For short training jobs or stable clusters, standard DDP (non-elastic) is simpler and sufficient.

Now that we've covered all the key concepts—from basic DDP setup to advanced features like asynchronous and elastic training—let's put everything together in a complete, production-ready example that demonstrates best practices.

## 12. Real-World Example: Training a Transformer with DDP

Let's put it all together with a complete example: training a transformer model (GPT-style) with DDP, mixed precision, and checkpointing. This example demonstrates best practices and shows how the various DDP features work together in practice. It integrates everything we've learned: proper setup, DistributedSampler usage, mixed precision, checkpointing, and error handling.

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for block in self.transformer_blocks:
            src = block(src)
        output = self.fc_out(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, local_rank, world_size

def get_dataloader(rank, world_size, batch_size=32, seq_len=128):
    # Dummy dataset for example
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10000, vocab_size=10000, seq_len=128):
            self.size = size
            self.vocab_size = vocab_size
            self.seq_len = seq_len
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return torch.randint(0, self.vocab_size, (self.seq_len,))
    
    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    return dataloader, sampler

def train():
    rank, local_rank, world_size = setup()
    
    # Create model
    model = TransformerModel(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=50)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Data
    dataloader, sampler = get_dataloader(rank, world_size, batch_size=32)
    
    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, src in enumerate(dataloader):
            src = src.to(local_rank)
            tgt = src[:, 1:]  # Shift for next-token prediction
            src = src[:, :-1]
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(src)
                output = output.view(-1, output.size(-1))
                tgt = tgt.reshape(-1)
                loss = criterion(output, tgt)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Save checkpoint
        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
    
    dist.destroy_process_group()

if __name__ == '__main__':
    train()
```

Launch with:

```bash
torchrun --nproc_per_node=8 train_transformer.py
```

This example includes:

- DDP setup and teardown
- DistributedSampler with set_epoch()
- Mixed precision training
- Checkpointing
- A realistic transformer model

## Conclusion

DDP is the foundation of distributed training in PyTorch. Throughout this chapter, we've covered how DDP works internally, how to set it up for single-node and multi-node training, how to debug common issues, how to profile and optimize performance, and how to handle advanced scenarios like checkpointing, asynchronous training, and elastic scaling.

Understanding how DDP works—gradient synchronization, bucketing, overlap—helps you write efficient training code and debug issues when they arise. The key concepts we've explored include:

- **Gradient synchronization**: DDP uses AllReduce to aggregate gradients across all processes, ensuring model consistency
- **Computation-communication overlap**: DDP overlaps gradient synchronization with computation to hide communication latency
- **Process management**: Using `torchrun` for launching and managing DDP processes
- **Data sharding**: Using `DistributedSampler` to ensure each process sees different data
- **Performance optimization**: Profiling, bucket tuning, mixed precision, and other optimization techniques
- **Fault tolerance**: Checkpointing and elastic training for long-running jobs

Key takeaways:

- Use `torchrun` for launching DDP jobs—it handles process management and error recovery
- Always use `DistributedSampler` and call `set_epoch()` each epoch to ensure proper data shuffling
- Enable mixed precision (FP16/BF16) for better performance—it's usually a free win
- Profile before optimizing—use `torch.profiler.profile` to identify bottlenecks
- Save checkpoints regularly—long training jobs will fail, and checkpoints let you resume
- Test single-process before scaling—validate correctness before adding complexity
- Understand your communication topology—NVLink for intra-node, InfiniBand for inter-node

DDP is mature, well-optimized, and suitable for most distributed training scenarios. However, for very large models that don't fit on a single GPU, you'll need to move beyond DDP to techniques like FSDP (Fully Sharded Data Parallel), which we'll cover in the next chapter. FSDP extends DDP by sharding model parameters across GPUs, enabling training of models that are too large for any single GPU's memory.
