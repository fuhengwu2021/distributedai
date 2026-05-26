\fancydividerwithicon[center]{hand.png}


## Exercises


### Process Group Initialization

Implement a function to initialize a distributed process group with proper error handling.

__Requirements:__

- Function signature: `setup_distributed(rank, world_size, backend='nccl')`
- Initialize the process group using `torch.distributed.init_process_group()`
- Set the CUDA device for the current rank: `torch.cuda.set_device(rank)`
- Add error handling for cases where CUDA is not available
- Return `True` if successful, `False` otherwise
- Print a message indicating successful initialization with rank and world_size

__Test your function:__
```python
import torch.distributed as dist
import os

# Simulate rank and world_size from environment
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

if setup_distributed(rank, world_size):
    print(f"Rank {rank}/{world_size} initialized successfully")
    dist.destroy_process_group()
```

### Manual AllReduce Implementation

Implement a manual gradient averaging function using AllReduce, simulating what DDP does internally.

__Requirements:__

- Function signature: `average_gradients(model, world_size)`
- Iterate through all parameters in the model
- For each parameter with `requires_grad=True`:
  - Use `dist.all_reduce()` with `op=dist.ReduceOp.SUM` to sum gradients across all ranks
  - Divide the gradient by `world_size` to get the average
- Handle the case where gradients might be `None` (skip those parameters)

__Test your function:__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

# Create a simple model
model = nn.Linear(10, 1).cuda()
loss_fn = nn.MSELoss()

# Forward and backward pass
x = torch.randn(32, 10).cuda()
y = torch.randn(32, 1).cuda()
output = model(x)
loss = loss_fn(output, y)
loss.backward()

# Average gradients across ranks
average_gradients(model, world_size=2)

# Verify gradients are averaged (should be same on all ranks after all_reduce)
print(f"Gradient on rank {rank}: {model.weight.grad}")
```

### Broadcast with Verification

Implement a function that broadcasts a tensor from rank 0 to all other ranks and verifies the result.

__Requirements:__

- Function signature: `broadcast_and_verify(tensor, root=0)`
- If current rank is root: create or use the provided tensor
- If current rank is not root: create a zero tensor of the same shape
- Use `dist.broadcast()` to send data from root to all ranks
- After broadcast, verify that all ranks have identical data
- Return the broadcasted tensor and a boolean indicating if verification passed

__Test your function:__
```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

if rank == 0:
    # Root rank creates data
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
else:
    data = None

result, verified = broadcast_and_verify(data, root=0)
print(f"Rank {rank}: {result}, Verified: {verified}")
```

### DistributedSampler Wrapper

Create a wrapper class for `DistributedSampler` that automatically handles epoch setting.

__Requirements:__

- Class name: `AutoEpochDistributedSampler`
- Inherit from `torch.utils.data.distributed.DistributedSampler`
- Override `__iter__()` to automatically call `set_epoch()` with an internal epoch counter
- Add a method `reset()` to reset the epoch counter
- Add a method `get_epoch()` to return the current epoch number
- The sampler should increment the epoch counter each time `__iter__()` is called

__Test your function:__
```python
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = list(range(size))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = DummyDataset(100)
sampler = AutoEpochDistributedSampler(
    dataset, 
    num_replicas=world_size,
    rank=rank
)
dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

# Iterate through epochs - epoch should auto-increment
for epoch in range(3):
    for batch in dataloader:
        pass  # Process batch
    print(f"Epoch {sampler.get_epoch()} completed")
```

### Gradient Synchronization Timing

Write a script that measures the time taken for gradient synchronization using AllReduce.

__Requirements:__

- Function signature: `measure_sync_time(model, num_iterations=10)`
- Create a model with configurable size (number of parameters)
- Run forward and backward pass
- Time the `dist.all_reduce()` operation for all gradients
- Return average synchronization time in milliseconds
- Test with different model sizes (1M, 10M, 100M parameters) and different world sizes
- Print results showing how sync time scales with model size and world size

__Test your function:__
```python
import torch
import torch.nn as nn
import time
import torch.distributed as dist

def create_model(num_params):
    """Create a simple model with approximately num_params parameters"""
    # Simple linear layers to approximate parameter count
    layers = []
    # Rough calculation: for Linear(in, out), params = in * out + out
    # This is simplified - adjust as needed
    return nn.Sequential(
        nn.Linear(1000, num_params // 2000),
        nn.ReLU(),
        nn.Linear(num_params // 2000, 10)
    ).cuda()

model_sizes = [1e6, 10e6, 100e6]  # 1M, 10M, 100M parameters

for size in model_sizes:
    model = create_model(int(size))
    sync_time = measure_sync_time(model, num_iterations=10)
    print(f"Model size: {size/1e6:.1f}M params, "
          f"Avg sync time: {sync_time:.2f} ms, "
          f"World size: {dist.get_world_size()}")
```



## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Calculate memory requirements for different model sizes and precisions
- Initialize distributed process groups correctly
- Understand how gradient synchronization works in DDP
- Use collective operations (AllReduce, Broadcast) effectively
- Work with DistributedSampler for data partitioning
- Measure and analyze communication overhead in distributed training