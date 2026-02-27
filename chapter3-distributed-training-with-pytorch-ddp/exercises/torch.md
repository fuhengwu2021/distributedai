\fancydividerwithicon[center]{hand.png}


## Exercises


### Implement Manual Gradient Synchronization

Implement gradient synchronization manually using `torch.distributed` primitives to understand what DDP does internally.

__Requirements:__

- Function signature:
```python
def sync_gradients(model: nn.Module, world_size: int) -> None:
    """Synchronize gradients across all ranks using AllReduce."""
    pass
```
- Iterate through all model parameters with `requires_grad=True`
- Use `dist.all_reduce()` with `ReduceOp.SUM` to sum gradients
- Divide by `world_size` to compute the average
- Handle parameters with `None` gradients (skip them)

__Test your implementation:__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

# Simple model
model = nn.Linear(100, 10).cuda()

# Forward and backward
x = torch.randn(32, 100).cuda()
y = torch.randn(32, 10).cuda()
loss = nn.MSELoss()(model(x), y)
loss.backward()

# Sync gradients manually
sync_gradients(model, world_size=dist.get_world_size())

# Verify: gradients should be identical across all ranks
print(f"Rank {dist.get_rank()}: weight grad sum = {model.weight.grad.sum().item():.4f}")
```

### Compare DP vs DDP Performance

Write a benchmark script that compares DataParallel (DP) and DistributedDataParallel (DDP) performance.

__Requirements:__

- Create a ResNet-50 model for benchmarking
- Implement training loop for both DP and DDP
- Measure:
  - Throughput (samples/second)
  - GPU memory usage per device
  - GPU utilization percentage
- Run with 2, 4, and 8 GPUs (if available)
- Plot scaling efficiency: `efficiency = (N_gpu * throughput_N) / (1 * throughput_1)`

__Test your implementation:__
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50
import time

def benchmark_dp(model, batch_size, num_iterations):
    """Benchmark DataParallel."""
    model = nn.DataParallel(model)
    # ... training loop
    return throughput

def benchmark_ddp(model, batch_size, num_iterations, rank, world_size):
    """Benchmark DistributedDataParallel."""
    model = DDP(model, device_ids=[rank])
    # ... training loop
    return throughput

# Compare results
dp_throughput = benchmark_dp(resnet50().cuda(), batch_size=64, num_iterations=100)
ddp_throughput = benchmark_ddp(resnet50().cuda(), batch_size=64, num_iterations=100, rank, world_size)

print(f"DP throughput: {dp_throughput:.1f} samples/sec")
print(f"DDP throughput: {ddp_throughput:.1f} samples/sec")
print(f"DDP speedup: {ddp_throughput/dp_throughput:.2f}x")
```

### Implement Gradient Bucketing

Implement a simplified version of DDP's gradient bucketing to understand communication optimization.

__Requirements:__

- Class signature:
```python
class GradientBucketer:
    def __init__(self, model: nn.Module, bucket_size_mb: float = 25.0):
        """Initialize bucketer with model and bucket size."""
        pass
    
    def create_buckets(self) -> list[list[nn.Parameter]]:
        """Group parameters into buckets based on size."""
        pass
    
    def sync_bucket(self, bucket: list[nn.Parameter]) -> None:
        """Synchronize gradients for a single bucket."""
        pass
    
    def sync_all(self) -> None:
        """Synchronize all buckets in reverse order."""
        pass
```
- Group parameters into buckets of approximately `bucket_size_mb` megabytes
- Flatten gradients within each bucket for efficient AllReduce
- Sync buckets in reverse order (last layer first, like DDP)
- Measure and report communication time per bucket

__Test your implementation:__
```python
from torchvision.models import resnet50

model = resnet50().cuda()
bucketer = GradientBucketer(model, bucket_size_mb=25.0)

# Show bucket structure
buckets = bucketer.create_buckets()
for i, bucket in enumerate(buckets):
    size_mb = sum(p.numel() * 4 / 1e6 for p in bucket)
    print(f"Bucket {i}: {len(bucket)} params, {size_mb:.1f} MB")

# Forward/backward
x = torch.randn(32, 3, 224, 224).cuda()
loss = model(x).sum()
loss.backward()

# Sync with timing
import time
start = time.time()
bucketer.sync_all()
print(f"Total sync time: {(time.time() - start) * 1000:.1f} ms")
```

### Implement Fault-Tolerant DDP Training

Create a training script with checkpointing and automatic recovery from failures.

__Requirements:__

- Save checkpoints periodically (every N steps)
- Checkpoint should include:
  - Model state dict
  - Optimizer state dict
  - Learning rate scheduler state
  - Current epoch and step
  - Random states (torch, numpy, python)
- Implement automatic resume from latest checkpoint
- Handle rank 0 saving with barrier synchronization
- Test recovery by killing and restarting training

__Test your implementation:__
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class FaultTolerantTrainer:
    def __init__(self, model, optimizer, checkpoint_dir, save_every=100):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.step = 0
        self.epoch = 0
    
    def save_checkpoint(self):
        """Save checkpoint (rank 0 only)."""
        pass
    
    def load_checkpoint(self) -> bool:
        """Load latest checkpoint if exists. Return True if loaded."""
        pass
    
    def train_step(self, batch):
        """Single training step with periodic checkpointing."""
        pass

# Usage
trainer = FaultTolerantTrainer(model, optimizer, "./checkpoints")
if trainer.load_checkpoint():
    print(f"Resumed from step {trainer.step}")

for epoch in range(trainer.epoch, num_epochs):
    for batch in dataloader:
        trainer.train_step(batch)
```

### Analyze Communication Patterns with NCCL

Write a script that profiles NCCL communication patterns during DDP training.

__Requirements:__

- Use `torch.cuda.Event` to measure communication time
- Profile different collective operations:
  - AllReduce (gradient sync)
  - Broadcast (parameter sync)
  - AllGather (if used)
- Measure bandwidth utilization: `bandwidth = data_size / time`
- Compare with theoretical peak bandwidth
- Generate a report showing communication bottlenecks

__Test your implementation:__
```python
import torch
import torch.distributed as dist

def profile_allreduce(tensor_size_mb: float, num_iterations: int = 10):
    """Profile AllReduce performance."""
    tensor = torch.randn(int(tensor_size_mb * 1e6 / 4)).cuda()
    
    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    # Profile
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        dist.all_reduce(tensor)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / num_iterations
    bandwidth_gbps = (tensor_size_mb * 2 / 1000) / (time_ms / 1000)  # Ring AllReduce: 2x data
    
    return time_ms, bandwidth_gbps

# Test different sizes
for size_mb in [1, 10, 100, 500]:
    time_ms, bw = profile_allreduce(size_mb)
    print(f"Size: {size_mb:4d} MB, Time: {time_ms:6.2f} ms, Bandwidth: {bw:.1f} GB/s")
```


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Understand how gradient synchronization works at a low level
- Compare and contrast DP vs DDP performance characteristics
- Implement gradient bucketing for communication optimization
- Build fault-tolerant distributed training systems
- Profile and analyze NCCL communication patterns
- Identify and resolve communication bottlenecks in DDP training
