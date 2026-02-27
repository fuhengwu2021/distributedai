\fancydividerwithicon[center]{hand.png}


## Exercises


### Compare ZeRO Stages

Implement a benchmark comparing DeepSpeed ZeRO stages 1, 2, and 3 on the same model.

__Requirements:__

- Train a 7B parameter model (or smaller if GPU-limited)
- Test ZeRO Stage 1 (optimizer state partitioning)
- Test ZeRO Stage 2 (+ gradient partitioning)
- Test ZeRO Stage 3 (+ parameter partitioning)
- Measure for each stage:
  - Peak GPU memory per device
  - Training throughput (tokens/second)
  - Communication overhead
- Generate comparison table and memory breakdown

__Test your implementation:__
```python
import deepspeed
import torch

def benchmark_zero_stage(model, stage: int, num_steps: int = 100):
    """Benchmark a specific ZeRO stage."""
    ds_config = {
        "train_batch_size": 32,
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "bf16": {"enabled": True},
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    
    # Benchmark training
    memory, throughput = run_training(model_engine, num_steps)
    return memory, throughput

# Compare stages
results = {}
for stage in [1, 2, 3]:
    model = create_model()  # Fresh model for each stage
    memory, throughput = benchmark_zero_stage(model, stage)
    results[f"ZeRO-{stage}"] = {"memory_gb": memory, "throughput": throughput}

print("Stage | Memory (GB) | Throughput (tok/s)")
for name, metrics in results.items():
    print(f"{name} | {metrics['memory_gb']:.1f} | {metrics['throughput']:.1f}")
```

### Implement CPU Offloading

Configure and benchmark DeepSpeed ZeRO-Offload for training models larger than GPU memory.

__Requirements:__

- Configure ZeRO Stage 3 with CPU offloading
- Test optimizer state offloading
- Test parameter offloading
- Measure:
  - Maximum trainable model size
  - Training throughput vs GPU-only
  - CPU memory usage
  - PCIe bandwidth utilization

__Test your implementation:__
```python
import deepspeed

# ZeRO-3 with full CPU offload
offload_config = {
    "train_batch_size": 8,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "bf16": {"enabled": True},
}

# Test with increasingly large models
model_sizes = ["1B", "3B", "7B", "13B"]
for size in model_sizes:
    try:
        model = create_model(size)
        engine, _, _, _ = deepspeed.initialize(model=model, config=offload_config)
        
        # Measure memory and throughput
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        cpu_mem = get_cpu_memory_usage()
        throughput = benchmark_throughput(engine, num_steps=10)
        
        print(f"{size}: GPU={gpu_mem:.1f}GB, CPU={cpu_mem:.1f}GB, {throughput:.1f} tok/s")
    except RuntimeError as e:
        print(f"{size}: OOM - {e}")
        break
```

### Implement Tensor Parallelism

Implement a simplified tensor parallel linear layer to understand Megatron-style parallelism.

__Requirements:__

- Class signature:
```python
class ColumnParallelLinear(nn.Module):
    """Linear layer with column-wise parallelism."""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class RowParallelLinear(nn.Module):
    """Linear layer with row-wise parallelism."""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```
- Column parallel: Split output features across ranks
- Row parallel: Split input features across ranks
- Implement AllReduce for row parallel output
- Verify correctness against standard linear layer

__Test your implementation:__
```python
import torch
import torch.distributed as dist

# Create parallel layers
world_size = dist.get_world_size()
rank = dist.get_rank()

col_linear = ColumnParallelLinear(1024, 4096, world_size, rank).cuda()
row_linear = RowParallelLinear(4096, 1024, world_size, rank).cuda()

# Test forward pass
x = torch.randn(32, 1024).cuda()
y = col_linear(x)  # Shape: (32, 4096 // world_size)
z = row_linear(y)  # Shape: (32, 1024) after AllReduce

print(f"Rank {rank}: input={x.shape}, after col={y.shape}, after row={z.shape}")

# Verify correctness (gather outputs and compare with standard linear)
```

### Implement Pipeline Parallelism

Create a simple pipeline parallel training loop with micro-batching.

__Requirements:__

- Split a model into N pipeline stages
- Implement 1F1B (one forward, one backward) schedule
- Handle micro-batch accumulation
- Measure pipeline bubble overhead
- Compare with data parallel baseline

__Test your implementation:__
```python
class PipelineStage(nn.Module):
    """A single pipeline stage."""
    def __init__(self, layers: nn.ModuleList, stage_id: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PipelineParallel:
    def __init__(self, model: nn.Module, num_stages: int, num_microbatches: int):
        self.stages = self.split_model(model, num_stages)
        self.num_microbatches = num_microbatches
    
    def split_model(self, model, num_stages) -> list:
        """Split model into pipeline stages."""
        pass
    
    def forward_backward(self, batch):
        """Execute 1F1B schedule."""
        pass

# Test pipeline parallel
model = create_transformer(num_layers=24)
pp = PipelineParallel(model, num_stages=4, num_microbatches=8)

batch = torch.randn(64, 512, 1024).cuda()
loss = pp.forward_backward(batch)
print(f"Pipeline loss: {loss.item():.4f}")
```

### Configure Hybrid Parallelism

Set up a training configuration combining data, tensor, and pipeline parallelism.

__Requirements:__

- Configure 3D parallelism (DP × TP × PP)
- Example: 8 GPUs with DP=2, TP=2, PP=2
- Verify correct process group setup
- Measure scaling efficiency vs single-dimension parallelism
- Document communication patterns for each dimension

__Test your implementation:__
```python
import torch.distributed as dist

def setup_3d_parallelism(world_size: int, dp: int, tp: int, pp: int):
    """Set up process groups for 3D parallelism."""
    assert dp * tp * pp == world_size, "DP × TP × PP must equal world_size"
    
    rank = dist.get_rank()
    
    # Calculate position in 3D grid
    dp_rank = rank // (tp * pp)
    tp_rank = (rank // pp) % tp
    pp_rank = rank % pp
    
    # Create process groups
    # Data parallel group: same TP and PP position
    # Tensor parallel group: same DP and PP position
    # Pipeline parallel group: same DP and TP position
    
    return dp_rank, tp_rank, pp_rank, dp_group, tp_group, pp_group

# Test with 8 GPUs: DP=2, TP=2, PP=2
dp_rank, tp_rank, pp_rank, dp_group, tp_group, pp_group = setup_3d_parallelism(
    world_size=8, dp=2, tp=2, pp=2
)

print(f"Rank {dist.get_rank()}: DP={dp_rank}, TP={tp_rank}, PP={pp_rank}")
print(f"DP group size: {dist.get_world_size(dp_group)}")
print(f"TP group size: {dist.get_world_size(tp_group)}")
print(f"PP group size: {dist.get_world_size(pp_group)}")
```


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Compare and choose appropriate ZeRO stages for your workload
- Configure CPU offloading for training large models on limited GPU memory
- Understand tensor parallelism and implement parallel linear layers
- Implement pipeline parallelism with efficient scheduling
- Configure hybrid 3D parallelism for maximum scaling
- Analyze communication patterns in different parallelism strategies
