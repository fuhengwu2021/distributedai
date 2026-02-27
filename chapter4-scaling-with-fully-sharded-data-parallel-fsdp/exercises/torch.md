\fancydividerwithicon[center]{hand.png}


## Exercises


### Implement Manual Parameter Sharding

Implement a simplified version of FSDP's parameter sharding to understand how it enables training models larger than GPU memory.

__Requirements:__

- Function signature:
```python
def shard_parameters(model: nn.Module, world_size: int, rank: int) -> dict:
    """Shard model parameters across ranks, keeping only local shard."""
    pass

def gather_parameters(sharded_params: dict, world_size: int) -> dict:
    """Gather all parameter shards to reconstruct full parameters."""
    pass
```
- Flatten all parameters into a single tensor
- Divide the flattened tensor into `world_size` equal shards
- Each rank keeps only its shard (1/world_size of total parameters)
- Implement gather to reconstruct full parameters for forward pass

__Test your implementation:__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 100)
)

# Shard parameters
sharded = shard_parameters(model, world_size=4, rank=dist.get_rank())
print(f"Rank {dist.get_rank()}: sharded size = {sharded['shard'].numel()} elements")

# Gather for forward pass
full_params = gather_parameters(sharded, world_size=4)
print(f"Full params size: {sum(p.numel() for p in full_params.values())} elements")
```

### Compare FSDP Sharding Strategies

Write a benchmark comparing different FSDP sharding strategies and their memory/performance tradeoffs.

__Requirements:__

- Test three sharding strategies:
  - `FULL_SHARD`: Shard parameters, gradients, and optimizer states
  - `SHARD_GRAD_OP`: Shard gradients and optimizer states only
  - `NO_SHARD`: No sharding (like DDP)
- Measure for each strategy:
  - Peak GPU memory usage
  - Training throughput (samples/second)
  - Communication volume
- Use a model that doesn't fit in single GPU memory with `NO_SHARD`
- Generate comparison table and plots

__Test your implementation:__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

strategies = [
    ShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.NO_SHARD,
]

results = {}
for strategy in strategies:
    model = create_large_model()  # e.g., 7B parameter model
    fsdp_model = FSDP(model, sharding_strategy=strategy)
    
    memory, throughput = benchmark_training(fsdp_model, num_steps=100)
    results[strategy.name] = {"memory_gb": memory, "throughput": throughput}

# Print comparison table
print("Strategy | Memory (GB) | Throughput (samples/s)")
for name, metrics in results.items():
    print(f"{name:15} | {metrics['memory_gb']:10.1f} | {metrics['throughput']:10.1f}")
```

### Implement Custom FSDP Wrapping Policy

Create a custom auto-wrap policy that wraps transformer layers individually for optimal memory efficiency.

__Requirements:__

- Class signature:
```python
def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    min_num_params: int = 1e6
) -> bool:
    """Custom wrap policy for transformer models."""
    pass
```
- Wrap each transformer block (attention + FFN) as a single FSDP unit
- Don't wrap embedding layers (they should be in root FSDP)
- Don't wrap final output projection
- Allow configurable minimum parameter threshold

__Test your implementation:__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import _module_wrap_policy
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

fsdp_model = FSDP(
    model,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        min_num_params=1e6
    )
)

# Verify wrapping
def count_fsdp_units(module, depth=0):
    count = 1 if isinstance(module, FSDP) else 0
    for child in module.children():
        count += count_fsdp_units(child, depth + 1)
    return count

print(f"Number of FSDP units: {count_fsdp_units(fsdp_model)}")
```

### Implement Mixed Precision with FSDP

Configure and benchmark FSDP with different mixed precision policies.

__Requirements:__

- Test three precision configurations:
  - Full FP32 (baseline)
  - BF16 compute with FP32 parameters
  - BF16 compute with BF16 parameters and gradients
- Measure:
  - Memory savings vs FP32
  - Training loss convergence
  - Throughput improvement
- Handle gradient scaling for FP16 (if applicable)

__Test your implementation:__
```python
from torch.distributed.fsdp import MixedPrecision
import torch

# Define precision policies
fp32_policy = MixedPrecision()

bf16_compute_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

bf16_full_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

policies = {
    "FP32": fp32_policy,
    "BF16 Compute": bf16_compute_policy,
    "BF16 Full": bf16_full_policy,
}

for name, policy in policies.items():
    model = FSDP(create_model(), mixed_precision=policy)
    memory, loss, throughput = train_and_measure(model, num_steps=100)
    print(f"{name}: Memory={memory:.1f}GB, Loss={loss:.4f}, Throughput={throughput:.1f}")
```

### Implement FSDP Checkpointing

Create a robust checkpointing system for FSDP models with support for resharding.

__Requirements:__

- Save checkpoints that can be loaded with different world sizes
- Support both full state dict and sharded state dict formats
- Implement checkpoint validation (verify saved == loaded)
- Handle optimizer state saving/loading
- Support resuming training from checkpoint

__Test your implementation:__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

class FSDPCheckpointer:
    def __init__(self, model: FSDP, optimizer, checkpoint_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
    
    def save(self, step: int, full_state: bool = True):
        """Save checkpoint."""
        pass
    
    def load(self, step: int) -> dict:
        """Load checkpoint and return metadata."""
        pass
    
    def validate(self, step: int) -> bool:
        """Validate checkpoint integrity."""
        pass

# Test saving and loading
checkpointer = FSDPCheckpointer(fsdp_model, optimizer, "./checkpoints")

# Train for a few steps
train_steps(fsdp_model, optimizer, num_steps=10)

# Save checkpoint
checkpointer.save(step=10)

# Validate
assert checkpointer.validate(step=10), "Checkpoint validation failed!"

# Load and resume
metadata = checkpointer.load(step=10)
print(f"Resumed from step {metadata['step']}")
```


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Understand how FSDP shards parameters across GPUs
- Compare and choose appropriate sharding strategies for your workload
- Implement custom wrapping policies for transformer models
- Configure mixed precision training with FSDP
- Build robust checkpointing systems for distributed training
- Optimize memory usage while maintaining training throughput
