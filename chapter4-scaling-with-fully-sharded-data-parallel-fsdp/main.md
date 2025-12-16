# Chapter 4 — Scaling with Fully Sharded Data Parallel (FSDP)

When your model doesn't fit on a single GPU, you need to split it across multiple GPUs. FSDP (Fully Sharded Data Parallel) shards parameters, gradients, and optimizer states across ranks, letting you train models that are much larger than what fits in a single GPU's memory.

PyTorch has two FSDP implementations: the original `FullyShardedDataParallel` wrapper class (which uses a flat-parameter approach), and the newer per-parameter-sharding design accessed via `fully_shard()`. This chapter focuses on the newer design—it's simpler, more flexible, and is the direction PyTorch is moving. The original FSDP still works, but for new projects, you should use the per-parameter-sharding API.

## Why FSDP Enables Larger-Than-Memory Models

The basic idea is straightforward: instead of keeping the full model on every GPU, you split it up. Each GPU holds a shard of the parameters. During forward pass, you all-gather the parameters you need. During backward, you compute gradients on the local shard, then reduce-scatter to aggregate across GPUs.

The memory savings come from three places: parameters, gradients, and optimizer states. With FSDP, you're only storing 1/N of each on each GPU (where N is the number of GPUs). For a 70B parameter model with Adam optimizer, that's the difference between needing 8 GPUs versus 64 GPUs.

The per-parameter-sharding design (introduced in PyTorch issue #114299) shards each parameter individually on dimension 0. This is simpler than the original flat-parameter approach and enables several useful features: flexible fp8 all-gather, frozen parameters in the same group, communication-free sharded state dicts, and better compiler integration.

## Understanding FSDP2: The Per-Parameter-Sharding API

The new API uses `fully_shard()` as a function that modifies modules in place. No wrapper class needed—it's more functional and composable.

Here's what the API looks like:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

# Initialize device mesh
mesh = init_device_mesh("cuda", (world_size,))

# Apply FSDP to your model
fully_shard(
    model,
    mesh=mesh,
    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
)
```

Key parameters:

- `mesh`: The `DeviceMesh` over which to shard. For normal FSDP, this is a 1D mesh. For HSDP (hybrid sharding), it can be 2D.
- `reshard_after_forward`: Controls when parameters are resharded. `True` (default) reshards after forward (like ZeRO-3), `False` keeps them unsharded (like ZeRO-2), or an `int` reshards to an intermediate size (like ZeRO++ hpZ).
- `mp_policy`: Mixed precision settings—parameter dtype, reduce dtype, output dtype.
- `offload_policy`: CPU/NVMe offloading configuration.

The `reshard_after_forward` parameter is important. If you set it to `True`, parameters are resharded after forward, saving memory but requiring all-gather in backward. If `False`, parameters stay unsharded after forward, using more memory but avoiding all-gather in backward. The default (`True`) is usually the right choice unless you have memory headroom and want to reduce communication.

## A Complete Working Example

Let's build a complete example that you can run on 2 GPUs. This trains a model that's intentionally too large for a single GPU, demonstrating FSDP in action.

```python
"""
FSDP2 training example - can run on 2 GPUs to demonstrate sharding

Usage:
    torchrun --nproc_per_node=2 code/train_fsdp2.py
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
```

Save this as `code/train_fsdp2.py` and run it with:

```bash
torchrun --nproc_per_node=2 code/train_fsdp2.py
```

This example creates a model with about 135M parameters—too large to fit comfortably on a single GPU with optimizer states. With FSDP2, it shards across 2 GPUs, so each GPU only holds half the parameters, gradients, and optimizer states.

## Key Features of FSDP2

The per-parameter-sharding design enables several things that the original FSDP couldn't do easily:

**Flexible fp8 all-gather**: You can mix fp8 weights and non-fp8 parameters in the same all-gather. This is useful for models that use fp8 for some layers but not others.

**Frozen parameters**: You can have frozen and trainable parameters in the same communication group without extra memory overhead. This is handy for fine-tuning where you freeze some layers.

**Simpler checkpointing**: Sharded state dicts match the training representation, so saving and loading is straightforward. Each rank saves its shard, and loading is just reading the shards back.

**Better compiler integration**: The per-parameter design works better with `torch.compile` and other graph compilers that want to optimize communication patterns.

## Activation Checkpointing and Offloading

Activation checkpointing is almost always used with FSDP. Instead of storing all activations during forward, you recompute them during backward. This can cut activation memory by 50-80%.

```python
from torch.utils.checkpoint import checkpoint

# After applying FSDP
fully_shard(model, mesh=mesh)

# Use checkpointing in forward
def forward_with_checkpoint(x):
    return checkpoint(model, x)
```

CPU offloading moves optimizer states to CPU memory, freeing GPU memory at the cost of slower training. The FSDP2 API supports this:

```python
from torch.distributed.fsdp import OffloadPolicy

fully_shard(
    model,
    mesh=mesh,
    offload_policy=OffloadPolicy(offload_type="cpu"),
)
```

When to use offloading: if you're still OOM after full-shard and checkpointing, offloading can help, but expect 20-50% slowdown. For most cases, full-shard + checkpointing is enough.

## Multi-Node FSDP Training

Multi-node FSDP works the same way as multi-node DDP—you need process group initialization and proper networking. The main difference is checkpointing: with FSDP2, sharded state dicts are straightforward—each rank writes its shard, and you can load them back without all-gather.

For checkpointing, you'll want shared storage (NFS, S3, etc.) or parallel writes. The sharded approach helps since each rank only writes its shard.

## Comparing FSDP2 with ZeRO and DDP

**DDP** is the simplest—every GPU has a full copy of everything. Use it when your model fits on a single GPU.

**ZeRO (DeepSpeed)** does staged sharding similar to FSDP, but it's part of the DeepSpeed ecosystem. ZeRO-1 shards optimizer states, ZeRO-2 adds gradients, ZeRO-3 adds parameters.

**FSDP2** is PyTorch-native and integrates tightly with autograd. The per-parameter-sharding design is simpler (about 3k lines of code versus 14k for the original) and more flexible.

Which to choose? If you're using PyTorch and want native integration, FSDP2 is a good fit. The new design is the direction PyTorch is moving, so new projects should use it.

## Practical Tips

A few things that trip people up:

**State dict handling**: With FSDP2, sharded state dicts match the training representation, so saving and loading is straightforward. Each rank saves its shard.

**Shared parameters**: If you have shared parameters (same tensor used in multiple places), they need to be in the same FSDP group. This is a limitation—there's no way to preserve sharedness after parameter swapping.

**Memory profiling**: Use `torch.profiler` or `nvidia-smi` to see where memory is actually being used. Sometimes the bottleneck isn't what you think—it could be activations, not parameters.

**Tuning reshard_after_forward**: The default (`True`) is usually right. But if you have memory headroom and want to reduce communication, try `False`. You can also use an intermediate size (like `int` for ZeRO++ hpZ style).

Start simple: use full-shard with checkpointing, see if that's enough. Only add offloading if you're still hitting memory limits.

The code example above shows a complete working setup. Run it on 2 GPUs to see FSDP2 in action—you'll see the model sharded across GPUs, with each rank only holding its portion of the parameters.

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Per-Parameter-Sharding FSDP RFC](https://github.com/pytorch/pytorch/issues/114299)
- [TorchTitan FSDP Guide](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
