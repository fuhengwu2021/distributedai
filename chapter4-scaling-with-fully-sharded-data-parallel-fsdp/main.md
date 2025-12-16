# Chapter 4 — Scaling with Fully Sharded Data Parallel (FSDP)

In the previous chapter, we covered DistributedDataParallel (DDP), which replicates the entire model on each GPU and synchronizes gradients. DDP works great when your model fits on a single GPU—it's simple, efficient, and well-optimized. But what happens when your model is too large for a single GPU's memory?

This is where Fully Sharded Data Parallel (FSDP) comes in. FSDP extends DDP by sharding model parameters, gradients, and optimizer states across GPUs. Instead of each GPU holding a full copy of the model, each GPU only holds a shard. This lets you train models that are much larger than what fits in a single GPU's memory.

If you've been following along with the DDP examples in Chapter 3, you've seen how to scale training across multiple GPUs when the model fits on one GPU. Now we'll tackle the next challenge: training models that don't fit on a single GPU, even with mixed precision and activation checkpointing.

PyTorch has two FSDP implementations: the original `FullyShardedDataParallel` wrapper class (which uses a flat-parameter approach), and the newer per-parameter-sharding design accessed via `fully_shard()`. This chapter focuses on the newer design—it's simpler, more flexible, and is the direction PyTorch is moving. The original FSDP still works, but for new projects, you should use the per-parameter-sharding API, often called FSDP2.

## Why FSDP Enables Larger-Than-Memory Models

The basic idea is straightforward: instead of keeping the full model on every GPU, you split it up. Each GPU holds a shard of the parameters. During forward pass, you all-gather the parameters you need. During backward, you compute gradients on the local shard, then reduce-scatter to aggregate across GPUs.

But let's dig deeper into why this matters. When training a large model with DDP, each GPU needs to store:

1. **Model parameters**: The weights themselves. For a 70B parameter model in FP32, that's 280 GB. In BF16, it's 140 GB.
2. **Gradients**: Same size as parameters. Another 140 GB for BF16.
3. **Optimizer states**: For Adam, this includes momentum and variance buffers, which are 2x the parameter size. That's another 280 GB for BF16 parameters.
4. **Activations**: Depends on batch size and sequence length, but can easily be 50-200 GB for large models.

Total: For a 70B model with Adam, you're looking at 140 + 140 + 280 = 560 GB just for parameters, gradients, and optimizer states. Even the largest GPUs (H100 with 80 GB) can't fit this.

With FSDP, you shard all three across GPUs. With 8 GPUs, each GPU only holds 1/8 of the parameters, gradients, and optimizer states. That's 560 / 8 = 70 GB per GPU, which fits comfortably on an A100 (80 GB) or H100.

The memory savings come from three places: parameters, gradients, and optimizer states. With FSDP, you're only storing 1/N of each on each GPU (where N is the number of GPUs). For a 70B parameter model with Adam optimizer, that's the difference between needing 8 GPUs versus 64 GPUs.

### The Memory Breakdown

Let's look at a concrete example. Say you're training a transformer model with 7 billion parameters:

**Without FSDP (DDP):**
- Parameters (BF16): 7B × 2 bytes = 14 GB
- Gradients (BF16): 14 GB
- Optimizer states (Adam, FP32): 7B × 4 bytes × 2 (momentum + variance) = 56 GB
- **Total per GPU**: 84 GB

This won't fit on most GPUs, and you haven't even accounted for activations yet.

**With FSDP on 8 GPUs:**
- Parameters (BF16): 14 GB / 8 = 1.75 GB
- Gradients (BF16): 14 GB / 8 = 1.75 GB
- Optimizer states (FP32): 56 GB / 8 = 7 GB
- **Total per GPU**: 10.5 GB

Now you have headroom for activations, and the model fits comfortably on A100s or even V100s.

### How FSDP Works: All-Gather and Reduce-Scatter

FSDP uses two key collective operations:

1. **All-Gather**: During forward pass, when you need parameters that aren't on the current GPU, FSDP all-gathers them from all GPUs. After all-gather, all GPUs have a full copy of the needed parameters, but only temporarily.

2. **Reduce-Scatter**: During backward pass, gradients are computed locally, then reduce-scattered across GPUs. Each GPU ends up with its shard of the aggregated gradients.

The key insight is that you don't need all parameters at once. During forward pass, you process layers sequentially. FSDP can all-gather parameters for the current layer, use them, then free them before moving to the next layer. This is why activation checkpointing is so important with FSDP—it reduces activation memory so you have room for the all-gathered parameters.

The per-parameter-sharding design (introduced in PyTorch issue #114299) shards each parameter individually on dimension 0. This is simpler than the original flat-parameter approach and enables several useful features: flexible fp8 all-gather, frozen parameters in the same group, communication-free sharded state dicts, and better compiler integration.

## Understanding FSDP2: The Per-Parameter-Sharding API

The new API uses `fully_shard()` as a function that modifies modules in place. No wrapper class needed—it's more functional and composable. This is a significant departure from the original FSDP, which used a wrapper class similar to DDP.

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

The key difference from the original FSDP is that `fully_shard()` modifies the model in place. It doesn't return a wrapped model—your model becomes an FSDP model. This makes it easier to compose with other transformations and works better with `torch.compile`.

### Device Mesh: The Foundation

The `DeviceMesh` is a new abstraction in PyTorch that represents a logical arrangement of devices. For FSDP, you typically use a 1D mesh (all GPUs in a single dimension):

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh for standard FSDP
mesh = init_device_mesh("cuda", (world_size,))
```

This creates a mesh where all GPUs are arranged in a single dimension. For 8 GPUs, this is `[0, 1, 2, 3, 4, 5, 6, 7]`.

For hybrid sharding (HSDP), you can use a 2D mesh:

```python
# 2D mesh for hybrid sharding
# 4 nodes × 8 GPUs per node = 32 GPUs total
mesh = init_device_mesh("cuda", (4, 8))
```

This arranges GPUs in a 2D grid, which is useful for very large scale training where you want to shard within a node but replicate across nodes.

### Key Parameters

**`mesh`**: The `DeviceMesh` over which to shard. For normal FSDP, this is a 1D mesh. For HSDP (hybrid sharding), it can be 2D.

**`reshard_after_forward`**: Controls when parameters are resharded. This is one of the most important parameters:

- `True` (default): Parameters are resharded after forward pass. This saves memory but requires all-gather in backward. This is like ZeRO-3.
- `False`: Parameters stay unsharded after forward. Uses more memory but avoids all-gather in backward. This is like ZeRO-2.
- `int`: Reshards to an intermediate size. For example, `reshard_after_forward=2` means parameters are sharded across 2 GPUs instead of all GPUs. This is like ZeRO++ hpZ (hybrid parameter zero).

The default (`True`) is usually the right choice unless you have memory headroom and want to reduce communication. If you're memory-constrained, stick with `True`. If you have extra memory and communication is your bottleneck, try `False`.

**`mp_policy`**: Mixed precision settings. You can specify:
- `param_dtype`: Dtype for parameters (e.g., `torch.bfloat16`)
- `reduce_dtype`: Dtype for gradient reduction (e.g., `torch.float32`)
- `output_dtype`: Dtype for outputs (optional)

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,  # Keep gradients in FP32 for precision
)
```

**`offload_policy`**: CPU/NVMe offloading configuration. This moves optimizer states or parameters to CPU/NVMe to save GPU memory:

```python
from torch.distributed.fsdp import OffloadPolicy

fully_shard(
    model,
    mesh=mesh,
    offload_policy=OffloadPolicy(offload_type="cpu"),
)
```

Offloading comes with a performance cost (20-50% slowdown), so only use it if you're still OOM after trying everything else.

### Hierarchical Sharding

You can apply `fully_shard()` at different levels of your model hierarchy. For example, you might shard individual transformer layers but not the embedding layer:

```python
# Shard each transformer layer individually
for layer in model.transformer.layers:
    fully_shard(layer, mesh=mesh)

# Don't shard the embedding layer (it's small)
# fully_shard(model.embedding, mesh=mesh)  # Skip this
```

This gives you fine-grained control over what gets sharded. Small layers (like embeddings) might not benefit from sharding and can add communication overhead, so you can leave them unsharded.

## A Complete Working Example: Training a Transformer with FSDP2

Let's build a complete example based on the PyTorch FSDP2 examples. This trains a transformer model that's intentionally too large for a single GPU, demonstrating FSDP in action.

The example uses a transformer architecture similar to GPT, with configurable layers, heads, and dimensions. Here's the model definition:

```python
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    n_layers: int = 10
    vocab_size: int = 1024
    max_seq_len: int = 64
    dim: int = 512
    n_heads: int = 8
    dropout_p: float = 0.1


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=self.dropout_p if self.training else 0,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens):
        bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output
```

Now here's the training script that uses FSDP2:

```python
import argparse
import os
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from model import ModelArgs, Transformer


def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cpu")

    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group(backend=backend, device_id=device)

    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    
    model_args = ModelArgs(
        n_layers=10,
        n_heads=8,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    
    # Create model on meta device first (no memory allocation)
    with torch.device("meta"):
        model = Transformer(model_args)
    
    # Setup FSDP
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    
    # Shard each layer individually
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    
    # Shard the entire model
    fully_shard(model, **fsdp_kwargs)

    # Move model to actual device and initialize parameters
    world_size = torch.distributed.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,))
    
    model.to_empty(device=device)
    model.reset_parameters()

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training loop
    for step in range(10):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()
        
        if torch.distributed.get_rank() == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    args = parser.parse_args()
    
    main(args)
```

Save this as `code/train_fsdp2_transformer.py` and run it with:

```bash
torchrun --nproc_per_node=2 code/train_fsdp2_transformer.py
```

To enable mixed precision:

```bash
torchrun --nproc_per_node=2 code/train_fsdp2_transformer.py --mixed-precision
```

This example creates a transformer with 10 layers, 512 dimensions, and 8 attention heads. With the default configuration, this model has roughly 50M parameters. While this might fit on a single GPU, the example demonstrates the FSDP2 workflow. For larger models (100M+ parameters), FSDP becomes essential.

Key points from this example:

1. **Meta device initialization**: We create the model on the meta device first, which doesn't allocate memory. This is useful for very large models where you can't even create the model on a single GPU.

2. **Hierarchical sharding**: We shard each transformer layer individually, then shard the entire model. This gives fine-grained control.

3. **Parameter initialization**: After applying FSDP, we call `to_empty()` to move the model to the actual device, then `reset_parameters()` to initialize weights. This works with the sharded representation.

4. **Mixed precision**: The example shows how to enable mixed precision training with FSDP2, which is essential for large models.

## Key Features of FSDP2

The per-parameter-sharding design enables several things that the original FSDP couldn't do easily:

**Flexible fp8 all-gather**: You can mix fp8 weights and non-fp8 parameters in the same all-gather. This is useful for models that use fp8 for some layers but not others. The original FSDP required all parameters in a group to have the same dtype, which was limiting.

**Frozen parameters**: You can have frozen and trainable parameters in the same communication group without extra memory overhead. This is handy for fine-tuning where you freeze some layers. With the original FSDP, you'd need separate groups, which added complexity.

**Simpler checkpointing**: Sharded state dicts match the training representation, so saving and loading is straightforward. Each rank saves its shard, and loading is just reading the shards back. No need to gather all parameters to rank 0, save, then shard again on load.

**Better compiler integration**: The per-parameter design works better with `torch.compile` and other graph compilers that want to optimize communication patterns. The compiler can see individual parameters and optimize all-gather/reduce-scatter operations more effectively.

**Communication-free state dicts**: When saving checkpoints, you can save sharded state dicts directly without gathering. This is faster and uses less memory. When loading, you can load sharded state dicts directly without broadcasting.

## Checkpointing with FSDP2

Checkpointing is crucial for long training runs, and FSDP2 makes it straightforward. The key insight is that you can save sharded state dicts directly—each rank saves its shard, and you can load them back without gathering.

There are two approaches: using the Distributed Checkpoint (DCP) API, or manually handling sharded state dicts. Let's look at both.

### Using the DCP API (Recommended)

The DCP API is the recommended way to save and load FSDP2 checkpoints. It handles all the complexity of sharded state dicts:

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
import os


def save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir):
    """Save checkpoint using DCP API."""
    rank = torch.distributed.get_rank()
    
    # Get state dicts with full_state_dict=False to get sharded dicts
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=False,  # Keep sharded
            cpu_offload=True,  # Offload to CPU for saving
        ),
    )
    
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        ),
    )
    
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save using DCP
    writer = FileSystemWriter(checkpoint_path)
    writer.save(
        {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
            "epoch": epoch,
        }
    )
    
    if rank == 0:
        print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint_dcp(model, optimizer, checkpoint_dir, epoch):
    """Load checkpoint using DCP API."""
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    
    # Load using DCP
    reader = FileSystemReader(checkpoint_path)
    checkpoint = reader.load()
    
    # Set state dicts
    set_model_state_dict(
        model=model,
        model_state_dict=checkpoint["model"],
        options=StateDictOptions(
            full_state_dict=False,
        ),
    )
    
    set_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        optim_state_dict=checkpoint["optimizer"],
        options=StateDictOptions(
            full_state_dict=False,
        ),
    )
    
    loaded_epoch = checkpoint["epoch"]
    if torch.distributed.get_rank() == 0:
        print(f"Checkpoint loaded from {checkpoint_path}, epoch {loaded_epoch}")
    
    return loaded_epoch
```

The DCP API handles all the complexity of sharded state dicts. Each rank saves its shard, and loading is just reading the shards back. No gathering, no broadcasting.

### Manual Sharded Checkpointing

If you need more control, you can manually handle sharded state dicts. Here's how:

```python
def save_checkpoint_manual(model, optimizer, epoch, checkpoint_dir):
    """Manually save sharded checkpoint."""
    rank = torch.distributed.get_rank()
    
    # Get sharded state dict
    model_sd = model.state_dict()  # Already sharded
    
    # Get optimizer state dict (also sharded)
    optim_sd = optimizer.state_dict()
    
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Each rank saves its shard
    model_path = os.path.join(checkpoint_path, f"model_rank_{rank}.pt")
    optim_path = os.path.join(checkpoint_path, f"optim_rank_{rank}.pt")
    
    torch.save(model_sd, model_path)
    torch.save(optim_sd, optim_path)
    
    # Save metadata on rank 0
    if rank == 0:
        metadata = {"epoch": epoch, "world_size": torch.distributed.get_world_size()}
        torch.save(metadata, os.path.join(checkpoint_path, "metadata.pt"))
        print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint_manual(model, optimizer, checkpoint_dir, epoch):
    """Manually load sharded checkpoint."""
    rank = torch.distributed.get_rank()
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    
    # Each rank loads its shard
    model_path = os.path.join(checkpoint_path, f"model_rank_{rank}.pt")
    optim_path = os.path.join(checkpoint_path, f"optim_rank_{rank}.pt")
    
    model_sd = torch.load(model_path, map_location="cpu")
    optim_sd = torch.load(optim_path, map_location="cpu")
    
    model.load_state_dict(model_sd)
    optimizer.load_state_dict(optim_sd)
    
    if rank == 0:
        metadata = torch.load(os.path.join(checkpoint_path, "metadata.pt"))
        print(f"Checkpoint loaded from {checkpoint_path}, epoch {metadata['epoch']}")
    
    return epoch
```

The manual approach gives you more control but requires careful handling of sharded state dicts. The DCP API is recommended for most use cases.

### Full State Dict for Evaluation

Sometimes you need a full (unsharded) state dict, for example to save a final model for inference or to share with others. You can gather all shards to rank 0:

```python
def save_full_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save full (unsharded) checkpoint on rank 0."""
    rank = torch.distributed.get_rank()
    
    # Get full state dict (gathers all shards to rank 0)
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,  # Gather all shards
            cpu_offload=True,
        ),
    )
    
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    
    # Only rank 0 saves
    if rank == 0:
        checkpoint = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
            "epoch": epoch,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Full checkpoint saved to {checkpoint_path}")
    
    torch.distributed.barrier()
```

This gathers all shards to rank 0, which uses more memory but gives you a single checkpoint file that can be loaded on any number of GPUs.

## Prefetching: Optimizing Communication

FSDP2 supports explicit prefetching to overlap communication with computation. The idea is to prefetch parameters for the next layer while computing the current layer.

### Forward Prefetching

Forward prefetching all-gathers parameters for upcoming layers while processing the current layer:

```python
def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Set forward prefetching for transformer layers."""
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)
```

This tells each layer to prefetch parameters for the next `num_to_forward_prefetch` layers. While layer `i` is computing, it's already all-gathering parameters for layers `i+1`, `i+2`, etc.

### Backward Prefetching

Backward prefetching is similar but for the backward pass. It prefetches parameters needed for gradient computation:

```python
def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Set backward prefetching for transformer layers."""
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)
```

This prefetches parameters for previous layers while computing gradients for the current layer.

### When to Use Prefetching

Prefetching helps when:
- Your model has many layers (10+)
- Communication bandwidth is limited
- Layers are large enough that prefetching can overlap with computation

Prefetching doesn't help much if:
- Your model is small (few layers)
- Communication is already fast (NVLink, high-bandwidth InfiniBand)
- Layers are too small (prefetch overhead exceeds benefit)

Start without prefetching, then add it if profiling shows communication is a bottleneck. A good starting point is `num_to_forward_prefetch=2` and `num_to_backward_prefetch=2`.

## Activation Checkpointing and Offloading

Activation checkpointing is almost always used with FSDP. Instead of storing all activations during forward, you recompute them during backward. This can cut activation memory by 50-80%, which is crucial when you're already memory-constrained.

### Why Activation Checkpointing Matters with FSDP

With FSDP, you're already sharding parameters, gradients, and optimizer states. Activations can still be a memory bottleneck, especially for large batch sizes or long sequences. Activation checkpointing trades computation for memory: you recompute activations during backward instead of storing them.

For a transformer with sequence length 2048 and batch size 8, activations can easily be 50-100 GB. With checkpointing, you might reduce this to 10-20 GB, at the cost of recomputing activations (roughly 30% slower forward pass, but backward is similar since you'd compute gradients anyway).

### Using Activation Checkpointing

The simplest approach is to checkpoint the entire model:

```python
from torch.utils.checkpoint import checkpoint

# After applying FSDP
fully_shard(model, mesh=mesh)

# Wrap forward pass with checkpointing
def forward_with_checkpoint(x):
    return checkpoint(model, x)
```

But you can be more selective. For transformers, you typically checkpoint individual layers:

```python
class TransformerBlockWithCheckpoint(nn.Module):
    def __init__(self, args: ModelArgs, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        if self.use_checkpoint:
            # Checkpoint the entire block
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

Then enable checkpointing for some layers:

```python
# Checkpoint every other layer to balance memory and speed
for i, layer in enumerate(model.layers):
    layer.use_checkpoint = (i % 2 == 0)
```

### CPU Offloading

CPU offloading moves optimizer states (or parameters) to CPU memory, freeing GPU memory at the cost of slower training. The FSDP2 API supports this:

```python
from torch.distributed.fsdp import OffloadPolicy

fully_shard(
    model,
    mesh=mesh,
    offload_policy=OffloadPolicy(offload_type="cpu"),
)
```

This offloads optimizer states to CPU. When the optimizer needs to update parameters, it transfers them from CPU to GPU, updates, then transfers back. This adds significant overhead (20-50% slowdown) but can be necessary for very large models.

You can also offload parameters (not just optimizer states), but this is even slower and rarely needed:

```python
# Offload parameters too (very slow, rarely needed)
fully_shard(
    model,
    mesh=mesh,
    offload_policy=OffloadPolicy(offload_type="cpu", offload_params=True),
)
```

### When to Use Offloading

Use CPU offloading only if:
- You're still OOM after full-shard and activation checkpointing
- You can't reduce batch size or sequence length further
- You're willing to accept 20-50% slowdown

For most cases, full-shard + activation checkpointing is enough. Only add offloading as a last resort.

### NVMe Offloading

For very large models, you can offload to NVMe (SSD) instead of CPU. This is slower than CPU but allows even larger models:

```python
fully_shard(
    model,
    mesh=mesh,
    offload_policy=OffloadPolicy(offload_type="nvme", offload_path="/path/to/nvme"),
)
```

NVMe offloading is useful when:
- Your model is too large even with CPU offloading
- You have fast NVMe storage (PCIe 4.0 or better)
- Training time is less important than being able to train at all

Expect 50-100% slowdown with NVMe offloading, so use it only when necessary.

## Performance Optimization

Once you have FSDP working, you'll want to optimize performance. The main bottlenecks are communication (all-gather/reduce-scatter) and activation memory. Let's look at how to profile and optimize.

### Profiling FSDP Training

Use PyTorch's profiler to understand where time is spent:

```python
from torch.profiler import profile, record_function, ProfilerActivity

def train_with_profiling(model, dataloader, optimizer, num_iterations=10):
    rank = torch.distributed.get_rank()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("training_loop"):
            for i, (data, target) in enumerate(dataloader):
                if i >= num_iterations:
                    break
                
                data = data.cuda(rank, non_blocking=True)
                target = target.cuda(rank, non_blocking=True)
                
                with record_function("forward"):
                    output = model(data)
                    loss = criterion(output, target)
                
                with record_function("backward"):
                    loss.backward()
                
                with record_function("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()
    
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        prof.export_chrome_trace("fsdp_trace.json")
```

Look for:
- **All-gather operations**: Should overlap with computation. If they don't, prefetching might help.
- **Reduce-scatter operations**: Should overlap with gradient computation.
- **Activation memory**: Use `profile_memory=True` to see peak memory usage.

### Optimizing Communication

If communication is a bottleneck, try:

1. **Enable prefetching**: As discussed earlier, prefetching can overlap communication with computation.

2. **Tune reshard_after_forward**: If you have memory headroom, try `reshard_after_forward=False` to avoid all-gather in backward:

```python
fully_shard(
    model,
    mesh=mesh,
    reshard_after_forward=False,  # Keep parameters unsharded after forward
)
```

This uses more memory but reduces communication. Only use if you have headroom.

3. **Use faster interconnects**: NVLink for intra-node, InfiniBand for inter-node. Make sure NCCL is using them:

```bash
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_DEBUG=INFO    # Check what NCCL is using
```

### Optimizing Activation Memory

If activation memory is a bottleneck:

1. **Reduce batch size**: Smaller batches = less activation memory.

2. **Reduce sequence length**: For transformers, shorter sequences = less memory.

3. **Use gradient accumulation**: Instead of large batches, use smaller batches with gradient accumulation:

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

This gives you the effective batch size of `batch_size * accumulation_steps` but with the memory footprint of `batch_size`.

4. **Selective checkpointing**: Checkpoint only some layers, not all. Experiment to find the right balance.

### Memory Profiling

Use `nvidia-smi` or PyTorch's memory profiler to see where memory is used:

```python
import torch

def print_memory_usage(step_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{step_name}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

# Before model creation
print_memory_usage("Before model")

# After model creation
print_memory_usage("After model")

# After FSDP
print_memory_usage("After FSDP")

# After forward
print_memory_usage("After forward")

# After backward
print_memory_usage("After backward")
```

This helps you understand where memory is being used and where you can optimize.

## Multi-Node FSDP Training

Multi-node FSDP works the same way as multi-node DDP—you need process group initialization and proper networking. The main difference is checkpointing: with FSDP2, sharded state dicts are straightforward—each rank writes its shard, and you can load them back without all-gather.

### Setting Up Multi-Node FSDP

The setup is similar to multi-node DDP. On each node, you need to:

1. Set environment variables for process group initialization
2. Launch training with `torchrun`
3. Ensure network connectivity between nodes

On the master node (node 0):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_ip> --master_port=29500 train_fsdp2.py
```

On worker node (node 1):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_ip> --master_port=29500 train_fsdp2.py
```

Replace `<master_ip>` with the actual IP address of the master node. You can find it with:

```bash
hostname -I
```

### Network Configuration for Multi-Node

For multi-node FSDP, network bandwidth and latency are critical. FSDP does more communication than DDP (all-gather and reduce-scatter), so fast interconnects are even more important.

**InfiniBand is preferred** over Ethernet because:
- Higher bandwidth: 200-400 Gb/s per link vs 10-100 Gb/s for Ethernet
- Lower latency: Sub-microsecond vs microseconds
- RDMA support: Direct GPU-to-GPU memory access

Make sure NCCL is using InfiniBand:

```bash
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
```

Check the NCCL logs to verify it's using InfiniBand. You should see messages like:

```
NCCL INFO NET/IB: Using [device] for node [rank]
```

### Checkpointing on Multi-Node

With FSDP2, checkpointing is straightforward even on multi-node. Each rank saves its shard, so you need shared storage accessible from all nodes.

**Option 1: Shared filesystem (NFS, Lustre, etc.)**

If all nodes mount the same filesystem, each rank can write directly:

```python
checkpoint_dir = "/shared/checkpoints"  # Mounted on all nodes
save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir)
```

**Option 2: Parallel writes to local storage**

Each node writes locally, then you sync later:

```python
# Each node writes to local storage
local_checkpoint_dir = f"/local/checkpoints/node_{node_rank}"
save_checkpoint_dcp(model, optimizer, epoch, local_checkpoint_dir)
```

Then sync to shared storage after training (or use a distributed filesystem).

**Option 3: Object storage (S3, etc.)**

Use a library like `s3fs` to write directly to S3:

```python
import s3fs

fs = s3fs.S3FileSystem()
checkpoint_path = f"s3://bucket/checkpoints/epoch_{epoch}"
# Save using DCP with S3 backend
```

The sharded approach helps since each rank only writes its shard (smaller files, less bandwidth).

### SLURM Integration

Most HPC clusters use SLURM. Here's a SLURM script for multi-node FSDP:

```bash
#!/bin/bash
#SBATCH --job-name=fsdp_train
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
export NODE_RANK=$SLURM_NODEID

# Launch training
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_fsdp2.py
```

Or using `torchrun` directly with SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=fsdp_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_fsdp2.py
```

### Scaling Considerations

When scaling to many nodes, consider:

1. **Communication overhead**: With more nodes, communication overhead increases. Make sure your model is large enough that computation dominates.

2. **Checkpoint size**: With many nodes, you'll have many shard files. Use a distributed filesystem or object storage that handles many small files well.

3. **Fault tolerance**: With many nodes, failures are more likely. Save checkpoints frequently and use elastic training if supported.

4. **Network topology**: For best performance, ensure nodes are on the same network segment and use InfiniBand if available.

## Debugging FSDP Issues

FSDP adds complexity, and debugging can be challenging. Here are common issues and how to debug them.

### Issue: Out of Memory (OOM)

OOM errors are common with FSDP, especially when first setting it up. Here's how to debug:

1. **Check if FSDP is actually sharding**: Print parameter shapes to verify sharding:

```python
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, device={param.device}")
    # Sharded parameters should have smaller shapes
```

2. **Check activation memory**: Use memory profiler to see if activations are the issue:

```python
print_memory_usage("After forward")
```

3. **Reduce batch size or sequence length**: If activations are the issue, reduce them.

4. **Enable activation checkpointing**: If you haven't already, enable it.

5. **Check for memory leaks**: Make sure you're not accumulating tensors across iterations.

### Issue: Hanging or Deadlock

FSDP can hang if processes get out of sync. Common causes:

1. **Different code paths**: Make sure all ranks execute the same code. Conditional logic based on rank can cause hangs:

```python
# BAD: Different code paths
if rank == 0:
    model.some_operation()  # Only rank 0 executes

# GOOD: All ranks execute
model.some_operation()
```

2. **Unbalanced data**: If one rank runs out of data before others, it can cause hangs. Make sure all ranks have the same number of batches.

3. **Checkpoint loading issues**: If loading checkpoints, make sure all ranks load successfully.

4. **NCCL issues**: Check NCCL logs for errors:

```bash
export NCCL_DEBUG=INFO
```

### Issue: Slow Training

If training is slower than expected:

1. **Profile to find bottlenecks**: Use profiler to see where time is spent.

2. **Check communication overlap**: Verify all-gather/reduce-scatter overlap with computation. If not, try prefetching.

3. **Check network**: For multi-node, verify InfiniBand is being used and bandwidth is good.

4. **Check activation checkpointing overhead**: If using checkpointing, it adds ~30% overhead. Make sure the memory savings are worth it.

5. **Verify mixed precision**: Make sure mixed precision is enabled if your hardware supports it.

### Issue: Incorrect Results

If you're getting incorrect results:

1. **Check random seed**: Make sure all ranks use the same seed for reproducibility:

```python
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

2. **Verify data sharding**: Make sure `DistributedSampler` is used correctly:

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
sampler.set_epoch(epoch)  # Important: call this each epoch
```

3. **Check gradient synchronization**: Verify gradients are being synchronized. You can print gradients to check:

```python
if rank == 0:
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm()}")
```

4. **Compare with single-GPU**: Run the same model on a single GPU to verify correctness, then scale up.

### Debugging Tools

Use these tools to debug FSDP:

1. **NCCL debug**: Enable detailed NCCL logging:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

2. **PyTorch profiler**: Profile to see communication patterns:

```python
with torch.profiler.profile(...) as prof:
    # Training code
```

3. **Memory profiler**: Track memory usage:

```python
print(torch.cuda.memory_summary())
```

4. **Distributed debugger**: Use `torch.distributed` debugging utilities:

```python
torch.distributed.set_debug_level(torch.distributed.DebugLevel.DETAIL)
```

## Comparing FSDP2 with ZeRO and DDP

When choosing a distributed training strategy, you need to understand the tradeoffs. Let's compare DDP, ZeRO (DeepSpeed), and FSDP2.

### DDP: When Your Model Fits on One GPU

**DDP** is the simplest—every GPU has a full copy of everything. Use it when your model fits on a single GPU.

**Pros:**
- Simple to use: just wrap your model with `DDP`
- Well-optimized: mature, battle-tested, excellent performance
- Low communication overhead: only gradient synchronization
- Works with any PyTorch model

**Cons:**
- Requires model to fit on a single GPU
- Doesn't help with very large models

**When to use:** Models that fit on a single GPU (even with mixed precision and activation checkpointing). This covers most models up to ~7B parameters on modern GPUs.

### ZeRO (DeepSpeed): The DeepSpeed Ecosystem

**ZeRO** does staged sharding similar to FSDP, but it's part of the DeepSpeed ecosystem. ZeRO has three stages:

- **ZeRO-1**: Shards optimizer states only
- **ZeRO-2**: Shards optimizer states + gradients
- **ZeRO-3**: Shards optimizer states + gradients + parameters

**Pros:**
- Part of DeepSpeed ecosystem: integrates with other DeepSpeed features (ZeRO-Offload, ZeRO-Infinity, etc.)
- Well-documented: extensive documentation and examples
- Production-tested: used by many organizations

**Cons:**
- Requires DeepSpeed: adds dependency, not pure PyTorch
- More complex: DeepSpeed has many features, can be overwhelming
- Less flexible: tied to DeepSpeed's APIs

**When to use:** If you're already using DeepSpeed or need DeepSpeed-specific features (like ZeRO-Offload for CPU offloading, or ZeRO-Infinity for NVMe offloading).

### FSDP2: PyTorch-Native Sharding

**FSDP2** is PyTorch-native and integrates tightly with autograd. The per-parameter-sharding design is simpler (about 3k lines of code versus 14k for the original) and more flexible.

**Pros:**
- PyTorch-native: no external dependencies, integrates with PyTorch ecosystem
- Flexible: per-parameter sharding enables features like fp8 all-gather, frozen parameters
- Simpler API: functional approach with `fully_shard()`, easier to compose
- Better compiler integration: works well with `torch.compile`
- Active development: PyTorch team is actively improving it

**Cons:**
- Newer: less battle-tested than ZeRO, fewer examples
- Limited offloading: CPU/NVMe offloading is available but less mature than DeepSpeed's ZeRO-Offload/Infinity

**When to use:** If you're using PyTorch and want native integration. The new design is the direction PyTorch is moving, so new projects should use it.

### Memory Comparison

Let's compare memory usage for a 7B parameter model with Adam optimizer on 8 GPUs:

**DDP:**
- Parameters: 14 GB (BF16) × 8 GPUs = 112 GB total
- Gradients: 14 GB × 8 GPUs = 112 GB total
- Optimizer states: 56 GB × 8 GPUs = 448 GB total
- **Per GPU**: 84 GB (doesn't fit on most GPUs)

**ZeRO-3 / FSDP2:**
- Parameters: 14 GB / 8 = 1.75 GB per GPU
- Gradients: 14 GB / 8 = 1.75 GB per GPU
- Optimizer states: 56 GB / 8 = 7 GB per GPU
- **Per GPU**: 10.5 GB (fits comfortably)

### Performance Comparison

For models that fit on a single GPU, DDP is usually fastest (lowest communication overhead). For models that don't fit, FSDP2 and ZeRO-3 have similar performance:

- **Communication overhead**: Both do all-gather and reduce-scatter. Performance depends on network topology and implementation details.
- **Computation**: Same (both use data parallelism for computation).
- **Memory**: Same (both shard parameters, gradients, optimizer states).

In practice, performance is similar. Choose based on ecosystem fit, not performance.

### Migration Path

If you're using DDP and need to scale to larger models:

1. **Try optimization first**: Mixed precision, activation checkpointing, gradient accumulation. You might be able to fit a larger model with DDP.

2. **If still OOM, switch to FSDP2**: The API is similar to DDP, migration is straightforward. Just replace `DDP(model)` with `fully_shard(model, mesh=mesh)`.

3. **If you need advanced features**: Consider DeepSpeed ZeRO if you need features like ZeRO-Offload or ZeRO-Infinity.

### Recommendation

**Use DDP if:**
- Your model fits on a single GPU
- You want the simplest solution
- You don't need to scale beyond single-node

**Use FSDP2 if:**
- Your model doesn't fit on a single GPU
- You want PyTorch-native solution
- You're starting a new project

**Use ZeRO if:**
- You're already using DeepSpeed
- You need DeepSpeed-specific features (ZeRO-Offload, ZeRO-Infinity)
- You prefer the DeepSpeed ecosystem

For most PyTorch users, FSDP2 is the recommended choice for models that don't fit on a single GPU.

## Practical Tips

Here are practical tips from real-world FSDP usage:

### State Dict Handling

With FSDP2, sharded state dicts match the training representation, so saving and loading is straightforward. Each rank saves its shard.

**Key points:**
- Use DCP API for checkpointing (recommended) or handle sharded state dicts manually
- Don't try to gather all shards to rank 0 unless you need a full checkpoint for inference
- Loading is just reading shards back—no gathering needed

### Shared Parameters

If you have shared parameters (same tensor used in multiple places), they need to be in the same FSDP group. This is a limitation—there's no way to preserve sharedness after parameter swapping.

**Workaround:** If you have shared parameters, make sure they're in the same module hierarchy so FSDP treats them as a single parameter. Or restructure your model to avoid sharing.

### Memory Profiling

Use `torch.profiler` or `nvidia-smi` to see where memory is actually being used. Sometimes the bottleneck isn't what you think—it could be activations, not parameters.

**Common memory issues:**
- Activations: Use activation checkpointing
- Optimizer states: Already sharded with FSDP, but can offload to CPU if needed
- Temporary tensors: Make sure you're not accumulating tensors across iterations
- DataLoader: Use `pin_memory=False` if you're memory-constrained

### Tuning reshard_after_forward

The default (`True`) is usually right. But if you have memory headroom and want to reduce communication, try `False`. You can also use an intermediate size (like `int` for ZeRO++ hpZ style).

**When to use `False`:**
- You have extra GPU memory
- Communication is your bottleneck (slow network)
- You want to reduce all-gather operations in backward

**When to use `True` (default):**
- You're memory-constrained
- Communication is fast (NVLink, fast InfiniBand)
- You want maximum memory savings

### Initialization Best Practices

1. **Use meta device for large models**: Create model on meta device first, then move to actual device after FSDP:

```python
with torch.device("meta"):
    model = Transformer(args)

# Apply FSDP
fully_shard(model, mesh=mesh)

# Move to actual device and initialize
model.to_empty(device=device)
model.reset_parameters()
```

2. **Initialize on all ranks**: Make sure all ranks initialize parameters the same way (same seed).

3. **Don't initialize before FSDP**: If you initialize parameters before applying FSDP, you'll waste memory. Initialize after.

### Data Loading

Use `DistributedSampler` correctly:

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,  # Shuffle data
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster CPU->GPU transfer
)

# Important: call set_epoch each epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Ensures different shuffle each epoch
    for batch in dataloader:
        # Training code
```

### Gradient Clipping

Gradient clipping works with FSDP, but you need to unshard parameters first:

```python
# Unshard parameters for gradient clipping
model.unshard()

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reshard (happens automatically, but explicit is clearer)
model.reshard()

optimizer.step()
```

Or use the FSDP-aware gradient clipping:

```python
# FSDP handles unsharding/resharding automatically
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Mixed Precision Best Practices

1. **Use BF16 for parameters**: BF16 has better numerical stability than FP16.

2. **Keep gradients in FP32**: Use `reduce_dtype=torch.float32` for gradient reduction to maintain precision.

3. **Test without mixed precision first**: Get FSDP working without mixed precision, then add it.

### Starting Simple

Start simple: use full-shard with checkpointing, see if that's enough. Only add offloading if you're still hitting memory limits.

**Progressive optimization:**
1. Start with FSDP2 + mixed precision
2. Add activation checkpointing if OOM
3. Reduce batch size or sequence length if still OOM
4. Add CPU offloading as last resort

Don't optimize prematurely. Get it working first, then optimize.

### Common Mistakes

1. **Forgetting to call `set_epoch()`**: This causes all ranks to see the same data each epoch.

2. **Different code paths per rank**: Conditional logic based on rank can cause hangs.

3. **Initializing before FSDP**: Wastes memory. Initialize after applying FSDP.

4. **Not using DistributedSampler**: Each rank will see the same data, defeating the purpose of distributed training.

5. **Saving checkpoints incorrectly**: Make sure you understand sharded vs full state dicts.

## Advanced Topics

### Hybrid Sharding (HSDP)

Hybrid sharding combines data parallelism with model parallelism. You shard within a node but replicate across nodes. This is useful for very large scale training.

```python
# 2D mesh: 4 nodes × 8 GPUs per node
mesh = init_device_mesh("cuda", (4, 8))

# Shard across the second dimension (within node)
fully_shard(model, mesh=mesh, mesh_dim=1)
```

This shards parameters across 8 GPUs within each node, but replicates across 4 nodes. Useful when:
- You have many nodes
- Inter-node communication is slower than intra-node
- You want to reduce inter-node communication

### Compiler Integration

FSDP2 works well with `torch.compile`:

```python
fully_shard(model, mesh=mesh)
model = torch.compile(model)  # Compile after FSDP
```

The compiler can optimize all-gather and reduce-scatter operations, potentially improving performance.

### Custom Communication Hooks

You can customize communication behavior with hooks, though this is advanced and rarely needed:

```python
def custom_all_gather_hook(state, bucket):
    # Custom all-gather logic
    pass

# Register hook (advanced usage)
```

Most users won't need this, but it's available if you need fine-grained control.

### Integration with Other PyTorch Features

FSDP2 integrates with:
- **torch.compile**: As mentioned, works well together
- **Automatic Mixed Precision (AMP)**: Use `MixedPrecisionPolicy` instead
- **Gradient accumulation**: Works naturally with FSDP
- **Learning rate scheduling**: No special handling needed

### Monitoring and Observability

Use these tools to monitor FSDP training:

1. **PyTorch profiler**: Profile to see communication patterns
2. **NCCL logs**: Check NCCL debug output for communication issues
3. **Memory profiler**: Track memory usage over time
4. **Distributed logging**: Use `torch.distributed` logging utilities

## Conclusion

FSDP2 is PyTorch's solution for training models that don't fit on a single GPU. It shards parameters, gradients, and optimizer states across GPUs, enabling training of models that are 8x, 16x, or even larger than what fits on a single GPU.

Key takeaways:

1. **Use FSDP2 when your model doesn't fit on a single GPU**: If DDP works, use DDP. Only use FSDP2 when necessary.

2. **Start simple**: Use full-shard with activation checkpointing first. Add complexity only if needed.

3. **Understand the tradeoffs**: FSDP adds communication overhead (all-gather/reduce-scatter). Make sure your model is large enough that computation dominates.

4. **Use the DCP API for checkpointing**: It handles sharded state dicts correctly and is the recommended approach.

5. **Profile before optimizing**: Use PyTorch profiler to understand where time is spent. Don't optimize blindly.

6. **Test on small scale first**: Get FSDP2 working on 2-4 GPUs before scaling to many nodes.

FSDP2 is actively developed and is the direction PyTorch is moving for large model training. If you're starting a new project and need to train models that don't fit on a single GPU, FSDP2 is the recommended choice.

The code examples in this chapter show complete working setups. Run them on your hardware to see FSDP2 in action—you'll see the model sharded across GPUs, with each rank only holding its portion of the parameters, gradients, and optimizer states.

In the next chapter, we'll cover DeepSpeed and ZeRO optimization, which provides additional features like ZeRO-Offload and ZeRO-Infinity for even larger models.

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Per-Parameter-Sharding FSDP RFC](https://github.com/pytorch/pytorch/issues/114299)
- [TorchTitan FSDP Guide](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
- https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed
- https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2
- https://ggrigorev.me/posts/introduction-to-parallelism/
- https://arxiv.org/pdf/2304.11277
- https://arxiv.org/pdf/2411.00284
- /media/wukong/jackie/git.repo/distributed-ai/resources/torch-examples/distributed/FSDP2

