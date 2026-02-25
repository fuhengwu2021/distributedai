# Chapter 4: Scaling with Fully Sharded Data Parallel (FSDP) {-}

*Training models larger than single GPU memory with parameter sharding*

> The data center is the new unit of computing.
- Jensen Huang, CEO of NVIDIA

**Code Summary**

- `torch.distributed.fsdp.fully_shard(module, mesh=..., mp_policy=...)`: Shard a module in place with per-parameter sharding (DTensor)
- `torch.distributed.fsdp.MixedPrecisionPolicy`: Mixed precision configuration for FSDP2
- `torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,))`: Create a device mesh for sharding
- `torch.distributed.checkpoint`: Distributed Checkpoint (DCP) API for saving/loading sharded state dicts
- `torch.distributed.fsdp.FullyShardedDataParallel(module, ...)`: Wrapper class for sharding with FlatParameter
- `torch.distributed.fsdp.wrap()`: Wrap submodules with FSDP1
- `torch.distributed.fsdp.MixedPrecision`: Mixed precision configuration for FSDP1
- `torch.distributed.fsdp.set_state_dict_type()`: Configure state dict type for checkpointing
- `torch.distributed.fsdp.StateDictConfig` / `OptimStateDictConfig`: State dict configuration

**Fully Sharded Data Parallel (FSDP)** is a training strategy that shards model parameters, gradients, and optimizer state across multiple devices so that each device holds only a fraction of the full model. In Chapter~\ref{chap:distributed-training-with-pytorch-ddp}, we used DDP, which replicates the entire model on every GPU—effective when the model fits in a single GPU's memory. When the model (plus gradients and optimizer state) exceeds that memory, DDP is no longer viable. FSDP addresses this by distributing the model and its training state across GPUs, so you can train models that are larger than the memory of any one device.

PyTorch provides two main FSDP APIs for GPU training, plus a separate implementation for TPU:

- **FSDP1** (`FullyShardedDataParallel`): The original wrapper class using a flat-parameter approach.
- **FSDP2** (`fully_shard()`): The newer per-parameter-sharding design, accessed via `fully_shard()`. Simpler, more flexible, and the direction PyTorch is moving.
- **FSDP via SPMD** (`SpmdFullyShardedDataParallel`): For XLA/TPU devices, using GSPMD for automatic parallelization.

Throughout this chapter, we use **FSDP** (without a number) when discussing the general technique—sharding parameters, all-gather, reduce-scatter—that applies to both APIs. When the distinction matters, we say **FSDP1** or **FSDP2** explicitly.

This chapter focuses on FSDP2 for GPU training—it's the recommended approach for new projects on CUDA devices. FSDP1 still works and remains in use in production codebases; for example, Wan2.2 uses PyTorch FSDP together with DeepSpeed Ulysses for multi-GPU inference.[^wan22] We summarize FSDP1 below and then concentrate on FSDP2. For TPU training, see Section~\ref{sec:fsdp-spmd}.

[^wan22]: <https://github.com/Wan-Video/Wan2.2>
[^fsdp2-rfc]: <https://github.com/pytorch/pytorch/issues/114299>
[^t5-flan]: **T5** (Text-to-Text Transfer Transformer) is an encoder-decoder model from Google that frames NLP tasks as text-to-text. **FLAN-T5** is the instruction-tuned family of T5 models (e.g. flan-t5-small, flan-t5-xl, flan-t5-xxl), used for tasks like summarization and question answering.

## Why FSDP Enables Larger-Than-Memory Models

The basic idea is straightforward: instead of keeping the full model on every GPU, you split it up. Each GPU holds a shard of the parameters. During forward pass, you __all-gather__ the parameters you need. During backward, you compute gradients on the local shard, then __reduce-scatter__ to aggregate across GPUs.

But let's dig deeper into why this matters. When training a large model with DDP, each GPU needs to store:

1. **Model parameters**: The weights themselves. For a 7B parameter model in BF16, that's 7B × 2 bytes = 14 GB.
2. **Gradients**: Same size as parameters. Another 14 GB for BF16.
3. **Optimizer states**: For Adam, momentum and variance are 2× the parameter size in FP32. That's 7B × 4 bytes × 2 = 56 GB.
4. **Activations**: Depends on batch size and sequence length, but can easily be tens of GB for large models.

So for a 7B model with Adam, parameters, gradients, and optimizer states alone are 14 + 14 + 56 = 84 GB per GPU—more than an 80 GB H100 can hold, and activations are not yet counted. With FSDP, those three components are sharded across GPUs: each device holds 1/N of each (N = number of GPUs). With 8 GPUs, that is 84 / 8 = 10.5 GB per GPU, leaving plenty of headroom for activations so the model fits comfortably on A100s or even V100s. In practice, that is the difference between fitting the same 7B model on 8 GPUs with FSDP versus not fitting on a single 80 GB GPU with DDP.

![Per-GPU memory: DDP vs FSDP for a 7B model.](img/ddp_fsdp_mem.png){#fig:ddp-fsdp-mem .block width=100% align=center}

Figure~\ref{fig:ddp-fsdp-mem} illustrates the comparison: with DDP, each GPU holds the full 84 GB (parameters, gradients, and optimizer state) and exceeds an 80 GB device; with FSDP, memory per GPU falls as 84/N, and at 8 GPUs the 10.5 GB per GPU leaves room for activations.

>NOTE: FSDP shards only parameters, gradients, and optimizer state—not activations. Each GPU still stores activations for its share of the batch during forward and backward, so activation memory remains a per-GPU cost. Techniques like activation checkpointing (recomputing activations in backward instead of storing them) are often used with FSDP to free headroom for the temporarily all-gathered parameters.

### How FSDP Works

The core idea behind FSDP comes from the ZeRO (Zero Redundancy Optimizer) paper from Microsoft Research (2019).[^zero-paper] ZeRO observed that in data-parallel training, each GPU holds a full copy of the model, gradients, and optimizer state—most of which is redundant. By partitioning these across GPUs and gathering them only when needed, you can train much larger models without changing the underlying data-parallel algorithm. We cover ZeRO in detail in Chapter~\ref{chap:deepspeed-zero}; here we focus on PyTorch's native implementation of these ideas.

[^zero-paper]: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC 2020. <https://arxiv.org/abs/1910.02054>

PyTorch's FSDP implements this idea using two collective operations. During forward pass, when a layer needs its parameters, FSDP **all-gathers** them from all GPUs—temporarily reconstructing the full parameter tensor (see Section~\ref{sec:allgather} in Chapter~\ref{chap:introduction-to-modern-distributed-ai}). After the layer finishes, the gathered parameters are freed. During backward pass, gradients are computed locally on the full (temporarily gathered) parameters, then **reduce-scattered** across GPUs so each GPU ends up with its shard of the aggregated gradients (see Section~\ref{sec:reducescatter} in Chapter~\ref{chap:introduction-to-modern-distributed-ai}).

![FSDP: All-Gather in forward, Reduce-Scatter in backward.](img/fsdp_allgather_reducescatter.png){#fig:fsdp-allgather-reducescatter .block width=100% align=center}

Figure~\ref{fig:fsdp-allgather-reducescatter} illustrates these two steps. In the left panel (forward), each rank holds one parameter shard ($1/N$); after All-Gather, every rank has the full parameters temporarily. In the right panel (backward), each rank has full gradients; after Reduce-Scatter, each rank keeps only its shard of the reduced gradients ($1/N$).

The key insight is that you don't need all parameters at once. Neural networks process layers sequentially—forward through layer 1, then layer 2, and so on. FSDP exploits this by all-gathering parameters for the current layer, using them, then freeing them before moving to the next layer. This is why activation checkpointing pairs well with FSDP: it reduces activation memory, leaving room for the temporarily all-gathered parameters.

### FSDP1 vs FSDP2: The Evolution

PyTorch's original FSDP (released in 2021, often called FSDP1) used a **flat-parameter** design borrowed from FairScale's implementation. It flattens all parameters in a wrapped module into a single contiguous `FlatParameter` tensor, then shards that tensor across GPUs. This worked, but had limitations: all parameters in a group had to share the same dtype, frozen parameters needed separate groups, and the flattening made it harder for compilers to optimize communication patterns.

**FSDP2** (introduced in 2024 via PyTorch RFC #114299[^fsdp2-rfc]) takes a different approach: **per-parameter sharding** using DTensor with `Shard(0)`. Instead of flattening, it shards each parameter tensor individually on dimension 0. A linear layer weight of shape $(4096, 1024)$ with 4 GPUs becomes four shards of shape $(1024, 1024)$—each rank holds one quarter of the rows. No flattening, no `FlatParameter` class. When dimension 0 isn't evenly divisible by the world size, FSDP2 pads the tensor; very small parameters may be replicated instead of sharded.

![FSDP1 vs FSDP2 parameter layout.](img/fsdp1_vs_fsdp2_layout.png){#fig:fsdp1-vs-fsdp2-layout .block width=100% align=center}

Figure~\ref{fig:fsdp1-vs-fsdp2-layout} illustrates the difference. FSDP1 (left) concatenates parameters W1, W2, W3 into a single flat tensor before sharding across ranks. FSDP2 (right) shards each parameter independently on dimension 0—each rank holds a slice of every parameter.

The per-parameter design is simpler (~3k lines of code vs ~14k for FSDP1) and more flexible. You can mix dtypes (some parameters in fp8, others in bf16), keep frozen and trainable parameters in the same group, save sharded checkpoints without gathering, and give compilers visibility into individual parameters for better optimization. The collectives are the same—All-Gather and Reduce-Scatter—but the parameter layout is fundamentally different.


### Original FSDP (FSDP1)

The **original FSDP** (often called FSDP1) is the wrapper class `FullyShardedDataParallel` in `torch.distributed.fsdp`. It flattens the parameters of each wrapped module into a single `FlatParameter` object (one instance of the class) and shards that across ranks; the same all-gather and reduce-scatter ideas apply. Usage is similar to DDP: you wrap the model (or submodules via `wrap()`), then train as usual.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 style
    device_id=torch.cuda.current_device(),
)
```

You can use `FSDP.set_state_dict_type()` and `StateDictConfig` / `OptimStateDictConfig` for checkpointing; mixed precision is configured via `MixedPrecision`.

Although FSDP2 is the recommended API for new projects, FSDP1 remains relevant for technical reasons beyond simple legacy inertia. Consider **Wan2.2**, a cutting-edge 2025 video generative model that still uses FSDP1.[^wan22] Several factors drive this choice.

First, Wan2.2 relies on DeepSpeed Ulysses for sequence parallelism to handle high-resolution video frames. Ulysses uses specific all-to-all patterns for attention head distribution that were hardened against FSDP1's interface. While FSDP2's `DeviceMesh` is designed for multi-dimensional parallelism, hybrid FSDP + Ulysses setups often find FSDP1's hooks into `distributed_c10d` group calls more stable and predictable.

Second, the Mixture-of-Experts architecture (27B total / 14B active parameters) requires precise control over non-uniform sharding. FSDP1's manual wrapping strategy via `ModuleWrapPolicy` gives developers granular control when experts are swapped or offloaded dynamically across timesteps—often easier to debug than FSDP2's `fully_shard` annotations for this use case.

Third, the model uses UMT5-XXL for text encoding, a massive frozen parameter block. Many optimized T5 wrappers in the Hugging Face ecosystem are built for FSDP1's `ShardedGradScaler` and auto-wrap policies. Moving the entire pipeline (encoder + DiT + VAE) to FSDP2 would require rewriting the T5 integration to avoid mixed-version distributed errors.

When you encounter or extend such projects, understanding the wrapper-class style and flat-parameter behavior of FSDP1 becomes essential.

## FSDP2: The Per-Parameter-Sharding API

For new projects, FSDP2 offers a cleaner design. Instead of wrapping modules in a class, FSDP2 uses `fully_shard()` as a function that modifies modules in place—more functional and composable. This is a significant departure from the original FSDP, which used a wrapper class similar to DDP.

Here's what the API looks like. A complete runnable example is in `code/train_fsdp2.py`:

```bash
torchrun --nproc_per_node=2 code/train_fsdp2.py
```

The core pattern:

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

For a minimal transformer example with hierarchical sharding (each block wrapped separately), see `code/fsdp2_basic.py`:

```bash
torchrun --nproc_per_node=2 code/fsdp2_basic.py
```

### Device Mesh: The Foundation

The `DeviceMesh` is a new abstraction in PyTorch that represents a logical arrangement of devices. For FSDP, you typically use a 1D mesh (all GPUs in a single dimension):

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh for standard FSDP (4 GPUs)
mesh = init_device_mesh("cuda", (4,))
```

This creates a mesh where all GPUs are arranged in a single dimension. For 4 GPUs, this is `[0, 1, 2, 3]`.

For very large clusters, full sharding across all GPUs can create excessive cross-node communication. Hybrid Sharded Data Parallel (HSDP) addresses this by sharding parameters only within each node while replicating across nodes—trading some memory for reduced inter-node traffic. We cover HSDP in detail in Section~\ref{sec:hsdp}; for now, here's how to set up a 2D mesh:

```python
# 2D mesh for hybrid sharding
# 2 nodes × 4 GPUs per node = 8 GPUs total
mesh = init_device_mesh("cuda", (2, 4))
```

This arranges GPUs in a 2D grid, which is useful for very large scale training where you want to shard within a node but replicate across nodes.

![Device Mesh: 1D (FSDP) vs 2D (HSDP).](img/device_mesh.png){#fig:device-mesh .block width=100% align=center}

Figure~\ref{fig:device-mesh} shows the two mesh configurations. In the 1D mesh (left panel), four GPUs labeled R0–R3 form a single sharding group—each GPU holds a different shard of every parameter. In the 2D mesh (right panel), N0 and N1 represent two physical nodes (e.g., two servers connected via InfiniBand). Within each node, GPUs are sharded along dim 1 (the green arrow), so R0–R3 in N0 each hold different parameter shards. Across nodes, GPUs at the same position share identical shards along dim 0 (the red arrow)—R0 in N0 and R4 in N1 hold the same data. This hybrid approach keeps the heavy all-gather traffic within the fast intra-node interconnect (NVLink) while only exchanging gradients across the slower inter-node network.

### Key Parameters

The `fully_shard()` function accepts several parameters that control sharding behavior. The `mesh` parameter specifies the `DeviceMesh` over which to shard—typically a 1D mesh for standard FSDP, or a 2D mesh for hybrid sharding (HSDP).

The most important parameter is `reshard_after_forward`, which controls the memory-communication tradeoff. When set to `True` (the default), parameters are resharded immediately after each layer's forward pass, freeing memory but requiring an additional all-gather during backward. This corresponds to ZeRO-3 behavior. Setting it to `False` keeps parameters in memory after forward, which uses more memory but eliminates the backward all-gather—similar to ZeRO-2. You can also pass an integer to reshard to an intermediate size; for example, `reshard_after_forward=2` shards across only 2 GPUs instead of all GPUs, mimicking ZeRO++'s hybrid parameter zero (hpZ).

For most memory-constrained scenarios, the default `True` is the right choice. If you have memory headroom and communication is your bottleneck, try `False`.

![reshard_after_forward: True vs False.](img/reshard_after_forward.png){#fig:reshard-after-forward .block width=100% align=center}

Figure~\ref{fig:reshard-after-forward} compares the two modes. Each row shows the forward (Fwd) and backward (Bwd) passes for a two-layer model. The colored blocks represent: AG (All-Gather, purple) for collecting sharded parameters, L1/L2 (Compute, yellow) for layer computation, and RS (Reduce-Scatter, red) for distributing gradients. With `reshard_after_forward=True` (top), parameters are freed after each layer's forward pass (marked "free") and must be all-gathered again in backward—this keeps memory low but doubles the all-gather communication. With `False` (bottom), parameters stay in memory after forward (marked "keep"), so the backward pass skips all-gather entirely—higher peak memory but less communication.

Mixed precision is configured through `mp_policy`. You specify `param_dtype` for parameter storage (e.g., `torch.bfloat16`), `reduce_dtype` for gradient reduction (often `torch.float32` for numerical stability), and optionally `output_dtype` for layer outputs:

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
```

Because FSDP2 shards per-parameter rather than flattening into a single buffer, you can mix dtypes freely—some layers in fp8, others in bf16. The original FSDP required all parameters in a group to share the same dtype.

Finally, `offload_policy` enables CPU or NVMe offloading when GPU memory is exhausted:

```python
from torch.distributed.fsdp import OffloadPolicy

fully_shard(
    model,
    mesh=mesh,
    offload_policy=OffloadPolicy(offload_type="cpu"),
)
```

Offloading introduces a performance cost, so treat it as a last resort after exhausting other memory optimizations. The slowdown varies widely depending on PCIe generation, CPU memory bandwidth, and optimizer state size—rough empirical ranges are 20-50% for CPU offloading.

### Hierarchical Sharding

Beyond these per-call parameters, FSDP2 gives you control over *where* in the model hierarchy sharding boundaries are placed. You can apply `fully_shard()` at different levels—for example, wrapping individual transformer blocks while leaving the embedding layer unsharded:

```python
# Shard each transformer layer individually
for layer in model.transformer.layers:
    fully_shard(layer, mesh=mesh)

# Don't shard the embedding layer (it's small)
# fully_shard(model.embedding, mesh=mesh)  # Skip this
```

This gives you fine-grained control over what gets sharded. Small layers (like embeddings) might not benefit from sharding and can add communication overhead, so you can leave them unsharded.

Figure~\ref{fig:fsdp-hierarchical-sharding} contrasts the two approaches. Both panels show a transformer model with an embedding layer (Embed), four transformer blocks (Block 0–3), and an output head (Head). The red dashed boxes indicate FSDP unit boundaries. With hierarchical sharding (left), each transformer block is wrapped as a separate FSDP unit by calling `fully_shard(block)` in a loop, while the embedding and head remain unsharded. This means all-gather and reduce-scatter happen at block boundaries, enabling prefetching (the next block's parameters can be gathered while the current block computes) and fine-grained memory management (only one block's full parameters need to be in memory at a time). With flat sharding (right), a single `fully_shard(model)` call wraps the entire model as one FSDP unit. This is simpler but requires gathering all parameters at once, leading to higher peak memory.

![Hierarchical vs flat sharding.](img/fsdp_hierarchical_sharding.png){#fig:fsdp-hierarchical-sharding .block width=100% align=center}

## A Complete Working Example: T5 Summarization with FSDP

A complete runnable example in `code/FSDP/` trains **T5 (FLAN-T5)**[^t5-flan] for text summarization with both FSDP1 and FSDP2, including checkpointing, mixed precision, and example training logs.

The example provides three entry points so you can compare single-GPU training, FSDP1, and FSDP2 on the same task and model:

- **FSDP1**: `T5_training_FSDP1.py` wraps FLAN-T5 with `FullyShardedDataParallel`, `ShardingStrategy.FULL_SHARD`, mixed precision, and FSDP1-style state dict handling. Use it when working with codebases that still rely on the original FSDP API.
- **FSDP2**: `T5_training_FSDP2.py` uses `fully_shard()` and the Distributed Checkpoint (DCP) API; this is the recommended script for new projects.
- **Single-GPU baseline**: `T5_training_Single.py` trains the same model on one GPU (no FSDP), useful for checking correctness and for comparing memory and throughput.

All scripts live in `code/FSDP/`. Before running the training scripts, download the WikiHow dataset from the `code/FSDP/` directory by running:

```
bash download_dataset.sh
```

to fetch the CSV files into `data/`. The README there describes model choices (flan-t5-small through flan-t5-xxl), VRAM requirements, and command-line options.

**Running the example.** Single-GPU baseline:

```bash
python code/FSDP/T5_training_Single.py
```

FSDP1 on 2 GPUs:

```bash
torchrun --nnodes 1 --nproc_per_node 2 code/FSDP/T5_training_FSDP1.py
```

FSDP2 on 2 GPUs:

```bash
torchrun --nnodes 1 --nproc_per_node 2 code/FSDP/T5_training_FSDP2.py
```

Larger models (e.g. `--model-name google/flan-t5-xl` or `google/flan-t5-xxl`) may require a smaller batch size.

**Comparison.** The numbers below are from example runs on H200 GPUs. Single-GPU training keeps the full model on one device; when the model fits, it can have the highest iteration throughput (it/s) per GPU, but with 2 GPUs the epoch completes in less wall-clock time because the batch is distributed (e.g. XL: 49 s vs 91 s). Single-GPU does not scale to models that exceed one GPU’s memory such as FLAN-T5-XXL. FSDP1 and FSDP2 shard parameters, gradients, and optimizer state across GPUs, so memory per GPU drops and larger models can be trained.

For the smaller FLAN-T5-XL (3B) model, single-GPU training fits on one H200; with 2 GPUs, FSDP1 reduces memory per GPU and completes each epoch in less wall-clock time (49 s vs 91 s). Table~\ref{tab:fsdp-t5-xl-comparison} gives the numbers.

| Mode   | GPUs | Mem/GPU | Peak mem/GPU | Throughput | Time/epoch |
|--------|------|---------|----------------------|------------|------------|
| Single | 1    | ~43 GB  | ~58 GB       | ~4.36 it/s | ~91 s      |
| FSDP1  | 2    | ~22 GB  | ~33 GB       | ~4.09 it/s | ~49 s      |

Table: FLAN-T5-XL (3B): single-GPU vs FSDP1 (2 GPUs). Example runs on H200 GPUs. {#tab:fsdp-t5-xl-comparison}

For FLAN-T5-XXL (11B parameters), single-GPU training runs out of memory (OOM) even on an H200 (140 GB). Table~\ref{tab:fsdp-t5-comparison} compares FSDP1 and FSDP2 on 2 GPUs.

| Mode   | GPUs | Mem/GPU | Peak mem/GPU | Throughput | Time/epoch |
|--------|------|---------|----------------------|------------|------------|
| Single | 1    | OOM     | OOM          | —          | —          |
| FSDP1  | 2    | ~84 GB  | ~105 GB      | ~1.96 it/s | ~101 s     |
| FSDP2  | 2    | ~84 GB  | ~105 GB      | ~1.86 it/s | ~106 s     |

Table: Comparison of single-GPU, FSDP1, and FSDP2 on FLAN-T5-XXL (11B). Example runs on H200 GPUs. {#tab:fsdp-t5-comparison}

**Code Analysis:** The following snippets show how the T5 example implements loading, hierarchical sharding, and mixed precision.

*FSDP1: policies and wrapper.* In FSDP1, a **policy** is a configuration object you pass into the wrapper: the **mixed-precision policy** specifies dtypes for parameters and gradients (e.g. bfloat16) to save memory and speed up compute; the **wrap policy** is a callable that tells FSDP which submodules to wrap as separate FSDP units (here, each `T5Block`), so sharding is hierarchical and all-gather/reduce-scatter happen at block boundaries. The script obtains both from `get_policies`, then wraps the already-loaded model with the `FSDP` wrapper class. The code snippet below is from `T5_training_FSDP1.py` and `policies/wrapping.py`.

```python
# get_policies (T5_training_FSDP1.py): mixed precision + wrap policy
def get_policies(cfg, rank):
    mixed_precision_policy = None
    if cfg.mixed_precision:
        # e.g. policies.bfSixteen for bfloat16
        mixed_precision_policy = policies.bfSixteen
    wrapping_policy = policies.get_t5_wrapper()  # targets T5Block
    return mixed_precision_policy, wrapping_policy

# Wrap policy (policies/wrapping.py): wrap each T5Block as an FSDP unit
def get_t5_wrapper():
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={T5Block},
    )

# Apply FSDP (T5_training_FSDP1.py): model already from setup_model()
mixed_precision_policy, t5_auto_wrap_policy = get_policies(train_config, rank)
model = FSDP(model,
    auto_wrap_policy=t5_auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=fsdp_config.sharding_strategy,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=fsdp_config.limit_all_gathers)
```

*FSDP2: mixed precision policy.* FSDP2 uses `MixedPrecisionPolicy` (not the FSDP1 `MixedPrecision` object). The script calls `get_policies(train_config, rank)` to build the policy from config; that policy is then passed into every `fully_shard(...)` call via `fsdp_kwargs["mp_policy"]`. The code snippet below is from `T5_training_FSDP2.py`.

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def get_policies(cfg, rank):
    """Establish mixed precision policy for FSDP2 (no wrap policy; sharding is explicit)."""
    mp_policy = None
    if cfg.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.use_fp16:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            )
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using MixedPrecisionPolicy")
        elif cfg.use_fp16:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
            )
    return mp_policy

# In fsdp_main: get policy once, then pass into every fully_shard call
mp_policy = get_policies(train_config, rank)
fsdp_kwargs = {}
if mp_policy is not None:
    fsdp_kwargs["mp_policy"] = mp_policy
# ... later: fully_shard(block, **fsdp_kwargs) and fully_shard(model, **fsdp_kwargs)
```

*FSDP2: load model, then shard each block and the root.* The model is loaded with `from_pretrained` and moved to the device. There is no auto wrap policy: the script explicitly loops over `model.encoder.block` and `model.decoder.block` (each element is a `T5Block`) and calls `fully_shard(block, **fsdp_kwargs)` so each block becomes a separate sharded unit, then calls `fully_shard(model, **fsdp_kwargs)` to wrap the root. Order matters—children are sharded before the root.

Note that this example omits the `mesh` parameter. When running with a single process group (the common case), `fully_shard()` infers the default mesh from the world size. For multi-dimensional parallelism (e.g., HSDP), you would explicitly pass `mesh` as shown in earlier sections. The code snippet below is from `T5_training_FSDP2.py`.

```python
from torch.distributed.fsdp import fully_shard

model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)

# fsdp_kwargs already holds mp_policy from get_policies()

# Shard encoder blocks (each T5Block becomes one FSDP unit)
if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
    for block in model.encoder.block:
        fully_shard(block, **fsdp_kwargs)
# Shard decoder blocks
if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
    for block in model.decoder.block:
        fully_shard(block, **fsdp_kwargs)
# Shard the entire model (root); children must already be sharded
fully_shard(model, **fsdp_kwargs)
```

For models that do not fit in memory even for loading, see Section~\ref{sec:fsdp-initialization-best-practices} (meta device, `to_empty`, and `reset_parameters`).

## Checkpointing with FSDP2

Once your model is sharded and training, you'll want to save checkpoints. Long runs can fail—hardware errors, preemptions, bugs—and losing days of progress is painful. With FSDP2, checkpointing is simpler than before: because sharded state dicts match the training representation, each rank just saves its shard directly. No gathering to rank 0, no resharding on load. Saving and loading happen locally, which is faster and uses less memory.

There are two approaches: using the Distributed Checkpoint (DCP) API, or manually handling sharded state dicts.

### Using the DCP API (Recommended)

The DCP API is the recommended way to save and load FSDP2 checkpoints. It handles all the complexity of sharded state dicts. A complete runnable example is in `code/fsdp2_checkpoint_dcp.py`:

```bash
torchrun --nproc_per_node=2 code/fsdp2_checkpoint_dcp.py
```

The key functions are `save_checkpoint_dcp` and `load_checkpoint_dcp`:

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
)
import torch.distributed.checkpoint as dcp

def save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir):
    """Save checkpoint using DCP API."""
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    optim_state_dict = get_optimizer_state_dict(
        model=model, optimizers=optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    dcp.save({"model": model_state_dict, "optimizer": optim_state_dict, "epoch": epoch},
             checkpoint_id=checkpoint_path)

def load_checkpoint_dcp(model, optimizer, checkpoint_dir, epoch):
    """Load checkpoint using DCP API."""
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    opt = get_optimizer_state_dict(model, optimizers=optimizer, options=StateDictOptions(full_state_dict=False))
    state_dict = {"model": get_model_state_dict(model, options=StateDictOptions(full_state_dict=False)),
                  "optimizer": opt,
                  "epoch": 0}
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    set_model_state_dict(model, state_dict["model"], options=StateDictOptions(full_state_dict=False))
    set_optimizer_state_dict(model, optimizers=optimizer, optim_state_dict=state_dict["optimizer"],
                             options=StateDictOptions(full_state_dict=False))
    return state_dict["epoch"]
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

With checkpointing sorted out, the next concern is performance. FSDP adds communication overhead—every forward pass needs an all-gather to reconstruct parameters, and every backward pass needs a reduce-scatter for gradients. Fast interconnects like NVLink reduce this overhead, but they don't eliminate it.

One way to hide this latency is prefetching: start fetching parameters for the next layer while the current layer is still computing. The key insight is that modern GPUs can execute computation and communication concurrently on different hardware units (CUDA streams for compute, NVLink/PCIe for transfers). If computation takes longer than communication, the all-gather finishes before it's needed and you pay no latency penalty.

![Prefetching timeline: without vs with.](img/fsdp_prefetch_timeline.png){#fig:fsdp-prefetch-timeline .block width=100% align=center}

Figure~\ref{fig:fsdp-prefetch-timeline} illustrates the difference. Without prefetching (top), each layer must wait for its all-gather (AG) to complete before computing—the operations are sequential. With prefetching (bottom), while layer L₀ computes, the all-gather for L₁ runs in parallel on a separate stream. By the time L₀ finishes, L₁'s parameters are already available. The total time shrinks because communication is hidden behind computation.

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

Prefetching works best when your model has many layers (10+) and each layer does enough computation to hide the communication latency. If layers are small or your interconnect is already fast (NVLink, high-bandwidth InfiniBand), the benefit shrinks—you might even add overhead from the extra scheduling logic.

The practical advice: start without prefetching. Profile your training loop, and if communication shows up as a bottleneck, try adding prefetching with `num_to_forward_prefetch=2` and `num_to_backward_prefetch=2`. Adjust from there based on what the profiler tells you.

## Activation Checkpointing and Offloading

Activation checkpointing is almost always used with FSDP. Instead of storing all activations during forward, you recompute them during backward. This can cut activation memory by 50-80%, which is crucial when you're already memory-constrained.

### Why Activation Checkpointing Matters with FSDP

With FSDP, you're already sharding parameters, gradients, and optimizer states. Activations can still be a memory bottleneck, especially for large batch sizes or long sequences. Activation checkpointing trades computation for memory: you recompute activations during backward instead of storing them.

Activation memory scales with model architecture and batch size. A rough estimate for a transformer is:

$$\text{Activation Memory} \approx L \times B \times S \times H \times \text{bytes per element} \times k$$

where $L$ is the number of layers, $B$ is batch size, $S$ is sequence length, $H$ is hidden dimension, and $k$ is a factor (typically 10–20) accounting for intermediate tensors in attention and MLP blocks. For a 7B model ($L=32$, $H=4096$) with sequence length 2048 and batch size 8 in fp16:

$$32 \times 8 \times 2048 \times 4096 \times 2 \times 12 \approx 52\text{ GB}$$

With activation checkpointing, you store only the inputs to each checkpointed block rather than all intermediate tensors, reducing memory by 50–80%. The tradeoff is recomputing activations during backward (roughly 30% slower forward pass, but backward is similar since you'd compute gradients anyway).

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

This offloads optimizer states to CPU. When the optimizer needs to update parameters, it transfers them from CPU to GPU, updates, then transfers back. This adds significant overhead but can be necessary for very large models. The actual slowdown depends on PCIe generation (3.0 vs 4.0 vs 5.0), CPU memory bandwidth, NUMA topology, and optimizer state size—empirically 20-50% is common, but your mileage will vary.

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

CPU offloading should come late in your optimization sequence. If you've already enabled full-shard and activation checkpointing, reduced batch size and sequence length as much as you can, and you're still hitting OOM—then offloading makes sense. For most models, full-shard plus activation checkpointing is enough without touching offloading.

### NVMe Offloading

For very large models, you can offload to NVMe (SSD) instead of CPU. This is slower than CPU but allows even larger models:

```python
fully_shard(
    model,
    mesh=mesh,
    offload_policy=OffloadPolicy(offload_type="nvme", offload_path="/path/to/nvme"),
)
```

NVMe offloading goes one step further—useful when even CPU memory isn't enough. The slowdown depends heavily on NVMe bandwidth (PCIe 3.0 vs 4.0 vs 5.0), sequential vs random access patterns, and how much data is being transferred. Empirically, expect 50-100% longer training times with fast NVMe (PCIe 4.0+), but slower drives or suboptimal access patterns can be worse. The tradeoff is clear: slower, but at least possible.

## Performance Optimization

Once you have FSDP working, you'll want to optimize performance. The main bottlenecks are communication (all-gather/reduce-scatter) and activation memory. Let's look at how to profile and optimize.

### Profiling FSDP Training

Use PyTorch's profiler to understand where time is spent. A complete runnable example is in `code/fsdp2_profile.py`:

```bash
torchrun --nproc_per_node=2 code/fsdp2_profile.py
```

The key pattern is wrapping your training loop with `profile()` and using `record_function()` to label different phases:

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for i in range(num_iterations):
        with record_function("forward"):
            output = model(data)
            loss = criterion(output, target)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
prof.export_chrome_trace("fsdp_trace.json")
```

The trace file is in Chrome trace format. See Section~\ref{sec:ddp-profiling} in Chapter 3 for how to open and interpret these traces. The short version: open `chrome://tracing` in Chrome, click "Load", and select the `.json` file. In the timeline, look for all-gather and reduce-scatter operations—ideally they overlap with computation. If you see them blocking, prefetching might help. Also check peak memory usage (`profile_memory=True` enables this) to see if activations are eating more than expected.

### Optimizing Communication

If the profiler shows communication as a bottleneck, you have a few options. Prefetching (covered earlier) can overlap communication with computation. If you have memory headroom, setting `reshard_after_forward=False` avoids the all-gather in backward—parameters stay unsharded after forward, so backward doesn't need to fetch them again:

```python
fully_shard(model, mesh=mesh, reshard_after_forward=False)
```

This trades memory for speed. Only use it if you have headroom after profiling.

Hardware matters too. NVLink for intra-node and InfiniBand for inter-node make a big difference. Check that NCCL is actually using them:

```bash
export NCCL_IB_DISABLE=0 && export NCCL_DEBUG=INFO
```

The debug output will show which interconnects NCCL detected.

### Optimizing Activation Memory

If activation memory is the bottleneck, the simplest fix is reducing batch size or sequence length—smaller inputs mean fewer activations to store. But if you need a large effective batch size for convergence, gradient accumulation lets you get there without the memory cost:

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    loss = criterion(model(data), target) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

This gives you the effective batch size of `batch_size * accumulation_steps` but with the memory footprint of a single `batch_size`. You can also try selective checkpointing—checkpoint only some layers instead of all, and experiment to find the right balance between memory and recomputation overhead.

### Memory Profiling

To understand where memory goes, sprinkle some print statements through your code:

```python
def print_memory_usage(step_name):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{step_name}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
print_memory_usage("Before model")
print_memory_usage("After FSDP")
print_memory_usage("After forward")
print_memory_usage("After backward")
```

This tells you how much memory is used at each stage. If "After forward" is much higher than "After FSDP", activations are the culprit. If "After backward" stays high, gradients or optimizer state might be the issue. For a more detailed breakdown, `nvidia-smi` shows total GPU memory usage, and PyTorch's `torch.cuda.memory_summary()` gives a full allocation report.

## Multi-Node FSDP Training

Multi-node FSDP works the same way as multi-node DDP—you need process group initialization and proper networking. The main difference is checkpointing: with FSDP2, sharded state dicts are straightforward—each rank writes its shard, and you can load them back without all-gather.

### Setting Up Multi-Node FSDP

The setup is similar to multi-node DDP. On each node, you need to:

1. Set environment variables for process group initialization
2. Launch training with `torchrun`
3. Ensure network connectivity between nodes

On the master node (node 0):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_ip> --master_port=29500 code/train_fsdp2.py
```

On worker node (node 1):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_ip> --master_port=29500 code/train_fsdp2.py
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
export NCCL_IB_DISABLE=0 && export NCCL_DEBUG=INFO
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

Most HPC clusters use SLURM for job scheduling. Below is a minimal example for multi-node FSDP:

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

We cover SLURM in detail in Chapter~\ref{chap:running-distributed-training-with-slurm}.

### Scaling Considerations

As you scale to more nodes, communication overhead grows. Make sure your model is large enough that computation still dominates—otherwise you're paying for GPUs that spend most of their time waiting on the network.

Checkpointing also gets trickier. With many nodes, you end up with many shard files. A distributed filesystem or object storage that handles small files well (like Lustre or S3) helps here. And with more hardware comes more failures—save checkpoints frequently, and consider elastic training if your cluster supports it.

Network topology matters too. Nodes on the same network segment with InfiniBand will outperform nodes scattered across racks connected by slower links.

## Debugging FSDP Issues

FSDP adds complexity, and when things go wrong, the error messages aren't always helpful. Here's how to approach common problems.

### Out of Memory (OOM)

OOM errors are common when first setting up FSDP. Start by checking whether FSDP is actually sharding your model—print parameter shapes and verify they're smaller than the full model:

```python
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, device={param.device}")
```

If parameters look right but you're still OOM, activations are likely the culprit. Use the memory profiling approach from earlier to confirm, then reduce batch size, sequence length, or enable activation checkpointing. Also watch for memory leaks—tensors accumulating across iterations because you forgot to detach or delete them.

### Hanging or Deadlock

FSDP hangs when processes get out of sync. The most common cause is conditional logic that only some ranks execute:

```python
# BAD: only rank 0 runs this collective-triggering code
if rank == 0:
    model.some_operation()

# GOOD: all ranks execute the same code
model.some_operation()
```

Other causes: unbalanced data (one rank runs out of batches before others), checkpoint loading failures on some ranks, or NCCL issues. For NCCL problems, enable debug logging with `export NCCL_DEBUG=INFO` and look for errors in the output.

### Slow Training

If training is slower than expected, profile first—guessing wastes time. Check whether communication overlaps with computation; if not, try prefetching. For multi-node, verify InfiniBand is being used (`NCCL_DEBUG=INFO` shows this). Activation checkpointing adds ~30% overhead, so make sure the memory savings justify it. And confirm mixed precision is actually enabled if your hardware supports it.

### Incorrect Results

Incorrect results usually come from one of a few places. First, check that all ranks use the same random seed:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Second, verify `DistributedSampler` is set up correctly and you're calling `sampler.set_epoch(epoch)` each epoch—forgetting this means all epochs see the same shuffle. Third, check that gradients are being synchronized by printing gradient norms across ranks. If all else fails, run the same model on a single GPU to establish a baseline, then scale up and compare.

### Debugging Tools

A few tools help with FSDP debugging. For NCCL issues, `export NCCL_DEBUG=INFO` (or `NCCL_DEBUG_SUBSYS=ALL` for more detail) shows what's happening at the communication layer. The PyTorch profiler (covered earlier) reveals communication patterns and bottlenecks. For memory issues, `torch.cuda.memory_summary()` gives a detailed breakdown. And for distributed-specific problems, `torch.distributed.set_debug_level(torch.distributed.DebugLevel.DETAIL)` enables verbose logging from the distributed runtime.

## Comparing FSDP2 with ZeRO and DDP

Choosing between DDP, ZeRO (DeepSpeed), and FSDP2 depends on your model size and ecosystem preferences.

**DDP** is the simplest option. Every GPU holds a full copy of the model, gradients, and optimizer state. Communication is just gradient synchronization (AllReduce), which is well-optimized. Use DDP when your model fits on a single GPU—this covers most models up to ~7B parameters on modern GPUs with mixed precision.

**ZeRO** (from DeepSpeed) does staged sharding: ZeRO-1 shards optimizer states, ZeRO-2 adds gradients, and ZeRO-3 shards everything including parameters. It's production-tested and integrates with DeepSpeed's other features like ZeRO-Offload (CPU) and ZeRO-Infinity (NVMe). The tradeoff is adding DeepSpeed as a dependency and learning its APIs.

**FSDP2** is PyTorch-native and does full sharding like ZeRO-3. The per-parameter design is simpler (~3k lines vs ~14k for FSDP1), integrates well with `torch.compile`, and doesn't require external dependencies. It's newer than ZeRO, so there are fewer battle-tested examples, but it's the direction PyTorch is moving.

### Memory and Performance

To make this concrete: a 7B parameter model with Adam optimizer needs ~84 GB per GPU with DDP (parameters + gradients + optimizer states, all replicated). With FSDP2 or ZeRO-3 on 8 GPUs, that drops to ~10.5 GB per GPU—everything is sharded 8 ways.

Performance-wise, DDP is fastest when the model fits (less communication). For models that don't fit, FSDP2 and ZeRO-3 perform similarly—both do all-gather and reduce-scatter, and the difference comes down to implementation details and network topology rather than fundamental design. Choose based on ecosystem fit, not performance.

### When to Use What

If your model fits on a single GPU, use DDP—it's simpler and faster. If it doesn't fit, try optimization first (mixed precision, activation checkpointing, gradient accumulation). Still OOM? Switch to FSDP2; the migration is straightforward since the API is similar. If you need DeepSpeed-specific features like ZeRO-Offload or ZeRO-Infinity, or you're already in the DeepSpeed ecosystem, use ZeRO instead.

## Practical Tips

A few lessons from real-world FSDP usage are worth highlighting.

When saving checkpoints with FSDP2, the sharded state dict matches the training representation—each rank saves only its shard. The DCP API (covered earlier) handles this efficiently. Avoid gathering all shards to rank 0 for a full checkpoint unless you specifically need it for inference; the gather is slow and can OOM on large models.

Shared parameters require some care. If the same tensor appears in multiple places in your model (e.g., tied embeddings), those uses must live in the same FSDP group. FSDP's parameter swapping doesn't preserve sharedness across groups, so structure your model to keep shared parameters in the same module hierarchy, or avoid sharing altogether.

Memory profiling often reveals surprises. The bottleneck isn't always where you expect. Use `torch.profiler` or `nvidia-smi` to investigate. Common culprits include activations (address with checkpointing), temporary tensors that accumulate across iterations (detach or explicitly delete them), and DataLoader with `pin_memory=True` on memory-constrained systems.

Finally, remember that `reshard_after_forward` defaults to `True`, which saves memory by resharding after forward but requires an additional all-gather in backward. If you have memory headroom and communication is your bottleneck, try setting it to `False` to keep parameters unsharded between passes.

### Initialization Best Practices {#sec:fsdp-initialization-best-practices}

For very large models, create on the meta device first, apply FSDP, then move to the actual device and initialize:

```python
with torch.device("meta"):
    model = Transformer(args)
fully_shard(model, mesh=mesh)
model.to_empty(device=device)
model.reset_parameters()
```

This avoids ever materializing the full model on a single device. Make sure all ranks use the same seed so parameters initialize identically.

### Data Loading and Gradient Clipping

Use `DistributedSampler` and call `sampler.set_epoch(epoch)` each epoch—forgetting this means all epochs see the same shuffle. Gradient clipping works with FSDP; just call `torch.nn.utils.clip_grad_norm_` as usual and FSDP handles the unsharding/resharding automatically.

### Mixed Precision

BF16 is generally better than FP16 for parameters (wider dynamic range, less overflow risk). Keep gradient reduction in FP32 (`reduce_dtype=torch.float32`) for numerical stability. Get FSDP working without mixed precision first, then add it.

### Progressive Optimization

Start simple: FSDP2 with mixed precision. If you hit OOM, add activation checkpointing. Still OOM? Reduce batch size or sequence length. CPU offloading is the last resort—it works, but the slowdown is significant. Don't optimize prematurely; get it working first.

## Advanced Topics

### Hybrid Sharding (HSDP) {#sec:hsdp}

At very large scale, you might want to shard within a node but replicate across nodes—this reduces inter-node communication, which is typically slower than intra-node (NVLink vs InfiniBand). Use a 2D mesh:

```python
mesh = init_device_mesh("cuda", (4, 8))  # 4 nodes × 8 GPUs per node
fully_shard(model, mesh=mesh, mesh_dim=1)  # shard within node (dim 1)
```

This shards parameters across 8 GPUs within each node, but replicates across 4 nodes. The tradeoff: more memory usage (4× replication) but less cross-node traffic.

### Compiler Integration

FSDP2 works well with `torch.compile`—apply FSDP first, then compile:

```python
fully_shard(model, mesh=mesh)
model = torch.compile(model)
```

The per-parameter design helps here. The compiler can see individual parameters rather than a flattened buffer, so it can optimize all-gather and reduce-scatter patterns more effectively.

### Other Integrations

FSDP2 works naturally with gradient accumulation and learning rate scheduling—no special handling needed. For mixed precision, use `MixedPrecisionPolicy` instead of the standard AMP context manager. Custom communication hooks are available for fine-grained control, but most users won't need them.

## FSDP via SPMD for TPU/XLA {#sec:fsdp-spmd}

This chapter has focused on FSDP2 for GPU training using CUDA devices. However, PyTorch also provides FSDP via SPMD for TPU/XLA devices, which uses a different approach based on GSPMD (Generalized Single-Program Multiple-Data) for automatic parallelization.

**Key differences from GPU FSDP2:**

- **Uses SPMD mode**: The XLA compiler automatically partitions computation based on sharding annotations, rather than explicit all-gather/reduce-scatter operations.
- **Mesh-based sharding**: Uses PyTorch/XLA's `Mesh` abstraction with named dimensions (e.g., `('fsdp', 'model')`).
- **Compiler-driven**: The XLA compiler handles communication optimization, similar to how JAX's `pmap` works.

A complete example is in `code/fsdp_spmd_tpu.py`. Note that this requires TPU hardware—it will not run on GPU:

```bash
# On a TPU VM:
python code/fsdp_spmd_tpu.py
```

The core pattern:

```python
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2
)

xr.use_spmd()  # Enable SPMD mode

# Create mesh with 'fsdp' axis
num_devices = xr.global_runtime_device_count()
mesh = xs.Mesh(np.array(range(num_devices)), (num_devices, 1), ('fsdp', 'model'))

# Shard inputs and wrap model
x = xs.mark_sharding(x, mesh, ('fsdp', None))
model = FSDPv2(model, mesh)
```

Use FSDP via SPMD when training on TPU devices or when you want compiler-optimized communication patterns. The XLA compiler handles communication optimization automatically, similar to JAX's `pmap`. For GPU training, use the `fully_shard()` API covered in this chapter—it gives you explicit control over communication patterns and doesn't require XLA.

For more details, see the PyTorch/XLA SPMD documentation.[^xla-spmd]

[^xla-spmd]: <https://docs.pytorch.org/xla/master/spmd.html>

## Conclusion

FSDP2 is PyTorch's answer to training models that don't fit on a single GPU. By sharding parameters, gradients, and optimizer states across GPUs, it lets you train models 8×, 16×, or larger than what a single GPU can hold.

It's worth stepping back to understand what FSDP changes—and what it doesn't. FSDP transforms the memory ceiling from a single-GPU constraint to a cluster-wide constraint. A model that requires 160 GB of memory (parameters + gradients + optimizer states) can run on 8 GPUs with 24 GB each, because each GPU holds only 1/8 of the total. This is a fundamental shift: you're no longer limited by the largest GPU you can buy, but by how many GPUs you can connect.

However, FSDP is still data parallelism at its core. Each GPU processes different data batches, and the model computation itself isn't split across devices—every GPU executes the same operations, just on different shards that get all-gathered when needed. This distinguishes FSDP from model parallelism (tensor parallelism, pipeline parallelism), where different GPUs compute different parts of the model simultaneously. FSDP scales memory, not compute per sample. For compute scaling, you still rely on larger batch sizes across more GPUs, just like DDP.

This architectural distinction matters when choosing your parallelism strategy. FSDP alone can take you surprisingly far—models up to hundreds of billions of parameters on large clusters. But for the largest models (trillion+ parameters) or when you need to reduce per-sample latency, you'll combine FSDP with tensor or pipeline parallelism. We cover these combinations in later chapters.

The practical advice is simple: if DDP works, use DDP—it's faster and simpler. When your model outgrows a single GPU, try optimization first (mixed precision, activation checkpointing, gradient accumulation). If you're still OOM, switch to FSDP2. Start with full sharding and the DCP API for checkpointing, profile to find bottlenecks, and test on 2-4 GPUs before scaling to many nodes. Don't optimize blindly—let the profiler guide you.

The code examples in this chapter are complete and runnable. Try them on your hardware to see sharding in action: each rank holds only its portion of the model, and the all-gather/reduce-scatter operations happen automatically.

FSDP2 handles most large model training scenarios well. But what if even full sharding isn't enough? What if you need CPU or NVMe offloading to push memory limits further, or optimized communication patterns for training across many nodes? That's where DeepSpeed's ZeRO comes in. In the next chapter, we'll explore ZeRO-Offload, ZeRO-Infinity, and ZeRO++—features that extend beyond what FSDP2 currently offers—and when to choose DeepSpeed over PyTorch-native solutions.

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
- https://docs.pytorch.org/xla/master/spmd.html
- https://github.com/Wan-Video/Wan2.2 (Wan2.2: FSDP + DeepSpeed Ulysses for multi-GPU inference)
- /media/wukong/jackie/git.repo/distributed-ai/resources/torch-examples/distributed/FSDP2

