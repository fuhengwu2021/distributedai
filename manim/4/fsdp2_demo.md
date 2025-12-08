I'll search for any information about `fully_shard` implementation in the codebase, and then explain how it works internally.

[2 tools called]

Let me look at the actual PyTorch implementation of `fully_shard`:

[1 tool called]

Let me also look at the `FSDPParamGroup` class which handles the actual parameter sharding:

[1 tool called]

Let me also look at how individual parameters are sharded:

[1 tool called]

Based on the PyTorch source code, here's how `fully_shard` works internally:

## High-Level Architecture

`fully_shard` implements FSDP2 (the new per-parameter-sharding API) through several key components:

### 1. **Main Entry Point** (`fully_shard` function)

```python
@contract(state_cls=FSDPState)
def fully_shard(module, *, mesh, reshard_after_forward, mp_policy, offload_policy, ...):
```

**What it does:**

1. **Creates device mesh** - Establishes the communication topology (1D for FSDP, 2D for HSDP)
2. **Attaches FSDPState** - Adds state object to the module (via decorator)
3. **Identifies managed parameters** - Collects all parameters to shard
4. **Creates FSDPParamGroup** - Groups parameters for collective communication
5. **Dynamic class replacement** - Changes module's class to `FSDPModule` by prepending it to the MRO

```186:250:code/chapter4/fsdp2_basic.py
Returns:
        FSDPModule: The module with FSDP applied (in-place).
    """
    torch._C._log_api_usage_once("torch.distributed.fsdp.fully_shard")
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        if mesh.mesh_dim_names is None:
            raise AssertionError(
                "Please init the 2D mesh for HSDP with mesh_dim_names specified"
            )
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    auto_reshard_after_forward = reshard_after_forward is None
    # If the user does not provide ``reshard_after_forward``, we set it to True.
    # During lazy_init, we identify which module is the root and override its value to False
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward if not auto_reshard_after_forward else True,  # type: ignore[arg-type]
        mesh_info,
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined] # see [1]
    state.init(modules, device, mp_policy, auto_reshard_after_forward)

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module
```

### 2. **FSDPParamGroup** - Parameter Sharding Manager

This is where the actual sharding happens:

- **Groups parameters together** for efficient collective communication
- **Creates FSDPParam objects** for each parameter, which:
  - Converts parameters to **DTensor sharded on dim-0** (or custom dim)
  - Stores both sharded and unsharded versions
  - Manages memory allocation/deallocation via storage resizing

- **Sets up communication streams**:
  - `all_gather_stream` - overlaps all-gather with compute
  - `all_gather_copy_in_stream` - overlaps next copy-in with current all-gather
  - `reduce_scatter_stream` - for gradient reduction
  - `all_reduce_stream` - for HSDP (hybrid sharding)

### 3. **FSDPModule** - Hook-Based Orchestration

The dynamically created `FSDPModule` class adds hooks to the original module:

**Pre-forward hook:**
- **All-gather parameters** from all ranks → reconstruct full parameter
- **Convert to compute dtype** (e.g., bfloat16 → float32)
- **Register unsharded params** to module

**Post-forward hook:**
- If `reshard_after_forward=True`: **Free unsharded parameters** (resize storage to 0)
- **Register sharded params** back to module
- **Trigger prefetching** for next module (overlap)

**Pre-backward hook:**
- **Re-all-gather parameters** if they were freed after forward

**Post-backward hook:**
- **Reduce-scatter gradients** across ranks
- **Free unsharded parameters** 
- **Keep only sharded gradients**

### 4. **Key Memory Optimization: Storage Resizing**

FSDP2 uses **storage resizing** (not creating new tensors) to free/allocate memory:

```python
# Simplified version
unsharded_param.untyped_storage().resize_(0)  # Free memory
unsharded_param.untyped_storage().resize_(full_size)  # Allocate memory
```

This preserves **tensor identity** which is critical because:
- Autograd might hold references to the tensor
- Views into the tensor remain valid
- No need to re-register parameters with the optimizer

### 5. **Communication Pattern**

```
Initial State: Each GPU has 1/N of parameters (sharded)

Forward:
  ┌─────────────────────────────────────────┐
  │ 1. Pre-forward: All-gather params       │  GPU0: [shard0] → [param_full]
  │    (all GPUs get full params)           │  GPU1: [shard1] → [param_full]
  ├─────────────────────────────────────────┤
  │ 2. Forward compute with full params     │
  ├─────────────────────────────────────────┤
  │ 3. Post-forward: Free full params       │  GPU0: [param_full] → [shard0]
  │    (keep only local shard)              │  GPU1: [param_full] → [shard1]
  └─────────────────────────────────────────┘

Backward:
  ┌─────────────────────────────────────────┐
  │ 1. Pre-backward: All-gather params      │  GPU0: [shard0] → [param_full]
  │    (if freed after forward)             │  GPU1: [shard1] → [param_full]
  ├─────────────────────────────────────────┤
  │ 2. Backward compute with full params    │  GPU0: grad_full
  │    (generates full gradients)           │  GPU1: grad_full
  ├─────────────────────────────────────────┤
  │ 3. Reduce-scatter gradients             │  GPU0: [grad_full] → [grad_shard0]
  │    (each GPU keeps 1/N of gradients)    │  GPU1: [grad_full] → [grad_shard1]
  ├─────────────────────────────────────────┤
  │ 4. Free full params again               │
  └─────────────────────────────────────────┘
```

### 6. **Bottom-Up Application Pattern**

```132:137:code/chapter4/fsdp2_basic.py
# Apply fully_shard to each TransformerBlock and then the whole model
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module, **fsdp_cfg)
fully_shard(model, **fsdp_cfg)
```

**Why bottom-up?** 
- Each `fully_shard` call creates **one communication group**
- Parameters already in a submodule's group are **excluded** from parent group
- This enables **layer-by-layer communication** for better overlap

### 7. **Key Differences from FSDP1**

| FSDP1 (Wrapper) | FSDP2 (fully_shard) |
|-----------------|---------------------|
| Flattens all params into one tensor | Per-parameter sharding |
| Wrapper class architecture | Functional/composable API |
| ~14k lines of code | ~3k lines of code |
| Complex state management | Simpler with hooks |
| Harder to compile | Better torch.compile support |

The internal implementation is elegant: it uses **DTensor for sharding**, **storage resizing for memory management**, **PyTorch hooks for orchestration**, and **multiple CUDA streams for overlap**. The result is a powerful system that transparently shards your model while maintaining the familiar PyTorch interface!


----

技術示意圖：PyTorch FSDP2 (fully_shard) 內部運作


```
                +------------------------------------------------+
                |                User Model (nn.Module)          |
                +------------------------------------------------+
                                  |
                                  | apply fully_shard()
                                  v
+--------------------------------------------------------------------------------+
|                            FSDP2 Internal Architecture                         |
+--------------------------------------------------------------------------------+

1. DeviceMesh & MeshInfo
   (1D or 2D mesh, e.g., [GPU0, GPU1, GPU2, GPU3])

2. Parameter Discovery
   +---------------------------+
   | _get_managed_parameters() |
   +---------------------------+
                |
                v
3. FSDPParamGroup
   +-----------------------------------------------------------------------+
   |  - FSDPParam objects per parameter                                   |
   |  - Create DTensor shard on mesh[shard_dim]                           |
   |  - Maintain:                                                         |
   |       param_sharded   (1/N shard)                                   |
   |       param_unsharded (full param; storage-resized)                  |
   |  - Allocate CUDA streams for overlap:                                |
   |       all_gather_stream                                              |
   |       all_gather_copy_in_stream                                      |
   |       reduce_scatter_stream                                          |
   +-----------------------------------------------------------------------+

4. FSDPModule (injected via MRO rewrite)
   +-----------------------+
   | Pre-forward hook      |----> AllGather weights into full param
   +-----------------------+
   | Forward compute       |
   +-----------------------+
   | Post-forward hook     |----> Reshard (resize storage to keep only shard)
   +-----------------------+
   | Pre-backward hook     |----> Re-all-gather params
   +-----------------------+
   | Backward compute      |
   +-----------------------+
   | Post-backward hook    |----> ReduceScatter gradients
   +-----------------------+

5. Optimizer Step
   - Each rank only updates its *sharded* parameters/gradients.
```