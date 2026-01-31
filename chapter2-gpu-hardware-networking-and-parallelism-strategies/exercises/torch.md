\fancydividerwithicon[center]{hand.png}


## Exercises


### GPU Hardware Inspector

Implement a comprehensive GPU hardware inspection function that gathers detailed information about available GPUs.

__Requirements:__

- Function signature:
```python
inspect_gpu_hardware()
```
- Return a dictionary with the following keys for each GPU:
  - `name`: GPU name
  - `memory_total_gb`: Total memory in GB
  - `memory_free_gb`: Free memory in GB
  - `compute_capability`: Compute capability as a string (e.g., "9.0")
  - `multiprocessor_count`: Number of streaming multiprocessors
  - `cuda_version`: CUDA version
- Handle the case where CUDA is not available (return empty list)
- Print a formatted summary table showing all GPU information

__Test your function:__
```python
import torch

gpu_info = inspect_gpu_hardware()
print(f"Found {len(gpu_info)} GPU(s)")

for i, info in enumerate(gpu_info):
    print(f"\nGPU {i}:")
    print(f"  Name: {info['name']}")
    print(f"  Memory: {info['memory_total_gb']:.1f} GB total, "
          f"{info['memory_free_gb']:.1f} GB free")
    print(f"  Compute Capability: {info['compute_capability']}")
    print(f"  Multiprocessors: {info['multiprocessor_count']}")
```

### Memory Requirement Calculator

Implement a function to calculate memory requirements for training models with different precisions and optimizer configurations.

__Requirements:__

- Function signature:
```python
calculate_training_memory(num_params, precision='bf16', optimizer='adam', batch_size=1, seq_length=2048)
```
- Calculate memory for:
  - Model parameters (weights)
  - Gradients (same size as parameters)
  - Optimizer states:
    - Adam: 2× parameter size (momentum + variance)
    - SGD: 1× parameter size (momentum only)
  - Activations: approximate as `batch_size × seq_length × hidden_size × num_layers × bytes_per_element`
- Precision options: 'fp32' (4 bytes), 'bf16'/'fp16' (2 bytes), 'int8' (1 byte)
- Return a dictionary with breakdown: `{'parameters': ..., 'gradients': ..., 'optimizer': ..., 'activations': ..., 'total': ...}`
- All values in GB

__Test your function:__
```python
# 70B parameter model with BF16, Adam optimizer
memory = calculate_training_memory(
    num_params=70e9,
    precision='bf16',
    optimizer='adam',
    batch_size=4,
    seq_length=2048
)

print("Memory Requirements (GB):")
for key, value in memory.items():
    print(f"  {key}: {value:.2f} GB")
```

### Topology Detection and Analysis

Implement a function that parses `nvidia-smi topo -m` output to detect GPU interconnect topology.

__Requirements:__

- Function signature:
```python
analyze_gpu_topology()
```
- Use `subprocess` to run `nvidia-smi topo -m` and capture output
- Parse the topology matrix to identify:
  - GPUs connected via NVLink (look for NV18, NV12, NV4)
  - GPUs connected only via PCIe (look for PIX, PXB)
  - All-to-all connectivity (all GPUs can reach all other GPUs via NVLink)
- Return a dictionary with:
  - `num_gpus`: Number of GPUs
  - `nvlink_pairs`: List of GPU pairs connected via NVLink
  - `pcie_only_pairs`: List of GPU pairs connected only via PCIe
  - `has_all_to_all_nvlink`: Boolean indicating if all GPUs have NVLink to all others
  - `recommended_parallelism`: Suggested parallelism strategy based on topology

__Test your function:__
```python
import subprocess

topology = analyze_gpu_topology()
print(f"Number of GPUs: {topology['num_gpus']}")
print(f"NVLink pairs: {topology['nvlink_pairs']}")
print(f"PCIe-only pairs: {topology['pcie_only_pairs']}")
print(f"All-to-all NVLink: {topology['has_all_to_all_nvlink']}")
print(f"Recommended strategy: {topology['recommended_parallelism']}")
```

### Parallelism Strategy Selector

Implement a function that recommends parallelism strategies based on model size, hardware topology, and training requirements.

__Requirements:__

- Function signature:
```python
recommend_parallelism_strategy(model_size_gb, num_gpus, topology_info, has_nvlink_all_to_all, training_type='training')
```
- Consider:
  - Model size vs. single GPU memory
  - Topology (NVLink all-to-all vs. PCIe-only)
  - Training vs. inference requirements
- Return a dictionary with:
  - `primary_strategy`: Main parallelism approach (DDP, FSDP, TP, PP, or hybrid)
  - `reasoning`: Explanation of why this strategy was chosen
  - `alternative_strategies`: List of other viable options
  - `estimated_gpu_count`: Minimum GPUs needed
  - `memory_per_gpu_gb`: Estimated memory per GPU

__Test your function:__
```python
# 70B model (140 GB in BF16) on 8 GPUs with NVLink all-to-all
strategy = recommend_parallelism_strategy(
    model_size_gb=140,
    num_gpus=8,
    topology_info={'has_all_to_all_nvlink': True},
    has_nvlink_all_to_all=True,
    training_type='training'
)

print(f"Recommended: {strategy['primary_strategy']}")
print(f"Reasoning: {strategy['reasoning']}")
print(f"Estimated GPUs needed: {strategy['estimated_gpu_count']}")
print(f"Memory per GPU: {strategy['memory_per_gpu_gb']:.1f} GB")
```



## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Inspect and analyze GPU hardware specifications and topology
- Calculate memory requirements for different model sizes, precisions, and optimizers
- Detect and interpret GPU interconnect topology (NVLink vs PCIe)
- Select appropriate parallelism strategies based on hardware capabilities and model requirements
