# Chapter 8 — Distributed AI Training in Action

So far, we've covered the theory and implementation of distributed training (DDP, FSDP, DeepSpeed, Megatron) and inference (vLLM, SGLang). But understanding the concepts is only part of the equation—you also need to know how to actually run these systems in practice. Most HPC clusters and cloud providers use job schedulers like Slurm to manage GPU resources and coordinate multi-node jobs.

This chapter provides a practical guide to running distributed AI training workloads using Slurm as the job scheduler and resource manager. We'll cover setting up a Slurm cluster, submitting distributed training jobs, integrating with PyTorch DDP and FSDP, managing multi-node training, and best practices for production workloads.

**Audience:** ML engineers and researchers who need to run distributed training jobs on shared GPU clusters, and platform engineers setting up Slurm for AI workloads.

## 1. Introduction to Slurm for Distributed Training

Slurm (Simple Linux Utility for Resource Management) is a widely-used open-source job scheduler and resource manager for HPC and AI clusters. It excels at:

- **Resource allocation**: Allocate GPUs, CPUs, and memory across multiple nodes
- **Job scheduling**: Queue and schedule training jobs based on priority and resource availability
- **Multi-node coordination**: Automatically set up environment variables for distributed training
- **GPU management**: Track and allocate GPU resources via Generic Resources (GRES)

### Why Slurm for Distributed Training?

- **Industry standard**: Used by most HPC centers and cloud providers
- **PyTorch integration**: Native support via `torch.distributed` and `torchrun`
- **Resource isolation**: Ensures jobs don't interfere with each other
- **Fair scheduling**: Prevents resource hoarding and enables fair-share scheduling
- **Checkpointing support**: Built-in mechanisms for job preemption and resumption

## 2. Setting Up Slurm for Multi-GPU Training

### 2.1 Single-Node Multi-GPU Setup

For development and testing, you can run multiple Slurm compute nodes (slurmd daemons) on a single physical machine. This allows you to simulate a multi-node cluster for testing distributed training code.

**Key configuration** (see `code/config/slurm.conf`):

```bash
# Enable multiple slurmd support
# Use $HOSTNAME or $(hostname) to get the actual hostname
NodeName=node6 NodeHostname=$HOSTNAME Port=17016 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN

NodeName=node7 NodeHostname=$HOSTNAME Port=17017 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
```

**GPU mapping** (see `code/config/gres.conf`):

```bash
NodeName=node6 Name=gpu File=/dev/nvidia6
NodeName=node7 Name=gpu File=/dev/nvidia7
```

### 2.2 Quick Setup

Use the provided setup script:

```bash
cd code/chapter8
bash slurm_setup.sh
```

This script:
1. Creates required directories
2. Generates configuration files
3. Starts slurmctld (controller)
4. Starts slurmd daemons for each virtual node

### 2.3 Verifying the Setup

```bash
# Set PATH to use compiled Slurm
# Replace $SLURM_PREFIX with your Slurm installation prefix (e.g., /opt/slurm or $HOME/slurm)
export PATH=$SLURM_PREFIX/bin:$PATH

# Check cluster status
sinfo

# Expected output:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      2   idle node[6-7]

# Check node details
scontrol show nodes

# Test simple job
srun -N 1 hostname
srun -N 2 hostname
```

## 3. Submitting Distributed Training Jobs

### 3.1 Basic Job Submission with `srun`

The simplest way to run a distributed training job:

```bash
# Two nodes, 1 GPU each
srun -N 2 --gres=gpu:1 --cpus-per-task=4 python train.py
```

### 3.2 Batch Jobs with `sbatch`

For longer-running jobs, use batch submission:

**Example: `train_ddp.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=ddp-training
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Get node list and master address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE, Rank: $RANK, Local rank: $LOCAL_RANK"

# Run training
srun python train_ddp.py
```

Submit the job:

```bash
sbatch train_ddp.sh
```

Check job status:

```bash
squeue                    # List all jobs
squeue -u $USER          # List your jobs
scontrol show job <job_id>  # Detailed job info
```

### 3.3 Environment Variables

Slurm automatically sets these environment variables for distributed training:

- `SLURM_JOB_NODELIST`: List of allocated nodes (e.g., `node[6-7]`)
- `SLURM_JOB_NUM_NODES`: Number of nodes allocated
- `SLURM_NTASKS`: Total number of tasks
- `SLURM_PROCID`: Global process rank (0 to NTASKS-1)
- `SLURM_LOCALID`: Local rank on the node (0 to tasks-per-node-1)
- `SLURM_NODEID`: Node index (0 to NUM_NODES-1)

## 4. PyTorch Distributed Training with Slurm

This section provides an overview of different distributed training frameworks and their integration with SLURM. For hands-on examples with complete code, see [Section 9: Hands-on: Complete Distributed Training Workflow](#9-hands-on-complete-distributed-training-workflow).

### 4.1 PyTorch DDP (Distributed Data Parallel)

PyTorch DDP replicates the model across multiple GPUs and synchronizes gradients during backward pass. It's the simplest distributed training approach.

**Key characteristics:**
- Model replicated on each GPU
- Gradients synchronized via all-reduce
- Works with `torch.distributed.launch` or `torchrun`
- Suitable for models that fit in single GPU memory

For complete code examples and SLURM scripts, see [Section 9.1: PyTorch DDP](#91-pytorch-ddp-distributed-data-parallel).

### 4.2 PyTorch FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across GPUs, enabling training of larger models.

**Key characteristics:**
- Parameters sharded across GPUs
- Memory efficient for large models
- Supports CPU offloading
- Uses `torchrun` for distributed launch

For complete code examples and SLURM scripts, see [Section 9.2: PyTorch FSDP](#92-pytorch-fsdp-fully-sharded-data-parallel).

### 4.3 DeepSpeed ZeRO-3 with CPU Offload

DeepSpeed ZeRO-3 provides advanced memory optimization with optional CPU offloading for training very large models.

**Key characteristics:**
- Automatic distributed setup (no manual initialization)
- ZeRO-3 shards parameters, gradients, and optimizer states
- CPU offload enables training models larger than total GPU memory
- Works seamlessly with HuggingFace models

For complete code examples, configuration files, and SLURM scripts, see [Section 9.3: DeepSpeed ZeRO-3](#93-deepspeed-zero-3-with-cpu-offload).

### 4.4 Megatron-LM Training with SLURM

Megatron-LM is NVIDIA's framework for training large language models with advanced parallelism strategies.

**Key characteristics:**
- Multiple parallelism strategies: Tensor, pipeline, context, and data parallelism
- Production-ready optimizations
- Supports various model architectures (GPT, BERT, T5, etc.)
- Built-in FP8 support and other cutting-edge features

For complete code examples, installation instructions, and SLURM scripts, see [Section 9.4: Megatron-LM](#94-megatron-lm-training-with-slurm).

### 4.5 Using Slurm's Built-in MPI Support

Slurm can automatically set up the process group via MPI:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

# Slurm automatically sets up MPI environment
srun python train_ddp.py
```

In your Python code:

```python
import os
import torch.distributed as dist

# Use environment variables set by Slurm
dist.init_process_group(
    backend='nccl',
    init_method='env://',  # Use environment variables
)
```

## 5. Advanced Slurm Features for Training

### 5.1 Job Arrays for Hyperparameter Tuning

Run multiple training jobs with different hyperparameters:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Each array task gets different hyperparameters
LR=$(echo "0.001 0.0001 0.00001 0.000001" | cut -d' ' -f$((SLURM_ARRAY_TASK_ID % 4 + 1)))
BATCH_SIZE=$((32 * (SLURM_ARRAY_TASK_ID / 4 + 1)))

python train.py --lr $LR --batch_size $BATCH_SIZE
```

Submit:

```bash
sbatch train_array.sh
```

### 5.2 Interactive Jobs with `salloc`

Allocate resources interactively for debugging:

```bash
# Allocate 2 nodes, 1 GPU each, for 1 hour
salloc -N 2 --gres=gpu:1 --time=1:00:00

# Once allocated, run commands
srun hostname
srun nvidia-smi
srun python train.py

# Release when done
exit
```

### 5.3 Job Dependencies

Chain jobs so one starts after another completes:

```bash
# Submit first job
JOB1=$(sbatch --parsable train_stage1.sh)

# Submit second job that depends on first
sbatch --dependency=afterok:$JOB1 train_stage2.sh
```

### 5.4 Checkpointing and Job Resumption

Slurm supports job preemption and resumption:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90  # Send signal 90 seconds before time limit

# Handle checkpoint signal
trap 'echo "Checkpointing..."; python checkpoint.py' SIGUSR1

python train.py --resume --checkpoint_dir=/path/to/checkpoints
```

## 6. Monitoring and Debugging

### 6.1 Job Monitoring

```bash
# Watch job queue
watch -n 1 squeue

# Watch specific job
watch -n 1 scontrol show job <job_id>

# View job output in real-time
tail -f slurm-<job_id>.out

# Check GPU usage across nodes
srun -N 2 nvidia-smi
```

### 6.2 Logging and Output

Slurm captures stdout and stderr:

```bash
#SBATCH --output=train_%j.out    # %j = job ID
#SBATCH --error=train_%j.err
```

For distributed training, each rank writes to the same file. Use rank-specific logging:

```python
import logging
import torch.distributed as dist

rank = dist.get_rank() if dist.is_initialized() else 0
logging.basicConfig(
    filename=f'train_rank_{rank}.log',
    level=logging.INFO
)
```

### 6.3 Profiling Distributed Training

Use PyTorch profiler with Slurm:

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Training step
    output = model(input)

# Save trace (only on rank 0)
if dist.get_rank() == 0:
    prof.export_chrome_trace("trace.json")
```

## 7. Best Practices

### 7.1 Resource Allocation

- **Always specify resources explicitly**: Don't rely on defaults
- **Use `--exclusive` for full node**: When you need all resources on a node
- **Request appropriate memory**: Use `--mem` or `--mem-per-gpu` to avoid OOM

```bash
srun -N 2 --gres=gpu:1 --mem=200G --cpus-per-task=28 python train.py
```

### 7.2 Multi-Node Communication

- **Use high-speed interconnects**: InfiniBand or high-speed Ethernet for multi-node
- **Set appropriate NCCL environment variables**:

```bash
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0  # Specify network interface
export NCCL_DEBUG=INFO  # For debugging
```

### 7.3 Checkpointing Strategy

- **Frequent checkpoints**: Save every N steps, not just at epoch boundaries
- **Distributed checkpointing**: Use `torch.distributed.checkpoint` for FSDP
- **Resume capability**: Always implement `--resume` flag in training scripts

### 7.4 Error Handling

- **Handle node failures**: Implement retry logic for transient failures
- **Validate data loading**: Ensure data is accessible from all nodes
- **Monitor for deadlocks**: Use timeouts and health checks

## 8. Troubleshooting Common Issues

### 8.1 Nodes Not Available

```bash
# Check node status
sinfo -N -l

# Resume down nodes
scontrol update NodeName=node[6-7] State=RESUME

# Drain nodes for maintenance
scontrol update NodeName=node6 State=DRAIN Reason="maintenance"
```

### 8.2 GPU Allocation Issues

```bash
# Check GPU availability
scontrol show nodes | grep Gres

# Verify GPU mapping
# Replace $SLURM_PREFIX with your Slurm installation prefix
cat $SLURM_PREFIX/etc/gres.conf

# Test GPU allocation
srun -N 1 --gres=gpu:1 nvidia-smi -L
```

### 8.3 Communication Errors

- **Check network connectivity**: `srun -N 2 ping -c 3 <other_node>`
- **Verify NCCL setup**: Set `NCCL_DEBUG=INFO` for detailed logs
- **Check firewall**: Ensure required ports are open

### 8.4 Job Hanging

- **Check for deadlocks**: Look for processes waiting on barriers
- **Verify data loading**: Ensure all ranks can access data
- **Check logs**: Review both stdout and stderr from all ranks

## 9. Hands-on: Complete Distributed Training Workflow

This section provides hands-on examples for running distributed training with different frameworks on SLURM clusters. All code examples are available in the `code/` directory.

### 9.1 PyTorch DDP (Distributed Data Parallel)

**Method 1: Using `torch.distributed.launch`**

```python
# train_ddp.py
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Initialize process group
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    
    # Create model and wrap with DDP
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    # Training loop
    for epoch in range(10):
        # ... training code ...
        if rank == 0:
            print(f"Epoch {epoch} completed")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

**Slurm batch script:**

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

srun python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_ddp.py
```

**Method 2: Using `torchrun` (Recommended)**

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_ddp.py
```

### 9.2 PyTorch FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across GPUs:

```python
# train_fsdp.py
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    # Create model
    model = MyLargeModel()
    
    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )
    
    # Training loop
    # ...

if __name__ == '__main__':
    main()
```

**Slurm batch script for FSDP:**

```bash
#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_fsdp.py
```

### 9.3 DeepSpeed ZeRO-3 with CPU Offload

DeepSpeed ZeRO-3 enables training models larger than GPU memory by sharding parameters, gradients, and optimizer states across GPUs, with optional CPU offloading for even larger models.

**Key features:**
- Automatic distributed setup (no manual `torch.distributed` initialization needed)
- ZeRO-3 shards parameters, gradients, and optimizer states
- CPU offload enables training models larger than total GPU memory
- Works seamlessly with HuggingFace models

**Training script** (`code/deepspeed/train.py`):

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Initialize distributed (DeepSpeed handles this internally)
    deepspeed.init_distributed()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_zero3_offload.json"
    )
    
    # Training loop
    for epoch in range(10):
        # ... training code ...
        model_engine.backward(loss)
        model_engine.step()

if __name__ == "__main__":
    main()
```

**DeepSpeed configuration** (`code/deepspeed/ds_zero3_offload.json`):

```json
{
  "train_batch_size": 2,
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16
  },

  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": "auto"
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },

  "wall_clock_breakdown": false
}
```

**SLURM batch script** (`code/deepspeed/run.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=deepspeed-zero3
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_%j_%N.out
#SBATCH --error=logs/train_%j_%N.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research

# Distributed setup
export MASTER_ADDR=127.0.0.1  # For single physical node with virtual nodes
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker,lo
export GLOO_SOCKET_IFNAME=eth0

# Launch training
srun --chdir="$SLURM_SUBMIT_DIR" --label \
    bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate research
        export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        cd \"$SLURM_SUBMIT_DIR\"
        python train.py --deepspeed --deepspeed_config ds_zero3_offload.json
    "
```

**Usage:**

```bash
# Submit job
cd chapter8-distributed-ai-training-in-action/code/deepspeed
sbatch run.slurm

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/train_*.out
```

**Key differences from DDP/FSDP:**

1. **No manual distributed setup**: DeepSpeed handles distributed initialization internally
2. **Direct Python execution**: Run `python train.py` directly, not through `torchrun`
3. **SLURM environment variables**: The script reads `SLURM_PROCID`, `SLURM_NTASKS`, etc.
4. **CPU offload support**: Can train models larger than GPU memory by offloading to CPU
5. **Automatic optimizer creation**: Can specify optimizer in config file

**Important notes:**

- DeepSpeed requires `LOCAL_RANK` environment variable (set from `SLURM_LOCALID`)
- GPU mapping: For virtual nodes, map node name to GPU number (e.g., `node6` → GPU 6)
- IPv6 resolution: Set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to avoid IPv6 issues
- Conda activation: Ensure conda environment is activated on each compute node via `srun`

### 9.4 Megatron-LM Training with SLURM

Megatron-LM is NVIDIA's framework for training large language models with advanced parallelism strategies including tensor parallelism (TP), pipeline parallelism (PP), context parallelism (CP), and data parallelism (DP).

**Key features:**
- **Multiple parallelism strategies**: Tensor, pipeline, context, and data parallelism
- **Efficient memory management**: Optimized for large model training
- **Production-ready**: Used by NVIDIA for training state-of-the-art models
- **Flexible configuration**: Supports various model architectures (GPT, BERT, T5, etc.)
- **Built-in optimizations**: FP8 support, activation recomputation, gradient accumulation

**Prerequisites:**

1. **Install Megatron-LM from source** (required for `megatron.training` module):
   ```bash
   conda activate research
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   pip install --no-build-isolation .[mlm,dev]
   ```
   
   **Note**: The PyPI package `megatron-core` only includes `megatron.core`, not `megatron.training`. 
   Since `pretrain_gpt.py` requires `megatron.training`, you must install from source.

2. **Copy training scripts** to your working directory:
   - `pretrain_gpt.py` - Main training script
   - `gpt_builders.py` - Model builder utilities
   - `model_provider.py` - Model provider functions

**SLURM batch script** (`code/megatron/run.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=megatron-gpt
#SBATCH --nodes=2                    # 2 nodes (one GPU per node)
#SBATCH --gres=gpu:1                # 1 GPU per node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_%j_%N.out
#SBATCH --error=logs/train_%j_%N.err

# Activate conda environment
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate research || {
    echo "ERROR: Failed to activate conda environment 'research'"
    exit 1
}

# Get the directory where this script is located
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$SCRIPT_DIR"
mkdir -p logs

# Use pretrain_gpt.py from the same directory
PRETRAIN_SCRIPT="${SCRIPT_DIR}/pretrain_gpt.py"

# Distributed training setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-6000}
export WORLD_SIZE=$SLURM_NTASKS

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Training configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/gpt_8b}"
TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-${SCRIPT_DIR}/tensorboard_logs/gpt_8b}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${SCRIPT_DIR}/data_cache}"

# Model configuration
NUM_LAYERS=${NUM_LAYERS:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-14336}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
SEQ_LENGTH=${SEQ_LENGTH:-2048}

# Parallelism configuration
TP_SIZE=${TP_SIZE:-1}      # Tensor parallelism
CP_SIZE=${CP_SIZE:-1}       # Context parallelism
PP_SIZE=${PP_SIZE:-1}       # Pipeline parallelism

# Training hyperparameters
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LR=${LR:-0.00015}
MIN_LR=${MIN_LR:-0.00001}

# Use mock data for demonstration
USE_MOCK_DATA=${USE_MOCK_DATA:-1}

# Launch training with torchrun
srun --chdir="$SCRIPT_DIR" --label \
    bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
            source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate research
        
        # Set CUDA_VISIBLE_DEVICES
        export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        
        cd \"$SCRIPT_DIR\"
        
        # Launch with torchrun
        torchrun \\
            --nproc_per_node=1 \\
            --nnodes=\$SLURM_JOB_NUM_NODES \\
            --node_rank=\$SLURM_NODEID \\
            --master_addr=\"$MASTER_ADDR\" \\
            --master_port=\"$MASTER_PORT\" \\
            \"$PRETRAIN_SCRIPT\" \\
            --use-mcore-models \\
            --num-layers $NUM_LAYERS \\
            --hidden-size $HIDDEN_SIZE \\
            --ffn-hidden-size $FFN_HIDDEN_SIZE \\
            --num-attention-heads $NUM_ATTENTION_HEADS \\
            --group-query-attention \\
            --num-query-groups 8 \\
            --seq-length $SEQ_LENGTH \\
            --max-position-embeddings $SEQ_LENGTH \\
            --position-embedding-type rope \\
            --micro-batch-size $MICRO_BATCH_SIZE \\
            --global-batch-size $GLOBAL_BATCH_SIZE \\
            --train-samples 1000000 \\
            --lr $LR \\
            --min-lr $MIN_LR \\
            --lr-decay-style cosine \\
            --tensor-model-parallel-size $TP_SIZE \\
            --context-parallel-size $CP_SIZE \\
            --pipeline-model-parallel-size $PP_SIZE \\
            --sequence-parallel \\
            --use-distributed-optimizer \\
            --bf16 \\
            --mock-data \\
            --tokenizer-type NullTokenizer \\
            --vocab-size 128256 \\
            --save \"$CHECKPOINT_PATH\" \\
            --load \"$CHECKPOINT_PATH\" \\
            --tensorboard-dir \"$TENSORBOARD_LOGS_PATH\"
    "
```

**Usage:**

```bash
# Submit job
cd chapter8-distributed-ai-training-in-action/code/megatron
sbatch run.slurm

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/train_*.out
```

**Key differences from DDP/FSDP/DeepSpeed:**

1. **Multiple parallelism strategies**: Supports tensor, pipeline, context, and data parallelism simultaneously
2. **torchrun launcher**: Uses `torchrun` for distributed initialization (like DDP/FSDP)
3. **Model architecture**: Designed specifically for transformer-based language models
4. **Advanced features**: Built-in support for FP8, MoE (Mixture of Experts), and other cutting-edge techniques
5. **Production optimizations**: Includes many production-ready optimizations out of the box

**Important notes:**

- **Installation requirement**: Must install from source to get `megatron.training` module
- **Script dependencies**: Requires `pretrain_gpt.py`, `gpt_builders.py`, and `model_provider.py` in the same directory
- **Parallelism configuration**: Adjust `TP_SIZE`, `PP_SIZE`, `CP_SIZE` based on your hardware and model size
- **Mock data**: The example uses mock data (`--mock-data`). For real training, provide data paths and tokenizer
- **Memory requirements**: Large models may require adjusting batch sizes and sequence lengths

**Checkpoint File Size Analysis:**

When training with Megatron-LM, checkpoint files can be quite large. For an 8B parameter model, you might see checkpoint directories like:

```
code/megatron/checkpoints/gpt_8b/iter_0000010/
27G     __0_0.distcp
27G     __0_1.distcp
27G     __1_0.distcp
27G     __1_1.distcp
24K     common.pt
4.0K    metadata.json
```

**Why are checkpoints so large?**

**Theoretical size calculation:**
- **Model parameters (bf16)**: 8.03B × 2 bytes = 16.06 GB
- **Optimizer states (Adam, fp32)**: 8.03B × 8 bytes = 64.24 GB
  - Momentum (exp_avg): 4 bytes/param
  - Variance (exp_avg_sq): 4 bytes/param
- **Theoretical total**: 80.30 GB
- **Actual size**: ~108 GB (4 files × 27 GB)

**Additional overhead (~27.70 GB) explained:**

1. **Distributed optimizer sharding:**
   - Using `--use-distributed-optimizer` shards parameters and optimizer states across multiple ranks
   - Each rank saves its own shard, which may include some redundancy for efficient loading

2. **File format overhead:**
   - PyTorch distributed checkpoint format includes metadata
   - Index and mapping information for distributed loading
   - Alignment and padding for efficient I/O

3. **Shard structure:**
   - `__0_0.distcp`: rank 0, shard 0
   - `__0_1.distcp`: rank 0, shard 1
   - `__1_0.distcp`: rank 1, shard 0
   - `__1_1.distcp`: rank 1, shard 1
   - Each rank has multiple shards to enable parallel save/load operations

**Is this normal?**

Yes, this is expected behavior:
- 8B model + Adam optimizer ≈ 80GB is the theoretical minimum
- Distributed checkpoints have additional overhead for parallel I/O
- Optimizer states are typically 4× larger than model parameters (fp32 vs bf16)
- The distributed checkpoint format enables efficient multi-node checkpointing and resuming

**Tips for managing checkpoint size:**
- Use `--save-interval` to control checkpoint frequency
- Consider using optimizer state offloading if available
- For production, implement checkpoint rotation to keep only recent checkpoints
- Use distributed storage (e.g., shared filesystem) for checkpoint directories

**Checkpoint Format Conversion:**

Megatron-LM checkpoints are saved in a distributed format (`.distcp` files) that requires Megatron-LM to load. For use with other frameworks or standalone PyTorch models, you can convert checkpoints to standard formats.

**Converting to PyTorch Format:**

Use the provided conversion script (`code/megatron/convert_megatron_checkpoint.py`):

```bash
# Convert Megatron checkpoint to standard PyTorch format
python convert_megatron_checkpoint.py \
    --checkpoint-dir checkpoints/gpt_8b/iter_0000010 \
    --output-dir exported_checkpoint \
    --format pytorch \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --vocab-size 128256 \
    --max-position-embeddings 2048 \
    --use-mcore-models \
    --bf16
```

**Converting to HuggingFace Format:**

```bash
# Convert to HuggingFace format (simplified)
python convert_megatron_checkpoint.py \
    --checkpoint-dir checkpoints/gpt_8b/iter_0000010 \
    --output-dir huggingface_checkpoint \
    --format huggingface \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --vocab-size 128256 \
    --max-position-embeddings 2048 \
    --use-mcore-models \
    --bf16
```

**Using Converted Checkpoints:**

The exported PyTorch checkpoint is **completely independent** and does NOT require Megatron-LM to load:

```python
import torch

# Load checkpoint - NO MEGATRON NEEDED!
checkpoint = torch.load('exported_checkpoint/model.pt', map_location='cpu')

# View model configuration
print(checkpoint['model_config'])

# Access state dict
state_dict = checkpoint['model_state_dict']
print(f"Total keys: {len(state_dict)}")
print(f"First key: {list(state_dict.keys())[0]}")
```

**Checkpoint Structure:**

The exported checkpoint contains:

```python
{
    'model_state_dict': {
        # All model weights in standard PyTorch format
        'embedding.word_embeddings.weight': tensor(...),
        'decoder.layers.0.self_attention.linear_proj.weight': tensor(...),
        # ... etc
    },
    'model_config': {
        'num_layers': 32,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'vocab_size': 128256,
        'max_position_embeddings': 2048,
    }
}
```

**Key Benefits of Conversion:**

- ✅ **Standalone**: No Megatron-LM required to load the checkpoint
- ✅ **Standard format**: Can be used with any PyTorch model
- ✅ **Smaller size**: Exported checkpoints only contain model weights (no optimizer state)
- ✅ **Compatible**: Can be loaded by other frameworks (vLLM, SGLang, etc.) with proper model initialization

**Note**: Full HuggingFace format conversion may require additional layer name mapping and tensor reshaping. For production use, consider using tools like [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for complete format conversion.

**Building a wheel package** (optional):

If you want to create a standalone wheel that includes `megatron.training`, see `code/megatron/BUILD_PACKAGE.md` for instructions on building a custom package.


## References

- [Slurm Documentation](https://slurm.schedmd.com/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Single-Node Slurm Cluster Docker](https://github.com/minyang-chen/single-node-slurm-cluster-docker) - Fully dockerized single-node Slurm cluster with GPU support
- [ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates](https://arxiv.org/html/2505.12242v3) - Importance-aware offloading framework that decouples GPU and CPU updates to eliminate GPU stalls
- [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/html/2409.15241v1) - Generic approach to hide communication behind computation in tensor parallelism training
- [Optimizing Language Model Training: A Practical Guide to SLURM](https://medium.com/@viktorciroski/optimizing-language-model-training-a-practical-guide-to-slurm-a6621d3c1bf2) - Practical guide to using SLURM for fine-tuning large language models across multiple GPUs
- [DeepOps & SLURM: Your GPU Cluster Guide](https://arxiv.org/pdf/2405.00030) - Research paper on distributed training (check arXiv for latest version)
- [Deploy an Auto-Scaling HPC Cluster with Slurm on GCP](https://codelabs.developers.google.com/codelabs/hpc-slurm-on-gcp#0) - Step-by-step guide to deploying Slurm clusters on Google Cloud Platform
- [Slurm Workload Manager (Official GitHub Repository)](https://github.com/SchedMD/slurm) - Official source code repository for Slurm maintained by SchedMD
- [Running Multiple Worker Daemons in Slurm](https://stackoverflow.com/questions/40695348/running-multiple-worker-daemons-slurm) - Stack Overflow discussion on configuring multiple slurmd daemons on a single machine
- https://github.com/NVIDIA-NeMo/Megatron-Bridge


