---
title: "Distributed AI Training in Action"
---

# Chapter 8 — Distributed AI Training in Action

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

**Key configuration** (see `code/chapter8/config/slurm.conf`):

```bash
# Enable multiple slurmd support
NodeName=node6 NodeHostname=moirai-h200 Port=17016 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN

NodeName=node7 NodeHostname=moirai-h200 Port=17017 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
```

**GPU mapping** (see `code/chapter8/config/gres.conf`):

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
export PATH=/home/fuhwu/slurm/bin:$PATH

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
# Single node, 2 GPUs
srun -N 1 --gres=gpu:2 python train.py

# Two nodes, 1 GPU each
srun -N 2 --gres=gpu:1 python train.py

# Two nodes, 2 GPUs each (4 GPUs total)
srun -N 2 --gres=gpu:2 python train.py
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

### 4.1 PyTorch DDP (Distributed Data Parallel)

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

### 4.2 PyTorch FSDP (Fully Sharded Data Parallel)

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

### 4.3 Using Slurm's Built-in MPI Support

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
cat /home/fuhwu/slurm/etc/gres.conf

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

## 9. Example: Complete Distributed Training Workflow

### 9.1 Training Script

```python
# train_distributed.py
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    
    # Create model
    model = MyModel().to(device)
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(f'{args.checkpoint_dir}/latest.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    
    # Training loop
    for epoch in range(start_epoch, 100):
        train_one_epoch(model, epoch, device)
        
        # Save checkpoint (rank 0 only)
        if rank == 0 and epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, f'{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pt')
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### 9.2 Slurm Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=distributed-training
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --signal=SIGUSR1@90

# Setup distributed training environment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# Handle checkpoint signal
trap 'echo "Checkpointing..."; python checkpoint.py' SIGUSR1

# Run training
srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed.py \
    --resume \
    --checkpoint_dir=/path/to/checkpoints
```

### 9.3 Submit and Monitor

```bash
# Submit job
sbatch train_distributed.sh

# Monitor
watch -n 1 squeue
tail -f train_<job_id>.out

# Check GPU usage
srun -N 2 nvidia-smi
```

## 10. Configuration Files Reference

All configuration files are available in `code/chapter8/config/`:

- **`slurm.conf`**: Main Slurm configuration
- **`gres.conf`**: GPU resource mapping
- **`cgroup.conf`**: Cgroup settings (disabled for simplicity)
- **`slurm_setup.sh`**: Automated setup script
- **`README.md`**: Detailed configuration documentation

## 11. Next Steps

- **Scale to more nodes**: Add more physical nodes to the cluster
- **Enable cgroup constraints**: For production resource limits
- **Set up fair-share scheduling**: For multi-user environments
- **Integrate with MLflow/Weights & Biases**: For experiment tracking
- **Implement job preemption**: For better resource utilization

## References

- [Slurm Documentation](https://slurm.schedmd.com/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- Configuration files: `code/chapter8/config/`

---

**Summary**: This chapter demonstrated how to use Slurm for distributed AI training, covering setup, job submission, PyTorch DDP/FSDP integration, and best practices. The provided configuration files and examples serve as a practical starting point for running distributed training workloads.
