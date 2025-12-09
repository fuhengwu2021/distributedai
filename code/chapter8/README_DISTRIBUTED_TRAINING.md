# Distributed Training Example

This directory contains a complete example of distributed training using Slurm with 2 nodes (node6 and node7), each with 1 GPU.

## Overview

The example demonstrates:
- **PyTorch DDP (Distributed Data Parallel)**: Training across 2 GPUs on 2 nodes
- **Slurm integration**: Using Slurm's environment variables for distributed setup
- **DistributedSampler**: Proper data sharding across nodes
- **Model checkpointing**: Saving model after training

## Files

- `train_distributed_example.py`: PyTorch training script with DDP
- `train_distributed_example.sh`: Slurm batch script for job submission
- `README_DISTRIBUTED_TRAINING.md`: This file

## Prerequisites

1. **Slurm cluster running**: Ensure node6 and node7 are in IDLE state
   ```bash
   export PATH=/home/fuhwu/slurm/bin:$PATH
   sinfo
   ```

2. **PyTorch installed**: With CUDA and distributed support
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

3. **NCCL**: For multi-GPU communication (usually comes with PyTorch)

## Quick Start

### 1. Submit the Job

```bash
cd /home/fuhwu/workspace/distributedai/code/chapter8
sbatch train_distributed_example.sh
```

### 2. Monitor the Job

```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# Watch output in real-time
tail -f train_distributed_<job_id>.out
```

### 3. Check Results

After completion, you should see:
- `train_distributed_<job_id>.out`: Standard output
- `train_distributed_<job_id>.err`: Standard error
- `model_distributed.pt`: Saved model (if training completes)

## Understanding the Example

### Training Script (`train_distributed_example.py`)

**Key components:**

1. **Distributed Setup**:
   ```python
   rank, world_size, local_rank, device = setup_distributed()
   ```
   - Gets rank, world_size from Slurm environment variables
   - Initializes NCCL process group
   - Sets up CUDA device for each rank

2. **Model Wrapping**:
   ```python
   model = DDP(model, device_ids=[local_rank])
   ```
   - Wraps model with DistributedDataParallel
   - Handles gradient synchronization automatically

3. **Data Sharding**:
   ```python
   sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
   ```
   - Ensures each rank gets a different subset of data
   - Prevents data duplication across nodes

4. **Synchronization**:
   ```python
   dist.barrier()  # Wait for all ranks
   ```
   - Ensures all processes stay synchronized

### Slurm Script (`train_distributed_example.sh`)

**Key settings:**

- `--nodes=2`: Allocate 2 nodes (node6 and node7)
- `--gres=gpu:1`: 1 GPU per node (2 GPUs total)
- `--ntasks-per-node=1`: 1 training process per node
- `--cpus-per-task=28`: CPUs per process
- `--mem=200G`: Memory per node

**Environment variables set:**

- `MASTER_ADDR`: IP/hostname of the first node
- `MASTER_PORT`: Port for process group initialization
- `WORLD_SIZE`: Total number of processes (2)
- `RANK`: Global process rank (0 or 1)
- `LOCAL_RANK`: Local rank on node (always 0 in this case)

## Expected Output

When running successfully, you should see:

```
==========================================
Distributed Training Example
==========================================
Job ID: 123
Number of nodes: 2
Node list: node[6-7]
Master address: moirai-h200
World size: 2
Rank: 0
Local rank: 0

Node information:
moirai-h200
moirai-h200

GPU information:
0, NVIDIA H200, 141309 MB
0, NVIDIA H200, 141309 MB

==========================================
Starting training...
==========================================
Starting distributed training
World size: 2, Rank: 0, Local rank: 0
Device: cuda:0
Epochs: 10, Batch size: 32, LR: 0.001
Batch 0, Loss: 0.8234
Batch 10, Loss: 0.7123
...
Epoch 1/10, Average Loss: 0.6543
...
Training completed successfully!
```

## Customization

### Change Number of Nodes

Edit `train_distributed_example.sh`:
```bash
#SBATCH --nodes=4  # Use 4 nodes instead of 2
```

### Change Number of GPUs per Node

Edit `train_distributed_example.sh`:
```bash
#SBATCH --gres=gpu:2  # 2 GPUs per node
```

And update the training script to handle multiple GPUs per node:
```python
# In train_distributed_example.py, change:
torch.cuda.set_device(local_rank % torch.cuda.device_count())
```

### Adjust Training Parameters

Edit the `srun` command in `train_distributed_example.sh`:
```bash
srun python train_distributed_example.py \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.0001 \
    --data_size 5000
```

## Troubleshooting

### Job Stuck in PENDING

```bash
# Check why job is pending
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Check node availability
sinfo
```

### Nodes Not Responding

```bash
# Check node status
scontrol show nodes

# Resume nodes if down
scontrol update NodeName=node[6-7] State=RESUME
```

### NCCL Communication Errors

```bash
# Enable detailed NCCL logging
export NCCL_DEBUG=INFO

# Check network connectivity
srun -N 2 ping -c 3 $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
```

### CUDA Out of Memory

Reduce batch size or model size:
```bash
srun python train_distributed_example.py --batch_size 16
```

## Advanced Usage

### Using torchrun Instead of srun

You can also use `torchrun` for more control:

```bash
srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed_example.py
```

### Multi-GPU per Node

For 2 GPUs per node (4 GPUs total):

```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
```

Update training script:
```python
# Use all GPUs on the node
local_rank = int(os.environ.get('SLURM_LOCALID', 0))
torch.cuda.set_device(local_rank)
```

### Checkpointing

Add checkpoint saving to the training script:

```python
if rank == 0 and epoch % 5 == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

## Next Steps

1. **Try FSDP**: Modify the example to use Fully Sharded Data Parallel
2. **Add validation**: Include validation loop with proper metric aggregation
3. **Experiment tracking**: Integrate with MLflow or Weights & Biases
4. **Mixed precision**: Add automatic mixed precision (AMP) training
5. **Gradient accumulation**: For larger effective batch sizes

## References

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Slurm Documentation](https://slurm.schedmd.com/)
- Configuration files: `code/chapter8/config/`
