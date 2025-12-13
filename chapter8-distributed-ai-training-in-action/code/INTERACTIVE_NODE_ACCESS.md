# How to Manually Log Into a Slurm Node

Since node6 and node7 are virtual nodes on the same physical machine (`moirai-h200`), you can't SSH into them separately. However, you can get an interactive session on a specific node using Slurm commands.

## Method 1: Interactive Shell with `srun` (Recommended)

Get an interactive shell on a specific node:

```bash
# Set PATH to use Slurm
export PATH=/home/fuhwu/slurm/bin:$PATH

# Get interactive shell on node6
srun -N 1 -w node6 --gres=gpu:1 --pty bash

# Get interactive shell on node7
srun -N 1 -w node7 --gres=gpu:1 --pty bash

# Get interactive shell on any available node
srun -N 1 --gres=gpu:1 --pty bash
```

**Options:**
- `-N 1`: Allocate 1 node
- `-w node6`: Specify which node (node6 or node7)
- `--gres=gpu:1`: Request 1 GPU
- `--pty`: Allocate a pseudo-terminal for interactive use
- `bash`: Command to run (interactive shell)

## Method 2: Allocate Resources with `salloc` Then Use `srun`

Allocate resources first, then run commands:

```bash
# Allocate 1 node with 1 GPU for 1 hour
salloc -N 1 -w node6 --gres=gpu:1 --time=1:00:00

# Once allocated, you'll get a new shell prompt
# Now you can run commands on the allocated node:
srun hostname
srun nvidia-smi
srun python -c "import torch; print(torch.__version__)"

# When done, exit to release the allocation
exit
```

## Method 3: Direct SSH to Physical Machine

Since both virtual nodes are on the same physical machine, you can SSH directly:

```bash
# SSH to the physical machine
ssh moirai-h200
# or
ssh fuhwu@moirai-h200
```

However, this doesn't give you the Slurm environment variables and GPU isolation.

## Method 4: Interactive Job with Specific Resources

Request an interactive job with specific resources:

```bash
# Interactive job on node6 with GPU
srun -N 1 -w node6 --gres=gpu:1 --cpus-per-task=28 --mem=200G --time=1:00:00 --pty bash

# Once you get the shell, you're on node6 with:
# - 1 GPU allocated
# - 28 CPUs
# - 200GB memory
# - Slurm environment variables set
```

## Useful Commands Once You're In

Once you have an interactive session:

```bash
# Check which node you're on
hostname
echo $SLURM_JOB_NODELIST

# Check GPU
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Check Slurm environment variables
env | grep SLURM

# Run Python with PyTorch
python -c "import torch; print(torch.cuda.device_count())"

# Test distributed training setup
python -c "
import os
print('RANK:', os.environ.get('SLURM_PROCID', 'N/A'))
print('LOCAL_RANK:', os.environ.get('SLURM_LOCALID', 'N/A'))
print('WORLD_SIZE:', os.environ.get('SLURM_NTASKS', 'N/A'))
"
```

## Quick Examples

### Example 1: Quick Test on node6
```bash
srun -N 1 -w node6 --gres=gpu:1 --pty bash -c "nvidia-smi && hostname"
```

### Example 2: Interactive Python Session
```bash
srun -N 1 -w node6 --gres=gpu:1 --pty python
```

### Example 3: Run a Script
```bash
srun -N 1 -w node6 --gres=gpu:1 python my_script.py
```

### Example 4: Debug Training Script
```bash
# Get interactive shell
srun -N 1 -w node6 --gres=gpu:1 --pty bash

# Inside the shell, activate conda and test
conda activate wan22
python train_distributed_example.py --epochs 1
```

## Check Node Availability

Before requesting a node, check if it's available:

```bash
sinfo -N -l
scontrol show nodes
```

If a node is DOWN or DRAINED, you can't allocate it. Resume it first:

```bash
scontrol update NodeName=node6 State=RESUME
```

## Notes

- **Virtual nodes**: node6 and node7 are virtual nodes on the same physical machine
- **GPU isolation**: Using `--gres=gpu:1` ensures you get exclusive access to one GPU
- **Environment**: Slurm sets environment variables like `SLURM_PROCID`, `SLURM_LOCALID`, etc.
- **Time limit**: Interactive sessions may have time limits; use `--time` to specify

## Troubleshooting

### "Node not available"
```bash
# Check node status
sinfo -N -l

# Resume if down
scontrol update NodeName=node6 State=RESUME
```

### "Insufficient resources"
```bash
# Check what's available
sinfo
squeue

# Try without specifying node
srun -N 1 --gres=gpu:1 --pty bash
```

### "Permission denied"
```bash
# Make sure you're using the correct Slurm
export PATH=/home/fuhwu/slurm/bin:$PATH
which srun
```
