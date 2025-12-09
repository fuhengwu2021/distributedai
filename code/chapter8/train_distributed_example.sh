#!/bin/bash
#SBATCH --job-name=distributed-training-example
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=train_distributed_%j.out
#SBATCH --error=train_distributed_%j.err

# This script demonstrates distributed training across 2 nodes (node6 and node7)
# Each node gets 1 GPU, for a total of 2 GPUs

echo "=========================================="
echo "Distributed Training Example"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo ""

# Get master node address (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Set distributed training environment variables
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local rank: $LOCAL_RANK"
echo ""

# Display node and GPU information
echo "Node information:"
srun hostname
echo ""

echo "GPU information:"
srun nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# NCCL settings for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0  # Adjust based on your network interface

# Set CUDA visible devices (each rank sees only its assigned GPU)
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Setup conda environment (adjust environment name as needed)
# Uncomment and modify if you use conda:
# source ~/miniconda3/etc/profile.d/conda.sh  # or ~/anaconda3/etc/profile.d/conda.sh
# conda activate <your_env_name>  # e.g., pytorch, base, etc.

# Check if PyTorch is available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Please:"
    echo "  1. Install PyTorch: pip install torch"
    echo "  2. Or activate conda environment with PyTorch"
    echo "  3. Or modify this script to activate your conda environment"
    exit 1
fi

echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run distributed training
# Using srun to launch the training script on all allocated nodes
srun python train_distributed_example.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --data_size 1000

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
