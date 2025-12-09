# Slurm Quick Start - 2 GPU Setup

## TL;DR - Run This Now

```bash
# 1. Run the setup script
bash /home/fuhwu/workspace/distributedai/code/chapter8/slurm_setup.sh

# 2. Fix PATH (CRITICAL!)
export PATH=/home/fuhwu/slurm/bin:$PATH

# 3. Verify
sinfo
# Expected: 2 nodes (node6, node7) idle

# 4. Test
srun -N 2 hostname
```

---

## What This Sets Up

```
Physical Setup:
  1 machine (moirai-h200)
  ├── GPU 6 (H200)
  └── GPU 7 (H200)

Virtual Slurm Cluster:
  node6 (port 17016) → GPU 6
  node7 (port 17017) → GPU 7
```

---

## Step-by-Step (3 minutes)

### 1. Run Setup Script

```bash
cd /home/fuhwu/workspace/distributedai/code/chapter8
bash slurm_setup.sh
```

**What it does:**
- ✅ Creates all directories
- ✅ Generates gres.conf (GPU6, GPU7)
- ✅ Generates cgroup.conf
- ✅ Creates slurm.conf (node6, node7)
- ✅ Stops old daemons
- ✅ Starts slurmctld
- ✅ Starts 2 slurmd daemons

### 2. Fix PATH (CRITICAL!)

```bash
# Temporary (current session)
export PATH=/home/fuhwu/slurm/bin:$PATH

# Permanent (add to ~/.bashrc)
echo 'export PATH=/home/fuhwu/slurm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify you're using the right version
which sinfo
# Should show: /home/fuhwu/slurm/bin/sinfo

sinfo --version
# Should show: slurm 25.11.0
```

### 3. Verify Cluster

```bash
# Check nodes
sinfo
# Expected output:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      2   idle node6,node7

# Detailed node info
scontrol show nodes
```

### 4. Test Jobs

```bash
# Test 1: Single node
srun -N 1 -w node6 hostname
# Output: moirai-h200

# Test 2: Both nodes
srun -N 2 hostname
# Output: moirai-h200 (twice)

# Test 3: GPU on node6
srun -N 1 -w node6 --gres=gpu:1 nvidia-smi -L
# Expected: GPU 6: NVIDIA H200 (or similar)

# Test 4: GPU on node7
srun -N 1 -w node7 --gres=gpu:1 nvidia-smi -L
# Expected: GPU 7: NVIDIA H200 (or similar)

# Test 5: Both GPUs
srun -N 2 --gres=gpu:1 nvidia-smi -L
# Expected: Both GPU 6 and GPU 7
```

---

## Common Issues

### Issue 1: "Invalid Protocol Version"

**Cause:** Using system Slurm (v21.08.5) instead of compiled (v25.11.0)

**Fix:**
```bash
export PATH=/home/fuhwu/slurm/bin:$PATH
which sinfo  # Should show /home/fuhwu/slurm/bin/sinfo
```

### Issue 2: Nodes in DOWN state

**Fix:**
```bash
scontrol update NodeName=node6,node7 State=RESUME
```

### Issue 3: No nodes showing

**Check:**
```bash
# Controller running?
pgrep slurmctld

# Slurmd daemons running?
ps aux | grep slurmd | grep -v grep
# Should show 2 processes

# Logs
tail /home/fuhwu/slurm/var/log/slurmctld.log
tail /home/fuhwu/slurm/var/log/slurmd.node6.log
```

### Issue 4: Can't allocate GPU

**Check GRES:**
```bash
scontrol show node node6 | grep Gres
# Should show: Gres=gpu:1

# Verify gres.conf exists
cat /home/fuhwu/slurm/etc/gres.conf
```

---

## Running Distributed PyTorch Jobs

### Example: 2-GPU Training

```bash
# Create test script
cat > test_distributed.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test-2gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=test_%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun python -c "
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f'Rank {rank}/{world_size} - GPU: {torch.cuda.get_device_name(0)}')
dist.destroy_process_group()
"
EOF

# Submit
sbatch test_distributed.sh

# Check status
squeue

# View output
cat test_*.out
```

---

## Management Commands

```bash
# Check cluster
sinfo
sinfo -N -l

# Check queue
squeue
squeue -u $USER

# Check specific node
scontrol show node node6

# Restart cluster
pkill slurmctld slurmd
/home/fuhwu/slurm/sbin/slurmctld
/home/fuhwu/slurm/sbin/slurmd -N node6
/home/fuhwu/slurm/sbin/slurmd -N node7

# View logs
tail -f /home/fuhwu/slurm/var/log/slurmctld.log
tail -f /home/fuhwu/slurm/var/log/slurmd.node6.log
```

---

## Summary

**Before running the script, you have:**
- ❌ 0/2 slurmd daemons running
- ❌ Missing config files (gres.conf, cgroup.conf)
- ❌ Wrong PATH (using old system Slurm)

**After running the script:**
- ✅ 2/2 slurmd daemons running (node6, node7)
- ✅ All config files created
- ✅ Cluster ready for jobs
- ⚠️  Still need to fix PATH manually

**Total time:** ~1 minute setup + ~2 minutes testing = **3 minutes total**

Ready to run? Execute:
```bash
bash /home/fuhwu/workspace/distributedai/code/chapter8/slurm_setup.sh
```

Then fix PATH and test! 🚀

