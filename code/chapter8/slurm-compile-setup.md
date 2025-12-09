# Slurm Single-Node Setup Guide

This guide covers setting up Slurm 25.11.0 on a single node with multiple virtual nodes for testing distributed AI workloads.

## Prerequisites

You have already compiled Slurm from source with:
```bash
./configure \
  --prefix=/home/fuhwu/slurm \
  --sysconfdir=/home/fuhwu/slurm/etc \
  --with-munge \
  --enable-multiple-slurmd

make -j$(nproc)
make install
```

**Key flags:**
- `--enable-multiple-slurmd`: Allows running multiple slurmd daemons on one node
- `--with-munge`: Enables authentication via Munge

## Installation Verification

```bash
# Check Slurm version
/home/fuhwu/slurm/bin/scontrol --version
# Output: slurm 25.11.0

# Check installed binaries
ls /home/fuhwu/slurm/bin/
# Should show: srun, sbatch, squeue, sinfo, etc.

ls /home/fuhwu/slurm/sbin/
# Should show: slurmctld, slurmd, slurmdbd

# Check Munge is running
systemctl status munge
# Should show: active (running)
```

## Directory Structure

```bash
/home/fuhwu/slurm/
├── bin/          # User commands (srun, sbatch, squeue, etc.)
├── sbin/         # Daemons (slurmctld, slurmd)
├── etc/          # Configuration files
├── lib/          # Libraries
├── include/      # Header files
├── share/        # Documentation and man pages
└── var/          # Runtime data (logs, state, spool)
    ├── log/
    ├── state/
    └── spool/
```

## Step 1: Create Required Directories

```bash
# Create directories for Slurm runtime data
mkdir -p /home/fuhwu/slurm/var/log
mkdir -p /home/fuhwu/slurm/var/state
mkdir -p /home/fuhwu/slurm/var/spool

# For multiple slurmd support, create per-node directories
for i in {0..7}; do
    mkdir -p /home/fuhwu/slurm/var/spool/node${i}
    mkdir -p /home/fuhwu/slurm/var/log/node${i}
done
```

## Step 2: Create Slurm Configuration

Create `/home/fuhwu/slurm/etc/slurm.conf`:

```bash
cat > /home/fuhwu/slurm/etc/slurm.conf << 'EOF'
# slurm.conf - Slurm configuration file for single-node multi-GPU setup
# Generated for Slurm 25.11.0

# CLUSTER CONFIGURATION
ClusterName=distributed-ai
SlurmctldHost=moirai-h200

# AUTHENTICATION
AuthType=auth/munge
CryptoType=crypto/munge

# PATHS AND FILES
SlurmctldPidFile=/home/fuhwu/slurm/var/state/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/home/fuhwu/slurm/var/spool/slurmd.%n.pid
SlurmdPort=6818
SlurmdSpoolDir=/home/fuhwu/slurm/var/spool/%n
StateSaveLocation=/home/fuhwu/slurm/var/state

# LOGGING
SlurmctldDebug=info
SlurmctldLogFile=/home/fuhwu/slurm/var/log/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/home/fuhwu/slurm/var/log/slurmd.%n.log

# PROCESS TRACKING
ProctrackType=proctrack/cgroup
TaskPlugin=task/affinity,task/cgroup

# SCHEDULING
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# RESOURCE LIMITS
DefMemPerCPU=4000
MaxMemPerCPU=0

# TIMEOUTS
SlurmctldTimeout=120
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0

# COMPUTE NODES
# Multiple virtual nodes on single physical host
# Each node gets 1 GPU (8 H200 GPUs total)
NodeName=node[0-7] NodeHostname=moirai-h200 Port=1701[0-7] \
    CPUs=28 Boards=1 SocketsPerBoard=1 CoresPerSocket=28 ThreadsPerCore=1 \
    RealMemory=240000 Gres=gpu:1 State=UNKNOWN

# PARTITIONS
PartitionName=gpu Nodes=node[0-7] Default=YES MaxTime=INFINITE State=UP \
    OverSubscribe=NO

# GRES CONFIGURATION
GresTypes=gpu
EOF
```

### Key Configuration Points

**Multiple slurmd support:**
- `%n` in paths: Replaced with NodeName (node0, node1, etc.)
- `Port=1701[0-7]`: Each slurmd uses different port (17010-17017)
- `NodeHostname=moirai-h200`: All nodes on same physical host

**Resource allocation:**
- 8 virtual nodes (node0-node7)
- Each node: 28 CPUs, 240GB RAM, 1 GPU
- Adjust based on your actual H200 specs

## Step 3: Create GRES Configuration

Create `/home/fuhwu/slurm/etc/gres.conf`:

```bash
cat > /home/fuhwu/slurm/etc/gres.conf << 'EOF'
# gres.conf - Generic Resource (GRES) configuration
# Maps GPUs to virtual nodes

# Node 0 - GPU 0
NodeName=node0 Name=gpu File=/dev/nvidia0

# Node 1 - GPU 1
NodeName=node1 Name=gpu File=/dev/nvidia1

# Node 2 - GPU 2
NodeName=node2 Name=gpu File=/dev/nvidia2

# Node 3 - GPU 3
NodeName=node3 Name=gpu File=/dev/nvidia3

# Node 4 - GPU 4
NodeName=node4 Name=gpu File=/dev/nvidia4

# Node 5 - GPU 5
NodeName=node5 Name=gpu File=/dev/nvidia5

# Node 6 - GPU 6
NodeName=node6 Name=gpu File=/dev/nvidia6

# Node 7 - GPU 7
NodeName=node7 Name=gpu File=/dev/nvidia7
EOF
```

## Step 4: Create Cgroup Configuration

Create `/home/fuhwu/slurm/etc/cgroup.conf`:

```bash
cat > /home/fuhwu/slurm/etc/cgroup.conf << 'EOF'
# cgroup.conf - Cgroup configuration for resource isolation

CgroupAutomount=yes
ConstrainCores=yes
ConstrainDevices=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
EOF
```

## Step 5: Set Up Environment

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Add to ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# Slurm environment
export SLURM_HOME=/home/fuhwu/slurm
export PATH=$SLURM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SLURM_HOME/lib:$LD_LIBRARY_PATH
export MANPATH=$SLURM_HOME/share/man:$MANPATH
EOF

# Reload
source ~/.bashrc
```

## Step 6: Start Slurm Daemons

### Start the Controller (slurmctld)

```bash
# Start controller daemon
/home/fuhwu/slurm/sbin/slurmctld -D &

# Check it's running
ps aux | grep slurmctld

# Check logs
tail -f /home/fuhwu/slurm/var/log/slurmctld.log
```

### Start Multiple slurmd Daemons

Start one slurmd for each virtual node:

```bash
# Start all 8 slurmd daemons (one per GPU)
for i in {0..7}; do
    /home/fuhwu/slurm/sbin/slurmd -N node${i} -D &
    echo "Started slurmd for node${i}"
    sleep 1
done

# Verify all are running
ps aux | grep slurmd | grep -v grep

# Check logs
tail -f /home/fuhwu/slurm/var/log/slurmd.node0.log
```

### Alternative: Start with systemd (Production Setup)

Create systemd service files for automatic startup (optional):

```bash
# Create controller service
sudo tee /etc/systemd/system/slurmctld.service << 'EOF'
[Unit]
Description=Slurm controller daemon
After=network.target munge.service
Requires=munge.service

[Service]
Type=forking
User=fuhwu
ExecStart=/home/fuhwu/slurm/sbin/slurmctld
ExecReload=/bin/kill -HUP $MAINPID
PIDFile=/home/fuhwu/slurm/var/state/slurmctld.pid
LimitNOFILE=65536
LimitMEMLOCK=infinity
LimitSTACK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Create slurmd template service
sudo tee /etc/systemd/system/slurmd@.service << 'EOF'
[Unit]
Description=Slurm node daemon for %i
After=network.target munge.service
Requires=munge.service

[Service]
Type=forking
User=fuhwu
ExecStart=/home/fuhwu/slurm/sbin/slurmd -N %i
PIDFile=/home/fuhwu/slurm/var/spool/%i/slurmd.pid
LimitNOFILE=65536
LimitMEMLOCK=infinity
LimitSTACK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable and start services
sudo systemctl enable slurmctld
sudo systemctl start slurmctld

# Start all slurmd instances
for i in {0..7}; do
    sudo systemctl enable slurmd@node${i}
    sudo systemctl start slurmd@node${i}
done
```

## Step 7: Verify Cluster Status

```bash
# Check node status
sinfo

# Expected output:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      8   idle node[0-7]

# Check detailed node info
scontrol show nodes

# Check partition info
scontrol show partition

# Check if all nodes are responding
sinfo -N -l
```

## Step 8: Test Job Submission

### Test 1: Simple Hostname Job

```bash
# Submit job to each node
for i in {0..7}; do
    srun -N 1 -w node${i} hostname
done

# Should print: moirai-h200 (8 times)
```

### Test 2: GPU Check

```bash
# Check GPU on each node
for i in {0..7}; do
    echo "=== Node ${i} ==="
    srun -N 1 -w node${i} --gres=gpu:1 nvidia-smi -L
done

# Should show different GPUs (GPU 0, GPU 1, ..., GPU 7)
```

### Test 3: Multi-Node Job

```bash
# Run across all 8 nodes
srun -N 8 hostname

# Run with 2 nodes
srun -N 2 hostname

# Run with GPU allocation
srun -N 4 --gres=gpu:1 nvidia-smi -L
```

### Test 4: Batch Job

```bash
# Create test script
cat > test_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --output=test_%j.out

echo "Running on nodes: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
hostname
nvidia-smi -L
EOF

# Submit job
sbatch test_job.sh

# Check queue
squeue

# Check output
cat test_*.out
```

## Step 9: Configure for PyTorch Distributed Training

### Test PyTorch Multi-Node

Create `test_pytorch.sh`:

```bash
cat > test_pytorch.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=pytorch-test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=pytorch_%j.out

# Get node list
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node ($head_node_ip)"
echo "All nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"

# Run PyTorch distributed test
srun python -c "
import torch
import torch.distributed as dist
import os

# Initialize process group
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

print(f'Rank {rank}/{world_size} on {os.uname().nodename}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  CUDA available: {torch.cuda.is_available()}')

# Simple all-reduce test
tensor = torch.ones(1).cuda() * rank
dist.all_reduce(tensor)
print(f'Rank {rank}: All-reduce result = {tensor.item()}')

dist.destroy_process_group()
"
EOF

# Submit
sbatch test_pytorch.sh
```

## Troubleshooting

### Issue 1: Nodes in DOWN state

```bash
# Check node status
sinfo

# If nodes are DOWN:
scontrol update NodeName=node[0-7] State=RESUME

# Check logs
tail -f /home/fuhwu/slurm/var/log/slurmd.node0.log
```

### Issue 2: Munge authentication errors

```bash
# Check Munge is running
systemctl status munge

# Test Munge
munge -n | unmunge

# Restart if needed
sudo systemctl restart munge
```

### Issue 3: Slurmd won't start

```bash
# Check if port is already in use
netstat -tuln | grep 17010

# Kill old processes
pkill -9 slurmd

# Check logs
tail -f /home/fuhwu/slurm/var/log/slurmd.node0.log
```

### Issue 4: GPU not accessible

```bash
# Check GRES configuration
scontrol show node node0 | grep Gres

# Verify GPU device exists
ls -la /dev/nvidia*

# Check cgroup configuration
cat /home/fuhwu/slurm/etc/cgroup.conf
```

## Management Commands

### Start/Stop Daemons

```bash
# Stop all Slurm daemons
pkill slurmctld
pkill slurmd

# Start controller
/home/fuhwu/slurm/sbin/slurmctld -D &

# Start compute daemons
for i in {0..7}; do
    /home/fuhwu/slurm/sbin/slurmd -N node${i} -D &
done
```

### Useful Commands

```bash
# Check cluster status
sinfo
sinfo -N -l

# Check node details
scontrol show nodes
scontrol show node node0

# Check partition details
scontrol show partition

# Check running jobs
squeue
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# Drain a node (maintenance)
scontrol update NodeName=node0 State=DRAIN Reason="maintenance"

# Resume a node
scontrol update NodeName=node0 State=RESUME
```

## Configuration for Distributed AI Training

### Optimize for Multi-GPU Training

Add to `/home/fuhwu/slurm/etc/slurm.conf`:

```bash
# MPI/PMI support for distributed training
MpiDefault=pmix
PrologFlags=X11

# Fair-share scheduling
PriorityType=priority/multifactor
PriorityWeightAge=1000
PriorityWeightFairshare=10000
PriorityWeightJobSize=1000
PriorityWeightPartition=1000
PriorityWeightQOS=0

# Job accounting
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30

# Preemption (optional)
PreemptType=preempt/partition_prio
PreemptMode=REQUEUE
```

### Create GPU Partition

Add to `/home/fuhwu/slurm/etc/slurm.conf`:

```bash
# High-priority GPU partition for urgent jobs
PartitionName=gpu-high Nodes=node[0-3] Priority=100 MaxTime=12:00:00 State=UP

# Standard GPU partition
PartitionName=gpu Nodes=node[0-7] Default=YES MaxTime=INFINITE State=UP

# Development partition (1 node only)
PartitionName=dev Nodes=node7 MaxTime=1:00:00 State=UP
```

## Advanced: Simulating Multi-Node Cluster

For testing multi-node behavior on a single machine:

### Configuration Tips

```bash
# In slurm.conf, create "nodes" that represent different network locations
# Use different hostnames (via /etc/hosts)

# Add to /etc/hosts
127.0.0.1 node0-ib node1-ib node2-ib node3-ib
127.0.0.1 node4-ib node5-ib node6-ib node7-ib

# Update slurm.conf NodeHostname
NodeName=node0 NodeHostname=node0-ib Port=17010 ...
NodeName=node1 NodeHostname=node1-ib Port=17011 ...
```

## Example: Distributed Training Job

```bash
cat > train_fsdp.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=fsdp_%j.out
#SBATCH --error=fsdp_%j.err

# Load environment
source ~/.bashrc
conda activate wan22

# Get master node info
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE, Rank: $RANK"

# Run training
srun python -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_fsdp.py
EOF

# Submit
sbatch train_fsdp.sh
```

## Monitoring and Debugging

### Real-time Monitoring

```bash
# Watch cluster status
watch -n 1 sinfo

# Watch queue
watch -n 1 squeue

# Watch specific job
watch -n 1 scontrol show job <job_id>

# Watch GPU usage across all nodes
watch -n 1 'for i in {0..7}; do echo "=== Node $i ==="; srun -N 1 -w node$i nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader; done'
```

### Log Files

```bash
# Controller logs
tail -f /home/fuhwu/slurm/var/log/slurmctld.log

# Node logs
tail -f /home/fuhwu/slurm/var/log/slurmd.node0.log

# All node logs
tail -f /home/fuhwu/slurm/var/log/slurmd.node*.log
```

### Debugging Failed Jobs

```bash
# Get job info
scontrol show job <job_id>

# Check job output
cat slurm-<job_id>.out

# Check job error
cat slurm-<job_id>.err

# Check job accounting
sacct -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

## Best Practices

### 1. Resource Allocation

```bash
# Always specify resources explicitly
srun -N 2 --gres=gpu:1 --mem=100G --cpus-per-task=28 your_command

# Use --exclusive for full node
srun -N 1 --exclusive your_command
```

### 2. Job Arrays for Hyperparameter Tuning

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Each array task gets different hyperparameters
LR=$(echo "0.001 0.0001 0.00001" | cut -d' ' -f$((SLURM_ARRAY_TASK_ID % 3 + 1)))
python train.py --lr $LR
```

### 3. Interactive Development

```bash
# Allocate interactive session
salloc -N 2 --gres=gpu:1 --time=1:00:00

# Once allocated, run commands
srun hostname
srun nvidia-smi

# Release when done
exit
```

## Quick Reference

### Common Commands

```bash
# Submit batch job
sbatch job_script.sh

# Run interactive job
srun -N 2 --gres=gpu:1 python train.py

# Allocate resources
salloc -N 4 --gres=gpu:1

# Check queue
squeue
squeue -u $USER

# Cancel job
scancel <job_id>
scancel -u $USER  # Cancel all your jobs

# Check cluster
sinfo
sinfo -N  # Per-node view

# Node details
scontrol show node node0
```

### Environment Variables in Jobs

```bash
SLURM_JOB_ID          # Job ID
SLURM_JOB_NODELIST    # Allocated nodes (e.g., node[0-3])
SLURM_JOB_NUM_NODES   # Number of nodes
SLURM_NTASKS          # Total tasks
SLURM_PROCID          # Process rank (0 to NTASKS-1)
SLURM_LOCALID         # Local rank on node
SLURM_NODEID          # Node index (0 to NUM_NODES-1)
```

## Maintenance

### Reconfigure Slurm

```bash
# After editing slurm.conf
scontrol reconfigure

# Or restart daemons
pkill slurmctld slurmd
# Then start again
```

### Clean Up

```bash
# Stop all Slurm processes
pkill -9 slurmctld
pkill -9 slurmd

# Clean state files
rm -rf /home/fuhwu/slurm/var/state/*
rm -rf /home/fuhwu/slurm/var/spool/node*/*

# Clean logs (optional)
rm -f /home/fuhwu/slurm/var/log/*.log
```

## References

- [Slurm Documentation](https://slurm.schedmd.com/)
- [Multiple slurmd Support](https://slurm.schedmd.com/programmer_guide.html#multiple_slurmd_support)
- [Slurm Configuration Tool](https://slurm.schedmd.com/configurator.html)
- [Slurm Quick Start Guide](https://slurm.schedmd.com/quickstart_admin.html)

---

**Version:** Slurm 25.11.0  
**Installation:** From source with `--enable-multiple-slurmd`  
**Purpose:** Single-node multi-GPU testing for distributed AI training
