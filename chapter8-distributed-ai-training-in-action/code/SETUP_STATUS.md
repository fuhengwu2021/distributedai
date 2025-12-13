# Slurm Setup Status Report

## Current Status

### ✅ What You've Completed

1. **Compiled Slurm from source**
   - Version: 25.11.0 (latest!)
   - Location: `/home/fuhwu/slurm/`
   - Flags: `--enable-multiple-slurmd`, `--with-munge`
   - Status: ✅ Successfully compiled and installed

2. **Created basic slurm.conf**
   - Location: `/home/fuhwu/slurm/etc/slurm.conf`
   - Nodes configured: 2 (node1, node2)
   - Status: ✅ Exists, but needs update for 8 nodes

3. **Started slurmctld (controller)**
   - Process: Running (PID 2859839)
   - Version: 25.11.0
   - Status: ✅ Running with verbose logging

4. **Munge authentication**
   - Service: Active and running
   - Status: ✅ Working correctly

---

### ❌ What's Missing

1. **Directory structure incomplete**
   - ❌ `/home/fuhwu/slurm/var/` not created
   - ❌ Per-node spool directories missing
   - Need: `var/{log,state,spool}`

2. **gres.conf not created**
   - ❌ No GPU-to-node mapping
   - Needed for: GPU resource allocation
   - Without it: Slurm can't assign GPUs to jobs

3. **cgroup.conf not created**
   - ❌ No resource isolation config
   - Needed for: Memory/CPU constraints
   - Impact: Jobs might interfere with each other

4. **No slurmd daemons running**
   - ❌ No compute node daemons
   - Need: 8 slurmd instances (one per virtual node)
   - Without them: Can't run any jobs!

5. **slurm.conf needs expansion**
   - Current: Only 2 nodes (node1, node2)
   - Need: 8 nodes (node0-node7) for all GPUs
   - Missing: Port configuration for multiple slurmd

6. **PATH configuration issue**
   - Running: System Slurm 21.08.5 (`/usr/bin/sinfo`)
   - Should run: Compiled Slurm 25.11.0 (`/home/fuhwu/slurm/bin/sinfo`)
   - Issue: **Version mismatch causing protocol errors**

---

## Critical Issue: Version Mismatch 🚨

```
Error: Invalid Protocol Version 11264
Cause: sinfo (client, v21.08.5) ← → slurmctld (server, v25.11.0)
```

**Your slurmctld is running v25.11.0**, but when you type `sinfo`, it runs the **system version 21.08.5**, which is incompatible!

---

## Quick Fix: Run This Script

I've created an automated setup script:

```bash
# Run the setup script
bash /home/fuhwu/workspace/distributedai/code/chapter8/slurm_setup.sh
```

This will:
1. ✅ Create all required directories
2. ✅ Generate gres.conf for 8 GPUs
3. ✅ Generate cgroup.conf
4. ✅ Update slurm.conf to 8 nodes
5. ✅ Restart slurmctld
6. ✅ Start all 8 slurmd daemons

---

## Manual Step-by-Step Fix

If you prefer manual setup:

### 1. Create Directories

```bash
mkdir -p /home/fuhwu/slurm/var/{log,state,spool}
# Only need node6 and node7 for 2-GPU setup
for i in 6 7; do
    mkdir -p /home/fuhwu/slurm/var/spool/node${i}
done
```

### 2. Create gres.conf

```bash
cat > /home/fuhwu/slurm/etc/gres.conf << 'EOF'
# GPU mapping for 2-node setup (GPU6 and GPU7)
NodeName=node6 Name=gpu File=/dev/nvidia6
NodeName=node7 Name=gpu File=/dev/nvidia7
EOF
```

### 3. Create cgroup.conf

```bash
cat > /home/fuhwu/slurm/etc/cgroup.conf << 'EOF'
CgroupAutomount=yes
ConstrainCores=yes
ConstrainDevices=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
EOF
```

### 4. Update slurm.conf for 2 nodes (GPU6 and GPU7)

Replace `/home/fuhwu/slurm/etc/slurm.conf` with:

```bash
cat > /home/fuhwu/slurm/etc/slurm.conf << 'EOF'
# Slurm configuration for 2-GPU setup (GPU6 and GPU7)
ClusterName=distributed-ai
SlurmctldHost=moirai-h200

# Authentication
AuthType=auth/munge
CryptoType=crypto/munge

# Paths with %n for per-node substitution
SlurmctldPidFile=/home/fuhwu/slurm/var/state/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/home/fuhwu/slurm/var/spool/%n/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/home/fuhwu/slurm/var/spool/%n
StateSaveLocation=/home/fuhwu/slurm/var/state

# Logging
SlurmctldDebug=info
SlurmctldLogFile=/home/fuhwu/slurm/var/log/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/home/fuhwu/slurm/var/log/slurmd.%n.log

# Process tracking
ProctrackType=proctrack/cgroup
TaskPlugin=task/affinity,task/cgroup

# Scheduling
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# Resource limits
DefMemPerCPU=4000
MaxMemPerCPU=0

# Timeouts
SlurmctldTimeout=120
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0

# User
SlurmUser=fuhwu

# MPI for distributed training
MpiDefault=pmix

# GRES
GresTypes=gpu

# 2 virtual nodes for GPU6 and GPU7 with different ports
NodeName=node6 NodeHostname=moirai-h200 Port=17016 \
    CPUs=28 Boards=1 SocketsPerBoard=1 CoresPerSocket=28 ThreadsPerCore=1 \
    RealMemory=240000 Gres=gpu:1 State=UNKNOWN

NodeName=node7 NodeHostname=moirai-h200 Port=17017 \
    CPUs=28 Boards=1 SocketsPerBoard=1 CoresPerSocket=28 ThreadsPerCore=1 \
    RealMemory=240000 Gres=gpu:1 State=UNKNOWN

# Partition with 2 nodes
PartitionName=gpu Nodes=node6,node7 Default=YES MaxTime=INFINITE State=UP OverSubscribe=NO
EOF
```

### 5. Fix PATH (Critical!)

```bash
# Add to ~/.bashrc
echo 'export PATH=/home/fuhwu/slurm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/home/fuhwu/slurm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Apply immediately
export PATH=/home/fuhwu/slurm/bin:$PATH
export LD_LIBRARY_PATH=/home/fuhwu/slurm/lib:$LD_LIBRARY_PATH

# Verify
which sinfo
# Should show: /home/fuhwu/slurm/bin/sinfo
```

### 6. Restart slurmctld

```bash
# Stop old controller
pkill slurmctld
sleep 2

# Start new controller
/home/fuhwu/slurm/sbin/slurmctld
sleep 2

# Verify
pgrep -a slurmctld
```

### 7. Start slurmd daemons for GPU6 and GPU7

```bash
# Start 2 slurmd instances
for i in 6 7; do
    /home/fuhwu/slurm/sbin/slurmd -N node${i}
    echo "Started slurmd for node${i} (GPU${i})"
    sleep 1
done

# Verify
ps aux | grep slurmd | grep -v grep
# Should show 2 slurmd processes
```

### 8. Verify cluster

```bash
# Use YOUR compiled version (not system version!)
/home/fuhwu/slurm/bin/sinfo

# Expected output:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      8   idle node[0-7]
```

---

## Summary Checklist

### ✅ Already Done
- [x] Compiled Slurm 25.11.0 with `--enable-multiple-slurmd`
- [x] Installed to `/home/fuhwu/slurm/`
- [x] Created basic `slurm.conf` (needs update)
- [x] Munge authentication running
- [x] Started slurmctld

### ❌ Still Need to Do (2-GPU Setup)
- [ ] Create directory structure (`var/log`, `var/state`, `var/spool`)
- [ ] Create `gres.conf` (GPU6, GPU7 mapping)
- [ ] Create `cgroup.conf` (resource isolation)
- [ ] Update `slurm.conf` to use node6, node7 with ports 17016, 17017
- [ ] Fix PATH to use compiled Slurm (critical!)
- [ ] Start 2 slurmd daemons (node6, node7)
- [ ] Verify with `sinfo` and test jobs

---

## Recommended Next Steps

### Option 1: Automated (Fastest)

```bash
# Run the setup script
bash /home/fuhwu/workspace/distributedai/code/chapter8/slurm_setup.sh

# Fix PATH
export PATH=/home/fuhwu/slurm/bin:$PATH

# Test
sinfo
```

### Option 2: Manual (More control)

Follow steps 1-8 above manually.

---

## Expected Final State

```
Cluster: distributed-ai
Nodes:   node6, node7 (2 virtual nodes)
GPUs:    GPU6, GPU7 (2 H200 GPUs, 1 per node)
Ports:   17016, 17017 (one per slurmd)
Status:  Both nodes idle, ready for jobs

Processes running:
- 1 slurmctld (controller)
- 2 slurmd (node6, node7)

Configuration files:
- slurm.conf    ✅ (2 nodes with ports)
- gres.conf     ✅ (GPU6, GPU7 mapping)
- cgroup.conf   ✅ (resource limits)
```

---

## Verification Commands

```bash
# Use compiled version explicitly (not system version!)
/home/fuhwu/slurm/bin/sinfo
# Expected: 2 nodes (node6, node7) in idle state

# Check both nodes
/home/fuhwu/slurm/bin/scontrol show nodes

# Test on node6
/home/fuhwu/slurm/bin/srun -N 1 -w node6 hostname

# Test on node7
/home/fuhwu/slurm/bin/srun -N 1 -w node7 hostname

# Test both nodes together
/home/fuhwu/slurm/bin/srun -N 2 hostname

# Test GPU on node6
/home/fuhwu/slurm/bin/srun -N 1 -w node6 --gres=gpu:1 nvidia-smi -L
# Expected: GPU 6

# Test GPU on node7
/home/fuhwu/slurm/bin/srun -N 1 -w node7 --gres=gpu:1 nvidia-smi -L
# Expected: GPU 7

# Test 2-GPU job
/home/fuhwu/slurm/bin/srun -N 2 --gres=gpu:1 nvidia-smi -L
# Expected: GPU 6 and GPU 7
```

Ready to complete the setup? Run the script or follow the manual steps! 🚀
