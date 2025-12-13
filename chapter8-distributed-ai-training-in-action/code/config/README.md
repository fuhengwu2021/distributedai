# Slurm Configuration Files

This directory contains the key configuration files for the Slurm single-node multi-GPU setup.

## Files

### `slurm.conf`
Main Slurm configuration file defining:
- Cluster name: `distributed-ai`
- 2 virtual nodes: `node6` and `node7` (for GPU6 and GPU7)
- Process tracking: `proctrack/pgid` (works with virtual nodes)
- MPI: `pmi2` (PMI2 plugin)
- Partition: `gpu` with 2 nodes

**Key settings:**
- `ProctrackType=proctrack/pgid` - Process tracking that works with virtual nodes
- `TaskPlugin=task/affinity` - CPU affinity without cgroup
- `MpiDefault=pmi2` - MPI plugin (PMI2, not PMIx - requires recompile for PMIx)
- `SlurmUser=fuhwu` and `SlurmdUser=fuhwu` - Non-root execution
- Each node: 112 CPUs, 240GB RAM, 1 GPU

### `gres.conf`
Generic Resource (GRES) configuration mapping GPUs to nodes:
- `node6` → `/dev/nvidia6` (GPU6)
- `node7` → `/dev/nvidia7` (GPU7)

### `cgroup.conf`
Cgroup configuration - **DISABLED** to avoid systemd scope permission issues:
- `CgroupPlugin=disabled` - Explicitly disables cgroup plugin
- Slurm works perfectly without cgroup for job scheduling and GPU allocation
- Cgroup is only needed for resource limit enforcement (optional)

**Note:** Slurm was compiled with `--enable-cgroupv2`, but the plugin is disabled in config to avoid systemd scope permission issues. To enable cgroup v2:
1. Enable systemd lingering: `sudo loginctl enable-linger $(whoami)`
2. Change `CgroupPlugin=disabled` to `CgroupPlugin=cgroup/v2`
3. Add constraint lines (ConstrainCores, ConstrainDevices, etc.)
4. Restart slurmd daemons

### `slurm_setup.sh`
Automated setup script that:
1. Creates directory structure
2. Generates configuration files
3. Starts slurmctld (controller)
4. Starts slurmd daemons for node6 and node7

## Setup Details

**Hardware:**
- Single physical node: `moirai-h200`
- 2 H200 GPUs: GPU6 and GPU7
- 224 CPUs total (2 sockets × 56 cores × 2 threads)
- Divided into 2 virtual nodes: 112 CPUs each

**Ports:**
- slurmctld: 6817
- slurmd (node6): 17016
- slurmd (node7): 17017

**Installation:**
- Slurm compiled from source: `/home/fuhwu/slurm`
- Version: 25.11.0
- Compiled with: `--enable-multiple-slurmd --enable-cgroupv2`

## Usage

After setup, use Slurm commands:
```bash
export PATH=/home/fuhwu/slurm/bin:$PATH

# Check cluster status
sinfo

# Run job on 1 node
srun -N 1 hostname

# Run job on 2 nodes
srun -N 2 hostname

# Run job with GPU
srun -N 1 --gres=gpu:1 nvidia-smi -L
```

## Notes

- **No cgroup constraints**: Jobs can exceed CPU/memory limits (for development/testing)
- **PMI2 MPI**: Works with most MPI implementations (OpenMPI, MPICH, etc.)
- **Virtual nodes**: Both node6 and node7 share the same physical hardware
- **GPU isolation**: Each virtual node gets exclusive access to one GPU via GRES
