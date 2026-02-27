# Chapter 8: Running Distributed Training with SLURM {-}

*Managing GPU resources and coordinating multi-node jobs with SLURM*

> In a room full of top software designers, if two agree on the same thing, that’s a majority.
- Bill Curtis

**Code Summary**

- `sbatch`: SLURM command to submit batch jobs
- `srun`: SLURM command to run interactive jobs
- `squeue`: SLURM command to view job queue
- `scancel`: SLURM command to cancel jobs
- `sinfo`: SLURM command to view cluster information
- `sacct`: SLURM command to view job accounting information
- `scontrol`: SLURM command for cluster control and configuration
- `torchrun`: PyTorch launcher compatible with SLURM
- `SLURM_PROCID`: SLURM environment variable for process ID
- `SLURM_NTASKS`: SLURM environment variable for number of tasks

## Introduction to Clusters for HPC and AI Training

The previous chapters covered the theory and implementation of distributed training—DDP for gradient synchronization, FSDP for memory efficiency, DeepSpeed for ZeRO optimization, and Megatron for model parallelism. But understanding these frameworks is only half the challenge. The other half is actually running them on real hardware: allocating GPUs across nodes, coordinating processes, managing job queues, and handling the inevitable failures that occur at scale.

Modern AI training happens on clusters—collections of interconnected machines that pool their compute resources. A typical GPU cluster consists of compute nodes (each containing multiple GPUs, CPUs, and memory), a high-speed interconnect (InfiniBand or high-bandwidth Ethernet) linking the nodes, shared storage accessible from all nodes, and a head node that manages job submission and scheduling. The cluster might have dozens to thousands of nodes, representing millions of dollars in hardware that must be shared efficiently among many users and projects.

This shared nature creates a fundamental challenge: how do you allocate resources fairly, ensure jobs don't interfere with each other, and maximize utilization of expensive hardware? In the early days of computing, users would sign up for time slots on a shared machine. Modern clusters use job schedulers—software that accepts job requests, queues them based on priority and resource availability, allocates resources when they become available, monitors running jobs, and cleans up when jobs complete or fail.

Several job schedulers exist in the HPC ecosystem. PBS (Portable Batch System) and its derivatives (Torque, PBS Pro) were dominant in traditional HPC.[^pbs] LSF (Load Sharing Facility) is popular in enterprise environments.[^lsf] HTCondor excels at high-throughput computing workloads where many independent jobs need to be distributed across available machines.[^htcondor] Kubernetes has become the standard for cloud-native workloads.[^k8s] But for GPU clusters running AI training workloads, SLURM has emerged as the dominant choice, used by the majority of academic institutions, national labs, and increasingly by cloud providers offering HPC instances.

[^pbs]: PBS Professional is now maintained by Altair. See https://www.altair.com/pbs-professional/ for the commercial version, and https://github.com/openpbs/openpbs for the open-source OpenPBS.

[^lsf]: IBM Spectrum LSF is widely used in financial services and life sciences. See https://www.ibm.com/products/hpc-workload-management.

[^htcondor]: HTCondor is developed by the Center for High Throughput Computing at UW-Madison. See https://htcondor.org/ for the official site and https://www.cs.utexas.edu/facilities/documentation/condor for an example deployment at UT Austin.

[^k8s]: Kubernetes can be extended for HPC workloads using projects like Volcano (https://volcano.sh/) or the Kubernetes Job API with GPU scheduling plugins.

### Why SLURM for AI Training?

SLURM (Simple Linux Utility for Resource Management) started as a project at Lawrence Livermore National Laboratory in 2002 and has evolved into a sophisticated resource manager supporting clusters with millions of cores.[^slurm] Several factors make it particularly well-suited for AI training workloads.

[^slurm]: SLURM is maintained by SchedMD. Official documentation and downloads are available at https://slurm.schedmd.com/.

First, SLURM has first-class support for GPUs through its Generic Resource (GRES) system. You can request specific GPU types (`--gres=gpu:a100:4`), and SLURM ensures exclusive allocation—no other job can access your GPUs while your job runs. This is critical for training, where GPU memory fragmentation from shared access would cause out-of-memory errors.

Second, SLURM integrates seamlessly with PyTorch's distributed training. When SLURM launches your job across multiple nodes, it automatically sets environment variables that map directly to distributed training concepts: `SLURM_PROCID` becomes the global rank, `SLURM_LOCALID` becomes the local rank within a node, `SLURM_NTASKS` becomes the world size, and `SLURM_JOB_NODELIST` provides the node list needed to establish communication. PyTorch's `torchrun` launcher reads these variables and initializes the process group automatically.

Third, SLURM provides robust job management features essential for long-running training jobs: job arrays for hyperparameter sweeps, job dependencies for multi-stage pipelines, preemption and checkpointing support for handling time limits, and detailed accounting for tracking resource usage. These features become essential when training runs take days or weeks.

Fourth, SLURM scales efficiently. The same commands and scripts work whether you're running on a 4-node lab cluster or a 10,000-node supercomputer. This portability means skills transfer across institutions and cloud providers.

The integration between SLURM and PyTorch's distributed training is remarkably smooth once you understand the mapping. Your training code doesn't need to know whether it's running on a laptop with 2 GPUs or a cluster with 256—the abstraction handles the details. A job script specifies resource requirements (`--nodes=4 --gres=gpu:8`), SLURM allocates those resources and sets up the environment, and your training script initializes distributed communication using the environment variables SLURM provides.


### SLURM Architecture: A Brief Overview

Before diving into usage, it helps to understand SLURM's architecture at a high level. SLURM consists of several daemons that work together to manage the cluster:

- **slurmctld** (controller daemon): The central brain that runs on the head node. It manages the job queue, makes scheduling decisions, allocates resources, and monitors job state. In production clusters, slurmctld typically runs in a high-availability configuration with a backup controller.

- **slurmd** (compute daemon): Runs on each compute node. It receives job allocations from slurmctld, launches and monitors tasks, reports node status back to the controller, and enforces resource limits using Linux cgroups.

- **slurmdbd** (database daemon): Optional but common in production. It stores accounting data (job history, resource usage, user/project allocations) in a MySQL or MariaDB database, enabling fair-share scheduling and usage reporting.

When you submit a job with `sbatch`, the request goes to slurmctld, which queues it and eventually allocates resources based on scheduling policies (priority, fair-share, backfill). Once resources are available, slurmctld notifies the relevant slurmd daemons, which spawn your job's processes and set up the environment variables your training script reads.

SLURM's scheduling algorithms, partition configurations, QOS (Quality of Service) policies, and plugin architecture are rich topics that cluster administrators tune for their specific workloads. This chapter focuses on the **user-facing aspects**—how to submit jobs, request resources, and integrate with distributed training frameworks—rather than cluster administration. For SLURM internals and administration, the official documentation[^slurm] and the SchedMD training materials provide comprehensive coverage.

![SLURM Architecture: slurmctld, slurmd, and slurmdbd daemons.](img/slurm_architecture.png){#fig:slurm-architecture .block width=95% align=center}

Figure~\ref{fig:slurm-architecture} illustrates the overall architecture. Users interact with the head node through commands like `sbatch` (submit batch jobs), `srun` (run interactive commands), and `squeue` (query job status), all of which communicate with slurmctld. The controller daemon maintains the job queue, tracks node states, and makes scheduling decisions—when resources become available, it notifies the appropriate slurmd daemons to launch job processes. Each slurmd manages its local node: spawning tasks, enforcing resource limits via cgroups, monitoring process health, and reporting status back to the controller. The optional slurmdbd daemon persists accounting data (job history, resource consumption, user allocations) to a database, enabling fair-share scheduling policies that balance resource usage across users and projects over time.

## Setting Up SLURM for Multi-GPU Training

Most users won't need to install SLURM themselves—cluster administrators handle that. But understanding the configuration helps debug issues when jobs don't behave as expected, and setting up a local test environment is invaluable for developing and debugging distributed training scripts before submitting to a production cluster.

### Simulating a Multi-Node Cluster

For development and testing, you can simulate a multi-node cluster on a single physical machine by running multiple SLURM compute daemons (slurmd), each mapped to a different GPU. This lets you test multi-node distributed training code without access to an actual cluster.

The key insight is that SLURM's architecture separates the concept of a "node" from a physical machine. Each slurmd daemon represents one node, and by running multiple daemons on different ports, you can create virtual nodes that SLURM treats as independent machines. The configuration in `slurm.conf` defines these virtual nodes:

```bash
# Enable multiple slurmd support
# Use $HOSTNAME or $(hostname) to get the actual hostname
NodeName=node6 NodeHostname=$HOSTNAME Port=17016 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
NodeName=node7 NodeHostname=$HOSTNAME Port=17017 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
```

Each virtual node listens on a different port (17016, 17017) but shares the same hostname. The `Gres=gpu:1` declaration tells SLURM that each node has one GPU available. The corresponding `gres.conf` file maps these virtual GPUs to physical devices. In this example, we're using an 8-GPU machine and dedicating the last two GPUs (indices 6 and 7) to our virtual cluster:

```bash
NodeName=node6 Name=gpu File=/dev/nvidia6
NodeName=node7 Name=gpu File=/dev/nvidia7
```

This mapping ensures that when a job requests `--gres=gpu:1` on node6, SLURM sets `CUDA_VISIBLE_DEVICES` to expose only `/dev/nvidia6` to that job. You can adjust the GPU indices to use any available GPUs on your machine—for instance, `/dev/nvidia0` and `/dev/nvidia1` if you want to use the first two GPUs instead. On a real cluster, each physical node would have its own `gres.conf` entry mapping to its local GPUs.

![Virtual multi-node cluster on a single physical machine.](img/virtual_node_setup.png){#fig:virtual-node-setup .block width=90% align=center}

Figure~\ref{fig:virtual-node-setup} shows the virtual node setup. Two slurmd daemons (node6 and node7) run on the same physical machine but listen on different ports. Each virtual node is mapped to a specific GPU through `gres.conf`, allowing you to test multi-node distributed training code locally.

### Quick Setup and Verification

The provided setup script automates the configuration process—creating directories, generating configuration files, and starting the SLURM daemons:

```bash
cd code
bash slurm_setup.sh
```

Once the daemons are running, verify the cluster is working correctly. First, add SLURM to your PATH (replace `$SLURM_PREFIX` with your installation prefix, typically `/opt/slurm` or `$HOME/slurm`):

```bash
export PATH=$SLURM_PREFIX/bin:$PATH
```

The `sinfo` command shows the cluster's partition and node status:

```bash
sinfo
# Example output:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      2   idle node[6-7]
```

The example output shows a partition named "gpu" (the asterisk indicates it's the default) with 2 idle nodes. For more detailed node information, use `scontrol show nodes`. To verify that jobs can actually run, submit a simple test:

```bash
srun -N 1 hostname
srun -N 2 hostname
```

The first command runs `hostname` on one node; the second runs it on both nodes simultaneously. If both commands complete successfully and print the expected node names, your SLURM setup is ready for distributed training experiments.

## Submitting Distributed Training Jobs

With SLURM configured and verified, the next step is submitting actual training jobs. SLURM provides two primary submission methods: `srun` for interactive execution and `sbatch` for batch submission. Understanding when to use each—and how to structure your job scripts—is essential for productive cluster usage.

### Interactive Execution with `srun`

For quick tests and debugging, `srun` executes commands immediately on allocated resources. You specify what you need, and SLURM either runs your command right away (if resources are available) or waits until they become free. The `code/train.py` script is a self-contained example that automatically detects SLURM environment variables, initializes distributed training, and runs a simple 3-layer neural network (`SimpleModel`) on synthetic data (`SimpleDataset`) — useful for verifying your cluster setup:

```bash
# Two nodes, 1 GPU each
srun -N 2 --gres=gpu:1 --cpus-per-task=4 python code/train.py
```

The flags tell SLURM exactly what resources your job requires: `-N 2` requests two nodes, `--gres=gpu:1` requests one GPU per node, and `--cpus-per-task=4` allocates four CPU cores for data loading and preprocessing. SLURM finds nodes matching these requirements, sets up the environment, and runs your command across all allocated resources simultaneously.

Interactive execution is convenient for development, but it ties up your terminal and requires you to stay connected. For production training runs that may take hours or days, batch submission is the standard approach.

### Batch Submission with `sbatch`

Batch jobs are defined in shell scripts with special `#SBATCH` directives that specify resource requirements. You submit the script to SLURM's queue, and it runs whenever resources become available—even if you've logged off. Here's a typical batch script for distributed training (a runnable version is in `code/train_ddp.sh`):

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
srun python code/train_ddp.py
```

The `#SBATCH` directives at the top define the job's resource envelope: 2 nodes, 1 GPU each, 28 CPU cores per task, 200GB memory, and a 24-hour time limit. The `%j` in output filenames expands to the job ID, so each run gets unique log files. These directives are comments to the shell but are parsed by `sbatch` before execution.

The middle section sets up the distributed training environment. The critical piece is `MASTER_ADDR`—the hostname where rank 0 runs and where all other ranks connect to establish the process group. The `scontrol show hostnames` command converts SLURM's compressed node list format (e.g., `node[6-7]`) into individual hostnames, and `head -n 1` extracts the first one as the master. The corresponding Python training script is in `code/train_ddp.py`.

Submit the job with `sbatch`, and SLURM returns a job ID immediately:

```bash
sbatch code/train_ddp.sh
```

You can then monitor your job's progress through the queue:

```bash
squeue                    # List all jobs
squeue -u $USER          # List your jobs
scontrol show job <job_id>  # Detailed job info
```

![SLURM job state lifecycle](img/job_lifecycle.png){#fig:job-lifecycle .block width=90% align=center}

Figure~\ref{fig:job-lifecycle} shows the job state transitions. Jobs start in PENDING while waiting for resources, move to RUNNING when allocated, then COMPLETING during cleanup, and finally COMPLETED on success. Jobs can also transition to FAILED (on error), CANCELLED (user intervention), or TIMEOUT (exceeded time limit). Use `squeue` to see current state and `sacct` for historical job information.

### Understanding SLURM Environment Variables

When SLURM launches your job, it automatically populates environment variables that your training script can read to configure distributed communication. Understanding these variables is essential for writing portable code that works across different cluster configurations.

At the job level, SLURM provides variables describing the overall allocation. `SLURM_JOB_ID` gives a unique identifier useful for naming log files and checkpoints. `SLURM_JOB_NAME` contains the name you specified with `--job-name` (or defaults to the script name). `SLURM_JOB_NODELIST` lists allocated nodes in compressed format—for example, `node[6-7]` or `gpu-node-[001-004]`—while `SLURM_JOB_NUM_NODES` gives the count. `SLURM_SUBMIT_DIR` records the directory from which you submitted the job, useful for locating config files or data relative to your submission location.

For distributed training, the process-level variables are most critical. Each task launched by SLURM receives `SLURM_PROCID`, a globally unique rank from 0 to NTASKS-1 that identifies this process among all processes in the job. `SLURM_LOCALID` gives the local rank within the current node (0 to tasks-per-node-1), which you typically use for GPU binding—process with `SLURM_LOCALID=0` uses GPU 0 on that node, and so on. `SLURM_NODEID` identifies which node this process runs on (0 to NUM_NODES-1), and `SLURM_NTASKS` provides the total task count, equivalent to world size in distributed training terminology. `SLURM_TASKS_PER_NODE` indicates how many tasks run on each node, though this may vary across nodes in heterogeneous allocations.

Resource-related variables help you tune performance. `SLURM_CPUS_PER_TASK` tells you how many CPU cores are available per task—useful for setting `num_workers` in your DataLoader. `SLURM_GPUS_ON_NODE` reports the GPU count on the current node, and `SLURM_MEM_PER_NODE` gives the memory allocation in MB. When you request GPUs with `--gres=gpu:N`, SLURM's GRES plugin automatically sets `CUDA_VISIBLE_DEVICES` to expose only your allocated GPUs, preventing conflicts with other jobs on the same node.

For establishing network communication, `SLURM_LAUNCH_NODE_IPADDR` provides the IP address of the launching node, and `SLURM_STEP_NODELIST` lists nodes participating in the current job step when using `srun` within an allocation.

The mapping from SLURM to PyTorch distributed training is straightforward: `SLURM_PROCID` becomes `RANK`, `SLURM_LOCALID` becomes `LOCAL_RANK`, and `SLURM_NTASKS` becomes `WORLD_SIZE`. The `MASTER_ADDR`—the address where rank 0 listens for connections—is typically derived from `SLURM_JOB_NODELIST` by extracting the first node's hostname. A typical setup script exports these translations:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
```

Once these variables are set, `torchrun` or `torch.distributed.init_process_group(init_method='env://')` reads them and configures the process group automatically. This abstraction is powerful: your training code doesn't need to know whether it's running under SLURM, launched by `torchrun` on a single machine, or orchestrated by a cloud provider. The same script works everywhere because it relies on standard environment variables rather than SLURM-specific APIs.

Figure~\ref{fig:slurm-env-vars} illustrates this mapping visually. Your training script can either read SLURM variables directly or use the exported PyTorch-standard variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`). The `torchrun` launcher handles this translation automatically when used with SLURM. Note that SLURM provides many additional environment variables for specialized use cases—for a complete reference, consult the `srun` man page or the official SLURM documentation.[^slurm]

![SLURM to PyTorch environment variable mapping.](img/slurm_env_vars_mapping.png){#fig:slurm-env-vars .block width=85% align=center}

## Launching Distributed Training Frameworks with SLURM {#sec:slurm-frameworks}

With the SLURM basics covered, we now turn to the practical question: how do you launch different distributed training frameworks on a SLURM cluster? Each framework has its own launcher and initialization pattern, but they all rely on the same SLURM environment variables we discussed above.

![Multi-node distributed training with SLURM.](img/multi_node_training.png){#fig:multi-node-training .block width=90% align=center}

Figure~\ref{fig:multi-node-training} illustrates the common pattern: SLURM allocates nodes and GPUs, launches processes across the cluster, and sets environment variables that each framework reads to establish distributed communication. The differences lie in how each framework wraps this process.

The table below summarizes the key differences in SLURM integration:

| Framework | Launcher | Distributed Init | SLURM Env Handling |
|-----------|----------|------------------|-----------------------|
| DDP | `torchrun` or `srun` | Manual | Export to `RANK`, `WORLD_SIZE`, etc. |
| FSDP | `torchrun` | Manual | Same as DDP |
| DeepSpeed | `python` | Automatic | Reads SLURM vars directly |
| Megatron-LM | `torchrun` | Automatic | Reads SLURM vars directly |

"Manual" initialization means you call `dist.init_process_group()` explicitly in your training script and handle environment variable setup in your SLURM batch script. "Automatic" means the framework handles distributed initialization internally—DeepSpeed via `deepspeed.init_distributed()` and Megatron-LM through its own launcher infrastructure—reading SLURM environment variables without requiring explicit setup code.

For detailed explanations of each framework's concepts and internals, refer to the earlier chapters: DDP in Chapter~\ref{chap:distributed-training-with-pytorch-ddp}, FSDP in Chapter~\ref{chap:fsdp-memory-efficient-distributed-training}, and DeepSpeed in Chapter~\ref{chap:deepspeed-zero-and-advanced-optimization}. Here we focus specifically on the SLURM launch patterns and provide complete working examples.

### DDP with SLURM {#sec:slurm-ddp-example}

As covered in Chapter~\ref{chap:distributed-training-with-pytorch-ddp}, PyTorch DDP replicates the entire model on each GPU, distributes data across processes, and synchronizes gradients via AllReduce during the backward pass. Because each GPU holds a complete copy of the model, DDP works best when your model fits comfortably in a single GPU's memory. Here we focus on the SLURM-specific launch patterns.

The training script structure is straightforward (full version in `code/train_ddp.py`):

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    for epoch in range(10):
        # ... training code ...
        if rank == 0:
            print(f"Epoch {epoch} completed")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

The SLURM batch script sets up the environment and launches via `torchrun` (full version in `code/train_ddp.sh`):

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    code/train_ddp.py
```

Alternatively, you can use SLURM's built-in MPI support without `torchrun`:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

srun python code/train_ddp.py
```

This approach requires your Python code to use `init_method='env://'`, which reads `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` from environment variables.

### FSDP with SLURM {#sec:slurm-fsdp-example}

As discussed in Chapter~\ref{chap:fsdp-memory-efficient-distributed-training}, FSDP shards model parameters, gradients, and optimizer states across GPUs, dramatically reducing per-GPU memory requirements for large models. The SLURM launch pattern is identical to DDP—you use `torchrun` the same way. The difference is in the Python code where you wrap the model with `FullyShardedDataParallel` instead of `DistributedDataParallel`.

Training script structure (full version in `code/train_fsdp.py`):

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    model = MyLargeModel()
    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )
    
    # Training loop...

if __name__ == '__main__':
    main()
```

SLURM batch script (full version in `code/train_fsdp.sh`):

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
    code/train_fsdp.py
```

### DeepSpeed with SLURM {#sec:slurm-deepspeed-example}

As covered in Chapter~\ref{chap:deepspeed-zero-and-advanced-optimization}, DeepSpeed's ZeRO optimizer provides three stages of memory optimization: ZeRO-1 partitions optimizer states, ZeRO-2 adds gradient partitioning, and ZeRO-3 further partitions model parameters themselves. ZeRO-3 is conceptually similar to FSDP—both shard parameters across GPUs—but DeepSpeed offers additional features like CPU and NVMe offloading that can push the memory boundary even further. The example here uses ZeRO-3 with CPU offloading, but you can easily switch to ZeRO-1 or ZeRO-2 by changing `"stage": 3` to `1` or `2` in the configuration file if you don't need full parameter sharding.

Unlike DDP and FSDP where you explicitly call `dist.init_process_group()` and use `torchrun` as the launcher, DeepSpeed takes a different approach. It handles distributed initialization internally via `deepspeed.init_distributed()`, reading SLURM environment variables directly without requiring a separate launcher. This design simplifies the user experience—you just run `python train.py` with the appropriate environment variables set, and DeepSpeed figures out the distributed topology automatically.

The training script structure reflects this simplicity (full version in `code/deepspeed/train.py`):

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    deepspeed.init_distributed()
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_zero3_offload.json"
    )
    
    for epoch in range(10):
        # ... training code ...
        model_engine.backward(loss)
        model_engine.step()

if __name__ == "__main__":
    main()
```

Notice that the script doesn't import `torch.distributed` or call `init_process_group()`—DeepSpeed handles all of that internally. The `deepspeed.initialize()` call returns a `model_engine` that wraps your model with ZeRO optimization, and you use `model_engine.backward()` and `model_engine.step()` instead of the standard PyTorch optimizer methods.

The configuration file controls ZeRO behavior (`code/deepspeed/ds_zero3_offload.json`):

```json
{
  "train_batch_size": 2,
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "fp16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_param": { "device": "cpu", "pin_memory": true },
    "offload_optimizer": { "device": "cpu", "pin_memory": true }
  },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 5e-5, "weight_decay": 0.01 }
  }
}
```

The `stage: 3` setting enables full parameter sharding, and the `offload_param` and `offload_optimizer` sections configure CPU offloading—essential for training models larger than your total GPU memory. The `pin_memory: true` option uses pinned (page-locked) CPU memory for faster CPU-GPU transfers.

The SLURM batch script requires more setup than DDP because we need to manually export the environment variables that DeepSpeed expects (`code/deepspeed/run.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=deepspeed-zero3
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

# Replace with your conda path and environment name
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker,lo
export GLOO_SOCKET_IFNAME=eth0

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

The script structure deserves some explanation. We set `MASTER_ADDR`, `MASTER_PORT`, and `WORLD_SIZE` at the job level, then use `srun` to launch a bash subshell on each node. Inside that subshell, we set the per-process variables (`CUDA_VISIBLE_DEVICES`, `LOCAL_RANK`, `RANK`) from SLURM's task-specific environment variables. The `--label` flag prefixes each line of output with the task ID, making it easier to debug multi-node issues.

A few practical considerations when running DeepSpeed on SLURM clusters. DeepSpeed requires the `LOCAL_RANK` environment variable, which you must explicitly export from `SLURM_LOCALID`—unlike `torchrun` which sets this automatically. If you're using virtual nodes for testing (as described earlier), remember to map node names to GPU indices appropriately—for example, `node6` should use GPU 6. IPv6 can cause connection issues on some clusters; setting `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to exclude problematic interfaces (like `^docker,lo`) often resolves this. Finally, remember to replace the conda path and environment name in the script with your own setup.

### Megatron-LM with SLURM {#sec:slurm-megatron-example}

As introduced in Chapter~\ref{chap:megatron-lm-and-model-parallelism}, Megatron-LM provides NVIDIA's production-grade framework combining tensor parallelism, pipeline parallelism, sequence/context parallelism, and data parallelism—all composable in a single training run. This multi-dimensional parallelism is essential for training the largest language models where no single parallelism strategy suffices.

Before diving into the SLURM script, there's an important installation consideration (also covered in Chapter~\ref{chap:megatron-lm-and-model-parallelism}). Unlike PyTorch's built-in DDP and FSDP, Megatron-LM requires installation from source to get the full training infrastructure. The PyPI package `megatron-core` only includes `megatron.core` (the model building blocks), but the training scripts like `pretrain_gpt.py` require `megatron.training` which is only available when you install from the GitHub repository:

```bash
conda activate research  # Replace with your environment name
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

You'll also need to copy the training scripts (`pretrain_gpt.py`, `gpt_builders.py`, `model_provider.py`) to your working directory, as these aren't installed as part of the package.

With the prerequisites in place, let's look at the SLURM batch script (`code/megatron/run.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=megatron-gpt
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_%j_%N.out
#SBATCH --error=logs/train_%j_%N.err

# Replace with your conda path and environment name
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$SCRIPT_DIR"
mkdir -p logs

PRETRAIN_SCRIPT="${SCRIPT_DIR}/pretrain_gpt.py"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-6000}
export WORLD_SIZE=$SLURM_NTASKS

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Model and training configuration
NUM_LAYERS=32; HIDDEN_SIZE=4096; NUM_ATTENTION_HEADS=32
TP_SIZE=1; CP_SIZE=1; PP_SIZE=1
MICRO_BATCH_SIZE=1; GLOBAL_BATCH_SIZE=128

srun --chdir="$SCRIPT_DIR" --label \
    bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate research
        export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        
        torchrun --nproc_per_node=1 --nnodes=\$SLURM_JOB_NUM_NODES \\
            --node_rank=\$SLURM_NODEID --master_addr=\"$MASTER_ADDR\" \\
            --master_port=\"$MASTER_PORT\" \"$PRETRAIN_SCRIPT\" \\
            --use-mcore-models --num-layers $NUM_LAYERS \\
            --hidden-size $HIDDEN_SIZE --num-attention-heads $NUM_ATTENTION_HEADS \\
            --tensor-model-parallel-size $TP_SIZE --pipeline-model-parallel-size $PP_SIZE \\
            --micro-batch-size $MICRO_BATCH_SIZE --global-batch-size $GLOBAL_BATCH_SIZE \\
            --bf16 --mock-data --tokenizer-type NullTokenizer --vocab-size 128256
    "
```

The script structure follows the same pattern as DeepSpeed: set job-level environment variables, then use `srun` to launch a bash subshell that sets per-process variables and invokes `torchrun`. The model configuration variables (`NUM_LAYERS`, `HIDDEN_SIZE`, etc.) define an 8B parameter GPT model, while the parallelism variables (`TP_SIZE`, `PP_SIZE`, `CP_SIZE`) control how the model is distributed—adjust these based on your hardware and model size. The example uses mock data (`--mock-data`) for demonstration; for real training, you'd provide actual data paths and a proper tokenizer. Remember to replace the conda path and environment name in the script with your own setup before submitting:

```bash
cd code/megatron
sbatch run.slurm
```

One thing you'll notice when training with Megatron-LM is that checkpoint files can be quite large. For an 8B parameter model, you might see checkpoint directories like this:

```
code/megatron/checkpoints/gpt_8b/iter_0000010/
27G     __0_0.distcp
27G     __0_1.distcp
27G     __1_0.distcp
27G     __1_1.distcp
24K     common.pt
4.0K    metadata.json
```

Why so large? The math is straightforward: model parameters in bf16 consume 8.03B × 2 bytes = 16.06 GB, while Adam optimizer states in fp32 require 8.03B × 8 bytes = 64.24 GB (4 bytes each for momentum and variance). That's already ~80 GB theoretically, and the actual size of ~108 GB includes additional overhead from distributed optimizer sharding, file format metadata, and alignment padding for efficient parallel I/O. Each rank saves its own shard (`__0_0.distcp`, `__0_1.distcp`, etc.) to enable parallel save/load operations across the cluster.

To manage checkpoint storage, consider using `--save-interval` to control how frequently checkpoints are saved, implementing checkpoint rotation to keep only recent checkpoints, and using a distributed filesystem that can handle the I/O load.

Another practical consideration is checkpoint format conversion. Megatron-LM saves checkpoints in a distributed format (`.distcp` files) that requires Megatron-LM to load. If you want to use your trained model with other frameworks like vLLM or SGLang for inference, or simply load it with vanilla PyTorch, you'll need to convert the checkpoint. The provided conversion script (`code/megatron/convert_megatron_checkpoint.py`) handles this:

```bash
python code/megatron/convert_megatron_checkpoint.py \
    --checkpoint-dir code/megatron/checkpoints/gpt_8b/iter_0000010 \
    --output-dir exported_checkpoint \
    --format pytorch \
    --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
    --vocab-size 128256 --max-position-embeddings 2048 \
    --use-mcore-models --bf16
```

The exported checkpoint is completely standalone—no Megatron-LM required to load it:

```python
import torch
checkpoint = torch.load('exported_checkpoint/model.pt', map_location='cpu')
print(checkpoint['model_config'])
state_dict = checkpoint['model_state_dict']
```

The converted checkpoint contains only model weights (no optimizer state), making it significantly smaller and compatible with any PyTorch-based inference framework. For production HuggingFace format conversion with proper layer name mapping and tensor reshaping, consider using [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

## Advanced Slurm Features for Training

### Job Arrays for Hyperparameter Tuning

Run multiple training jobs with different hyperparameters. A runnable version is in `code/train_array.sh`:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Each array task gets different hyperparameters
LR=$(echo "0.001 0.0001 0.00001 0.000001" | cut -d' ' -f$((SLURM_ARRAY_TASK_ID % 4 + 1)))
BATCH_SIZE=$((32 * (SLURM_ARRAY_TASK_ID / 4 + 1)))

python code/train.py --lr $LR --batch_size $BATCH_SIZE
```

Submit:

```bash
sbatch code/train_array.sh
```

### Interactive Jobs with `salloc`

Allocate resources interactively for debugging:

```bash
# Allocate 2 nodes, 1 GPU each, for 1 hour
salloc -N 2 --gres=gpu:1 --time=1:00:00

# Once allocated, run commands
srun hostname
srun nvidia-smi
srun python code/train.py

# Release when done
exit
```

### Job Dependencies

Chain jobs so one starts after another completes:

```bash
# Submit first job
JOB1=$(sbatch --parsable train_stage1.sh)

# Submit second job that depends on first
sbatch --dependency=afterok:$JOB1 train_stage2.sh
```

### Checkpointing and Job Resumption

Slurm supports job preemption and resumption. A checkpoint utility script is in `code/checkpoint.py`, and a complete training script with checkpointing is in `code/train_distributed.sh`:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90  # Send signal 90 seconds before time limit

# Handle checkpoint signal
trap 'echo "Checkpointing..."; python code/checkpoint.py' SIGUSR1

python code/train.py --resume --checkpoint_dir=/path/to/checkpoints
```

## Monitoring and Debugging

### Job Monitoring

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

### Logging and Output

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

### Profiling Distributed Training

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

## Best Practices

### Resource Allocation

- **Always specify resources explicitly**: Don't rely on defaults
- **Use `--exclusive` for full node**: When you need all resources on a node
- **Request appropriate memory**: Use `--mem` or `--mem-per-gpu` to avoid OOM

```bash
srun -N 2 --gres=gpu:1 --mem=200G --cpus-per-task=28 python code/train.py
```

### Multi-Node Communication

- **Use high-speed interconnects**: InfiniBand or high-speed Ethernet for multi-node
- **Set appropriate NCCL environment variables**:

```bash
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0  # Specify network interface
export NCCL_DEBUG=INFO  # For debugging
```

### Checkpointing Strategy

- **Frequent checkpoints**: Save every N steps, not just at epoch boundaries
- **Distributed checkpointing**: Use `torch.distributed.checkpoint` for FSDP
- **Resume capability**: Always implement `--resume` flag in training scripts

### Error Handling

- **Handle node failures**: Implement retry logic for transient failures
- **Validate data loading**: Ensure data is accessible from all nodes
- **Monitor for deadlocks**: Use timeouts and health checks

## Troubleshooting Common Issues

### Nodes Not Available

```bash
# Check node status
sinfo -N -l

# Resume down nodes
scontrol update NodeName=node[6-7] State=RESUME

# Drain nodes for maintenance
scontrol update NodeName=node6 State=DRAIN Reason="maintenance"
```

### GPU Allocation Issues

```bash
# Check GPU availability
scontrol show nodes | grep Gres

# Verify GPU mapping
# Replace $SLURM_PREFIX with your Slurm installation prefix
cat $SLURM_PREFIX/etc/gres.conf

# Test GPU allocation
srun -N 1 --gres=gpu:1 nvidia-smi -L
```

### Communication Errors

- **Check network connectivity**: `srun -N 2 ping -c 3 <other_node>`
- **Verify NCCL setup**: Set `NCCL_DEBUG=INFO` for detailed logs
- **Check firewall**: Ensure required ports are open

### Job Hanging

- **Check for deadlocks**: Look for processes waiting on barriers
- **Verify data loading**: Ensure all ranks can access data
- **Check logs**: Review both stdout and stderr from all ranks

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
- [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models
- [DeepOps](https://github.com/NVIDIA/deepops) - Open-source tools for deploying and managing GPU-accelerated clusters using Kubernetes and Slurm


