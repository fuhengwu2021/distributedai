\fancydividerwithicon[center]{hand.png}


## Exercises


### Write a Basic SLURM Job Script

Create a SLURM job script for distributed PyTorch training with proper resource allocation.

__Requirements:__

- Request 2 nodes with 4 GPUs each
- Set appropriate time limit and memory
- Configure environment variables for distributed training
- Handle NCCL initialization
- Add logging and error handling

__Test your implementation:__
```bash
#!/bin/bash
#SBATCH --job-name=ddp-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Your implementation here:
# 1. Load modules (CUDA, Python, etc.)
# 2. Set MASTER_ADDR and MASTER_PORT
# 3. Set NCCL environment variables
# 4. Launch distributed training with srun

# Verify your script:
# sbatch train.slurm
# squeue -u $USER
# cat logs/<job_id>.out
```

### Implement Automatic Checkpointing with SLURM

Create a training script that handles SLURM preemption and time limits gracefully.

__Requirements:__

- Detect SLURM signals (SIGTERM for preemption)
- Save checkpoint before job ends
- Implement automatic job resubmission
- Resume from latest checkpoint
- Track total training time across restarts

__Test your implementation:__
```python
import signal
import os
import torch
import torch.distributed as dist

class SLURMCheckpointer:
    def __init__(self, checkpoint_dir: str, model, optimizer):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.step = 0
        self.should_stop = False
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGUSR1, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle SLURM preemption signal."""
        print(f"Received signal {signum}, saving checkpoint...")
        self.should_stop = True
    
    def save(self):
        """Save checkpoint (rank 0 only)."""
        pass
    
    def load(self) -> bool:
        """Load latest checkpoint. Return True if loaded."""
        pass
    
    def should_checkpoint(self) -> bool:
        """Check if we should save and exit."""
        return self.should_stop

# Usage in training loop
checkpointer = SLURMCheckpointer("./checkpoints", model, optimizer)
checkpointer.load()  # Resume if checkpoint exists

for step in range(checkpointer.step, max_steps):
    # Training step
    loss = train_step(model, batch)
    
    if checkpointer.should_checkpoint():
        checkpointer.save()
        # Resubmit job
        if int(os.environ.get("SLURM_PROCID", 0)) == 0:
            os.system("sbatch train.slurm")
        break
```

### Configure Multi-Node NCCL Communication

Write a diagnostic script that verifies NCCL communication across SLURM nodes.

__Requirements:__

- Verify all nodes can communicate
- Measure inter-node bandwidth
- Test different NCCL algorithms (Ring, Tree)
- Identify communication bottlenecks
- Generate diagnostic report

__Test your implementation:__
```python
import torch
import torch.distributed as dist
import os
import time

def diagnose_nccl():
    """Diagnose NCCL communication in SLURM environment."""
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    
    # Get node information
    node_name = os.environ.get("SLURMD_NODENAME", "unknown")
    
    print(f"Rank {rank}/{world_size} on {node_name}, GPU {local_rank}")
    
    # Test AllReduce bandwidth
    sizes_mb = [1, 10, 100, 500]
    results = []
    
    for size_mb in sizes_mb:
        tensor = torch.randn(int(size_mb * 1e6 / 4)).cuda()
        
        # Warmup
        for _ in range(3):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bandwidth = (size_mb * 2 * 10) / elapsed  # Ring AllReduce: 2x data
        results.append((size_mb, bandwidth))
        
        if rank == 0:
            print(f"Size: {size_mb:4d} MB, Bandwidth: {bandwidth:.1f} GB/s")
    
    # Test point-to-point (rank 0 to all others)
    if rank == 0:
        print("\nPoint-to-point latency:")
        for target in range(1, world_size):
            tensor = torch.zeros(1).cuda()
            
            start = time.time()
            for _ in range(100):
                dist.send(tensor, dst=target)
                dist.recv(tensor, src=target)
            elapsed = time.time() - start
            
            latency_us = (elapsed / 200) * 1e6
            print(f"  Rank 0 <-> Rank {target}: {latency_us:.1f} us")
    else:
        tensor = torch.zeros(1).cuda()
        for _ in range(100):
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    diagnose_nccl()
```

### Implement Job Array for Hyperparameter Search

Create a SLURM job array for parallel hyperparameter search.

__Requirements:__

- Define hyperparameter grid
- Map array task ID to hyperparameters
- Run independent training jobs
- Collect and compare results
- Find best hyperparameters

__Test your implementation:__
```bash
#!/bin/bash
#SBATCH --job-name=hp-search
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/hp_%A_%a.out

# Hyperparameter grid
LEARNING_RATES=(1e-4 3e-4 1e-3 3e-3)
BATCH_SIZES=(16 32 64)

# Calculate indices
NUM_LR=${#LEARNING_RATES[@]}
LR_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LR))
BS_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LR))

LR=${LEARNING_RATES[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: LR=$LR, BS=$BS"

python train.py --lr $LR --batch-size $BS --output-dir results/hp_$SLURM_ARRAY_TASK_ID
```

```python
# collect_results.py
import os
import json
import glob

def collect_hp_results(results_dir: str):
    """Collect and compare hyperparameter search results."""
    results = []
    
    for result_file in glob.glob(f"{results_dir}/hp_*/metrics.json"):
        with open(result_file) as f:
            metrics = json.load(f)
        results.append(metrics)
    
    # Sort by validation loss
    results.sort(key=lambda x: x["val_loss"])
    
    print("Top 5 configurations:")
    for i, r in enumerate(results[:5]):
        print(f"{i+1}. LR={r['lr']}, BS={r['batch_size']}, "
              f"Val Loss={r['val_loss']:.4f}")
    
    return results[0]  # Best config

best = collect_hp_results("results")
print(f"\nBest: LR={best['lr']}, BS={best['batch_size']}")
```

### Monitor and Profile SLURM Jobs

Create monitoring tools for tracking distributed training jobs on SLURM.

__Requirements:__

- Monitor GPU utilization across nodes
- Track training progress (loss, throughput)
- Detect stragglers and communication issues
- Generate real-time dashboard
- Alert on anomalies

__Test your implementation:__
```python
import subprocess
import time
import json
from datetime import datetime

class SLURMJobMonitor:
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.metrics_history = []
    
    def get_node_list(self) -> list[str]:
        """Get list of nodes for this job."""
        result = subprocess.run(
            ["squeue", "-j", str(self.job_id), "-o", "%N", "-h"],
            capture_output=True, text=True
        )
        return self._expand_nodelist(result.stdout.strip())
    
    def get_gpu_utilization(self, node: str) -> dict:
        """Get GPU utilization on a node."""
        result = subprocess.run(
            ["ssh", node, "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        # Parse output
        lines = result.stdout.strip().split("\n")
        gpus = []
        for line in lines:
            util, mem_used, mem_total = map(float, line.split(", "))
            gpus.append({
                "utilization": util,
                "memory_used_gb": mem_used / 1024,
                "memory_total_gb": mem_total / 1024,
            })
        return gpus
    
    def collect_metrics(self) -> dict:
        """Collect metrics from all nodes."""
        nodes = self.get_node_list()
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "nodes": {}
        }
        
        for node in nodes:
            metrics["nodes"][node] = self.get_gpu_utilization(node)
        
        self.metrics_history.append(metrics)
        return metrics
    
    def detect_anomalies(self) -> list[str]:
        """Detect potential issues."""
        anomalies = []
        
        if not self.metrics_history:
            return anomalies
        
        latest = self.metrics_history[-1]
        
        for node, gpus in latest["nodes"].items():
            for i, gpu in enumerate(gpus):
                if gpu["utilization"] < 50:
                    anomalies.append(f"{node} GPU {i}: Low utilization ({gpu['utilization']:.0f}%)")
                if gpu["memory_used_gb"] / gpu["memory_total_gb"] > 0.95:
                    anomalies.append(f"{node} GPU {i}: High memory usage")
        
        return anomalies

# Usage
monitor = SLURMJobMonitor(job_id=12345)

while True:
    metrics = monitor.collect_metrics()
    anomalies = monitor.detect_anomalies()
    
    print(f"\n[{metrics['timestamp']}]")
    for node, gpus in metrics["nodes"].items():
        avg_util = sum(g["utilization"] for g in gpus) / len(gpus)
        print(f"  {node}: {avg_util:.0f}% GPU util")
    
    if anomalies:
        print("  Anomalies:", anomalies)
    
    time.sleep(30)
```


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Write effective SLURM job scripts for distributed training
- Handle job preemption and implement automatic checkpointing
- Diagnose and optimize NCCL communication
- Run hyperparameter searches with SLURM job arrays
- Monitor and profile distributed training jobs
- Troubleshoot common issues in SLURM environments
