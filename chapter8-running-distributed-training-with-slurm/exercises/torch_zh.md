\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. 编写多节点分布式训练 SLURM 作业提交脚本

编写一个标准的 SLURM Batch 脚本，精准配置多机多卡资源，并正确导出 PyTorch / NCCL 分布式训练所需的环境变量。

__要求：__

- 申请 2 个计算节点，每节点申请 4 张 GPU（共 8 卡）
- 合理设置作业超时上限（Time limit）、CPU 核心数与主机内存配额
- 通过 `scontrol show hostnames` 解析主节点 IP 并动态生成端口
- 正确配置 NCCL 环境变量（`NCCL_DEBUG`、网卡接口过滤等）
- 使用 `srun` 调起 `torchrun` 启动分布式训练，并分离标准输出与错误日志

__测试代码：__
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

# 实战任务：
# 1. 加载集群环境模块（CUDA, Python 等）
# 2. 导出 MASTER_ADDR 和 MASTER_PORT
# 3. 配置 NCCL 网络通信环境变量
# 4. 使用 srun 启动分布式训练脚本

# 验证作业提交命令：
# sbatch train.slurm
# squeue -u $USER
# cat logs/<job_id>.out
```

### 2. 实现作业信号捕获与抢占安全自动存盘（SLURM Checkpointing）

在 Python 训练循环中捕获 SLURM 超时与抢占信号（`SIGTERM` / `SIGUSR1`），实现安全落盘与作业自动重新排队提交。

__要求：__

- 注册 `SIGTERM` 与 `SIGUSR1` 信号处理句柄
- 在作业被抢占或超时前 90 秒收到信号时，由 Rank 0 安全保存最新的模型权重与优化器状态
- 在退出前调用 `sbatch` 自动重新提交自身作业实现断点续训
- 在恢复启动时自动检测并加载最新的 Checkpoint

__测试代码：__
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
        
        # 注册 SLURM 提前预警信号
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGUSR1, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """捕获 SLURM 抢占或超时信号。"""
        print(f"收到信号 {signum}，正在紧急保存 Checkpoint...")
        self.should_stop = True
    
    def save(self):
        """仅由 Rank 0 保存权重与优化器状态。"""
        pass
    
    def load(self) -> bool:
        """加载最新 Checkpoint，成功返回 True。"""
        pass
    
    def should_checkpoint(self) -> bool:
        """判断是否需要紧急存盘并退出。"""
        return self.should_stop

# 训练主循环中使用示例
checkpointer = SLURMCheckpointer("./checkpoints", model, optimizer)
checkpointer.load()  # 若存在历史检查点则恢复

for step in range(checkpointer.step, max_steps):
    loss = train_step(model, batch)
    
    if checkpointer.should_checkpoint():
        checkpointer.save()
        # 由主节点自动重新提交作业
        if int(os.environ.get("SLURM_PROCID", 0)) == 0:
            os.system("sbatch train.slurm")
        break
```

### 3. 多节点跨机 NCCL 通信基准诊断与网络带宽压测

编写一个专用于在 SLURM 跨机环境下测试 NCCL 互联带宽与延迟的诊断工具脚本。

__要求：__

- 验证跨节点所有 GPU 是否能成功建立 NCCL 环形/树状连接
- 针对不同张量数据包大小（1MB, 10MB, 100MB, 500MB）实测 AllReduce 聚合带宽（GB/s）
- 测试主节点与其他所有 Worker 节点间的点对点（P2P）通信往返延迟（$\mu s$）
- 识别跨机网络是否存在慢节点（Straggler）或带宽瓶颈

__测试代码：__
```python
import torch
import torch.distributed as dist
import os
import time

def diagnose_nccl():
    """诊断 SLURM 集群环境下的 NCCL 通信效能。"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    node_name = os.environ.get("SLURMD_NODENAME", "unknown")
    print(f"Rank {rank}/{world_size} 运行在节点 {node_name}, 本地 GPU {local_rank}")
    
    # 压测 AllReduce 聚合带宽
    sizes_mb = [1, 10, 100, 500]
    for size_mb in sizes_mb:
        tensor = torch.randn(int(size_mb * 1e6 / 4)).cuda()
        
        # 热身 Warmup
        for _ in range(3):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # 计时压测
        start = time.time()
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Ring AllReduce 有效通信数据量为 2 * (N-1)/N * Size
        bandwidth = (size_mb * 2 * 10) / elapsed
        if rank == 0:
            print(f"张量大小: {size_mb:4d} MB | 测算有效带宽: {bandwidth:.1f} GB/s")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    diagnose_nccl()
```

### 4. 使用 SLURM Job Array 编排并发超参数搜索网格

利用 SLURM 的 Job Array 功能，通过单个脚本并发执行网格搜索，并编写汇总脚本分析最佳结果。

__要求：__

- 声明 `#SBATCH --array=0-11` 创建 12 个并发独立子任务
- 将 `SLURM_ARRAY_TASK_ID` 映射为学习率（Learning Rate）与 Batch Size 组合
- 各子任务独立运行训练并输出指标文件 `metrics.json`
- 编写 Python 脚本收集并按验证集 Loss 排序选出最优超参数组合

__测试代码：__
```bash
#!/bin/bash
#SBATCH --job-name=hp-search
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/hp_%A_%a.out

# 超参数网格定义
LEARNING_RATES=(1e-4 3e-4 1e-3 3e-3)
BATCH_SIZES=(16 32 64)

NUM_LR=${#LEARNING_RATES[@]}
LR_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LR))
BS_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LR))

LR=${LEARNING_RATES[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

echo "任务 ID $SLURM_ARRAY_TASK_ID: 学习率=$LR, 批大小=$BS"
python train.py --lr $LR --batch-size $BS --output-dir results/hp_$SLURM_ARRAY_TASK_ID
```

```python
# collect_results.py
import glob
import json

def collect_hp_results(results_dir: str):
    """汇总并对比网格搜索实验结果。"""
    results = []
    for result_file in glob.glob(f"{results_dir}/hp_*/metrics.json"):
        with open(result_file) as f:
            results.append(json.load(f))
    
    results.sort(key=lambda x: x["val_loss"])
    print("前 5 优超参数组合：")
    for i, r in enumerate(results[:5]):
        print(f"{i+1}. LR={r['lr']}, BatchSize={r['batch_size']}, 验证集 Loss={r['val_loss']:.4f}")
    return results[0]

best = collect_hp_results("results")
print(f"\n全局最优: LR={best['lr']}, BatchSize={best['batch_size']}")
```

### 5. 构建全集群分布式作业监控与掉队检测器（Straggler Detector）

编写一个 Python 监控脚本，实时获取 SLURM 分布式作业的所有节点 GPU 状态并对异常掉队节点进行告警。

__要求：__

- 通过 `squeue` 动态解析作业占用的所有物理节点列表
- 跨节点远程查询每张 GPU 的显存占用与计算利用率（GPU Utilization）
- 自动检测并输出平均利用率过低（<50%）的“拖后腿”异常节点（Straggler）
- 实时打印集群健康看板

__测试代码：__
```python
import subprocess
import time
from datetime import datetime

class SLURMJobMonitor:
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.metrics_history = []
    
    def get_node_list(self) -> list[str]:
        """获取当前作业分配的物理节点列表。"""
        result = subprocess.run(
            ["squeue", "-j", str(self.job_id), "-o", "%N", "-h"],
            capture_output=True, text=True
        )
        # 解析压缩节点名（例如 node[6-7]）
        return self._expand_nodelist(result.stdout.strip())
    
    def get_gpu_utilization(self, node: str) -> list[dict]:
        """远程获取指定节点的 GPU 利用率与显存使用。"""
        result = subprocess.run(
            ["ssh", node, "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
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
    
    def detect_anomalies(self, metrics: dict) -> list[str]:
        """检测利用率过低或显存异常的慢节点。"""
        anomalies = []
        for node, gpus in metrics["nodes"].items():
            for i, gpu in enumerate(gpus):
                if gpu["utilization"] < 50:
                    anomalies.append(f"{node} GPU {i}: 计算利用率偏低 ({gpu['utilization']:.0f}%)")
                if gpu["memory_used_gb"] / gpu["memory_total_gb"] > 0.95:
                    anomalies.append(f"{node} GPU {i}: 显存逼近溢出")
        return anomalies

# 启动持续监控循环
monitor = SLURMJobMonitor(job_id=12345)
while True:
    metrics = monitor.collect_metrics()
    anomalies = monitor.detect_anomalies(metrics)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 集群监控状态已刷新...")
    if anomalies:
        print("发现系统异常:", anomalies)
    time.sleep(30)
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 熟练编写规范的多机多卡 SLURM Batch 脚本并正确配置 GRES GPU 资源
- 深刻理解 `SLURM_PROCID`、`SLURM_LOCALID`、`SLURM_NTASKS` 到 PyTorch `RANK`、`WORLD_SIZE` 的映射机理
- 掌握在 SLURM 集群上编排与调优 PyTorch DDP、FSDP、DeepSpeed 以及 Megatron-LM 作业
- 实现基于系统信号（`SIGUSR1`/`SIGTERM`）的优雅紧急存盘与自动断点续训
- 使用 SLURM Job Array 批量并发执行超参数网格搜索实验
- 运用 Nsight Systems、NCCL 调试参数与跨机监控脚本排查分布式作业吊死（Hang）与通信拥塞瓶颈
