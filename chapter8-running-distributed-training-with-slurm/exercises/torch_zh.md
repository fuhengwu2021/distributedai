\fancydividerwithicon[center]{hand.png}


## 实战演练


### 编写标准 SLURM 分布式作业批处理脚本

编写用于提交分布式 PyTorch 训练任务的 SLURM 调度脚本，正确配置跨节点资源分配与网络拓扑参数。

__实战要求：__

- 申请 2 个物理节点，每个节点配置 4 块 GPU（共 8 卡）
- 合理配置作业最长运行时间、内存配额与任务 CPU 核心配比
- 动态获取主节点 IP 并自动设置分布式环境变量（`MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE` 等）
- 正确配置 NCCL 通信网卡与互连网络环境变量
- 完善日志重定向（分流 stdout 与 stderr）与错误追踪机制

__测试验证：__
```bash
#!/bin/bash
#SBATCH --job-name=ddp-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# 请在下方完善执行逻辑：
# 1. 加载集群环境依赖模块（CUDA、Conda 环境等）
# 2. 动态提取 Rank 0 节点地址设置 MASTER_ADDR 与 MASTER_PORT
# 3. 配置 NCCL 网络接口（如 NCCL_SOCKET_IFNAME、NCCL_IB_DISABLE 等）
# 4. 使用 srun 启动多节点分布式训练进程

# 脚本提交与状态验证：
# sbatch train.slurm
# squeue -u $USER
# cat logs/<job_id>.out
```

### 实现基于 SLURM 信号捕获的自动检查点续训

编写能够优雅响应 SLURM 抢占调度与超时信号的训练中断保护与自动重投系统。

__实战要求：__

- 捕获两类核心操作系统中断信号：
  - 超时预警信号：由 `#SBATCH --signal=SIGUSR1@90` 触发的 `SIGUSR1`（作业达到时限前 90 秒告警）
  - 任务取消与抢占信号：由 `scancel` 或节点维护触发的 `SIGTERM`
- 捕获到信号后平滑终止迭代并保存完整 Checkpoint
- 由 Rank 0 负责自动重新提交自身作业（`sbatch`），实现无限续期接力训练
- 从历史最新 Checkpoint 恢复训练步数与优化器内部状态
- 统计并累加跨中断任务的总体真实训练耗时

__测试验证：__
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
        
        # 注册集群信号捕获钩子
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGUSR1, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """处理 SLURM 抢占或超时预警信号。"""
        print(f"捕获到信号 {signum}，准备触发紧急保存...")
        self.should_stop = True
    
    def save(self):
        """保存检查点（仅限 Rank 0 负责磁盘写入）。"""
        pass
    
    def load(self) -> bool:
        """加载历史检查点，恢复成功返回 True。"""
        pass
    
    def should_checkpoint(self) -> bool:
        """检测当前步是否收到退出保存信号。"""
        return self.should_stop

# 训练主循环中的应用范例
checkpointer = SLURMCheckpointer("./checkpoints", model, optimizer)
checkpointer.load()  # 若存在历史检查点则无缝恢复

for step in range(checkpointer.step, max_steps):
    loss = train_step(model, batch)
    
    if checkpointer.should_checkpoint():
        checkpointer.save()
        # 仅由主控进程负责向集群发起重新排队提交
        if int(os.environ.get("SLURM_PROCID", 0)) == 0:
            os.system("sbatch train.slurm")
        break
```

### 多节点跨机 NCCL 通信诊断与基准测试

编写集群网络体检脚本，在 SLURM 分配的多个节点间精确测量 NCCL 集合通信带宽与点对点往返延迟。

__实战要求：__

- 验证所有分配节点间的网络连通性与路由状态
- 精确测量跨节点真实 AllReduce 吞吐带宽
- 压测对比环形（Ring）与树形（Tree）算法的延迟表现
- 探测网络拥塞与网卡抖动（Jitter）
- 自动生成多节点网络健康度体检报告

__测试验证：__
```python
import torch
import torch.distributed as dist
import os
import time

def diagnose_nccl():
    """诊断 SLURM 集群环境下的 NCCL 通信健康状态。"""
    
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    
    # 提取物理节点拓扑信息
    node_name = os.environ.get("SLURMD_NODENAME", "unknown")
    print(f"Rank {rank}/{world_size} 运行于主机节点 {node_name}, GPU设备 {local_rank}")
    
    # 压测不同数据量下的 AllReduce 吞吐带宽
    sizes_mb = [1, 10, 100, 500]
    results = []
    
    for size_mb in sizes_mb:
        tensor = torch.randn(int(size_mb * 1e6 / 4)).cuda()
        
        # 充分预热通信通道
        for _ in range(3):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # 高精度耗时统计
        start = time.time()
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # 环形 AllReduce 通信总线带宽换算公式
        bandwidth = (size_mb * 2 * 10) / elapsed
        results.append((size_mb, bandwidth))
        
        if rank == 0:
            print(f"数据体量: {size_mb:4d} MB, 有效带宽: {bandwidth:.1f} GB/s")
    
    # 点对点往返往复延迟测试（Rank 0 与其余所有 Rank 轮转收发）
    if rank == 0:
        print("\n点对点单向网络延迟测试:")
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

### 基于 SLURM Job Array 实现分布式超参数搜索

利用 SLURM 任务阵列（Job Array）功能，构建大规模并行超参寻优与自动化结果归并系统。

__实战要求：__

- 定义离散超参数网格（学习率、批次大小等）
- 依据 `SLURM_ARRAY_TASK_ID` 映射各子任务对应的超参数元组
- 并行执行互不干扰的独立训练任务
- 编写聚合脚本收集各子任务生成的 `metrics.json`
- 按照验证集 Loss 排序并输出最佳超参组合

__测试验证：__
```bash
#!/bin/bash
#SBATCH --job-name=hp-search
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/hp_%A_%a.out

# 定义超参数寻优网格
LEARNING_RATES=(1e-4 3e-4 1e-3 3e-3)
BATCH_SIZES=(16 32 64)

# 计算数组索引
NUM_LR=${#LEARNING_RATES[@]}
LR_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LR))
BS_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LR))

LR=${LEARNING_RATES[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

echo "任务阵列 ID $SLURM_ARRAY_TASK_ID: 学习率=$LR, 批大小=$BS"

python train.py --lr $LR --batch-size $BS --output-dir results/hp_$SLURM_ARRAY_TASK_ID
```

```python
# collect_results.py 自动聚合分析脚本
import os
import json
import glob

def collect_hp_results(results_dir: str):
    """自动收集并对齐所有阵列子任务的评估指标。"""
    results = []
    
    for result_file in glob.glob(f"{results_dir}/hp_*/metrics.json"):
        with open(result_file) as f:
            metrics = json.load(f)
        results.append(metrics)
    
    # 按验证集损失升序排列
    results.sort(key=lambda x: x["val_loss"])
    
    print("前 5 组最优参数组合:")
    for i, r in enumerate(results[:5]):
        print(f"{i+1}. LR={r['lr']}, BS={r['batch_size']}, "
              f"Val Loss={r['val_loss']:.4f}")
    
    return results[0]  # 返回全局最佳超参

best = collect_hp_results("results")
print(f"\n最佳推荐配置: 学习率={best['lr']}, 批大小={best['batch_size']}")
```

### SLURM 作业集群实时监控与异常诊断

编写轻量级监控工具，跨多计算节点跟踪分布式作业的 GPU 负载、训练指标并预警慢节点（Straggler）。

__实战要求：__

- 解析作业所属的全部物理节点清单
- 无需依赖 SSH 免密权限，通过 `srun` 实时轮询各节点的 `nvidia-smi` 显存与利用率
- 追踪训练 Loss 收敛与迭代吞吐曲线
- 自动识别落后慢节点（如利用率持续低于 50% 或显存爆满）
- 触发异常时在终端高亮告警

__测试验证：__
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
        """提取该作业目前被分配的所有计算节点列表。"""
        result = subprocess.run(
            ["squeue", "-j", str(self.job_id), "-o", "%N", "-h"],
            capture_output=True, text=True
        )
        return self._expand_nodelist(result.stdout.strip())
    
    def get_gpu_utilization(self, node: str) -> dict:
        """通过 srun 跨节点直接读取 nvidia-smi 统计指标（无需 SSH 免密）。"""
        result = subprocess.run(
            ["srun", "-w", node, "nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        # 格式化解析输出
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
        """收集所有节点的实时硬件指标快照。"""
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
        """规则化排查潜在慢节点与显存溢出风险。"""
        anomalies = []
        if not self.metrics_history:
            return anomalies
        
        latest = self.metrics_history[-1]
        for node, gpus in latest["nodes"].items():
            for i, gpu in enumerate(gpus):
                if gpu["utilization"] < 50:
                    anomalies.append(f"节点 {node} GPU {i}: 计算核心利用率偏低 ({gpu['utilization']:.0f}%)")
                if gpu["memory_used_gb"] / gpu["memory_total_gb"] > 0.95:
                    anomalies.append(f"节点 {node} GPU {i}: 显存占用逼近物理上限 (即将发生 OOM)")
        
        return anomalies

# 启动轮询监控
monitor = SLURMJobMonitor(job_id=12345)

while True:
    metrics = monitor.collect_metrics()
    anomalies = monitor.detect_anomalies()
    
    print(f"\n[{metrics['timestamp']}]")
    for node, gpus in metrics["nodes"].items():
        avg_util = sum(g["utilization"] for g in gpus) / len(gpus)
        print(f"  节点 {node}: GPU 平均利用率 {avg_util:.0f}%")
    
    if anomalies:
        print("  【异常告警】:", anomalies)
    
    time.sleep(30)
```


## 预期学习目标

完成本章实战练习后，你将能够：

- 独立编写并维护生产级多节点 SLURM 分布式批处理作业脚本
- 实现基于系统信号（SIGUSR1/SIGTERM）的优雅容错与自动无缝接力续训
- 掌握多机网络通信体检方法，精准辨识 InfiniBand/RoCE 跨机性能瓶颈
- 熟练运用 SLURM Job Array 架构设计万级超参数并行搜索流水线
- 掌握在异构生产集群上编写非侵入式实时资源巡检与慢节点（Straggler）侦测工具
- 具备快速定位并解决大规模集群训练中各种节点掉线与通信挂起（Hang）故障的实操经验
