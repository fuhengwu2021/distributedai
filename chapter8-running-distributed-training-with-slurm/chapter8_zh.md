# 第 8 章：基于 SLURM 的分布式训练集群运维与实战 {-}

*管理大规模 GPU 集群算力、作业调度与多节点训练协同编排*

> 在一间坐满顶级软件架构师的屋子里，如果两个人对同一件事达成了一致，那就已经是绝对多数了。  
> —— 比尔·柯蒂斯（Bill Curtis，软件工程专家）

**核心代码速查**

- `sbatch`：向 SLURM 调度器提交异步批处理训练作业（Batch Job）
- `srun`：向集群申请资源并即刻以交互式执行单条或多节点命令
- `squeue`：查询集群全局或指定用户的作业排队与运行状态
- `scancel`：取消或终止正在排队/运行的指定作业
- `sinfo`：查看集群各分区（Partition）及计算节点健康状态
- `sacct`：查询历史已结束作业的资源消耗记账数据
- `scontrol`：管理员/用户高级控制命令（查看作业详情、恢复节点状态等）
- `torchrun`：与 SLURM 环境变量无缝绑定的 PyTorch 分布式拉起器
- `SLURM_PROCID` / `SLURM_LOCALID`：SLURM 分配给进程的全局 Rank 与机内 Local Rank
- `SLURM_NTASKS` / `SLURM_JOB_NODELIST`：总并发进程数（World Size）与分配的主机列表


## 高性能计算（HPC）与 AI 训练集群

在前面的章节中，我们系统学习了分布式训练的核心原理与算法实现——用于梯度同步的 DDP、用于显存全分片的 FSDP、用于异构存储优化的 DeepSpeed 以及用于多维算力切分的 Megatron-LM。然而，掌握算法框架仅仅完成了大模型工程的一半。**另一半核心挑战在于如何将这些分布式代码稳定、高效地运行在真实的物理硬件集群上**：如何跨多机节点分配 GPU、如何自动编排进程网络、如何公平共享百万美元级算力资源，以及如何从容应对大规模训练中不可避免的节点故障与作业抢占。

现代大模型训练无一例外运行在高性能 GPU 计算集群上。一个标准的 AI 集群由数十至数千个计算节点（每台包含 8 张 GPU、海量 CPU 核心与数百 GB 内存）、超低延迟的高速网络互联（InfiniBand 或 RoCE 满血无损网络）、全集群挂载的高性能共享分布式存储（Lustre/GPFS/NFS）以及负责统筹全集群调度的主控管理节点组成。

为了在海量研发团队之间公平、高效地共享这些昂贵的硬件资产，集群必须依赖**作业调度系统（Job Scheduler / Resource Manager）**。在高性能计算（HPC）与 AI 大模型研发领域，**SLURM**（Simple Linux Utility for Resource Management）凭借其工业级的稳定性、极低的管理开销以及对 GPU 资源的完美支持，成为了全球国家实验室、顶级高校以及主流云厂商（如 AWS, GCP, Azure, Oracle Cloud）事实上的标准底座[^slurm]。

[^slurm]: SLURM 由 SchedMD 维护。官方文档与下载地址：https://slurm.schedmd.com/

### 为什么 SLURM 是大规模 AI 训练的首选？

1. **原生的 GPU 通用资源隔离（GRES）**：通过 `--gres=gpu:8`，SLURM 能够精准识别物理 GPU 型号并进行**独占式硬件分配**，绝不允许其他作业挤占正在训练的模型显存；
2. **与 PyTorch 分布式环境变量天然对齐**：SLURM 在派生多机进程时，自动注入环境变量（`SLURM_PROCID` 对应全局 Rank，`SLURM_LOCALID` 对应 Local Rank，`SLURM_NTASKS` 对应 World Size），PyTorch `torchrun` 或标准分布式后端无需修改一行代码即可直接对接；
3. **完善的长期作业运维能力**：支持 Job Array（超参数网格搜索）、作业依赖链（Pipeline Chaining）、抢占与超时预警信号（优雅 Checkpointing）以及详尽的资源记账审计；
4. **极高的可扩展性**：同一套 SLURM 脚本无论在 2 台节点的实验室小型测试集群，还是在 10,000 张 GPU 的超级计算中心，都能完全无缝运行。

---

## SLURM 核心架构与守护进程全景

![SLURM 核心架构：slurmctld、slurmd 与 slurmdbd 交互机制](img/slurm_architecture.png){#fig:slurm-architecture .block width=95% align=center}

如 @fig:slurm-architecture 所示，SLURM 架构由三大核心守护进程组成：

1. **`slurmctld`（中央控制守护进程）**：运行在集群 Master 管理节点上，是整个系统的“大脑”。负责维护作业等待队列、计算调度优先级（公平共享 Fair-Share、回填调度 Backfill）、分配计算资源并监控全局节点状态；
2. **`slurmd`（节点计算守护进程）**：运行在每个具体的物理 GPU 计算节点上。接收来自 `slurmctld` 的调度指令，负责拉起训练进程、通过 Linux cgroups 强制执行 CPU/显存资源配额，并定期向 Master 上报节点心跳；
3. **`slurmdbd`（数据库记账守护进程）**：可选服务，将历史作业运行记录、资源消耗与用户项目配额持久化至 MySQL/MariaDB 数据库。

用户在登录节点（Login Node）通过 `sbatch` 提交作业脚本给 `slurmctld`，调度器在资源就绪后通知对应的各台 `slurmd` 节点拉起进程，各 GPU 进程通过 NCCL 建立高速互联并执行分布式计算。

---

## 本地单机模拟多节点集群环境（测试与调试）

在将任务提交至真实生产集群前，在单台多卡机器上通过运行多个 `slurmd` 实例模拟多节点跨机集群，是开发调试分布式脚本极具性价比的方式。

![单台多卡物理机模拟多节点虚拟集群架构](img/virtual_node_setup.png){#fig:virtual-node-setup .block width=90% align=center}

如 @fig:virtual-node-setup 所示，我们在单台 8 卡物理机上启动两个独立的 `slurmd` 进程（分别监听 17016 和 17017 端口）：
- 虚拟节点 **node6**：绑定物理 GPU 6；
- 虚拟节点 **node7**：绑定物理 GPU 7。

在 `slurm.conf` 中配置虚拟节点：
```bash
NodeName=node6 NodeHostname=$HOSTNAME Port=17016 CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
NodeName=node7 NodeHostname=$HOSTNAME Port=17017 CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
```
在 `gres.conf` 中映射物理设备文件：
```bash
NodeName=node6 Name=gpu File=/dev/nvidia6
NodeName=node7 Name=gpu File=/dev/nvidia7
```

通过执行 `sinfo` 与 `srun` 即可验证虚拟跨机环境：
```bash
# 查看节点状态
sinfo
# 验证跨节点并发拉起
srun -N 2 hostname
```

---

## 提交与管理分布式训练作业

### 1. 交互式调试作业：`srun`

用于快速排查代码 Bug 与环境依赖：
```bash
# 申请 2 个节点，每节点 1 卡，即刻交互式执行
srun -N 2 --gres=gpu:1 --cpus-per-task=4 python code/train.py
```

### 2. 生产级批处理作业：`sbatch`

将包含资源声明指令与执行逻辑的 Shell 脚本提交至后台排队：

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

# 1. 动态解析主节点主机名与端口
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "全局总进程数: $WORLD_SIZE, 当前全局 Rank: $RANK, 本地 Local Rank: $LOCAL_RANK"

# 2. 启动训练主程序
srun python code/train_ddp.py
```

提交与监控命令：
```bash
sbatch train.sh           # 提交作业并获取 Job ID
squeue -u $USER          # 查看当前用户作业排队/运行状态
scontrol show job <ID>   # 查看指定作业的详细运行节点与资源分配
scancel <ID>             # 终止取消指定作业
```

![SLURM 作业生命周期状态流转图](img/job_lifecycle.png){#fig:job-lifecycle .block width=90% align=center}

如 @fig:job-lifecycle 所示，作业从提交后处于 **PENDING** 排队状态，资源满足后转为 **RUNNING** 运行状态，正常退出转为 **COMPLETED**，遭遇异常或超时则转为 **FAILED / TIMEOUT**。

---

## SLURM 环境变量与 PyTorch 分布式完美映射

![SLURM 环境变量向 PyTorch 分布式环境的无缝映射](img/slurm_env_vars_mapping.png){#fig:slurm-env-vars .block width=85% align=center}

如 @fig:slurm-env-vars 所示，SLURM 注入的环境变量可 1:1 映射为 PyTorch 标准通信变量：

| SLURM 环境变量 | PyTorch 对应变量 | 物理含义 |
|:---|:---|:---|
| `SLURM_PROCID` | `RANK` | 当前进程在全局所有卡中的唯一逻辑编号（$0 \sim \text{WorldSize}-1$） |
| `SLURM_LOCALID` | `LOCAL_RANK` | 当前进程在所在物理机内部的局部 GPU 编号（$0 \sim \text{NPROC}-1$） |
| `SLURM_NTASKS` | `WORLD_SIZE` | 全集群参与训练的总 GPU 进程总数 |
| `SLURM_JOB_NODELIST` | `MASTER_ADDR` | 经 `scontrol show hostnames` 解析出的 Rank 0 主节点通信 IP |
| `SLURM_CPUS_PER_TASK`| `num_workers` | 分配给该进程的 CPU 核心数，直接用于指导 DataLoader 并发数 |

---

## 四大主流分布式框架的 SLURM 启动实战 {#sec:slurm-frameworks}

![SLURM 多机多卡分布式训练整体调度与拓扑拉起流程](img/multi_node_training.png){#fig:multi-node-training .block width=90% align=center}

### 1. PyTorch 原生 DDP 作业配置 {#sec:slurm-ddp-example}

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))

# 每个节点通过 srun 调起 torchrun，单机拉起 4 个进程
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    code/train_ddp.py
```

### 2. PyTorch FSDP 全状态分片作业配置 {#sec:slurm-fsdp-example}

FSDP 在启动逻辑上与 DDP 完全一致，通过 `torchrun` 处理进程拓扑，Python 脚本内部通过 `FullyShardedDataParallel` 或 FSDP2 的 `fully_shard()` 包装模型层。

```bash
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    code/train_fsdp.py
```

### 3. DeepSpeed ZeRO-3 异构卸载作业配置 {#sec:slurm-deepspeed-example}

DeepSpeed 无需 `torchrun`，其内部 `deepspeed.init_distributed()` 会直接读取环境变量：

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker,lo

srun --chdir="$SLURM_SUBMIT_DIR" --label \
    bash -c "
        export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        python train.py --deepspeed --deepspeed_config ds_zero3_offload.json
    "
```

### 4. NVIDIA Megatron-LM 生产级千亿训练作业配置 {#sec:slurm-megatron-example}

```bash
#!/bin/bash
#SBATCH --job-name=megatron-gpt
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_DISABLE=0

# 配置 3D 并行维度：32 卡 = TP(4) × PP(2) × DP(4)
TP_SIZE=4; PP_SIZE=2; CP_SIZE=1

srun --label \
    bash -c "
        torchrun --nproc_per_node=8 --nnodes=\$SLURM_JOB_NUM_NODES \
            --node_rank=\$SLURM_NODEID --master_addr=\"$MASTER_ADDR\" \
            --master_port=\"$MASTER_PORT\" pretrain_gpt.py \
            --use-mcore-models --num-layers 32 --hidden-size 4096 \
            --tensor-model-parallel-size $TP_SIZE \
            --pipeline-model-parallel-size $PP_SIZE \
            --micro-batch-size 1 --global-batch-size 128 \
            --use-distributed-optimizer --bf16
    "
```

---

## 生产级高级特性与故障容灾机制

### 1. Job Array：批量网格搜索

通过 `#SBATCH --array=0-11` 一键派发 12 个独立的超参数探索实验，利用 `SLURM_ARRAY_TASK_ID` 索引不同的学习率与批大小。

### 2. 交互式调试沙箱：`salloc`

```bash
# 申请 2 节点 8 卡资源独占 1 小时
salloc -N 2 --gres=gpu:4 --time=1:00:00

# 在分配的专属资源中直接运行命令
srun nvidia-smi
srun python debug_script.py
exit  # 释放资源
```

### 3. 作业依赖链编排（Job Dependencies）

```bash
# 提交第一阶段数据预处理并获取 ID
JOB1=$(sbatch --parsable preprocess.sh)

# 仅在阶段一成功结束后才启动分布式训练
sbatch --dependency=afterok:$JOB1 train.sh
```

### 4. 抢占与超时安全存盘（Graceful Checkpointing）

针对抢占式队列（Preemptible Queue）或严苛的运行时间限制，配置预警信号：
```bash
#SBATCH --signal=SIGUSR1@90  # 在作业被强制 kill 前 90 秒发送 SIGUSR1 信号
```
在 Python 训练代码中捕获 `SIGUSR1`，紧急执行 Checkpoint 落盘并调用 `sbatch` 重新将自身排队，实现万无一失的无缝断点续训。

---

## 本章小结

本章打通了分布式 AI 从单机代码走向大型计算集群的工程闭环：
- 剖析了现代 HPC 与 AI 训练集群的基础设施构成与 SLURM 守护进程体系；
- 掌握了在单台机器上配置多个 `slurmd` 虚拟节点调试分布式跨机脚本的方法；
- 深入推导了 `SLURM_PROCID`、`SLURM_LOCALID` 到 PyTorch 分布式环境变量的映射；
- 给出了 DDP、FSDP、DeepSpeed 以及 Megatron-LM 在 SLURM 集群上的标准启动模板；
- 掌握了 Job Array 网格搜索、`salloc` 交互式沙箱以及基于 `SIGUSR1` 的抢占安全断点续训机制。

至此，全书的**分布式训练核心技术与集群运维调度**已全面建立。在大模型训练完成后，面对日活千万级的高并发流量，如何将模型部署为生产级服务？在下一章中，我们将进入 **生产级 LLM 推理与服务化技术栈**，全面剖析高可用网关、负载均衡与性能监控监控中枢。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
