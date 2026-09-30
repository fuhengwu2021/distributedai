# 第8章：使用 SLURM 运行分布式训练 {-}

*用 SLURM 管理 GPU 资源并协调多节点作业*

> 在一屋子顶尖软件设计师中，如果有两个人意见一致，那就是多数派。
- Bill Curtis

**Code Summary**

- `sbatch`：提交批处理作业的 SLURM 命令
- `srun`：运行交互式作业的 SLURM 命令
- `squeue`：查看作业队列的 SLURM 命令
- `scancel`：取消作业的 SLURM 命令
- `sinfo`：查看集群信息的 SLURM 命令
- `sacct`：查看作业记账信息的 SLURM 命令
- `scontrol`：用于集群控制和配置的 SLURM 命令
- `torchrun`：与 SLURM 兼容的 PyTorch 启动器
- `SLURM_PROCID`：进程 ID 的 SLURM 环境变量
- `SLURM_NTASKS`：任务数量的 SLURM 环境变量

## HPC 和 AI 训练集群简介

前面的章节涵盖了分布式训练的理论和实现——用于梯度同步的 DDP、用于内存效率的 FSDP、用于 ZeRO 优化的 DeepSpeed，以及用于模型并行的 Megatron。但理解这些框架只是挑战的一半。另一半是在真实硬件上实际运行它们：跨节点分配 GPU、协调进程、管理作业队列，以及处理在规模上不可避免发生的故障。

现代 AI 训练发生在集群上——将它们的计算资源汇集在一起的互连机器集合。一个典型的 GPU 集群由计算节点（每个包含多块 GPU、CPU 和内存）、连接节点的高速互连（InfiniBand 或高带宽以太网）、从所有节点可访问的共享存储，以及管理作业提交和调度的头节点组成。集群可能有几十到数千个节点，代表价值数百万美元的硬件，必须在许多用户和项目之间高效共享。

这种共享性质造成一个根本挑战：你如何公平地分配资源、确保作业不互相干扰，以及最大化昂贵硬件的利用率？在计算的早期，用户会在共享机器上预约时间段。现代集群使用作业调度器——接受作业请求、根据优先级和资源可用性排队它们、在资源可用时分配资源、监控运行的作业，以及在作业完成或失败时清理的软件。

HPC 生态系统中存在几个作业调度器。PBS（可移植批处理系统）及其衍生品（Torque、PBS Pro）在传统 HPC 中占主导。[^pbs] LSF（负载共享设施）在企业环境中流行。[^lsf] HTCondor 在需要将许多独立作业分布到可用机器上的高吞吐量计算工作负载中表现卓越。[^htcondor] Kubernetes 已成为云原生工作负载的标准。[^k8s] 但对于运行 AI 训练工作负载的 GPU 集群，SLURM 已成为主导选择，被大多数学术机构、国家实验室使用，并越来越多地被提供 HPC 实例的云服务商使用。

[^pbs]: PBS Professional 现由 Altair 维护。商业版见 https://www.altair.com/pbs-professional/，开源 OpenPBS 见 https://github.com/openpbs/openpbs。

[^lsf]: IBM Spectrum LSF 在金融服务和生命科学中广泛使用。见 https://www.ibm.com/products/hpc-workload-management。

[^htcondor]: HTCondor 由 UW-Madison 的高吞吐量计算中心开发。官方网站见 https://htcondor.org/，UT Austin 的示例部署见 https://www.cs.utexas.edu/facilities/documentation/condor。

[^k8s]: Kubernetes 可以用像 Volcano（https://volcano.sh/）这样的项目或带 GPU 调度插件的 Kubernetes Job API 扩展用于 HPC 工作负载。

### 为什么 AI 训练用 SLURM？

SLURM（Simple Linux Utility for Resource Management）于 2002 年作为 Lawrence Livermore 国家实验室的一个项目开始，已演变成一个支持数百万核心集群的复杂资源管理器。[^slurm] 几个因素使它特别适合 AI 训练工作负载。

[^slurm]: SLURM 由 SchedMD 维护。官方文档和下载见 https://slurm.schedmd.com/。

首先，SLURM 通过它的通用资源（GRES）系统对 GPU 有一等支持。你可以请求特定的 GPU 类型（`--gres=gpu:a100:4`），SLURM 确保独占分配——当你的作业运行时，没有其他作业可以访问你的 GPU。这对训练至关重要，共享访问的 GPU 内存碎片化会导致内存不足错误。

其次，SLURM 与 PyTorch 的分布式训练无缝集成。当 SLURM 跨多个节点启动你的作业时，它自动设置直接映射到分布式训练概念的环境变量：`SLURM_PROCID` 成为全局 rank，`SLURM_LOCALID` 成为节点内的本地 rank，`SLURM_NTASKS` 成为 world size，`SLURM_JOB_NODELIST` 提供建立通信所需的节点列表。PyTorch 的 `torchrun` 启动器读取这些变量并自动初始化进程组。

第三，SLURM 提供长时间运行训练作业必不可少的稳健作业管理特性：用于超参数扫描的作业数组、用于多阶段流水线的作业依赖、处理时间限制的抢占和检查点支持，以及跟踪资源使用的详细记账。当训练运行需要几天或几周时，这些特性变得必不可少。

第四，SLURM 高效扩展。无论你在 4 节点的实验室集群还是 10,000 节点的超级计算机上运行，相同的命令和脚本都工作。这种可移植性意味着技能在机构和云服务商之间迁移。

一旦理清了环境变量的映射关系，SLURM 与 PyTorch 分布式训练的对接便会极其自然顺畅。你的算法训练代码无需感知自身究竟是运行在配备 2 张 GPU 的本地开发机上，还是运行在跨越 256 张 GPU 的大规模算力集群中——底层的基础设施抽象已为你抹平了所有物理差异。你只需通过作业脚本声明算力资源配额（如 `--nodes=4 --gres=gpu:8`），SLURM 就会自动完成资源锁定与环境变量注入，而训练脚本则直接利用这些环境变量无缝初始化分布式通信拓扑。


### SLURM 架构：简要概述

在深入使用之前，在高层次理解 SLURM 的架构会有帮助。SLURM 由几个协同工作以管理集群的守护进程组成：

- **slurmctld**（控制器守护进程）：运行在头节点上的中央大脑。它管理作业队列、做调度决策、分配资源，并监控作业状态。在生产集群中，slurmctld 通常在带备份控制器的高可用配置中运行。

- **slurmd**（计算守护进程）：运行在每个计算节点上。它从 slurmctld 接收作业分配、启动和监控任务、向控制器报告节点状态，并使用 Linux cgroups 强制执行资源限制。

- **slurmdbd**（数据库守护进程）：可选但在生产中常见。它将记账数据（作业历史、资源使用、用户/项目分配）存储在 MySQL 或 MariaDB 数据库中，实现公平共享调度和使用报告。

![SLURM 架构：slurmctld、slurmd 和 slurmdbd 守护进程。](img/slurm_architecture_zh.png){#fig:slurm-architecture .block width=95% align=center}

当你用 `sbatch` 提交一个作业时，请求去 slurmctld，它排队它并最终根据调度策略（优先级、公平共享、回填）分配资源。一旦资源可用，slurmctld 通知相关的 slurmd 守护进程，它们生成你作业的进程并设置你的训练脚本读取的环境变量。

SLURM 的调度算法、分区配置、QOS（服务质量）策略和插件架构是集群管理员为他们的特定工作负载调整的丰富主题。本章专注于 **面向用户的方面**——如何提交作业、请求资源，以及与分布式训练框架集成——而非集群管理。关于 SLURM 内部和管理，官方文档[^slurm] 和 SchedMD 培训材料提供全面覆盖。

图~\ref{fig:slurm-architecture} 展示了整体架构。用户通过像 `sbatch`（提交批处理作业）、`srun`（运行交互式命令）和 `squeue`（查询作业状态）这样的命令与头节点交互。提交新作业或启动交互式分配经过 slurmctld；一旦你持有一个分配，其中的 `srun` 直接与分配节点上的本地 slurmd 守护进程对话。控制器维护作业队列、跟踪节点状态并做调度决策——当资源可用时，它通知合适的 slurmd 守护进程启动作业进程。每个 slurmd 管理它的本地节点：生成任务、通过 cgroups 强制执行资源限制、监控进程健康，并向控制器报告状态。可选的 slurmdbd 守护进程将记账数据（作业历史、资源消耗、用户分配）持久化到数据库，实现随时间平衡用户和项目间资源使用的公平共享调度策略。


## 为多 GPU 训练设置 SLURM

大多数用户不需要自己安装 SLURM——集群管理员处理那个。但理解配置有助于在作业行为不如预期时调试问题，而设置本地测试环境对在提交到生产集群之前开发和调试分布式训练脚本非常宝贵。

### 模拟多节点集群

对于开发和测试，你可以通过运行多个 SLURM 计算守护进程（slurmd）在单台物理机器上模拟多节点集群，每个映射到不同的 GPU。这让你在没有实际集群访问的情况下测试多节点分布式训练代码。

关键洞见是 SLURM 的架构将"节点"的概念与物理机器分离。每个 slurmd 守护进程代表一个节点，通过在不同端口上运行多个守护进程，你可以创建 SLURM 视为独立机器的虚拟节点。`slurm.conf` 中的配置定义这些虚拟节点：

```bash
# Enable multiple slurmd support
# Use $HOSTNAME or $(hostname) to get the actual hostname
NodeName=node6 NodeHostname=$HOSTNAME Port=17016 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
NodeName=node7 NodeHostname=$HOSTNAME Port=17017 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN
```

每个虚拟节点在不同端口（17016、17017）上监听，但共享相同的主机名。`Gres=gpu:1` 声明告诉 SLURM 每个节点有一块可用的 GPU。相应的 `gres.conf` 文件将这些虚拟 GPU 映射到物理设备。下面的路径（`/dev/nvidiaN`）是 NVIDIA 特定的；其他供应商使用不同的设备文件——为你的硬件见 [SLURM GRES 文档](https://slurm.schedmd.com/gres.html)。在这个例子中，我们使用一台 8-GPU 机器，将最后两块 GPU（索引 6 和 7）专用于我们的虚拟集群：

```bash
NodeName=node6 Name=gpu File=/dev/nvidia6
NodeName=node7 Name=gpu File=/dev/nvidia7
```

这个映射确保当一个作业在 node6 上请求 `--gres=gpu:1` 时，SLURM 将 `CUDA_VISIBLE_DEVICES` 设置为只向那个作业暴露 `/dev/nvidia6`。你可以调整 GPU 索引以使用你机器上任何可用的 GPU——例如，如果你想用前两块 GPU，则用 `/dev/nvidia0` 和 `/dev/nvidia1`。在真实集群上，每个物理节点会有它自己的 `gres.conf` 条目映射到它的本地 GPU。

![单台物理机器上的虚拟多节点集群。](img/virtual_node_setup_zh.png){#fig:virtual-node-setup .block width=90% align=center}

图~\ref{fig:virtual-node-setup} 显示虚拟节点设置。两个 slurmd 守护进程（node6 和 node7）在同一台物理机器上运行，但在不同端口上监听。每个虚拟节点通过 `gres.conf` 映射到特定的 GPU，允许你在本地测试多节点分布式训练代码。

### 快速设置和验证

提供的设置脚本自动化配置过程——创建 SLURM 配置目录、为虚拟节点写 `slurm.conf` 和 `gres.conf`、初始化状态目录，以及启动 `slurmctld` 加上每个虚拟节点一个 `slurmd`：

```bash
cd code
bash slurm_setup.sh
```

一旦守护进程运行，验证集群正确工作。首先，将 SLURM 添加到你的 PATH（用你的安装前缀替换 `$SLURM_PREFIX`，通常是 `/opt/slurm` 或 `$HOME/slurm`）：

```bash
export PATH=$SLURM_PREFIX/bin:$PATH
```

`sinfo` 命令显示集群的分区和节点状态：

```bash
sinfo
# Example output:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      2   idle node[6-7]
```

示例输出显示一个名为 "gpu" 的分区（星号表示它是默认的）带 2 个空闲节点。关于更详细的节点信息，使用 `scontrol show nodes`。要验证作业实际可以运行，提交一个简单测试：

```bash
srun -N 1 hostname
srun -N 2 hostname
```

第一个命令在一个节点上运行 `hostname`；第二个同时在两个节点上运行它。如果两个命令都成功完成并打印预期的节点名，你的 SLURM 设置就为分布式训练实验准备好了。

## 提交分布式训练作业

SLURM 配置和验证后，下一步是提交实际的训练作业。SLURM 提供两种主要的提交方法：交互式执行的 `srun` 和批处理提交的 `sbatch`。理解何时使用每种——以及如何构造你的作业脚本——对高效使用集群至关重要。

### 用 `srun` 进行交互式执行

对于快速测试和调试，`srun` 在分配的资源上立即执行命令。你指定你需要什么，SLURM 要么立即运行你的命令（如果资源可用），要么等到它们空闲。`code/train.py` 脚本是一个自包含的示例，它自动检测 SLURM 环境变量、初始化分布式训练，并在合成数据（`SimpleDataset`）上运行一个简单的 3 层神经网络（`SimpleModel`）——对验证你的集群设置有用：

```bash
# Two nodes, 1 GPU each
srun -N 2 --gres=gpu:1 --cpus-per-task=4 python code/train.py
```

标志告诉 SLURM 你的作业到底需要什么资源：`-N 2` 请求两个节点，`--gres=gpu:1` 每节点请求一块 GPU，`--cpus-per-task=4` 为数据加载和预处理分配四个 CPU 核心——适合快速冒烟测试的适度数量。本章后面的生产批处理脚本在 DataLoader worker 和预处理需要余量时请求更多核心（如 28）。SLURM 找到匹配这些需求的节点、设置环境，并同时跨所有分配的资源运行你的命令。

交互式执行对开发方便，但它占用你的终端并要求你保持连接。对于可能需要几小时或几天的生产训练运行，批处理提交是标准方法。

### 用 `sbatch` 进行批处理提交

批处理作业在带特殊 `#SBATCH` 指令的 shell 脚本中定义，这些指令指定资源需求。你将脚本提交到 SLURM 的队列，它在资源可用时运行——即使你已经登出。这是一个分布式训练的典型批处理脚本（可运行版本在 `code/train_ddp.sh`）：

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
export MASTER_PORT=$((29500 + RANDOM % 1000))
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE, Rank: $RANK, Local rank: $LOCAL_RANK"

# Run training
srun python code/train_ddp.py
```

顶部的 `#SBATCH` 指令定义作业的资源包络：2 个节点、每个 1 块 GPU、每任务 28 个 CPU 核心、200GB 内存和 24 小时时间限制。输出文件名中的 `%j` 展开为作业 ID，所以每次运行得到唯一的日志文件。这些指令对 shell 是注释，但在执行前被 `sbatch` 解析。

中间部分设置分布式训练环境。关键部分是 `MASTER_ADDR`——rank 0 运行且所有其他 rank 连接以建立进程组的主机名。`scontrol show hostnames` 命令将 SLURM 的压缩节点列表格式（如 `node[6-7]`）转换为单个主机名，`head -n 1` 提取第一个作为主节点。相应的 Python 训练脚本在 `code/train_ddp.py`。

用 `sbatch` 提交作业，SLURM 立即返回一个作业 ID：

```bash
sbatch code/train_ddp.sh
```

然后你可以通过队列监控你作业的进度：

```bash
squeue                    # List all jobs
squeue -u $USER          # List your jobs
scontrol show job <job_id>  # Detailed job info
```

![SLURM 作业状态生命周期](img/job_lifecycle_zh.png){#fig:job-lifecycle .block width=90% align=center}

图~\ref{fig:job-lifecycle} 显示作业状态转换。作业在等待资源时以 PENDING 开始，在分配时移到 RUNNING，然后在清理期间 COMPLETING，最后在成功时 COMPLETED。作业也可以转换到 FAILED（出错时）、CANCELLED（用户干预）或 TIMEOUT（超过时间限制）。使用 `squeue` 查看当前状态，`sacct` 查看历史作业信息。

### 理解 SLURM 环境变量

当 SLURM 启动你的作业时，它自动填充你的训练脚本可以读取以配置分布式通信的环境变量。理解这些变量对写在不同集群配置间工作的可移植代码至关重要。

在作业级别，SLURM 提供描述整体分配的变量。`SLURM_JOB_ID` 给出一个对命名日志文件和检查点有用的唯一标识符。`SLURM_JOB_NAME` 包含你用 `--job-name` 指定的名字（或默认为脚本名）。`SLURM_JOB_NODELIST` 以压缩格式列出分配的节点——例如 `node[6-7]` 或 `gpu-node-[001-004]`——而 `SLURM_JOB_NUM_NODES` 给出计数。`SLURM_SUBMIT_DIR` 记录你提交作业的目录，对相对于你提交位置定位配置文件或数据有用。

对于分布式训练，进程级变量最关键。SLURM 启动的每个任务接收 `SLURM_PROCID`，一个从 0 到 NTASKS-1 的全局唯一 rank，在作业中所有进程中标识这个进程。`SLURM_LOCALID` 给出当前节点内的本地 rank（0 到每节点任务数-1），你通常用它进行 GPU 绑定——带 `SLURM_LOCALID=0` 的进程使用那个节点上的 GPU 0，以此类推。`SLURM_NODEID` 标识这个进程运行在哪个节点上（0 到 `NUM_NODES-1`），`SLURM_NTASKS` 提供总任务计数，等价于分布式训练术语中的 world size。`SLURM_TASKS_PER_NODE` 指示每个节点运行多少任务，尽管这在异构分配中可能跨节点变化。

资源相关变量帮助你调优性能。`SLURM_CPUS_PER_TASK` 告诉你每任务有多少 CPU 核心可用——对在你的 DataLoader 中设置 `num_workers` 有用。`SLURM_GPUS_ON_NODE` 报告当前节点上的 GPU 计数，`SLURM_MEM_PER_NODE` 以 MB 给出内存分配。当你用 `--gres=gpu:N` 请求 GPU 时，SLURM 的 GRES 插件设置 `CUDA_VISIBLE_DEVICES`，使每个任务只看到它分配的 GPU——在那个任务内通常重新映射到 `cuda:0`、`cuda:1`……。你通常用 `LOCAL_RANK` 绑定而非从全局 rank 猜测；这防止多个作业共享节点时的冲突。

对于建立网络通信，`SLURM_LAUNCH_NODE_IPADDR` 提供启动节点的 IP 地址，`SLURM_STEP_NODELIST` 在分配内使用 `srun` 时列出参与当前作业步骤的节点。

从 SLURM 到 PyTorch 分布式训练的映射很直接：`SLURM_PROCID` 成为 `RANK`，`SLURM_LOCALID` 成为 `LOCAL_RANK`，`SLURM_NTASKS` 成为 `WORLD_SIZE`。`MASTER_ADDR`——rank 0 监听连接的地址——通常通过提取第一个节点的主机名从 `SLURM_JOB_NODELIST` 派生。一个典型的设置脚本导出这些转换：

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
```

一旦这些变量被设置，`torchrun` 或 `torch.distributed.init_process_group(init_method='env://')` 读取它们并自动配置进程组。在共享集群上，像 29500 这样的固定端口可能与另一个作业冲突并以晦涩的 NCCL 错误失败；如上随机化 `MASTER_PORT` 避免大多数冲突。这种抽象很强大：你的训练代码不需要知道它是在 SLURM 下运行、被单台机器上的 `torchrun` 启动，还是被云服务商编排。相同的脚本到处都工作，因为它依赖标准环境变量而非 SLURM 特定的 API。

图~\ref{fig:slurm-env-vars} 视觉上展示这个映射。你的训练脚本可以直接读取 SLURM 变量或使用导出的 PyTorch 标准变量（`RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`）。与 SLURM 一起使用时，`torchrun` 启动器自动处理这个转换。注意 SLURM 为专门用例提供许多额外的环境变量——关于完整参考，查阅 `srun` man 页面或官方 SLURM 文档。[^slurm]

![SLURM 到 PyTorch 环境变量映射。](img/slurm_env_vars_mapping_zh.png){#fig:slurm-env-vars .block width=85% align=center}

## 用 SLURM 启动分布式训练框架 {#sec:slurm-frameworks}

涵盖了 SLURM 基础，我们现在转向实际问题：你如何在 SLURM 集群上启动不同的分布式训练框架？每个框架有它自己的启动器和初始化模式，但它们都依赖我们上面讨论的相同 SLURM 环境变量。

![用 SLURM 的多节点分布式训练。](img/multi_node_training_zh.png){#fig:multi-node-training .block width=90% align=center}

图~\ref{fig:multi-node-training} 展示了共同模式：SLURM 分配节点和 GPU、跨集群启动进程，并设置每个框架读取以建立分布式通信的环境变量。区别在于每个框架如何包装这个过程。

下表总结了 SLURM 集成中的关键区别：

| 框架 | 启动器 | 分布式初始化 | SLURM 环境处理 |
|-----------|----------|------------------|-----------------------|
| DDP | `torchrun` 或 `srun` | 手动 | 导出到 `RANK`、`WORLD_SIZE` 等 |
| FSDP | `torchrun` | 手动 | 与 DDP 相同 |
| DeepSpeed | `python` | 自动 | 直接读取 SLURM 变量 |
| Megatron-LM | `torchrun` | 自动 | 直接读取 SLURM 变量 |

"手动"初始化意味着你在训练脚本中显式调用 `dist.init_process_group()` 并在你的 SLURM 批处理脚本中处理环境变量设置。"自动"意味着框架在内部处理分布式初始化——DeepSpeed 通过 `deepspeed.init_distributed()`，Megatron-LM 通过它自己的启动器基础设施——读取 SLURM 环境变量而无需显式设置代码。

关于每个框架概念和内部的详细解释，参考前面的章节：DDP 在第~\ref{chap:distributed-training-with-pytorch-ddp}章，FSDP 在第~\ref{chap:scaling-with-fully-sharded-data-parallel-fsdp}章，DeepSpeed 在第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章。这里我们专门专注于 SLURM 启动模式并提供完整的可工作示例。

### 用 SLURM 的 DDP {#sec:slurm-ddp-example}

如第~\ref{chap:distributed-training-with-pytorch-ddp}章所涵盖，PyTorch DDP 在每块 GPU 上复制整个模型、跨进程分布数据，并在反向传播期间通过 AllReduce 同步梯度。因为每块 GPU 持有模型的完整副本，DDP 在你的模型舒适地装进单块 GPU 内存时工作最好。这里我们专注于 SLURM 特定的启动模式。

训练脚本结构很直接（完整版本在 `code/train_ddp.py`）：

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    device = torch.device(f'cuda:{local_rank}')
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    for epoch in range(10):
        # ... training code ...
        if rank == 0:
            print(f"Epoch {epoch} completed")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

SLURM 批处理脚本设置环境并通过 `torchrun` 启动（完整版本在 `code/train_ddp.sh`）：

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))

srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    code/train_ddp.py
```

或者，你可以使用 SLURM 的内建 MPI 支持而不用 `torchrun`：

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

srun python code/train_ddp.py
```

这种方法要求你的 Python 代码使用 `init_method='env://'`，它从环境变量读取 `RANK`、`WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT`。

### 用 SLURM 的 FSDP {#sec:slurm-fsdp-example}

如第~\ref{chap:scaling-with-fully-sharded-data-parallel-fsdp}章所讨论，FSDP 跨 GPU 分片模型参数、梯度和优化器状态，大幅减少大型模型的每 GPU 内存需求。SLURM 启动模式与 DDP 相同——你以相同方式使用 `torchrun`。区别在于 Python 代码中你用 `FullyShardedDataParallel` 而非 `DistributedDataParallel` 包装模型。下面的示例使用 FSDP1 的 `CPUOffload` 辅助器；在带 FSDP2 的 PyTorch 2.4+ 上，如第~\ref{chap:scaling-with-fully-sharded-data-parallel-fsdp}章那样换入 `fully_shard()` 和 `CPUOffloadPolicy`——只有 Python 包装器改变，而非 SLURM 脚本。

训练脚本结构（完整版本在 `code/train_fsdp.py`）：

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

SLURM 批处理脚本（完整版本在 `code/train_fsdp.sh`）：

```bash
#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))

srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    code/train_fsdp.py
```

### 用 SLURM 的 DeepSpeed {#sec:slurm-deepspeed-example}

如第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章所涵盖，DeepSpeed 的 ZeRO 优化器提供三个阶段的内存优化：ZeRO-1 分区优化器状态，ZeRO-2 添加梯度分区，ZeRO-3 进一步分区模型参数本身。ZeRO-3 概念上类似于 FSDP——两者都跨 GPU 分片参数——但 DeepSpeed 提供像 CPU 和 NVMe 卸载这样可以将内存边界推得更远的额外特性。这里的示例使用带 CPU 卸载的 ZeRO-3，但如果你不需要完全参数分片，你可以通过在配置文件中将 `"stage": 3` 改为 `1` 或 `2` 轻松切换到 ZeRO-1 或 ZeRO-2。

与你显式调用 `dist.init_process_group()` 并用 `torchrun` 作为启动器的 DDP 和 FSDP 不同，DeepSpeed 采取不同的方法。它通过 `deepspeed.init_distributed()` 在内部处理分布式初始化，直接读取 SLURM 环境变量而无需单独的启动器。这种设计简化了用户体验——你只需在适当的环境变量设置的情况下运行 `python train.py`，DeepSpeed 自动弄清分布式拓扑。

训练脚本结构反映这种简单性（完整版本在 `code/deepspeed/train.py`）：

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

注意脚本不导入 `torch.distributed` 或调用 `init_process_group()`——DeepSpeed 在内部处理所有那些。`deepspeed.initialize()` 调用返回一个用 ZeRO 优化包装你模型的 `model_engine`，你用 `model_engine.backward()` 和 `model_engine.step()` 而非标准的 PyTorch 优化器方法。

配置文件控制 ZeRO 行为（`code/deepspeed/ds_zero3_offload.json`）：

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

`stage: 3` 设置启用完全参数分片，`offload_param` 和 `offload_optimizer` 部分在模型超过总 GPU 内存时配置 CPU 卸载。`pin_memory: true` 选项使用固定（页锁定）CPU 内存以加快 CPU-GPU 传输。将这个 JSON 视为一个生产规模模板——先用 ZeRO 阶段 1 或 2 验证 NCCL 和 SLURM 接线，然后在作业干净运行后启用阶段 3 和 CPU 卸载。

SLURM 批处理脚本比 DDP 需要更多设置，因为我们需要手动导出 DeepSpeed 期待的环境变量（`code/deepspeed/run.slurm`）：

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

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
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

脚本在作业级别设置 `MASTER_ADDR`、`MASTER_PORT` 和 `WORLD_SIZE`——用 `scontrol` 派生主节点主机名，而非 `127.0.0.1`，否则每个节点形成它自己的单节点进程组，梯度从不跨节点同步。那个故障可以在本章前面的虚拟两节点设置上保持隐藏，那里两个 slurmd 守护进程共享一台物理主机。`srun` 然后在每个节点上启动一个 bash 子 shell；其中我们从 SLURM 的任务环境导出每进程变量（`CUDA_VISIBLE_DEVICES`、`LOCAL_RANK`、`RANK`）。`--label` 标志用任务 ID 为每行输出加前缀，这在调试多节点问题时有帮助。

在 SLURM 集群上运行 DeepSpeed 时的几个实用考量。DeepSpeed 需要 `LOCAL_RANK` 环境变量，你必须从 `SLURM_LOCALID` 显式导出它——与自动设置这个的 `torchrun` 不同。如果你用虚拟节点进行测试（如前所述），记得适当地将节点名映射到 GPU 索引——例如，`node6` 应使用 GPU 6。IPv6 可能在某些集群上导致连接问题；将 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 设置为排除有问题的接口（如 `^docker,lo`）常常解决这个。最后，记得用你自己的设置替换脚本中的 conda 路径和环境名。

### 用 SLURM 的 Megatron-LM {#sec:slurm-megatron-example}

如第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章所介绍，Megatron-LM 提供 NVIDIA 的生产级框架，结合张量并行、流水线并行、序列/上下文并行和数据并行——全都可在单个训练运行中组合。这种多维并行对训练没有单一并行策略足够的最大语言模型至关重要。

在深入 SLURM 脚本之前，有一个重要的安装考量（也在第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章涵盖）。与 PyTorch 的内建 DDP 和 FSDP 不同，Megatron-LM 需要从源码安装以获得完整的训练基础设施。PyPI 包 `megatron-core` 只包括 `megatron.core`（模型构建块），但像 `pretrain_gpt.py` 这样的训练脚本需要 `megatron.training`，它只在你从 GitHub 仓库安装时可用：

```bash
conda activate research  # Replace with your environment name
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation '.[mlm,dev]'
```

你还需要将训练脚本（`pretrain_gpt.py`、`gpt_builders.py`、`model_provider.py`）复制到你的工作目录，因为这些不作为包的一部分安装。

有了先决条件，让我们看看 SLURM 批处理脚本（`code/megatron/run.slurm`）：

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

脚本结构遵循与 DeepSpeed 相同的模式：设置作业级环境变量，然后用 `srun` 启动一个 bash 子 shell，它设置每进程变量并调用 `torchrun`。模型配置变量（`NUM_LAYERS`、`HIDDEN_SIZE` 等）定义一个 8B 参数 GPT 模型，而并行变量（`TP_SIZE`、`PP_SIZE`、`CP_SIZE`）控制模型如何分布——根据你的硬件和模型大小调整这些。示例用模拟数据（`--mock-data`）进行演示；对于真实训练，你会提供实际的数据路径和适当的分词器。提交前记得用你自己的设置替换脚本中的 conda 路径和环境名：

```bash
cd code/megatron
sbatch run.slurm
```

用 Megatron-LM 训练时你会注意到的一件事是检查点文件可能相当大。对于一个 8B 参数模型，你可能会看到像这样的检查点目录：

```
code/megatron/checkpoints/gpt_8b/iter_0000010/
27G     __0_0.distcp
27G     __0_1.distcp
27G     __1_0.distcp
27G     __1_1.distcp
24K     common.pt
4.0K    metadata.json
```

为什么这么大？算术很直接：bf16 中的模型参数消耗 8.03B × 2 字节 = 16.06 GB，而 fp32 中的 Adam 优化器状态需要 8.03B × 8 字节 = 64.24 GB（动量和方差各 4 字节）。那已经理论上约 80 GB，约 108 GB 的实际大小包括来自分布式优化器分片、文件格式元数据和高效并行 I/O 对齐填充的额外开销。每个 rank 保存它自己的分片（`__0_0.distcp`、`__0_1.distcp` 等）以实现跨集群的并行保存/加载操作。

为管理检查点存储，考虑使用 `--save-interval` 控制检查点保存的频率、实现检查点轮换以只保留最近的检查点，以及使用能处理 I/O 负载的分布式文件系统。

另一个实用考量是检查点格式转换。Megatron-LM 以需要 Megatron-LM 加载的分布式格式（`.distcp` 文件）保存检查点。如果你想将你训练的模型用于像 vLLM 或 SGLang 这样的其他框架进行推理，或简单地用原生 PyTorch 加载它，你需要转换检查点。提供的转换脚本（`code/megatron/convert_megatron_checkpoint.py`）处理这个：

```bash
python code/megatron/convert_megatron_checkpoint.py \
    --checkpoint-dir code/megatron/checkpoints/gpt_8b/iter_0000010 \
    --output-dir exported_checkpoint \
    --format pytorch \
    --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
    --vocab-size 128256 --max-position-embeddings 2048 \
    --use-mcore-models --bf16
```

导出的检查点完全独立——加载它不需要 Megatron-LM：

```python
import torch
checkpoint = torch.load('exported_checkpoint/model.pt', map_location='cpu')
print(checkpoint['model_config'])
state_dict = checkpoint['model_state_dict']
```

转换后的检查点只包含模型权重（无优化器状态），使它显著更小并与任何基于 PyTorch 的推理框架兼容。关于带适当层名映射和张量重塑的生产 HuggingFace 格式转换，考虑使用 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)。

## 训练的高级 SLURM 特性

除了基本作业提交，SLURM 提供几个对严肃训练工作流变得宝贵的高级特性。本节涵盖对 AI 从业者最有用的那些。

### 用于超参数调优的作业数组

当你需要用不同超参数运行相同的训练脚本时——超参数搜索的常见场景——一个一个提交作业很快变得繁琐。SLURM 的作业数组让你提交单个生成多个独立作业的脚本，每个带一个你可以用来选择不同配置的唯一任务 ID。

`#SBATCH --array=0-9` 指令告诉 SLURM 创建 10 个作业（索引 0 到 9）。每个作业在 `SLURM_ARRAY_TASK_ID` 环境变量中接收它的索引，你可以用它计算超参数。可运行版本在 `code/train_array.sh`：

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Each array task gets different hyperparameters
LRS=(0.001 0.0001 0.00001 0.000001)
LR=${LRS[$((SLURM_ARRAY_TASK_ID % 4))]}
BATCH_SIZE=$((32 * (SLURM_ARRAY_TASK_ID / 4 + 1)))

python code/train.py --lr $LR --batch_size $BATCH_SIZE
```

这个示例创建一个 4 个学习率和 3 个批大小的网格搜索（尽管 12 个组合中只有 10 个运行）。用 `sbatch code/train_array.sh` 提交，SLURM 调度所有 10 个作业——如果资源可用它们可能并行运行，否则排队。

### 用 `salloc` 的交互式作业

虽然 `sbatch` 对生产训练运行完美，调试分布式代码常常需要交互式访问。`salloc` 命令分配资源并给你一个可以直接运行命令的 shell：

```bash
# Allocate 2 nodes, 1 GPU each, for 1 hour
salloc -N 2 --gres=gpu:1 --time=1:00:00

# Once allocated, run commands interactively
srun hostname
srun nvidia-smi
srun python code/train.py

# Release when done
exit
```

这个工作流对调试非常宝贵——你可以运行你的训练脚本、看它失败、修复代码，并立即重试而无需再次在队列中等待。只需记住你的分配有时间限制，空闲时间仍然计入你的配额。

### 作业依赖

真实的训练流水线常常涉及多个阶段：数据预处理、训练、评估、检查点转换。你不用手动监控每个作业并提交下一个，而是可以用依赖链接作业：

```bash
# Submit first job and capture its ID
JOB1=$(sbatch --parsable train_stage1.sh)

# Submit second job that starts only after first succeeds
sbatch --dependency=afterok:$JOB1 train_stage2.sh
```

`--dependency=afterok:$JOB1` 标志告诉 SLURM 保持第二个作业直到第一个成功完成。其他依赖类型包括 `afterany`（无论退出状态都运行）、`afternotok`（只在第一个失败时运行）和 `singleton`（一次只运行一个带这个名字的作业）。

### 检查点与作业恢复

长时间训练运行不可避免地遇到中断——时间限制、节点故障、被更高优先级作业抢占。稳健的检查点至关重要，SLURM 提供一个机制来优雅地处理时间限制。

`--signal=SIGUSR1@90` 指令告诉 SLURM 在时间限制到期前 90 秒向你的作业发送一个 `SIGUSR1` 信号。你的脚本可以捕获这个信号并触发检查点保存。那与 SLURM 在立即取消或抢占（`scancel`、节点排空）时发送的 `SIGTERM` 分开——在生产检查点代码中处理两者。一个完整示例在 `code/train_distributed.sh`：

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

当信号到达时，trap 处理器运行你的检查点脚本，在 SLURM 终止作业之前给训练进程时间保存状态。结合你训练脚本中的 `--resume` 标志，你可以跨多次作业提交无缝继续训练。

## 监控与调试

当训练作业跨多个节点运行几小时或几天时，有效的监控变得必不可少。你需要知道你的作业是否实际运行、资源如何被利用，以及事情出错时去哪里看。

### 作业监控

最基本的监控从 `squeue` 开始，它显示队列中作业的状态。用 `watch` 包装它给你一个实时仪表板：

```bash
# Watch job queue, refreshing every second
watch -n 1 squeue -u $USER

# Get detailed information about a specific job
scontrol show job <job_id>

# Watch a specific job's state changes
watch -n 1 scontrol show job <job_id>
```

`scontrol show job` 输出包括像分配的节点、开始时间、时间限制和当前状态这样的有用细节。对于运行的作业，你可以检查所有分配节点的 GPU 利用率：

```bash
# Check GPU usage across all nodes in your allocation
srun -N 2 nvidia-smi

# For a running batch job, attach to its step instead of SSH
srun --jobid=<job_id> nvidia-smi
```

要实时监控作业输出，在输出文件上用 `tail -f`。默认情况下，SLURM 将输出写到提交目录中的 `slurm-<job_id>.out`：

```bash
tail -f slurm-<job_id>.out
```

对于长时间运行的作业，`sacct` 命令提供包括资源使用的历史信息：

```bash
# Show completed jobs with resource usage
sacct -j <job_id> --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State

# Show all your recent jobs
sacct -u $USER --starttime=2024-01-01
```

### 日志与输出

SLURM 捕获你作业的 stdout 和 stderr 并将它们写到文件。你可以使用特殊格式代码自定义文件名：

```bash
#SBATCH --output=train_%j.out    # %j = job ID
#SBATCH --error=train_%j.err     # Separate file for stderr
#SBATCH --output=train_%j_%N.out # %N = node name (useful for multi-node)
```

分布式训练的一个挑战是所有 rank 默认写到同一个输出文件，使输出交错且难以阅读。有几种策略处理这个。

最简单的方法是用 `srun` 的 `--label` 标志，它用任务 ID 为每行加前缀：

```bash
srun --label python train.py
```

关于更多控制，在你的 Python 代码中实现 rank 特定的日志：

```python
import logging
import torch.distributed as dist

def setup_logging():
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Each rank logs to its own file
    logging.basicConfig(
        filename=f'train_rank_{rank}.log',
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Optionally, only rank 0 logs to console
    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
```

这给你每个 rank 单独的日志文件，使调试 rank 特定问题容易得多。

### 剖析分布式训练

当你的训练比预期慢时，剖析有助于识别时间花在哪里。PyTorch 的内建性能分析器与 SLURM 作业无缝集成——你只需注意多个 rank 同时运行。

基本方法是用性能分析器上下文管理器包装几个训练步骤：

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # Profile a few training steps
    for step in range(5):
        with record_function("forward"):
            output = model(input)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()

# Save trace (only on rank 0 to avoid file conflicts)
if dist.get_rank() == 0:
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

`record_function` 上下文管理器为你的 trace 添加命名区域，使识别训练的哪个阶段是瓶颈更容易。`with_stack=True` 选项捕获 Python 调用栈，有助于将性能问题追溯到特定代码行。

导出的 trace 文件可以在 Chrome 的 `chrome://tracing` 或 TensorBoard 中查看（关于 trace 可视化的详细说明见第~\ref{sec:profiling-visualization}节）。对于分布式训练，特别注意 trace 中的通信操作。如果你看到 `ncclAllReduce` 或类似的集合操作主导你的剖析，你很可能有通信瓶颈。常见补救包括增加批大小以改善计算与通信的比率、使用梯度累积以减少同步频率，或如果你的框架支持则启用通信-计算重叠。

当性能分析器没有给你足够关于通信问题的信息时，NCCL 提供它自己的调试输出。将这些环境变量添加到你的 SLURM 脚本：

```bash
export NCCL_DEBUG=INFO        # Detailed NCCL logging
export NCCL_DEBUG_SUBSYS=ALL  # All subsystems
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debugging
```

这些产生显示 NCCL 到底在做什么的冗长输出——连接建立、环形拓扑、带宽测量和任何错误。这种细节级别在调试挂起或意外减速时非常宝贵，但输出量使它对生产运行不切实际。在调查特定问题时有选择地启用这些标志。

## 最佳实践

在完成上面的示例后，浮现出几个值得显式强调的模式。

### 资源分配

一个常见错误是依赖集群默认值进行资源分配。不同集群有不同的默认值，在你的实验室集群上工作的可能在共享 HPC 系统上静默失败。始终在你的批处理脚本中显式指定资源：`--nodes`、`--gres`、`--cpus-per-task`、`--mem` 和 `--time`。这使你的脚本可移植和自文档化。

当你需要对节点的保证独占访问时——大规模训练常见，你想避免其他作业的干扰——使用 `--exclusive` 标志。这确保没有其他作业共享你分配的节点，即使你没有使用它们的所有资源。

内存分配值得特别注意。GPU 内存不足错误很明显，但 CPU 内存耗尽可能导致静默失败或神秘崩溃。用 `--mem`（每节点）或 `--mem-per-cpu` 请求足够的内存，并记住数据加载 worker 也消耗 CPU 内存。

### 检查点策略

对于长时间训练运行，检查点策略可以决定丢失数天工作和中断后无缝恢复之间的区别。以固定步骤间隔而非只在 epoch 边界保存检查点——如果你的 epoch 很长，基于 epoch 的策略意味着在故障时丢失显著进度。

使用 FSDP 时，利用 `torch.distributed.checkpoint` 进行高效的分布式保存，不需要将完整模型收集到单个 rank。对于 DeepSpeed 和 Megatron-LM，使用它们正确处理分片状态的内建检查点机制。

最重要的是，在开始长时间运行之前始终测试你的恢复逻辑。提交一个短作业，让它检查点，取消它，并验证恢复产生相同的训练动态。在丢失一周训练后发现你检查点加载中的 bug 是痛苦的。

### 处理故障

在规模上，故障不可避免。节点崩溃、网络连接断开、作业被抢占。带着这个心态设计你的训练流水线。

实现检测挂起进程的健康检查——一个常见的故障模式，其中一个 rank 崩溃但其他在集合操作处无限期等待。PyTorch 的 `init_process_group` 接受一个 `timeout` 参数；将它设为合理的值（如 30 分钟），使挂起的作业最终失败而非无限期消耗资源。

确保你的数据加载对瞬态文件系统问题稳健。重负载下的共享文件系统偶尔可能返回错误；用带指数退避的重试逻辑包装数据加载防止这些瞬态问题杀死你的作业。

最后，考虑为可抢占队列实现自动作业重新提交。许多集群提供等待时间更短但可能被抢占的较低优先级队列。一个检测抢占并重新提交作业（用 `--dependency=singleton` 防止重复）的包装脚本可以大幅改善你在繁忙集群上的有效吞吐量。

## 常见问题排查

即使有仔细的设置，事情也会出错。本节涵盖你会遇到的最常见问题以及如何诊断它们。

### 节点不可用

有时你的作业在队列中以 `PD`（待处理）状态停留比预期长。第一步是检查你请求的节点是否实际可用：

```bash
sinfo -N -l
```

这显示每个节点的状态。常见状态包括 `idle`（可用）、`alloc`（使用中）、`down`（不可用）和 `drain`（管理上禁用）。如果节点 down 或 drained，你需要等待它们回来或调整你的作业以使用不同的节点。

如果你在运行本地测试集群（如虚拟节点设置一节所述），你可能需要在重启后手动恢复节点：

```bash
scontrol update NodeName=node[6-7] State=RESUME
```

### GPU 分配问题

当作业以 GPU 相关错误失败时，先验证 SLURM 正确看到 GPU：

```bash
scontrol show nodes | grep Gres
```

这显示为每个节点配置的通用资源（包括 GPU）。如果 GPU 没有显示，检查你 SLURM 配置目录中的 `gres.conf` 文件。你也可以直接测试 GPU 分配：

```bash
srun -N 1 --gres=gpu:1 nvidia-smi -L
```

如果这失败，问题很可能在 SLURM 的 GPU 配置而非你的训练脚本。

### 通信错误

分布式训练故障常常表现为集合操作期间的 NCCL 错误或超时。从验证节点之间的基本网络连接开始：

```bash
srun -N 2 bash -c 'echo "$(hostname): $(ping -c 1 node6 | grep time=)"'
```

如果节点不能互相到达，检查防火墙规则和网络配置。对于 NCCL 特定问题，用 `NCCL_DEBUG=INFO` 启用详细日志以准确看到通信在哪里失败。常见罪魁包括不正确的网络接口选择（用 `NCCL_SOCKET_IFNAME` 修复）、InfiniBand 配置问题（尝试 `NCCL_IB_DISABLE=1` 回退到以太网），以及端口冲突（如果默认在使用则更改 `MASTER_PORT`）。

### 作业挂起

也许最令人沮丧的问题是一个开始但然后无限期挂起的作业。这通常发生在一个 rank 崩溃或卡住而其他在集合操作处等待时。

首先，通过检查作业的输出文件并使用 `squeue -j <job_id>` 查看作业状态来检查所有进程是否实际运行。如果作业显示为运行但不产生输出，尝试 SSH 到分配的节点并用 `ps aux | grep python` 检查进程状态。

挂起的常见原因包括不匹配的 world size（一个 rank 认为有比实际启动更多的进程）、一个 rank 不能访问其他能访问的文件的数据加载问题，以及自定义代码中不正确同步的死锁。设置 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 并在 `init_process_group` 中使用合理的超时有助于诊断这些问题——至少作业会以错误消息失败而非永远挂起。

## 有用的链接

__SLURM 文档和工具__

- SLURM Workload Manager 文档：\url{https://slurm.schedmd.com/}
- SLURM GitHub 仓库：\url{https://github.com/SchedMD/slurm}
- 单节点 SLURM 集群 Docker：\url{https://github.com/minyang-chen/single-node-slurm-cluster-docker}
- DeepOps（GPU 集群部署）：\url{https://github.com/NVIDIA/deepops}

__PyTorch 分布式训练__

- PyTorch 分布式概述：\url{https://pytorch.org/tutorials/beginner/dist_overview.html}
- PyTorch FSDP 教程：\url{https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html}
- PyTorch 分布式检查点：\url{https://pytorch.org/docs/stable/distributed.checkpoint.html}

__DeepSpeed 和 Megatron-LM__

- DeepSpeed 文档：\url{https://www.deepspeed.ai/}
- DeepSpeed GitHub：\url{https://github.com/microsoft/DeepSpeed}
- Megatron-LM GitHub：\url{https://github.com/NVIDIA/Megatron-LM}
- Megatron-Bridge（检查点转换）：\url{https://github.com/NVIDIA-NeMo/Megatron-Bridge}

__教程和指南__

- Optimizing Language Model Training with SLURM (Medium, 2024)：\url{https://medium.com/@viktorciroski/optimizing-language-model-training-a-practical-guide-to-slurm-a6621d3c1bf2}
- Deploy an Auto-Scaling HPC Cluster with SLURM on GCP：\url{https://codelabs.developers.google.com/codelabs/hpc-slurm-on-gcp}

__研究__

- ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates (2025)：\url{https://arxiv.org/abs/2505.12242}
- Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping (2024)：\url{https://arxiv.org/abs/2409.15241}

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
