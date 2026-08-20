\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. DeepSpeed ZeRO 阶段（Stage 1/2/3）基准性能对比

编写一个基准测试脚本，量化对比 DeepSpeed ZeRO Stage 1、2、3 在相同模型架构下的显存占用与吞吐开销。

__要求：__

- 使用 7B 参数模型（若硬件受限可使用等比缩小版）
- 分别评测：
  - ZeRO Stage 1（仅优化器状态分片）
  - ZeRO Stage 2（优化器状态 + 梯度分片）
  - ZeRO Stage 3（参数 + 梯度 + 优化器状态全分片）
- 精确测量每种阶段下的：
  - 单卡峰值 GPU 显存（GB）
  - 训练吞吐量（Tokens/sec）
  - 跨卡集合通信时间占比
- 生成结构化对比表格与显存分解图

__测试代码：__
```python
import deepspeed
import torch

def benchmark_zero_stage(model, stage: int, num_steps: int = 100):
    """测试指定 ZeRO Stage 下的训练性能。"""
    ds_config = {
        "train_batch_size": 32,
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "bf16": {"enabled": True},
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    
    # 运行基准训练循环
    memory, throughput = run_training(model_engine, num_steps)
    return memory, throughput

# 对比各阶段
results = {}
for stage in [1, 2, 3]:
    model = create_model()  # 每个阶段新建独立模型
    memory, throughput = benchmark_zero_stage(model, stage)
    results[f"ZeRO-{stage}"] = {"memory_gb": memory, "throughput": throughput}

print("ZeRO 阶段   | 显存占用 (GB) | 训练吞吐 (tok/s)")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:10} | {metrics['memory_gb']:13.1f} | {metrics['throughput']:17.1f}")
```

### 2. 构建 ZeRO-Offload 异构内存卸载系统

配置并测试 DeepSpeed ZeRO-Offload，利用 CPU 内存突破 GPU 显存限制训练更大参数量的模型。

__要求：__

- 配置 ZeRO Stage 3 并启用 CPU 内存卸载
- 针对优化器状态（Optimizer Offload）开启卸载与 Pin Memory 锁页内存加速
- 针对模型参数（Param Offload）开启卸载
- 测量并分析：
  - 能够成功加载并训练的最大模型参数量
  - 相比纯 GPU 显存训练的吞吐下降比率
  - Host CPU 内存实际消耗峰值
  - PCIe 总线吞吐与利用率

__测试代码：__
```python
import deepspeed
import torch

# ZeRO-3 配合全量 CPU Offload
offload_config = {
    "train_batch_size": 8,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "bf16": {"enabled": True},
}

# 逐步增大模型规模进行压力测试
model_sizes = ["1B", "3B", "7B", "13B"]
for size in model_sizes:
    try:
        model = create_model(size)
        engine, _, _, _ = deepspeed.initialize(model=model, config=offload_config)
        
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        cpu_mem = get_cpu_memory_usage()
        throughput = benchmark_throughput(engine, num_steps=10)
        
        print(f"[{size}] GPU 显存: {gpu_mem:.1f}GB, CPU 内存: {cpu_mem:.1f}GB, 吞吐: {throughput:.1f} tok/s")
    except RuntimeError as e:
        print(f"[{size}] 触发 OOM: {e}")
        break
```

### 3. 从零手写 Megatron 张量并行（Tensor Parallelism）线性层

实现简化的列并行（ColumnParallelLinear）与行并行（RowParallelLinear）模块，深入理解 Megatron-LM 的算子级矩阵切分机制。

__要求：__

- 类签名：
```python
class ColumnParallelLinear(nn.Module):
    """沿输出特征维度（按列）进行切分的并行线性层。"""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class RowParallelLinear(nn.Module):
    """沿输入特征维度（按行）进行切分的并行线性层。"""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```
- 列并行：将输出维度等分到各卡，前向无通信，输出部分特征切片
- 行并行：将输入维度等分到各卡，前向通过 `dist.all_reduce()` 汇总各卡局部点积求和
- 验证组装出的双层 MLP 模块与标准全量 `nn.Linear` 算子的数学输出等价性

__测试代码：__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

world_size = dist.get_world_size()
rank = dist.get_rank()

col_linear = ColumnParallelLinear(1024, 4096, world_size, rank).cuda()
row_linear = RowParallelLinear(4096, 1024, world_size, rank).cuda()

# 前向传播测试
x = torch.randn(32, 1024).cuda()
y = col_linear(x)  # 局部形状: (32, 4096 // world_size)
z = row_linear(y)  # AllReduce 之后形状重构为: (32, 1024)

print(f"Rank {rank}: 输入={x.shape}, 列并行后={y.shape}, 行并行 AllReduce 后={z.shape}")
```

### 4. 实现 1F1B 调度流水线并行（Pipeline Parallelism）

编写一个带 Micro-batch 管道编排与 1F1B（One Forward One Backward）交错调度的流水线并行引擎。

__要求：__

- 将深度神经网络划分为 $N$ 个流水线 Stage，分配至不同 GPU 设备
- 实现 1F1B 稳态交错调度（前向与反向严格交替执行）
- 处理 Micro-batch 梯度的累加与反向传播依赖
- 量化度量流水线气泡（Pipeline Bubble）在不同 Micro-batch 数量下的开销占比

__测试代码：__
```python
class PipelineStage(nn.Module):
    """单个流水线阶段。"""
    def __init__(self, layers: nn.ModuleList, stage_id: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PipelineParallel:
    def __init__(self, model: nn.Module, num_stages: int, num_microbatches: int):
        self.stages = self.split_model(model, num_stages)
        self.num_microbatches = num_microbatches
    
    def split_model(self, model, num_stages) -> list:
        """切分模型为各 Stage。"""
        pass
    
    def forward_backward(self, batch):
        """执行 1F1B 交错调度。"""
        pass

# 测试 4 Stage 流水线
model = create_transformer(num_layers=24)
pp = PipelineParallel(model, num_stages=4, num_microbatches=8)

batch = torch.randn(64, 512, 1024).cuda()
loss = pp.forward_backward(batch)
print(f"流水线单步训练 Loss: {loss.item():.4f}")
```

### 5. 配置 3D 混合并行（DP × TP × PP）进程组拓扑

编写一个在集群多 GPU 上构建 3D 混合并行笛卡尔网格的进程组初始化与分配脚本。

__要求：__

- 支持配置 3D 并行维度：`DP × TP × PP = WORLD_SIZE`
- 例如在 8 卡集群上配置：`DP=2, TP=2, PP=2`
- 计算并分配每个 Rank 在 3D 网格中的坐标 `(dp_rank, tp_rank, pp_rank)`
- 正确创建正交的 DP 集合通信组、TP 集合通信组与 PP 点对点通信组
- 打印并验证各通信组的成员拓扑完整性

__测试代码：__
```python
import torch.distributed as dist

def setup_3d_parallelism(world_size: int, dp: int, tp: int, pp: int):
    """构建 3D 混合并行正交进程组网格。"""
    assert dp * tp * pp == world_size, "DP × TP × PP 乘积必须等于 world_size"
    
    rank = dist.get_rank()
    
    # 计算当前 Rank 在 3D 网格中的坐标
    dp_rank = rank // (tp * pp)
    tp_rank = (rank // pp) % tp
    pp_rank = rank % pp
    
    # 构建正交进程组
    # ...
    return dp_rank, tp_rank, pp_rank, dp_group, tp_group, pp_group

# 在 8 卡环境下测试 3D 拓扑
dp_rank, tp_rank, pp_rank, dp_group, tp_group, pp_group = setup_3d_parallelism(
    world_size=8, dp=2, tp=2, pp=2
)

print(f"Rank {dist.get_rank()}: 3D 坐标 -> (DP={dp_rank}, TP={tp_rank}, PP={pp_rank})")
print(f"DP 进程组规模: {dist.get_world_size(dp_group)}")
print(f"TP 进程组规模: {dist.get_world_size(tp_group)}")
print(f"PP 进程组规模: {dist.get_world_size(pp_group)}")
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 深刻理解 DeepSpeed ZeRO 1/2/3 的状态分片切分阶梯与显存节省机理
- 熟练配置 ZeRO-Offload 与 ZeRO-Infinity 异构分层存储系统
- 掌握 Megatron 张量并行（TP）与序列并行（SP）的算子级矩阵切分与 AllReduce 融合
- 深刻理解流水线并行（PP）的 1F1B 与虚拟交错（Interleaved/VPP）调度算法
- 理解超长上下文场景下 Ring Attention 与 DeepSpeed-Ulysses 的通信差异
- 掌握稀疏大模型专家并行（MoE EP）的 All-to-All 路由与负载均衡
- 熟练编排现代千亿/万亿参数大模型所需的 3D/4D 混合并行拓扑体系
