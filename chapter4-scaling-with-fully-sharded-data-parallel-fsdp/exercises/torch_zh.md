\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. 手动实现参数分片（Parameter Sharding）机制

实现一个简化版的 FSDP 参数分片与重构逻辑，深入理解系统如何突破单卡物理显存上限。

__要求：__

- 函数签名：
```python
def shard_parameters(model: nn.Module, world_size: int, rank: int) -> dict:
    """跨所有 Rank 对模型参数执行扁平化分片，各卡仅保留自身的本地分片。"""
    pass

def gather_parameters(sharded_params: dict, world_size: int) -> dict:
    """使用 All-Gather 集合通信汇聚所有分片，重构完整的模型参数。"""
    pass
```
- 将模型的所有参数张量打平（Flatten）为一个连续的一维大张量
- 将打平后的大张量等分为 `world_size` 份切片（Shard）
- 每个 Rank 仅将自身的切片驻留在 GPU 显存中（仅占总参数量的 $1 / \text{world\_size}$）
- 实现汇聚函数，在前向计算前通过 All-Gather 动态重构完整参数

__测试代码：__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 100)
)

# 参数分片
sharded = shard_parameters(model, world_size=4, rank=dist.get_rank())
print(f"Rank {dist.get_rank()}: 本地分片元素量 = {sharded['shard'].numel()}")

# 前向传播前动态汇聚
full_params = gather_parameters(sharded, world_size=4)
print(f"重构后的完整参数量: {sum(p.numel() for p in full_params.values())} 元素")
```

### 2. FSDP 分片策略（Sharding Strategies）量化基准对比

编写一个基准测试脚本，量化对比 FSDP 不同分片策略在显存占用与通信吞吐之间的权衡（Trade-off）。

__要求：__

- 对比以下三种核心分片策略：
  - `FULL_SHARD`：全分片（参数 + 梯度 + 优化器状态全面切分，ZeRO-3 等价）
  - `SHARD_GRAD_OP`：仅对梯度与优化器状态分片（ZeRO-2 等价）
  - `NO_SHARD`：不分片（与传统 DDP 等价）
- 精确测量每种策略下的：
  - 峰值 GPU 显存占用（Peak Memory in GB）
  - 训练吞吐量（Samples/sec）
  - 跨卡集合通信开销
- 使用在 `NO_SHARD` 下单卡会触发 OOM 的大模型进行压力测试
- 输出结构化对比表格与分析结果

__测试代码：__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

strategies = [
    ShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.NO_SHARD,
]

results = {}
for strategy in strategies:
    model = create_large_model()  # 例如 7B 参数模型
    fsdp_model = FSDP(model, sharding_strategy=strategy)
    
    memory, throughput = benchmark_training(fsdp_model, num_steps=100)
    results[strategy.name] = {"memory_gb": memory, "throughput": throughput}

# 格式化输出对比表格
print("分片策略        | 显存占用 (GB) | 训练吞吐 (samples/s)")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:15} | {metrics['memory_gb']:13.1f} | {metrics['throughput']:18.1f}")
```

### 3. 实现 Transformer 自定义分层包装策略（Auto-Wrap Policy）

编写一个针对 Transformer 架构的自定义自动包装策略，将每个 Transformer Block 独立作为 FSDP 分片单元，实现显存与通信的最佳平衡。

__要求：__

- 函数签名：
```python
def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    min_num_params: int = 1e6
) -> bool:
    """针对 Transformer 模型的细粒度包装策略。"""
    pass
```
- 将每个 Transformer Block（含 Attention + FFN）作为独立的 FSDP 单元进行包装
- 不单独包装 Embedding 层（将其保留在根 FSDP 单元中）
- 不单独包装最后的输出投影层（LM Head）
- 支持配置最小参数量阈值过滤

__测试代码：__
```python
import functools
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

fsdp_model = FSDP(
    model,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        min_num_params=1e6
    )
)

# 递归统计包装后的 FSDP 单元总数
def count_fsdp_units(module, depth=0):
    count = 1 if isinstance(module, FSDP) else 0
    for child in module.children():
        count += count_fsdp_units(child, depth + 1)
    return count

print(f"生成的 FSDP 独立分片单元总数: {count_fsdp_units(fsdp_model)}")
```

### 4. FSDP 混合精度（Mixed Precision）配置与调优

配置并测试 FSDP 在不同混合精度策略下的显存节约效果与数值稳定性。

__要求：__

- 测试以下三种精度配置：
  - 全 FP32 纯单精度基线
  - BF16 计算 + FP32 参数存储与梯度规约（Compute BF16）
  - 全流程纯 BF16（参数存储、梯度规约与计算均为 BF16）
- 量化度量：
  - 显存相比 FP32 的压缩率
  - 训练 Loss 的收敛平稳度
  - 训练 Step 耗时与加速比

__测试代码：__
```python
from torch.distributed.fsdp import MixedPrecision
import torch

# 定义各级精度策略
fp32_policy = MixedPrecision()

bf16_compute_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

bf16_full_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

policies = {
    "全 FP32 基准": fp32_policy,
    "BF16 混合计算": bf16_compute_policy,
    "纯 BF16 全分片": bf16_full_policy,
}

for name, policy in policies.items():
    model = FSDP(create_model(), mixed_precision=policy)
    memory, loss, throughput = train_and_measure(model, num_steps=100)
    print(f"[{name}] 显存: {memory:.1f}GB, Loss: {loss:.4f}, 吞吐: {throughput:.1f} steps/s")
```

### 5. 实现 FSDP 分布式检查点（DCP）保存与任意拓扑动态重分片加载

构建一个支持任意分布式拓扑（World Size 动态伸缩）的 FSDP 分布式检查点存储与加载系统。

__要求：__

- 保存分布式分片检查点，并支持使用**不同 GPU 卡数（World Size 改变）**重新加载
- 兼容分布式分片存储（Sharded State Dict）与全量聚合（Full State Dict）两种模式
- 包含 Checkpoint 数据一致性校验逻辑
- 完整包含优化器状态的分片保存与恢复

__测试代码：__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

class FSDPCheckpointer:
    def __init__(self, model: FSDP, optimizer, checkpoint_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
    
    def save(self, step: int, full_state: bool = True):
        """保存检查点。"""
        pass
    
    def load(self, step: int) -> dict:
        """加载检查点并返回元数据。"""
        pass
    
    def validate(self, step: int) -> bool:
        """校验检查点文件完整性。"""
        pass

# 测试验证流程
checkpointer = FSDPCheckpointer(fsdp_model, optimizer, "./checkpoints")

# 训练若干步并存盘
train_steps(fsdp_model, optimizer, num_steps=10)
checkpointer.save(step=10)

# 校验并恢复
assert checkpointer.validate(step=10), "检查点校验失败！"
metadata = checkpointer.load(step=10)
print(f"成功从 Checkpoint 恢复全局第 {metadata['step']} 步训练状态")
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 深刻理解 FSDP 如何通过 All-Gather 与 Reduce-Scatter 消除显存冗余
- 准确评估 `FULL_SHARD`、`SHARD_GRAD_OP` 与 `NO_SHARD` 的性能与显存拐点
- 熟练为现代 LLM（如 LLaMA、T5）编写高效的分层包装（Hierarchical Wrap Policy）策略
- 科学配置 FSDP2 的 `MixedPrecisionPolicy` 与 `DeviceMesh` 拓扑
- 熟练运用分布式检查点（DCP）技术，实现无单卡瓶颈的高吞吐并行 I/O 存盘与跨拓扑恢复
- 准确权衡前向/反向预取（Prefetching）、激活值重计算与 CPU Offloading 的系统开销
