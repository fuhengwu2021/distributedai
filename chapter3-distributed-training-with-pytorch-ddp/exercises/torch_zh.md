\fancydividerwithicon[center]{hand.png}


## 实战演练


### 手动实现梯度同步

使用 `torch.distributed` 底层原语手动实现梯度同步，深入理解 DDP 内部的通信执行细节。

__实战要求：__

- 函数签名：
```python
def sync_gradients(model: nn.Module, world_size: int) -> None:
    """使用 AllReduce 跨所有 rank 规约同步模型梯度。"""
    pass
```

- 遍历模型中所有满足 `requires_grad=True` 的可训练参数
- 使用 `dist.all_reduce()` 并指定规约操作为 `ReduceOp.SUM`
- 将累加结果除以 `world_size` 计算全局均值
- 妥善处理梯度为 `None` 的参数（自动跳过未被激活或无梯度的参数）

__测试验证：__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

# 构建简单测试模型
model = nn.Linear(100, 10).cuda()

# 前向传播与反向传播
x = torch.randn(32, 100).cuda()
y = torch.randn(32, 10).cuda()
loss = nn.MSELoss()(model(x), y)
loss.backward()

# 手动规约同步各卡梯度
sync_gradients(model, world_size=dist.get_world_size())

# 校验：各 rank 上的梯度数值应保持完全一致
print(f"Rank {dist.get_rank()}: 权重梯度总和 = {model.weight.grad.sum().item():.4f}")
```

### DP 与 DDP 性能基准对比

编写基准测试脚本，全方位对比 PyTorch 单进程多线程 DataParallel (DP) 与多进程 DistributedDataParallel (DDP) 的吞吐与资源开销差异。

__实战要求：__

- 基于 ResNet-50 构建基准训练工作负载
- 分别实现 DP 与 DDP 模式下的标准训练循环
- 综合度量以下指标：
  - 训练吞吐率（samples/second）
  - 每块 GPU 上的显存占用（观察 DP 主卡显存不均衡现象）
  - GPU 核心利用率百分比
- 分别在 2 卡、4 卡及 8 卡（若环境支持）规模下执行测试
- 计算并绘制扩展效率曲线：`efficiency = (N_gpu * throughput_N) / (1 * throughput_1)`

__测试验证：__
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50
import time

def benchmark_dp(model, batch_size, num_iterations):
    """评测 DataParallel 吞吐性能。"""
    model = nn.DataParallel(model)
    # ... 训练循环逻辑
    return throughput

def benchmark_ddp(model, batch_size, num_iterations, rank, world_size):
    """评测 DistributedDataParallel 吞吐性能。"""
    model = DDP(model, device_ids=[rank])
    # ... 训练循环逻辑
    return throughput

# 对比两者性能
dp_throughput = benchmark_dp(resnet50().cuda(), batch_size=64, num_iterations=100)
ddp_throughput = benchmark_ddp(resnet50().cuda(), batch_size=64, num_iterations=100, rank, world_size)

print(f"DP 模式吞吐率: {dp_throughput:.1f} samples/sec")
print(f"DDP 模式吞吐率: {ddp_throughput:.1f} samples/sec")
print(f"DDP 相较 DP 的加速比: {ddp_throughput/dp_throughput:.2f}x")
```

### 实现梯度分桶（Bucketing）聚合优化

实现一个简化版的梯度分桶管理器，深入掌握 DDP 如何通过融合通信来掩盖网络延迟。

__实战要求：__

- 类签名定义：
```python
class GradientBucketer:
    def __init__(self, model: nn.Module, bucket_size_mb: float = 25.0):
        """基于模型结构与目标分桶大小初始化分桶管理器。"""
        pass
    
    def create_buckets(self) -> list[list[nn.Parameter]]:
        """依据参数大小贪心聚合参数列表划分为若干分桶。"""
        pass
    
    def sync_bucket(self, bucket: list[nn.Parameter]) -> None:
        """对单个分桶中的梯度执行展平与融合 AllReduce 同步。"""
        pass
    
    def sync_all(self) -> None:
        """逆序（从尾层到首层）触发所有分桶的同步，模拟计算与通信重叠。"""
        pass
```

- 将模型参数按约 `bucket_size_mb` 兆字节阈值归并为多个连续桶
- 在执行通信前将同桶内的零散梯度打平成一维连续内存，以最大化 NCCL 传输带宽
- 严格遵循逆序策略（由输出层向输入层依次同步，与反向传播计算路径对齐）
- 统计并报告各分桶的纯通信耗时

__测试验证：__
```python
from torchvision.models import resnet50

model = resnet50().cuda()
bucketer = GradientBucketer(model, bucket_size_mb=25.0)

# 展示参数分桶划分详情
buckets = bucketer.create_buckets()
for i, bucket in enumerate(buckets):
    size_mb = sum(p.numel() * 4 / 1e6 for p in bucket)
    print(f"分桶 {i}: 包含 {len(bucket)} 个参数, 容量约 {size_mb:.1f} MB")

# 前向计算与反向传播
x = torch.randn(32, 3, 224, 224).cuda()
loss = model(x).sum()
loss.backward()

# 触发逆序梯度分桶同步并精确计时
import time
start = time.time()
bucketer.sync_all()
print(f"全局分桶同步总耗时: {(time.time() - start) * 1000:.1f} ms")
```

### 构建具备故障自愈能力的 DDP 训练流水线

编写包含周期性 Checkpoint 保存与崩溃自动断点续训的分布式训练脚本。

__实战要求：__

- 按照配置步长周期性持久化 Checkpoint 状态（每 $N$ 步触发）
- 检查点必须完整打包以下上下文：
  - 模型权重状态字典（`model.state_dict()`）
  - 优化器状态字典（`optimizer.state_dict()`）
  - 学习率调度器状态（`lr_scheduler.state_dict()`）
  - 当前训练轮次（epoch）与迭代步数（global step）
  - 全局随机数种子状态（涵盖 torch、numpy 与 python 原生 random）
- 启动时自动检测并恢复最新的可用 Checkpoint
- 严格限制仅由 Rank 0 负责落盘写磁盘，并通过分布式 `barrier()` 保证各卡读写时序同步
- 模拟进程强行中断退出，验证恢复后训练曲线与数据流的一致性

__测试验证：__
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class FaultTolerantTrainer:
    def __init__(self, model, optimizer, checkpoint_dir, save_every=100):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.step = 0
        self.epoch = 0
    
    def save_checkpoint(self):
        """保存检查点（仅限 Rank 0 写入，协同各卡同步）。"""
        pass
    
    def load_checkpoint(self) -> bool:
        """加载最新可用检查点，若加载成功返回 True。"""
        pass
    
    def train_step(self, batch):
        """单步迭代逻辑，包含周期性检查点写入。"""
        pass

# 使用示例
trainer = FaultTolerantTrainer(model, optimizer, "./checkpoints")
if trainer.load_checkpoint():
    print(f"成功从历史断点恢复: Step {trainer.step}")

for epoch in range(trainer.epoch, num_epochs):
    for batch in dataloader:
        trainer.train_step(batch)
```

### 基于 NCCL 原语剖析分布式通信模式

编写通信性能剖析脚本，量化 DDP 训练中底层 NCCL 集合通信的吞吐开销与瓶颈。

__实战要求：__

- 使用 `torch.cuda.Event` 高精度异步计时器测量纯 GPU 通信耗时
- 对比不同核心集合通信原语的开销特性：
  - AllReduce（梯度融合同步）
  - Broadcast（权重分发与参数初始化）
  - AllGather（指标汇总与全量状态拉取）
- 计算有效总线带宽利用率：`bandwidth = data_size / time`
- 对标硬件物理理论峰值带宽，核算总线带宽饱和率
- 格式化输出性能分析报告，识别通信阻塞点

__测试验证：__
```python
import torch
import torch.distributed as dist

def profile_allreduce(tensor_size_mb: float, num_iterations: int = 10):
    """精确评测 AllReduce 通信延迟与有效带宽。"""
    tensor = torch.randn(int(tensor_size_mb * 1e6 / 4)).cuda()
    
    # 充分执行预热（Warmup）
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    # 高精度事件剖析
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        dist.all_reduce(tensor)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / num_iterations
    bandwidth_gbps = (tensor_size_mb * 2 / 1000) / (time_ms / 1000)  # 环形 AllReduce 通信数据量按 2x 换算
    
    return time_ms, bandwidth_gbps

# 针对不同数据包规模执行阶梯压测
for size_mb in [1, 10, 100, 500]:
    time_ms, bw = profile_allreduce(size_mb)
    print(f"数据量: {size_mb:4d} MB, 平均延迟: {time_ms:6.2f} ms, 有效带宽: {bw:.1f} GB/s")
```


## 预期学习目标

完成本章实战练习后，你将能够：

- 深入透视数据并行底层梯度同步与参数规约的物理运行机制
- 从架构与性能层面精准对比 DP 与 DDP 的根本优劣，并在实战中规避单卡显存失衡陷阱
- 亲手实现梯度分桶机制，领会计算与通信重叠（Overlap）的系统级工程智慧
- 构建生产级容错训练管道，从容应对硬件节点掉线与集群作业抢占
- 使用 CUDA 异步事件工具高精度剖析 NCCL 集合通信的性能特征
- 准确定位分布式数据并行中的网络拥塞与通信延迟瓶颈并完成针对性调优
