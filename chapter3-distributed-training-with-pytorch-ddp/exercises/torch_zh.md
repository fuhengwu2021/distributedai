\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. 手动实现梯度同步（AllReduce）

使用 `torch.distributed` 集合通信原语手动编写一个梯度同步函数，深入理解 DDP 底层的同步机制。

__要求：__

- 函数签名：
```python
def sync_gradients(model: nn.Module, world_size: int) -> None:
    """使用 AllReduce 跨所有 Rank 同步并平均模型梯度。"""
    pass
```
- 遍历模型中所有 `requires_grad=True` 的参数
- 使用 `dist.all_reduce()` 配合 `ReduceOp.SUM` 对所有 Rank 上的梯度求和
- 将求和后的梯度除以 `world_size` 计算平均梯度
- 正确跳过梯度为 `None` 的参数

__测试代码：__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

# 构建简单模型
model = nn.Linear(100, 10).cuda()

# 前向传播与反向求导
x = torch.randn(32, 100).cuda()
y = torch.randn(32, 10).cuda()
loss = nn.MSELoss()(model(x), y)
loss.backward()

# 手动同步梯度
sync_gradients(model, world_size=dist.get_world_size())

# 校验：AllReduce 之后各卡上的梯度求和值应完全一致
print(f"Rank {dist.get_rank()}: 权重梯度总和 = {model.weight.grad.sum().item():.4f}")
```

### 2. DP 与 DDP 性能基准对比测试

编写一个基准测试脚本，量化对比旧版 `DataParallel` (DP) 与现代 `DistributedDataParallel` (DDP) 的实际性能差异。

__要求：__

- 使用 ResNet-50 模型作为测试负载
- 分别实现 DP 和 DDP 的完整训练循环
- 精确测量并对比：
  - 训练吞吐量（Samples/sec）
  - 每张 GPU 的显存占用（GB）
  - GPU 核心利用率百分比（%）
- 在 2 卡、4 卡及 8 卡（若环境支持）下运行测试
- 计算并绘制扩展效率曲线：`扩展效率 = (N_gpu * throughput_N) / (1 * throughput_1)`

__测试代码：__
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50
import time

def benchmark_dp(model, batch_size, num_iterations):
    """测试 DataParallel (DP) 性能。"""
    model = nn.DataParallel(model)
    # ... 执行训练循环
    return throughput

def benchmark_ddp(model, batch_size, num_iterations, rank, world_size):
    """测试 DistributedDataParallel (DDP) 性能。"""
    model = DDP(model, device_ids=[rank])
    # ... 执行训练循环
    return throughput

# 运行对比
dp_throughput = benchmark_dp(resnet50().cuda(), batch_size=64, num_iterations=100)
ddp_throughput = benchmark_ddp(resnet50().cuda(), batch_size=64, num_iterations=100, rank, world_size)

print(f"DP 吞吐量: {dp_throughput:.1f} samples/sec")
print(f"DDP 吞吐量: {ddp_throughput:.1f} samples/sec")
print(f"DDP 相比 DP 加速比: {ddp_throughput/dp_throughput:.2f}x")
```

### 3. 实现梯度分桶（Gradient Bucketing）机制

实现一个简化版的梯度分桶管理器，模拟 DDP 如何通过张量合并来降低通信开销。

__要求：__

- 类签名：
```python
class GradientBucketer:
    def __init__(self, model: nn.Module, bucket_size_mb: float = 25.0):
        """使用模型与分桶阈值初始化分桶器。"""
        pass
    
    def create_buckets(self) -> list[list[nn.Parameter]]:
        """按参数大小将模型参数划分为多个分桶。"""
        pass
    
    def sync_bucket(self, bucket: list[nn.Parameter]) -> None:
        """对单个分桶内的参数梯度执行扁平化 AllReduce 同步。"""
        pass
    
    def sync_all(self) -> None:
        """按逆序（反向传播顺序）同步所有分桶。"""
        pass
```
- 将模型参数划分为约 `bucket_size_mb` MB 大小的连续分桶
- 将每个分桶内的离散梯度张量扁平化（Flatten）为一个连续大张量后执行 AllReduce
- 严格按照反向传播的倒序触发各分桶的同步
- 测量并报告各分桶的通信时间开销

__测试代码：__
```python
from torchvision.models import resnet50

model = resnet50().cuda()
bucketer = GradientBucketer(model, bucket_size_mb=25.0)

# 查看分桶结构
buckets = bucketer.create_buckets()
for i, bucket in enumerate(buckets):
    size_mb = sum(p.numel() * 4 / 1e6 for p in bucket)
    print(f"分桶 {i}: 包含 {len(bucket)} 个参数, 容量 {size_mb:.1f} MB")

# 前向与反向传播
x = torch.randn(32, 3, 224, 224).cuda()
loss = model(x).sum()
loss.backward()

# 计时同步
import time
start = time.time()
bucketer.sync_all()
print(f"全部分桶同步耗时: {(time.time() - start) * 1000:.1f} ms")
```

### 4. 构建容错与断点续训 DDP 训练器

编写一个具备周期性 Checkpoint 保存与故障自动恢复能力的 DDP 训练系统。

__要求：__

- 支持每隔 $N$ 步定期保存检查点
- 检查点字典必须完整包含：
  - 模型权重状态字典 (`model.module.state_dict()`)
  - 优化器状态字典 (`optimizer.state_dict()`)
  - 学习率调度器状态
  - 当前 Epoch 与全局 Step
  - 各随机数生成器状态（PyTorch CPU/CUDA、NumPy、Python 原生）
- 实现启动时自动探测并从最新 Checkpoint 恢复
- 仅由 Rank 0 执行原子文件写入，并配合 `dist.barrier()` 进行跨进程同步
- 模拟进程中断并测试断点续训恢复的正确性

__测试代码：__
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
        """保存检查点（仅 Rank 0 执行原子写入）。"""
        pass
    
    def load_checkpoint(self) -> bool:
        """加载最新检查点（若存在），成功返回 True。"""
        pass
    
    def train_step(self, batch):
        """单步训练并执行周期性存盘。"""
        pass

# 使用示例
trainer = FaultTolerantTrainer(model, optimizer, "./checkpoints")
if trainer.load_checkpoint():
    print(f"成功从全局第 {trainer.step} 步恢复训练")

for epoch in range(trainer.epoch, num_epochs):
    for batch in dataloader:
        trainer.train_step(batch)
```

### 5. 使用 NCCL Profiling 分析通信通信特征

编写一个利用 `torch.cuda.Event` 精确度量 DDP 训练期间 NCCL 集合通信各阶段耗时的分析脚本。

__要求：__

- 使用 `torch.cuda.Event` 消除 Python 解释器开销，精确捕获 GPU 端通信耗时
- 测量不同集合通信操作：
  - AllReduce（梯度同步）
  - Broadcast（模型参数与 Buffer 初始化广播）
  - AllGather（若有调用）
- 计算实际有效带宽利用率：`有效带宽 = 传输数据量 / 耗时`
- 与硬件理论峰值总线带宽进行对比
- 生成量化报告，精准识别通信瓶颈所处的张量规模区间

__测试代码：__
```python
import torch
import torch.distributed as dist

def profile_allreduce(tensor_size_mb: float, num_iterations: int = 10):
    """评测 AllReduce 在特定张量规模下的通信性能。"""
    tensor = torch.randn(int(tensor_size_mb * 1e6 / 4)).cuda()
    
    # 热身 Warmup
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    # 评测计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        dist.all_reduce(tensor)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / num_iterations
    bandwidth_gbps = (tensor_size_mb * 2 / 1000) / (time_ms / 1000)  # Ring AllReduce: 2x 传输量修正
    
    return time_ms, bandwidth_gbps

# 测试不同张量尺寸
for size_mb in [1, 10, 100, 500]:
    time_ms, bw = profile_allreduce(size_mb)
    print(f"张量大小: {size_mb:4d} MB, 平均耗时: {time_ms:6.2f} ms, 实测有效带宽: {bw:.1f} GB/s")
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 深刻理解 DDP 底层梯度同步与反向求导的协作机制
- 准确阐明传统 DataParallel (DP) 与现代 DistributedDataParallel (DDP) 的架构差异与性能分水岭
- 熟练实现并调优梯度分桶（Gradient Bucketing）以最大化计算-通信重叠
- 构建具备工业级容错能力的断点续训与状态恢复系统
- 熟练使用 PyTorch Profiler 与 Chrome Tracing 精确分析 NCCL 集合通信时序图
- 诊断并迅速排查 DDP 训练中的死锁挂起（Hang）、显存溢出（OOM）与跨机通信低效问题
