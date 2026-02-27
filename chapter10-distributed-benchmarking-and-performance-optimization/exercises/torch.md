\fancydividerwithicon[center]{hand.png}


## Exercises


### Create a Training Benchmark

Write a script that measures forward, backward, communication, and optimizer times separately.

__Requirements:__

- Instrument a training loop to measure each phase:
  - Forward pass time
  - Backward pass time
  - Communication time (gradient sync)
  - Optimizer step time
- Run benchmarks with 1, 2, 4, and 8 GPUs
- Calculate scaling efficiency: `efficiency = (N × throughput_N) / (1 × throughput_1)`
- Generate comparison plots

__Test your implementation:__
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

class TrainingBenchmark:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.timings = {
            "forward": [],
            "backward": [],
            "comm": [],
            "optimizer": [],
        }
    
    def benchmark_step(self, batch):
        """Run one training step with timing."""
        # Forward
        torch.cuda.synchronize()
        start = time.time()
        output = self.model(batch)
        loss = output.mean()
        torch.cuda.synchronize()
        self.timings["forward"].append(time.time() - start)
        
        # Backward (includes communication for DDP)
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        self.timings["backward"].append(time.time() - start)
        
        # Optimizer
        start = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        self.timings["optimizer"].append(time.time() - start)
    
    def report(self):
        """Print timing summary."""
        for name, times in self.timings.items():
            avg = sum(times) / len(times) * 1000
            print(f"{name}: {avg:.2f} ms")

# Run benchmark
benchmark = TrainingBenchmark(ddp_model, optimizer)
for _ in range(100):
    batch = torch.randn(32, 1024).cuda()
    benchmark.benchmark_step(batch)

benchmark.report()
```

### Benchmark Inference Latency

Use genai-bench CLI to benchmark an inference server with different traffic scenarios.

__Requirements:__

- Set up a vLLM or SGLang server
- Run benchmarks with varying:
  - Concurrency levels (1, 4, 8, 16, 32)
  - Request rates (10, 50, 100, 200 req/s)
  - Input/output lengths
- Measure key metrics:
  - Time to First Token (TTFT)
  - End-to-End latency (E2E)
  - Time Per Output Token (TPOT)
- Generate percentile analysis (P50, P95, P99)

__Test your implementation:__
```bash
# Start inference server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000

# Run benchmark with genai-bench
genai-bench \
    --backend openai \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num-prompts 1000 \
    --request-rate 50 \
    --output-format excel \
    --output-dir results/
```

```python
# Analyze results
import pandas as pd
import matplotlib.pyplot as plt

def analyze_benchmark_results(results_file: str):
    """Analyze genai-bench results."""
    df = pd.read_excel(results_file)
    
    # Calculate percentiles
    metrics = ["ttft", "e2e_latency", "tpot"]
    percentiles = [50, 95, 99]
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for p in percentiles:
            value = df[metric].quantile(p / 100)
            print(f"  P{p}: {value:.2f} ms")
    
    # Plot latency distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, metrics):
        df[metric].hist(ax=ax, bins=50)
        ax.set_title(metric.upper())
        ax.set_xlabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig("latency_distribution.png")

analyze_benchmark_results("results/benchmark.xlsx")
```

### Benchmark Model Accuracy

Evaluate a model on standard benchmarks and compare accuracy across configurations.

__Requirements:__

- Evaluate on MMLU or GLUE benchmark
- Compare configurations:
  - Centralized training baseline
  - Distributed training (DDP, FSDP)
  - Quantized versions (INT8, INT4)
- Use statistical tests for significance
- Generate accuracy comparison report

__Test your implementation:__
```python
from lm_eval import evaluator
from lm_eval.models import get_model
import scipy.stats as stats

def evaluate_model(model_path: str, tasks: list[str]) -> dict:
    """Evaluate model on benchmark tasks."""
    model = get_model("hf", pretrained=model_path)
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=5,
    )
    return results

def compare_accuracy(results_a: dict, results_b: dict, task: str):
    """Compare accuracy between two models using statistical test."""
    acc_a = results_a["results"][task]["acc"]
    acc_b = results_b["results"][task]["acc"]
    
    # McNemar's test for paired comparison
    # (requires per-sample predictions)
    
    print(f"Model A: {acc_a:.4f}")
    print(f"Model B: {acc_b:.4f}")
    print(f"Difference: {abs(acc_a - acc_b):.4f}")

# Evaluate different configurations
configs = {
    "baseline": "meta-llama/Llama-3.2-1B",
    "quantized_int8": "meta-llama/Llama-3.2-1B-int8",
}

results = {}
for name, model_path in configs.items():
    results[name] = evaluate_model(model_path, ["mmlu"])
    print(f"{name}: {results[name]['results']['mmlu']['acc']:.4f}")
```

### Analyze Communication Overhead

Profile a distributed training job and identify communication vs computation time.

__Requirements:__

- Use PyTorch Profiler or Nsight Systems
- Measure:
  - Total training time
  - Computation time (forward + backward)
  - Communication time (AllReduce, AllGather)
  - Idle time (waiting for sync)
- Calculate communication overhead percentage
- Identify optimization opportunities

__Test your implementation:__
```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

def profile_distributed_training(model, dataloader, num_steps: int = 10):
    """Profile distributed training with detailed timing."""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler("./logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            # Training step
            output = model(batch)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            prof.step()
    
    # Analyze results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Extract communication time
    comm_time = 0
    compute_time = 0
    for event in prof.key_averages():
        if "nccl" in event.key.lower() or "all_reduce" in event.key.lower():
            comm_time += event.cuda_time_total
        else:
            compute_time += event.cuda_time_total
    
    total_time = comm_time + compute_time
    print(f"\nCommunication: {comm_time / 1e6:.2f} ms ({100 * comm_time / total_time:.1f}%)")
    print(f"Computation: {compute_time / 1e6:.2f} ms ({100 * compute_time / total_time:.1f}%)")

profile_distributed_training(ddp_model, dataloader)
```

### Optimize a Bottleneck

Identify and fix a performance bottleneck in a distributed system.

__Requirements:__

- Profile to identify the bottleneck
- Implement an optimization (e.g., gradient compression, overlap)
- Measure performance improvement
- Verify accuracy is maintained
- Document the optimization process

__Test your implementation:__
```python
import torch
import torch.distributed as dist

class GradientCompressor:
    """Top-k gradient compression for communication reduction."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.residuals = {}
    
    def compress(self, tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress gradient using top-k selection."""
        # Add residual from previous iteration
        if name in self.residuals:
            tensor = tensor + self.residuals[name]
        
        # Select top-k values
        k = int(tensor.numel() * self.compression_ratio)
        values, indices = torch.topk(tensor.abs().flatten(), k)
        
        # Store residual
        mask = torch.zeros_like(tensor.flatten())
        mask[indices] = 1
        self.residuals[name] = tensor.flatten() * (1 - mask)
        self.residuals[name] = self.residuals[name].view_as(tensor)
        
        # Return compressed representation
        compressed_values = tensor.flatten()[indices]
        return compressed_values, indices
    
    def decompress(self, values: torch.Tensor, indices: torch.Tensor, 
                   shape: tuple) -> torch.Tensor:
        """Decompress gradient."""
        tensor = torch.zeros(shape).flatten().to(values.device)
        tensor[indices] = values
        return tensor.view(shape)

# Benchmark with and without compression
def benchmark_compression(model, compressor=None):
    """Compare training with and without gradient compression."""
    import time
    
    times = []
    for _ in range(10):
        start = time.time()
        
        # Forward + backward
        output = model(torch.randn(32, 1024).cuda())
        loss = output.mean()
        loss.backward()
        
        # Sync gradients (with optional compression)
        for name, param in model.named_parameters():
            if param.grad is not None:
                if compressor:
                    values, indices = compressor.compress(param.grad, name)
                    dist.all_reduce(values)
                    param.grad = compressor.decompress(values, indices, param.grad.shape)
                else:
                    dist.all_reduce(param.grad)
        
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return sum(times) / len(times)

# Compare
baseline_time = benchmark_compression(model, compressor=None)
compressed_time = benchmark_compression(model, compressor=GradientCompressor(0.1))

print(f"Baseline: {baseline_time * 1000:.2f} ms")
print(f"Compressed: {compressed_time * 1000:.2f} ms")
print(f"Speedup: {baseline_time / compressed_time:.2f}x")
```


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Create comprehensive training benchmarks with phase-level timing
- Use genai-bench to benchmark inference servers
- Evaluate model accuracy on standard benchmarks
- Profile and analyze communication overhead in distributed training
- Identify and optimize performance bottlenecks
- Balance performance optimization with accuracy preservation
