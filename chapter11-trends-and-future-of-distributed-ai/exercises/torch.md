\fancydividerwithicon[center]{hand.png}


## Exercises


### Implement Basic MoE

Create a simple Mixture of Experts layer with expert routing.

__Requirements:__

- Implement an MoE layer with:
  - 4 expert networks (FFN layers)
  - Top-2 routing (each token uses 2 experts)
  - Load balancing loss
- Test on a small transformer model
- Measure expert utilization

__Test your implementation:__
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with expert routing."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute routing scores
        router_logits = self.router(x)  # (batch, seq, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                
                # Weight by routing score
                weight_mask = (top_k_indices == i).float()
                weights = (top_k_weights * weight_mask).sum(dim=-1, keepdim=True)
                output[mask] += expert_output * weights[mask]
        
        # Load balancing loss
        load_balance_loss = self._compute_load_balance_loss(routing_weights)
        
        return output, load_balance_loss
    
    def _compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        # Average routing probability per expert
        avg_prob = routing_weights.mean(dim=[0, 1])  # (num_experts,)
        
        # Fraction of tokens routed to each expert
        top_1_indices = routing_weights.argmax(dim=-1)
        expert_counts = torch.bincount(top_1_indices.flatten(), minlength=self.num_experts)
        expert_frac = expert_counts.float() / expert_counts.sum()
        
        # Load balance loss: encourage uniform distribution
        loss = self.num_experts * (avg_prob * expert_frac).sum()
        return loss

# Test MoE layer
moe = MoELayer(hidden_dim=256, num_experts=4, top_k=2).cuda()
x = torch.randn(8, 128, 256).cuda()

output, lb_loss = moe(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Load balance loss: {lb_loss.item():.4f}")
```

### Edge-Cloud Routing

Build a routing system that decides between edge and cloud inference.

__Requirements:__

- Implement routing logic based on:
  - Request complexity (input length, expected output)
  - Latency requirements
  - Edge device capabilities
  - Network conditions
- Simulate edge and cloud backends
- Measure routing decisions and latency

__Test your implementation:__
```python
import time
import random
from dataclasses import dataclass
from enum import Enum

class Backend(Enum):
    EDGE = "edge"
    CLOUD = "cloud"

@dataclass
class Request:
    prompt: str
    max_tokens: int
    latency_requirement_ms: float

class EdgeCloudRouter:
    def __init__(self, edge_capacity: int = 100, cloud_latency_ms: float = 50):
        self.edge_capacity = edge_capacity  # Max tokens edge can handle
        self.cloud_latency_ms = cloud_latency_ms  # Network latency to cloud
        self.edge_throughput = 10  # tokens/ms
        self.cloud_throughput = 50  # tokens/ms
    
    def route(self, request: Request) -> Backend:
        """Decide whether to route to edge or cloud."""
        # Estimate processing time
        edge_time = request.max_tokens / self.edge_throughput
        cloud_time = self.cloud_latency_ms + request.max_tokens / self.cloud_throughput
        
        # Check if edge can meet latency requirement
        if edge_time <= request.latency_requirement_ms:
            # Prefer edge if it can meet requirements
            if len(request.prompt) <= self.edge_capacity:
                return Backend.EDGE
        
        # Check if cloud can meet latency requirement
        if cloud_time <= request.latency_requirement_ms:
            return Backend.CLOUD
        
        # If neither can meet requirement, choose faster option
        return Backend.EDGE if edge_time < cloud_time else Backend.CLOUD
    
    def process(self, request: Request) -> tuple[str, float, Backend]:
        """Process request and return response with latency."""
        backend = self.route(request)
        
        if backend == Backend.EDGE:
            latency = request.max_tokens / self.edge_throughput
        else:
            latency = self.cloud_latency_ms + request.max_tokens / self.cloud_throughput
        
        # Simulate processing
        time.sleep(latency / 1000)
        
        response = f"Response from {backend.value}"
        return response, latency, backend

# Test router
router = EdgeCloudRouter()

# Generate test requests
requests = [
    Request("Short prompt", max_tokens=50, latency_requirement_ms=100),
    Request("Medium prompt " * 10, max_tokens=200, latency_requirement_ms=50),
    Request("Long prompt " * 50, max_tokens=500, latency_requirement_ms=200),
]

for req in requests:
    response, latency, backend = router.process(req)
    print(f"Tokens: {req.max_tokens}, Requirement: {req.latency_requirement_ms}ms, "
          f"Backend: {backend.value}, Actual: {latency:.1f}ms")
```

### Gradient Compression

Implement top-k gradient compression and measure communication reduction.

__Requirements:__

- Implement top-k gradient sparsification
- Add error feedback (residual accumulation)
- Measure:
  - Communication volume reduction
  - Training convergence impact
  - Compression overhead
- Compare with uncompressed baseline

__Test your implementation:__
```python
import torch
import torch.distributed as dist

class TopKCompressor:
    def __init__(self, k_ratio: float = 0.01):
        """Initialize top-k compressor.
        
        Args:
            k_ratio: Fraction of gradients to keep (0.01 = 1%)
        """
        self.k_ratio = k_ratio
        self.residuals = {}
    
    def compress(self, grad: torch.Tensor, name: str) -> dict:
        """Compress gradient using top-k selection with error feedback."""
        # Add residual from previous iteration
        if name in self.residuals:
            grad = grad + self.residuals[name]
        
        # Flatten for top-k selection
        flat_grad = grad.flatten()
        k = max(1, int(flat_grad.numel() * self.k_ratio))
        
        # Select top-k by magnitude
        _, indices = torch.topk(flat_grad.abs(), k)
        values = flat_grad[indices]
        
        # Store residual (unselected gradients)
        mask = torch.zeros_like(flat_grad)
        mask[indices] = 1
        self.residuals[name] = (flat_grad * (1 - mask)).view_as(grad)
        
        return {
            "values": values,
            "indices": indices,
            "shape": grad.shape,
            "original_size": grad.numel(),
            "compressed_size": k,
        }
    
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress gradient."""
        flat_grad = torch.zeros(compressed["original_size"], 
                                device=compressed["values"].device)
        flat_grad[compressed["indices"]] = compressed["values"]
        return flat_grad.view(compressed["shape"])

def benchmark_compression(model, compressor: TopKCompressor, num_steps: int = 10):
    """Benchmark gradient compression."""
    total_original = 0
    total_compressed = 0
    
    for step in range(num_steps):
        # Forward + backward
        x = torch.randn(32, 1024).cuda()
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Compress and sync gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                compressed = compressor.compress(param.grad, name)
                total_original += compressed["original_size"]
                total_compressed += compressed["compressed_size"]
                
                # Sync compressed gradients
                dist.all_reduce(compressed["values"])
                
                # Decompress
                param.grad = compressor.decompress(compressed)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    compression_ratio = total_original / total_compressed
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Communication reduction: {100 * (1 - 1/compression_ratio):.1f}%")

# Test compression
compressor = TopKCompressor(k_ratio=0.01)
benchmark_compression(model, compressor)
```

### Research Review

Read and summarize a recent paper on distributed AI.

__Requirements:__

- Select a paper on one of:
  - Mixture of Experts (e.g., DeepSeek-V3, Mixtral)
  - Edge AI (e.g., Apple Intelligence, on-device LLMs)
  - New parallelism strategies (e.g., context parallelism, expert parallelism)
- Summarize:
  - Problem addressed
  - Key technical contributions
  - Experimental results
  - Limitations and future work
- Discuss practical implications

__Template:__
```markdown
# Paper Review: [Paper Title]

## Problem Statement
- What problem does this paper address?
- Why is this problem important?

## Key Contributions
1. [Contribution 1]
2. [Contribution 2]
3. [Contribution 3]

## Technical Approach
- [Describe the main technical approach]
- [Key algorithms or architectures]

## Experimental Results
- [Main experimental findings]
- [Comparison with baselines]

## Limitations
- [What are the limitations of this work?]

## Practical Implications
- [How can this be applied in practice?]
- [What does this mean for distributed AI systems?]

## Questions for Discussion
1. [Question 1]
2. [Question 2]
```

__Suggested Papers:__
- DeepSeek-V3: Efficient MoE architecture with multi-head latent attention
- Mixtral 8x7B: Sparse mixture of experts for efficient inference
- Apple Intelligence: On-device LLM deployment strategies
- Ring Attention: Efficient context parallelism for long sequences


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Implement basic MoE layers with expert routing
- Design edge-cloud routing systems for hybrid inference
- Implement gradient compression for communication reduction
- Critically analyze research papers in distributed AI
- Understand emerging trends and their practical implications
