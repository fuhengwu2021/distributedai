# Chapter 11: Trends and Future of Distributed AI

## Overview

Throughout this book, we've covered the current state of distributed AI: DDP and FSDP for training, DeepSpeed ZeRO for very large models, vLLM and SGLang for inference, Slurm for job scheduling, production serving stacks, and benchmarking methodologies. These technologies form a solid foundation for building and deploying distributed AI systems today.

But the field is rapidly evolving. New architectures, optimization techniques, and deployment patterns emerge constantly. This chapter explores emerging trends and future directions in distributed AI systems. Readers will understand where the field is heading, evaluate new technologies and paradigms, and learn how to position themselves for the next wave of distributed AI innovations. The chapter covers MoE scaling, hybrid architectures, edge-cloud coordination, and emerging parallelism strategies, with practical code examples that demonstrate these concepts.

**Chapter Length:** 20 pages



## 1. The Evolution of Distributed AI: Where We Are and Where We're Going

### Current State (2024-2025)

**Key Achievements:**
- FSDP and DeepSpeed ZeRO enable training models with 100B+ parameters
- vLLM and SGLang have made distributed inference production-ready
- Kubernetes GPU scheduling is standard for cloud deployments
- Continuous batching and PagedAttention optimize memory efficiency

**Remaining Challenges:**
- Scaling to trillion-parameter models efficiently
- Reducing communication overhead in multi-node setups
- Balancing latency and throughput in inference
- Cost optimization for large-scale deployments

### Emerging Trends

**1. Mixture of Experts (MoE) Scaling**
- DeepSeek-V3 (671B MoE, 64 experts, 8 active)
- Efficient expert routing and load balancing
- Sparse activation patterns

**2. Hybrid Edge-Cloud Architectures**
- On-device LLM inference (Apple Intelligence, Samsung Gauss)
- Edge-cloud coordination and offloading
- Speculative decoding with edge-cloud collaboration

**3. Advanced Parallelism Strategies**
- Sequence parallelism for long contexts
- Expert parallelism for MoE models
- Hierarchical parallelism combinations

**4. Cost and Efficiency Focus**
- Model compression and quantization at scale
- Dynamic resource allocation
- Multi-tenant GPU sharing



## 2. Mixture of Experts (MoE) and Sparse Activation

### MoE Architecture Fundamentals

**Why MoE Matters:**
- Scales model capacity without proportional compute increase
- Only activates subset of experts per token
- Enables training larger models with same compute budget

**Basic MoE Implementation:**
```python
import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, expert_capacity, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        
        # Create expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Router (gating network)
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Router logits
        router_logits = self.router(x)  # [batch, seq, num_experts]
        
        # Top-k expert selection
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        # Flatten for processing
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k, 1)
        
        # Route to experts
        expert_outputs = []
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = (top_k_indices_flat == expert_id).any(dim=1)
            if mask.sum() == 0:
                continue  # Skip experts with no assigned tokens
            
            # Get tokens for this expert
            expert_tokens = x_flat[mask]
            
            # Get routing weights
            expert_weights = top_k_probs_flat[mask]
            expert_weights = expert_weights.squeeze(-1)
            
            # Process through expert
            expert_out = self.experts[expert_id](expert_tokens)
            
            # Weight by routing probability
            expert_out = expert_out * expert_weights.sum(dim=1, keepdim=True)
            
            expert_outputs.append((expert_out, mask))
        
        # Combine expert outputs
        output = torch.zeros_like(x_flat)
        for expert_out, mask in expert_outputs:
            output[mask] += expert_out
        
        return output.view(batch_size, seq_len, d_model)
```

### Expert Parallelism

**Distributing Experts Across GPUs:**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class ExpertParallelMoE(nn.Module):
    def __init__(self, d_model, num_experts, world_size, rank):
        super().__init__()
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        
        # Each rank owns a subset of experts
        experts_per_rank = num_experts // world_size
        self.local_expert_start = rank * experts_per_rank
        self.local_expert_end = (rank + 1) * experts_per_rank
        self.local_num_experts = experts_per_rank
        
        # Local experts
        self.local_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(self.local_num_experts)
        ])
        
        # Router (replicated on all ranks)
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Router logits (same on all ranks)
        router_logits = self.router(x)
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(router_logits, k=2, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        # Flatten
        x_flat = x.view(-1, d_model)
        top_k_indices_flat = top_k_indices.view(-1, 2)
        top_k_probs_flat = top_k_probs.view(-1, 2, 1)
        
        # Process local experts
        local_outputs = []
        for local_expert_id in range(self.local_num_experts):
            global_expert_id = self.local_expert_start + local_expert_id
            
            # Find tokens for this expert
            mask = (top_k_indices_flat == global_expert_id).any(dim=1)
            if mask.sum() == 0:
                continue
            
            expert_tokens = x_flat[mask]
            expert_weights = top_k_probs_flat[mask].squeeze(-1)
            
            # Process
            expert_out = self.local_experts[local_expert_id](expert_tokens)
            expert_out = expert_out * expert_weights.sum(dim=1, keepdim=True)
            
            local_outputs.append((expert_out, mask, global_expert_id))
        
        # All-gather expert outputs from all ranks
        # (Simplified - actual implementation needs proper communication)
        output = torch.zeros_like(x_flat)
        for expert_out, mask, expert_id in local_outputs:
            output[mask] += expert_out
        
        return output.view(batch_size, seq_len, d_model)
```

### Load Balancing in MoE

**Balancing Expert Utilization:**
```python
class LoadBalancedMoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
    
    def compute_load_balance_loss(self, router_probs):
        """
        Compute load balancing loss to encourage uniform expert usage.
        """
        # Average router probability per expert
        avg_probs = router_probs.mean(dim=[0, 1])  # [num_experts]
        
        # Target: uniform distribution
        target = torch.ones_like(avg_probs) / self.num_experts
        
        # Load balance loss (encourage uniform distribution)
        load_balance_loss = self.num_experts * torch.sum(avg_probs ** 2)
        
        return load_balance_loss
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        # Flatten for processing
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k, 1)
        
        # Route to experts and process
        output = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = (top_k_indices_flat == expert_id).any(dim=1)
            if mask.sum() == 0:
                continue
            
            # Get tokens for this expert
            expert_tokens = x_flat[mask]
            
            # Get routing weights
            expert_weights = top_k_probs_flat[mask]
            expert_weights = expert_weights.squeeze(-1)
            
            # Process through expert
            expert_out = self.experts[expert_id](expert_tokens)
            
            # Weight by routing probability
            expert_out = expert_out * expert_weights.sum(dim=1, keepdim=True)
            
            # Accumulate output
            output[mask] += expert_out
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute load balance loss
        load_balance_loss = self.compute_load_balance_loss(router_probs)
        
        return output, load_balance_loss
```



## 3. Hybrid Edge-Cloud Architectures

### Edge-Cloud Coordination Patterns

**Speculative Decoding with Edge-Cloud:**
```python
class EdgeCloudSpeculativeDecoding:
    """
    Use small edge model to generate draft tokens,
    large cloud model to verify and accept/reject.
    """
    def __init__(self, edge_model, cloud_model, tokenizer, max_draft_tokens=5):
        self.edge_model = edge_model  # Small, fast model
        self.cloud_model = cloud_model  # Large, accurate model
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self.max_draft_tokens = max_draft_tokens
        self.eos_token = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.pad_token_id
    
    def tokenize(self, text):
        """Tokenize text into token IDs"""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, return_tensors='pt')[0].tolist()
        else:
            # Fallback for simple tokenizers
            return self.tokenizer(text)
    
    def generate(self, prompt, max_tokens=100):
        tokens = self.tokenize(prompt)
        generated = []
        
        while len(generated) < max_tokens:
            # Edge: Generate draft tokens
            draft_tokens = self.edge_model.generate(
                tokens + generated,
                max_new_tokens=self.max_draft_tokens
            )
            
            # Cloud: Verify draft tokens
            verified_tokens = self.cloud_model.verify(
                tokens + generated,
                draft_tokens
            )
            
            generated.extend(verified_tokens)
            
            if verified_tokens and verified_tokens[-1] == self.eos_token:
                break
        
        return generated
```

### Dynamic Offloading Strategy

**Intelligent Edge-Cloud Routing:**
```python
class IntelligentOffloading:
    def __init__(self, edge_model, cloud_client, latency_threshold=100):
        self.edge_model = edge_model
        self.cloud_client = cloud_client
        self.latency_threshold = latency_threshold  # ms
    
    def route_request(self, request):
        """
        Decide whether to process on edge or cloud.
        """
        # Estimate edge latency
        edge_latency = self.estimate_edge_latency(request)
        
        # Estimate cloud latency
        cloud_latency = self.estimate_cloud_latency(request)
        
        # Consider confidence threshold
        edge_confidence = self.estimate_edge_confidence(request)
        
        # Decision logic
        if edge_latency < self.latency_threshold and edge_confidence > 0.8:
            return 'edge', self.edge_model.infer(request)
        else:
            return 'cloud', self.cloud_client.infer(request)
    
    def estimate_edge_latency(self, request):
        """Estimate edge processing latency"""
        # Based on request complexity, model size, etc.
        base_latency = 50  # ms
        complexity_factor = len(request) / 100
        return base_latency * (1 + complexity_factor)
    
    def estimate_cloud_latency(self, request):
        """Estimate cloud processing latency (includes network)"""
        network_latency = 20  # ms
        processing_latency = 30  # ms
        return network_latency + processing_latency
    
    def estimate_edge_confidence(self, request):
        """Estimate edge model confidence"""
        # Simplified: based on request characteristics
        return 0.85  # Placeholder
```



## 4. Advanced Parallelism Strategies

### Hierarchical Parallelism

**Combining Multiple Parallelism Strategies:**
```python
class HierarchicalParallelism:
    """
    Combine data parallelism, tensor parallelism, and pipeline parallelism.
    """
    def __init__(self, model, world_size, dp_size, tp_size, pp_size):
        self.world_size = world_size
        self.dp_size = dp_size  # Data parallel groups
        self.tp_size = tp_size  # Tensor parallel groups
        self.pp_size = pp_size  # Pipeline parallel stages
        
        assert dp_size * tp_size * pp_size == world_size
        
        # Create process groups
        self.dp_group = self.create_dp_group()
        self.tp_group = self.create_tp_group()
        self.pp_group = self.create_pp_group()
    
    def create_dp_group(self):
        """Create data parallel process group"""
        # Each data parallel group processes different data
        groups = []
        for dp_id in range(self.dp_size):
            ranks = [
                dp_id * self.tp_size * self.pp_size + 
                tp_id * self.pp_size + 
                pp_id
                for tp_id in range(self.tp_size)
                for pp_id in range(self.pp_size)
            ]
            groups.append(ranks)
        return groups
    
    def create_tp_group(self):
        """Create tensor parallel process group"""
        # Each tensor parallel group shards model parameters
        groups = []
        for tp_id in range(self.tp_size):
            ranks = [
                dp_id * self.tp_size * self.pp_size + 
                tp_id * self.pp_size + 
                pp_id
                for dp_id in range(self.dp_size)
                for pp_id in range(self.pp_size)
            ]
            groups.append(ranks)
        return groups
    
    def create_pp_group(self):
        """Create pipeline parallel process group"""
        # Each pipeline stage processes different layers
        groups = []
        for pp_id in range(self.pp_size):
            ranks = [
                dp_id * self.tp_size * self.pp_size + 
                tp_id * self.pp_size + 
                pp_id
                for dp_id in range(self.dp_size)
                for tp_id in range(self.tp_size)
            ]
            groups.append(ranks)
        return groups
```

### Sequence Parallelism for Long Contexts

**Handling Very Long Sequences:**
```python
class SequenceParallelAttention(nn.Module):
    """
    Split sequence dimension across GPUs for long contexts.
    """
    def __init__(self, d_model, num_heads, seq_parallel_size):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_parallel_size = seq_parallel_size
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, seq_rank, seq_world_size):
        """
        x: [batch, seq_len, d_model]
        seq_rank: rank in sequence parallel group
        seq_world_size: size of sequence parallel group
        """
        batch, seq_len, d_model = x.shape
        
        # Split sequence dimension
        local_seq_len = seq_len // seq_world_size
        start_idx = seq_rank * local_seq_len
        end_idx = (seq_rank + 1) * local_seq_len
        x_local = x[:, start_idx:end_idx, :]  # [batch, local_seq_len, d_model]
        
        # Local Q, K, V
        q_local = self.q_proj(x_local)
        k_local = self.k_proj(x_local)
        v_local = self.v_proj(x_local)
        
        # Reshape for attention
        q_local = q_local.view(batch, local_seq_len, self.num_heads, self.head_dim)
        k_local = k_local.view(batch, local_seq_len, self.num_heads, self.head_dim)
        v_local = v_local.view(batch, local_seq_len, self.num_heads, self.head_dim)
        
        # All-gather K and V from all sequence parallel ranks
        # (Need full K, V for attention computation)
        k_full = self.all_gather_sequence(k_local, seq_world_size)
        v_full = self.all_gather_sequence(v_local, seq_world_size)
        
        # Attention: Q_local @ K_full^T
        scores = torch.matmul(
            q_local.transpose(1, 2),  # [batch, num_heads, local_seq_len, head_dim]
            k_full.transpose(1, 2).transpose(-2, -1)  # [batch, num_heads, head_dim, full_seq_len]
        ) / (self.head_dim ** 0.5)
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, v_full.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, local_seq_len, d_model)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output
    
    def all_gather_sequence(self, tensor, world_size):
        """All-gather sequence chunks"""
        # Simplified - actual implementation uses dist.all_gather
        # This would gather from all sequence parallel ranks
        return tensor  # Placeholder
```



## 5. Cost Optimization and Multi-Tenancy

### Dynamic Resource Allocation

**Adaptive GPU Allocation:**
```python
class AdaptiveGPUAllocator:
    """
    Dynamically allocate GPUs based on workload characteristics.
    """
    def __init__(self, total_gpus, min_gpus_per_job=1):
        self.total_gpus = total_gpus
        self.min_gpus_per_job = min_gpus_per_job
        self.allocated_gpus = {}
        self.job_queue = []
    
    def allocate_for_job(self, job):
        """
        Allocate GPUs based on job requirements.
        """
        # Estimate required GPUs
        required_gpus = self.estimate_gpu_requirement(job)
        required_gpus = max(required_gpus, self.min_gpus_per_job)
        
        # Check availability
        available_gpus = self.total_gpus - sum(self.allocated_gpus.values())
        
        if available_gpus >= required_gpus:
            self.allocated_gpus[job.id] = required_gpus
            return required_gpus
        else:
            # Queue job or allocate partial resources
            self.job_queue.append(job)
            return None
    
    def estimate_gpu_requirement(self, job):
        """Estimate GPUs needed based on model size, batch size, etc."""
        model_size_gb = job.model_size / (1024 ** 3)  # GB
        batch_size = job.batch_size
        
        # Rough estimation
        gpus_needed = max(1, int(model_size_gb / 40))  # Assume 40GB per GPU
        gpus_needed = max(gpus_needed, int(batch_size / 32))  # Batch size consideration
        
        return min(gpus_needed, 8)  # Cap at 8 GPUs
```

### Multi-Tenant GPU Sharing

**Time-Sliced GPU Sharing:**
```python
class MultiTenantGPUScheduler:
    """
    Share GPUs across multiple tenants with time slicing.
    """
    def __init__(self, gpus, time_slice_ms=100):
        self.gpus = gpus
        self.time_slice_ms = time_slice_ms
        self.tenant_queues = {}
        self.current_tenant = None
        self.tenant_times = {}
    
    def add_tenant(self, tenant_id, priority=1):
        """Register a tenant"""
        self.tenant_queues[tenant_id] = []
        self.tenant_times[tenant_id] = 0
        self.tenant_priorities[tenant_id] = priority
    
    def schedule(self, current_time_ms):
        """
        Schedule GPU time across tenants.
        """
        # Round-robin with priority weighting
        if not self.tenant_queues:
            return
        
        # Select tenant based on priority and time since last allocation
        best_tenant = None
        best_score = -1
        
        for tenant_id in self.tenant_queues:
            if not self.tenant_queues[tenant_id]:
                continue
            
            priority = self.tenant_priorities[tenant_id]
            time_since_last = current_time_ms - self.tenant_times[tenant_id]
            score = priority * (1 + time_since_last / 1000)
            
            if score > best_score:
                best_score = score
                best_tenant = tenant_id
        
        if best_tenant:
            self.current_tenant = best_tenant
            self.tenant_times[best_tenant] = current_time_ms
            return best_tenant
        
        return None
```



## 6. Emerging Technologies and Research Directions

### Key Research Areas

**1. Communication-Efficient Training**
- Gradient compression techniques
- Sparse communication patterns
- Asynchronous updates

**2. Memory-Efficient Architectures**
- Flash Attention variants
- Memory-efficient attention mechanisms
- KV cache optimization

**3. Adaptive Systems**
- Dynamic batching strategies
- Adaptive parallelism selection
- Auto-tuning systems

**4. Sustainability and Green AI**
- Energy-efficient training
- Carbon-aware scheduling
- Model efficiency metrics

### Practical Code: Gradient Compression

```python
class GradientCompression:
    """
    Compress gradients to reduce communication overhead.
    """
    def __init__(self, compression_ratio=0.1, method='topk'):
        self.compression_ratio = compression_ratio
        self.method = method
    
    def compress(self, gradients):
        """Compress gradients"""
        if self.method == 'topk':
            return self.topk_compress(gradients)
        elif self.method == 'quantization':
            return self.quantize_compress(gradients)
        else:
            return gradients
    
    def quantize_compress(self, gradients):
        """Quantization-based compression"""
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Quantize to 8-bit
            grad_min = grad.min()
            grad_max = grad.max()
            scale = (grad_max - grad_min) / 255.0
            
            # Quantize
            quantized = ((grad - grad_min) / scale).round().byte()
            
            compressed[name] = {
                'quantized': quantized,
                'min': grad_min,
                'max': grad_max,
                'shape': grad.shape
            }
        
        return compressed
    
    def topk_compress(self, gradients):
        """Top-k sparsification"""
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            k = int(grad.numel() * self.compression_ratio)
            flat_grad = grad.flatten()
            
            # Select top-k values
            _, indices = torch.topk(flat_grad.abs(), k)
            values = flat_grad[indices]
            
            compressed[name] = {
                'values': values,
                'indices': indices,
                'shape': grad.shape
            }
        
        return compressed
    
    def decompress(self, compressed):
        """Decompress gradients"""
        gradients = {}
        for name, comp_data in compressed.items():
            if self.method == 'quantization':
                # Dequantize
                scale = (comp_data['max'] - comp_data['min']) / 255.0
                grad = comp_data['quantized'].float() * scale + comp_data['min']
                gradients[name] = grad.reshape(comp_data['shape'])
            else:
                # Top-k decompression
                grad = torch.zeros(comp_data['shape'])
                flat_grad = grad.flatten()
                flat_grad[comp_data['indices']] = comp_data['values']
                gradients[name] = grad
        
        return gradients
```



## 7. Preparing for the Future

### Skills to Develop

**1. Understanding Emerging Architectures**
- MoE models and expert routing
- Sparse activation patterns
- Hybrid model architectures

**2. Edge-Cloud Coordination**
- Speculative decoding
- Dynamic offloading
- Latency-accuracy tradeoffs

**3. Advanced Parallelism**
- Hierarchical parallelism
- Sequence parallelism
- Expert parallelism

**4. Cost and Efficiency**
- Multi-tenant systems
- Dynamic resource allocation
- Energy-efficient training

### Staying Current

**Resources:**
- Follow research papers (arXiv, conferences)
- Monitor open-source projects (vLLM, SGLang, DeepSpeed)
- Track industry developments (OpenAI, Anthropic, Google)
- Participate in communities (Hugging Face, PyTorch)

**Key Conferences:**
- NeurIPS, ICML, ICLR (research)
- MLSys, OSDI, SOSP (systems)
- Industry conferences (GTC, PyTorch Conference)



## Summary

This chapter has explored emerging trends and future directions in distributed AI:

1. **MoE architectures** enable scaling to larger models with efficient expert routing
2. **Hybrid edge-cloud systems** combine low latency with high accuracy
3. **Advanced parallelism strategies** handle increasingly complex workloads
4. **Cost optimization** becomes critical as systems scale
5. **Emerging technologies** continue to push boundaries

The field of distributed AI is rapidly evolving. The technologies covered in this book (DDP, FSDP, DeepSpeed, vLLM, SGLang) form the foundation, but new paradigms will continue to emerge. Staying current with research, experimenting with new tools, and understanding fundamental principles will prepare you for whatever comes next.



## Exercises

1. **Implement Basic MoE:** Create a simple MoE layer with 4 experts and top-2 routing. Test it on a small model.

2. **Edge-Cloud Routing:** Build a simple routing system that decides between edge and cloud based on request characteristics.

3. **Gradient Compression:** Implement top-k gradient compression and measure communication reduction.

4. **Research Review:** Read a recent paper on distributed AI (MoE, edge AI, or new parallelism) and summarize key insights.



## Further Reading

- DeepSeek-V3 Paper: Latest MoE architecture
- Apple Intelligence: On-device LLM deployment
- vLLM and SGLang: Latest inference optimizations
- DeepSpeed: Latest ZeRO and parallelism features
- Research papers on arXiv (cs.DC, cs.LG)
