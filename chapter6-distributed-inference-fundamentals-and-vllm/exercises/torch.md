\fancydividerwithicon[center]{hand.png}


## Exercises


### Implement KV Cache Management

Implement a simplified KV cache manager to understand how vLLM manages memory during inference.

__Requirements:__

- Class signature:
```python
class KVCacheManager:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, 
                 block_size: int, num_blocks: int):
        """Initialize KV cache with block-based allocation."""
        pass
    
    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a sequence. Return block indices."""
        pass
    
    def free(self, seq_id: int):
        """Free all blocks for a sequence."""
        pass
    
    def get_cache(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get K and V tensors for a sequence."""
        pass
```
- Use block-based allocation (like PagedAttention)
- Track free and allocated blocks
- Support dynamic growth as sequences generate tokens
- Implement defragmentation (optional)

__Test your implementation:__
```python
import torch

# Initialize cache manager
cache_mgr = KVCacheManager(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    block_size=16,
    num_blocks=1000
)

# Allocate for multiple sequences
seq1_blocks = cache_mgr.allocate(seq_id=1, num_tokens=100)
seq2_blocks = cache_mgr.allocate(seq_id=2, num_tokens=50)

print(f"Seq 1: {len(seq1_blocks)} blocks")
print(f"Seq 2: {len(seq2_blocks)} blocks")
print(f"Free blocks: {cache_mgr.num_free_blocks}")

# Free sequence 1
cache_mgr.free(seq_id=1)
print(f"After free: {cache_mgr.num_free_blocks} free blocks")
```

### Implement Continuous Batching

Create a simple continuous batching scheduler that processes requests as they arrive.

__Requirements:__

- Class signature:
```python
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int, max_seq_len: int):
        pass
    
    def add_request(self, request_id: int, prompt_tokens: list[int]):
        """Add a new request to the queue."""
        pass
    
    def step(self) -> dict:
        """Execute one generation step. Return completed requests."""
        pass
    
    def get_batch(self) -> list[int]:
        """Get current batch of request IDs."""
        pass
```
- Maintain a queue of pending requests
- Add new requests to running batch when space available
- Remove completed requests immediately
- Prioritize prefill for new requests

__Test your implementation:__
```python
scheduler = ContinuousBatchingScheduler(max_batch_size=8, max_seq_len=2048)

# Simulate incoming requests
import time
import random

for i in range(20):
    # Add request with random prompt length
    prompt_len = random.randint(10, 100)
    scheduler.add_request(request_id=i, prompt_tokens=list(range(prompt_len)))
    
    # Run generation steps
    for _ in range(5):
        completed = scheduler.step()
        if completed:
            print(f"Completed: {completed}")
    
    print(f"Step {i}: batch size = {len(scheduler.get_batch())}")
```

### Benchmark vLLM Throughput

Write a comprehensive benchmark for vLLM inference throughput under different configurations.

__Requirements:__

- Test variables:
  - Batch sizes: 1, 4, 8, 16, 32
  - Input lengths: 128, 512, 1024, 2048
  - Output lengths: 64, 128, 256, 512
  - Tensor parallelism: 1, 2, 4 GPUs
- Measure:
  - Throughput (tokens/second)
  - Time to first token (TTFT)
  - Time per output token (TPOT)
  - GPU memory utilization
- Generate plots and analysis

__Test your implementation:__
```python
from vllm import LLM, SamplingParams

def benchmark_vllm(
    model_name: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    tp_size: int = 1
) -> dict:
    """Benchmark vLLM with specific configuration."""
    llm = LLM(model=model_name, tensor_parallel_size=tp_size)
    
    # Generate test prompts
    prompts = ["Hello " * (input_len // 2)] * batch_size
    sampling_params = SamplingParams(max_tokens=output_len)
    
    # Warmup
    _ = llm.generate(prompts[:1], sampling_params)
    
    # Benchmark
    import time
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    return {
        "throughput": throughput,
        "latency": elapsed / batch_size,
        "ttft": outputs[0].metrics.first_token_time,
    }

# Run benchmarks
results = []
for batch_size in [1, 4, 8, 16]:
    for input_len in [128, 512, 1024]:
        metrics = benchmark_vllm(
            "Qwen/Qwen2.5-0.5B-Instruct",
            batch_size, input_len, output_len=128
        )
        results.append({"batch": batch_size, "input": input_len, **metrics})
        print(f"Batch={batch_size}, Input={input_len}: {metrics['throughput']:.1f} tok/s")
```

### Implement Speculative Decoding

Create a simple speculative decoding implementation to understand the technique.

__Requirements:__

- Use a small draft model to generate candidate tokens
- Verify candidates with the target model
- Implement rejection sampling for correctness
- Measure speedup vs standard decoding
- Handle variable acceptance rates

__Test your implementation:__
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, tokenizer, num_speculative: int = 4):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.num_speculative = num_speculative
    
    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate with speculative decoding."""
        pass
    
    def _draft_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate speculative tokens with draft model."""
        pass
    
    def _verify_tokens(self, input_ids: torch.Tensor, 
                       draft_tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Verify draft tokens with target model. Return accepted tokens."""
        pass

# Test speculative decoding
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").cuda()
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

decoder = SpeculativeDecoder(target, draft, tokenizer, num_speculative=4)

# Compare with standard decoding
prompt = "The capital of France is"
speculative_output = decoder.generate(prompt, max_tokens=50)
print(f"Speculative: {speculative_output}")
```

### Deploy vLLM with OpenAI-Compatible API

Set up a vLLM server with the OpenAI-compatible API and write a client to interact with it.

__Requirements:__

- Start vLLM server with OpenAI API
- Implement client for chat completions
- Support streaming responses
- Add request timeout and retry logic
- Measure latency distribution

__Test your implementation:__
```bash
# Start server (in terminal)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000
```

```python
import httpx
import asyncio
from typing import AsyncIterator

class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def chat(self, messages: list[dict], stream: bool = False) -> str | AsyncIterator[str]:
        """Send chat completion request."""
        pass
    
    async def list_models(self) -> list[str]:
        """List available models."""
        pass

async def main():
    client = VLLMClient()
    
    # List models
    models = await client.list_models()
    print(f"Available models: {models}")
    
    # Chat completion
    messages = [{"role": "user", "content": "What is machine learning?"}]
    response = await client.chat(messages)
    print(f"Response: {response}")
    
    # Streaming
    print("Streaming: ", end="")
    async for chunk in await client.chat(messages, stream=True):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Understand KV cache management and block-based allocation
- Implement continuous batching for efficient inference
- Benchmark and optimize vLLM throughput
- Understand speculative decoding and its tradeoffs
- Deploy and interact with vLLM's OpenAI-compatible API
- Analyze inference performance bottlenecks
