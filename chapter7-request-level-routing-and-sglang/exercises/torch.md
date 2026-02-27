\fancydividerwithicon[center]{hand.png}


## Exercises


### Implement RadixAttention Cache

Implement a simplified RadixAttention cache to understand SGLang's prefix caching mechanism.

__Requirements:__

- Class signature:
```python
class RadixCache:
    def __init__(self, max_size: int):
        """Initialize radix tree cache."""
        pass
    
    def insert(self, token_ids: list[int], kv_cache: torch.Tensor) -> int:
        """Insert KV cache for token sequence. Return cache ID."""
        pass
    
    def lookup(self, token_ids: list[int]) -> tuple[int, torch.Tensor | None]:
        """Find longest matching prefix. Return (match_length, kv_cache)."""
        pass
    
    def evict(self, num_entries: int):
        """Evict least recently used entries."""
        pass
```
- Use a trie (prefix tree) structure
- Store KV cache at each node
- Implement LRU eviction policy
- Track cache hit rate

__Test your implementation:__
```python
import torch

cache = RadixCache(max_size=1000)

# Insert some sequences
seq1 = [1, 2, 3, 4, 5]
seq2 = [1, 2, 3, 6, 7]
seq3 = [1, 2, 8, 9]

kv1 = torch.randn(5, 32, 128)  # 5 tokens, 32 heads, 128 dim
kv2 = torch.randn(5, 32, 128)
kv3 = torch.randn(4, 32, 128)

cache.insert(seq1, kv1)
cache.insert(seq2, kv2)
cache.insert(seq3, kv3)

# Test lookup
test_seq = [1, 2, 3, 4, 10, 11]
match_len, kv = cache.lookup(test_seq)
print(f"Query: {test_seq}")
print(f"Match length: {match_len}")  # Should be 4 (prefix [1,2,3,4])
print(f"KV cache shape: {kv.shape if kv is not None else None}")
```

### Benchmark Prefix Caching

Compare SGLang's prefix caching performance against vLLM for workloads with shared prefixes.

__Requirements:__

- Create test workloads with varying prefix sharing:
  - No sharing (random prompts)
  - Moderate sharing (same system prompt)
  - High sharing (few-shot examples)
- Measure:
  - Time to first token (TTFT)
  - Cache hit rate
  - Memory usage
  - Overall throughput
- Generate comparison plots

__Test your implementation:__
```python
import sglang as sgl
from vllm import LLM

def create_shared_prefix_workload(num_requests: int, prefix_len: int, unique_len: int):
    """Create workload with shared prefix."""
    shared_prefix = "You are a helpful assistant. " * (prefix_len // 30)
    prompts = [
        shared_prefix + f"Question {i}: What is {i}?" 
        for i in range(num_requests)
    ]
    return prompts

def benchmark_sglang(prompts: list[str]) -> dict:
    """Benchmark SGLang with prefix caching."""
    # Initialize SGLang runtime
    runtime = sgl.Runtime(model_path="Qwen/Qwen2.5-0.5B-Instruct")
    
    import time
    start = time.time()
    
    # Process requests
    for prompt in prompts:
        response = runtime.generate(prompt, max_new_tokens=50)
    
    elapsed = time.time() - start
    
    return {
        "total_time": elapsed,
        "cache_hit_rate": runtime.get_cache_stats()["hit_rate"],
    }

# Compare with different prefix sharing levels
for prefix_len in [0, 100, 500, 1000]:
    prompts = create_shared_prefix_workload(100, prefix_len, unique_len=50)
    
    sglang_results = benchmark_sglang(prompts)
    print(f"Prefix={prefix_len}: SGLang time={sglang_results['total_time']:.2f}s, "
          f"hit_rate={sglang_results['cache_hit_rate']:.2%}")
```

### Implement Constrained Decoding

Create a constrained decoding implementation that enforces JSON schema output.

__Requirements:__

- Support JSON schema constraints
- Implement grammar-based token filtering
- Handle nested structures (objects, arrays)
- Measure overhead vs unconstrained decoding

__Test your implementation:__
```python
import sglang as sgl
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

@sgl.function
def extract_person(s, text: str):
    s += "Extract person information from: " + text + "\n"
    s += "Output JSON:\n"
    s += sgl.gen("json_output", max_tokens=200, regex=Person.model_json_schema())

# Test constrained decoding
runtime = sgl.Runtime(model_path="Qwen/Qwen2.5-0.5B-Instruct")

text = "John Smith is a 35-year-old software engineer."
state = extract_person.run(text=text)

print(f"Output: {state['json_output']}")

# Validate output
try:
    person = Person.model_validate_json(state['json_output'])
    print(f"Parsed: name={person.name}, age={person.age}, occupation={person.occupation}")
except Exception as e:
    print(f"Validation failed: {e}")
```

### Implement Multi-Turn Conversation with State

Build a multi-turn conversation system using SGLang's state management.

__Requirements:__

- Maintain conversation history efficiently
- Reuse KV cache across turns
- Support branching conversations (explore multiple responses)
- Measure memory efficiency vs naive approach

__Test your implementation:__
```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, system_prompt: str):
    s += sgl.system(system_prompt)
    
    # First turn
    s += sgl.user("What is machine learning?")
    s += sgl.assistant(sgl.gen("response1", max_tokens=200))
    
    # Second turn (reuses KV cache from first turn)
    s += sgl.user("Can you give me a simple example?")
    s += sgl.assistant(sgl.gen("response2", max_tokens=200))
    
    # Third turn
    s += sgl.user("How is it different from traditional programming?")
    s += sgl.assistant(sgl.gen("response3", max_tokens=200))

# Run multi-turn conversation
runtime = sgl.Runtime(model_path="Qwen/Qwen2.5-0.5B-Instruct")

state = multi_turn_chat.run(
    system_prompt="You are a helpful AI assistant that explains concepts clearly."
)

print("Turn 1:", state["response1"][:100], "...")
print("Turn 2:", state["response2"][:100], "...")
print("Turn 3:", state["response3"][:100], "...")

# Check cache efficiency
print(f"Cache stats: {runtime.get_cache_stats()}")
```

### Implement Parallel Generation with Fork

Use SGLang's fork mechanism to generate multiple responses in parallel.

__Requirements:__

- Fork conversation state to explore multiple paths
- Generate diverse responses with different parameters
- Implement best-of-N selection
- Measure speedup vs sequential generation

__Test your implementation:__
```python
import sglang as sgl

@sgl.function
def parallel_generation(s, prompt: str, num_samples: int = 4):
    s += sgl.user(prompt)
    
    # Fork to generate multiple responses
    forks = s.fork(num_samples)
    
    for i, fork in enumerate(forks):
        fork += sgl.assistant(
            sgl.gen(f"response_{i}", max_tokens=200, temperature=0.8)
        )
    
    # Join forks
    s += sgl.join(forks)

@sgl.function
def best_of_n(s, prompt: str, n: int = 4):
    """Generate N responses and select the best one."""
    s += sgl.user(prompt)
    
    # Generate N candidates
    forks = s.fork(n)
    responses = []
    
    for i, fork in enumerate(forks):
        fork += sgl.assistant(sgl.gen(f"candidate_{i}", max_tokens=200))
        responses.append(fork[f"candidate_{i}"])
    
    # Score and select best (simplified: longest response)
    best_idx = max(range(n), key=lambda i: len(responses[i]))
    s += sgl.select("best", responses[best_idx])

# Test parallel generation
runtime = sgl.Runtime(model_path="Qwen/Qwen2.5-0.5B-Instruct")

import time

# Sequential baseline
start = time.time()
for i in range(4):
    state = runtime.generate("Explain quantum computing", max_new_tokens=200)
sequential_time = time.time() - start

# Parallel with fork
start = time.time()
state = parallel_generation.run(prompt="Explain quantum computing", num_samples=4)
parallel_time = time.time() - start

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time / parallel_time:.2f}x")
```


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Understand RadixAttention and prefix caching mechanisms
- Benchmark and compare prefix caching performance
- Implement constrained decoding with JSON schemas
- Build efficient multi-turn conversation systems
- Use SGLang's fork mechanism for parallel generation
- Optimize inference workloads with shared prefixes
