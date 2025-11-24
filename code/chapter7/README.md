# Chapter 7 code snippets

This folder contains runnable snippets extracted from Chapter 7 (`SGLang and Advanced Inference Architectures`).

Files:

- `chunked_prefill.py`: a runnable stub that simulates chunked prefill behavior and demonstrates how to split prompts into chunks and call a model's `forward_prefill` method.
- `router_example.py`: a tiny router session-affinity example that selects a runner with hot KV cache or picks an available runner.
- `genai_bench_run.sh`: wrapper for the `genai-bench` example command shown in the chapter (dry-run by default).
- `operator_fusion_profiling.py`: a small microbenchmark simulator to identify hot operators to target for fusion.

Quick run examples:

```bash
# Run the chunked prefill simulation
python3 chunked_prefill.py

# Run the router example
python3 router_example.py

# Simulate operator profiling
python3 operator_fusion_profiling.py

# Run genai-bench wrapper (dry-run)
bash genai_bench_run.sh my_sglang_workload.yaml results.json
```

Notes:
- Replace the dummy model and tokenizer in `chunked_prefill.py` with your real model runtime and ensure proper device placement.
- Use real profiling output (from `torch.profiler` or Nsight) to populate operator latencies for `operator_fusion_profiling.py`.
- The `genai_bench` wrapper is a convenience; install `genai-bench` and uncomment the actual command to run real benchmarks.
