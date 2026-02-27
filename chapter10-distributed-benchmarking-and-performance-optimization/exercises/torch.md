\fancydividerwithicon[center]{hand.png}


## Exercises


### Multiple Choice Questions

__1. What is the primary purpose of warmup iterations in benchmarking?__

A) To increase the total benchmark runtime
B) To eliminate cold-start effects like JIT compilation and cache warming
C) To reduce GPU memory usage
D) To test error handling code paths

__2. When measuring Time to First Token (TTFT), which phases are included?__

A) Only the model forward pass
B) Tokenization, prefill computation, and first token generation
C) All token generation including the final token
D) Only network latency

__3. A distributed training job shows 85% scaling efficiency at 8 GPUs. What does this mean?__

A) 85% of GPU memory is utilized
B) The job achieves 85% of ideal linear speedup (6.8x instead of 8x)
C) 85% of time is spent on computation
D) 15% of gradients are dropped

__4. Which metric best captures user-perceived responsiveness in a streaming LLM application?__

A) Tokens Per Second (TPS)
B) End-to-End Latency (E2E)
C) Time to First Token (TTFT)
D) GPU Utilization

__5. In PyTorch Profiler output, which operations indicate communication overhead?__

A) `aten::matmul` and `aten::linear`
B) `nccl:all_reduce` and `nccl:all_gather`
C) `aten::relu` and `aten::gelu`
D) `cuda::memcpy` only

__6. What does Amdahl's Law predict about scaling efficiency?__

A) Linear scaling is always achievable with more GPUs
B) Sequential portions of code limit maximum speedup regardless of parallelization
C) Communication overhead decreases with more GPUs
D) Memory usage scales linearly with GPU count

__7. When benchmarking inference throughput, why is it important to test with varying concurrency levels?__

A) To find the maximum batch size
B) To identify the saturation point where adding more requests doesn't increase throughput
C) To measure cold start latency
D) To test model accuracy

__8. Which approach correctly measures communication time separately from computation in DDP training?__

A) Measure total training time and divide by number of GPUs
B) Use `torch.cuda.synchronize()` before and after gradient synchronization
C) Count the number of parameters in the model
D) Measure only the forward pass time

__9. What is the relationship between Inter-token Latency (ITL) and Time Per Output Token (TPOT)?__

A) ITL measures the first token, TPOT measures subsequent tokens
B) They measure the same thing - the time between consecutive output tokens
C) ITL is for training, TPOT is for inference
D) TPOT includes network latency, ITL does not

__10. When comparing model accuracy between distributed and single-GPU training, what indicates a problem?__

A) Identical loss curves
B) Statistically significant accuracy degradation beyond expected variance
C) Faster convergence with more GPUs
D) Lower memory usage per GPU

__11. Which percentile metric is most important for SLA compliance in production inference?__

A) P50 (median) latency
B) P95 or P99 latency
C) Average latency
D) Minimum latency

__12. What is the correct formula for scaling efficiency?__

A) `efficiency = throughput_N / throughput_1`
B) `efficiency = (throughput_N × N) / throughput_1`
C) `efficiency = throughput_N / (throughput_1 × N)`
D) `efficiency = N / throughput_N`

__13. When profiling network communication, what does `nvidia-smi topo -m` show?__

A) GPU memory usage
B) GPU interconnect topology (NVLink, PCIe connections)
C) Network bandwidth usage
D) CUDA kernel execution times

__14. Why should benchmarks be run multiple times with statistical analysis?__

A) To increase total runtime
B) To account for variance and ensure reproducible, statistically significant results
C) To warm up the GPU
D) To test different model configurations

### Short Answer Questions

15. Explain why communication overhead typically increases as you scale to more GPUs in distributed training.


16. A benchmark shows P50 latency of 50ms and P99 latency of 500ms. What does this distribution suggest about the system's behavior?


17. Describe two ways to reduce communication overhead in distributed training without changing the model architecture.


18. Why might a quantized model (INT8) show different accuracy on different inference engines (vLLM vs SGLang) even with the same weights?


19. What is the difference between "samples per second" in training benchmarks and "tokens per second" in inference benchmarks?


## Answer Key

1. B - Warmup eliminates cold-start effects
2. B - TTFT includes tokenization through first token generation
3. B - 85% of ideal linear speedup
4. C - TTFT captures initial responsiveness
5. B - NCCL operations indicate communication
6. B - Sequential code limits maximum speedup
7. B - Find throughput saturation point
8. B - Use synchronization barriers for accurate timing
9. B - Both measure time between consecutive tokens
10. B - Statistically significant accuracy degradation
11. B - P95/P99 for SLA compliance
12. C - `efficiency = throughput_N / (throughput_1 × N)`
13. B - GPU interconnect topology
14. B - Statistical significance and reproducibility

__Short Answer Guidelines:__

15. More GPUs means more gradient synchronization across more nodes, increasing AllReduce communication volume and potentially crossing slower interconnects (PCIe vs NVLink).

16. High P99/P50 ratio (10x) indicates tail latency issues - most requests are fast but some experience significant delays, possibly due to garbage collection, resource contention, or cold cache effects.

17. Gradient accumulation (fewer sync points), gradient compression (reduced data volume), overlapping communication with computation, using faster interconnects.

18. Different quantization implementations, kernel optimizations, and numerical precision handling can cause slight accuracy variations even with identical weights.

19. Samples/second measures complete training examples processed; tokens/second measures individual token generation rate, which varies with sequence length and batching.


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Understand key benchmarking concepts and their practical implications
- Interpret benchmark metrics correctly (TTFT, ITL, TPS, scaling efficiency)
- Identify common pitfalls in distributed system benchmarking
- Analyze profiling output to diagnose performance bottlenecks
- Apply statistical thinking to benchmark results
