---
title: "DeepSpeed and ZeRO Optimization"
---

# Chapter 5 — DeepSpeed and ZeRO Optimization

This chapter explores DeepSpeed and its ZeRO optimizations — a set of techniques to shard optimizer states, gradients, and parameters across ranks. It covers practical config patterns, offloading options, pipeline/MoE integrations, and debugging tips.

## 1. ZeRO Optimization Fundamentals

ZeRO splits optimizer states, gradients, and parameters into stages to trade compute and communication for reduced memory footprint. Stages:

- Stage 1: shard optimizer states.  
- Stage 2: shard gradients and optimizer states.  
- Stage 3: shard parameters, gradients, and optimizer states (max memory reduction).

Each stage increases complexity but reduces memory requirements.

## 2. Configuring DeepSpeed for Multi-Node Training

DeepSpeed uses a JSON config to set ZeRO stage, offload targets, and optimizer settings. Important knobs:

- `zero_stage`, `offload_optimizer`, `offload_param`, `stage3_gather_16bit_weights_on_model_save`.
- IO tuning for NVMe offload and checkpointing strategy.

Example snippet (conceptual):

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "nvme", "nvme_path": "/nvme"}
  }
}
```

## 3. Using Pipeline and MoE Parallelism

DeepSpeed supports pipeline parallelism and mixture-of-experts (MoE) parallelism layered on top of ZeRO. When using pipeline parallelism, watch microbatch scheduling and memory locality. MoE requires routing large conditional compute and can increase communication due to expert shuffling.

## 4. CPU/NVMe Offloading Techniques

Offloading large optimizer or parameter state to CPU/NVMe enables training huge models on smaller GPU fleets. Key considerations:

- IO throughput and latency (prefer fast NVMe and parallel IO strategies).  
- Overlap IO with computation to mask offload cost.  
- Monitor CPU memory and system load to avoid contention.

## 5. Benchmarking and Debugging DeepSpeed

Use DeepSpeed's logging and throughput tools. Common issues:

- Checkpoint mismatches when using stage3_gather flags — validate model save/load across ranks.
- Performance regressions when offload IO is saturated — profile NVMe bandwidth.

## Hands-on Examples

1. DeepSpeed config templates for ZeRO Stages 1/2/3.  
2. Multi-node launch examples using `deepspeed --num_gpus` and SLURM.

## Best Practices

- Start with ZeRO Stage 1 or FSDP for simpler debugging, then move to Stage 2/3 as needed.
- Benchmark offload paths early to ensure IO or CPU won't bottleneck training.
- Combine offload with mixed precision and activation checkpointing for maximal memory savings.

---

References: DeepSpeed docs, ZeRO papers, community examples.
---
title: "DeepSpeed and ZeRO Optimization"
---

# Chapter 5 — DeepSpeed and ZeRO Optimization

Status: TODO — draft placeholder

Chapter headings:
1. ZeRO Optimization Fundamentals
2. Configuring DeepSpeed for Multi-Node Training
3. Using Pipeline and MoE Parallelism
4. CPU/NVMe Offloading Techniques
5. Benchmarking and Debugging DeepSpeed

TODO: Provide example DeepSpeed configs and benchmarks.
