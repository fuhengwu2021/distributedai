Packt Proposal Outline
Book Title (Working):
Modern Distributed AI Systems: Training, Inference, and Serving at Scale
-------------------------------------------------------------

**Note:** Every chapter includes hands-on, runnable code examples that readers can execute on their own infrastructure (single-node or multi-node setups). All examples are production-ready and tested.


PART I — Foundations and Distributed Training
============================================

### Chapter 1 — Introduction to Modern Distributed AI

**Summary:**
This chapter introduces why distributed architectures have become essential in the modern AI and LLM era. It explains the evolution from classic ML to large-scale models, the computational bottlenecks that arise, and how distributed training, inference, and serving fit into today's AI systems.

**Hands-on Exercises:**
• Set up a simple single-GPU baseline training script
• Profile memory and compute usage to identify bottlenecks
• Run a basic multi-GPU setup to see distribution benefits

**Learning Objectives:**
• Understand why distribution is required in LLM/AI workloads
• Recognize bottlenecks in compute, memory, and networking
• Differentiate training, inference, and serving workloads

**Key Topics:**
• Model scale, dataset scale, latency requirements
• Distributed-first system design
• Cloud-native AI and GPU clusters
• Overview of training vs inference vs serving
• RDMA & high-speed networking basics

**Code Examples:**
• Simple PyTorch model with memory profiling
• Basic NCCL communication test script
• GPU topology detection script

---

### Chapter 2 — GPU Hardware, Networking, and Parallelism Strategies

**Summary:**
This chapter explores modern GPU hardware, memory hierarchy, interconnect technologies, and parallelism fundamentals. Readers will understand why topology affects performance and learn to choose the right parallelism strategy.

**Hands-on Exercises:**
• Benchmark NVLink vs PCIe bandwidth with custom scripts
• Test AllReduce performance across different network topologies
• Implement and compare data parallelism vs tensor parallelism on a small model
• Profile communication overhead in distributed setups

**Learning Objectives:**
• Understand GPU memory and compute structure
• Compare NVLink, PCIe, NVSwitch
• Learn high-performance distributed network technologies
• Compare major parallelism strategies (DP, TP, PP, FSDP, ZeRO)
• Know when to use each parallelism approach

**Key Topics:**
• GPU architecture basics
• RDMA, InfiniBand, RoCE, PFC
• Multi-node topologies
• Communication bandwidth benchmarks
• Data, Tensor, Pipeline, Sequence Parallelism
• FSDP and ZeRO concepts
• AllReduce, Broadcast, Scatter/Gather patterns

**Code Examples:**
• GPU topology detection and visualization
• Network bandwidth benchmark script
• Simple data parallelism implementation from scratch
• Tensor parallelism example with manual sharding
• Communication pattern visualization tools

---

### Chapter 3 — Distributed Training with PyTorch DDP

**Summary:**
An in-depth, hands-on guide to PyTorch DistributedDataParallel (DDP) for single-node and multi-node training. Readers will build, debug, and optimize real DDP training jobs.

**Hands-on Exercises:**
• Implement single-node multi-GPU DDP training from scratch
• Set up multi-node DDP training with proper initialization
• Debug common DDP issues (hanging, deadlocks, mismatched shapes)
• Optimize DDP with gradient accumulation and mixed precision
• Profile and improve DDP communication efficiency
• Implement checkpointing and resuming for DDP jobs

**Learning Objectives:**
• Learn DDP architecture and gradient sync mechanisms
• Implement single- and multi-node DDP
• Debug and optimize DDP workloads
• Understand bucketization and communication overlap

**Key Topics:**
• DDP internals and initialization
• Bucketization strategies
• Overlapping compute & communication
• Multi-node debugging techniques
• Checkpointing and fault tolerance
• Mixed precision with DDP

**Code Examples:**
• Complete DDP training script for transformer model
• Multi-node launch script with SLURM/torchrun
• DDP debugging utilities
• Performance profiling and optimization examples
• Checkpoint save/load implementation

---

### Chapter 4 — Scaling with Fully Sharded Data Parallel (FSDP)

**Summary:**
This chapter provides hands-on experience with PyTorch FSDP, its sharding strategies, and memory optimization techniques. Readers will train large models that don't fit on a single GPU.

**Hands-on Exercises:**
• Train a model larger than single-GPU memory using FSDP
• Compare full-shard, grad-shard, and mixed-shard strategies
• Implement CPU and NVMe offloading
• Use activation checkpointing with FSDP
• Profile memory usage and optimize sharding strategy
• Build a multi-node FSDP training pipeline

**Learning Objectives:**
• Understand full-shard, grad-shard, and mixed-shard strategies
• Use activation checkpointing effectively
• Compare FSDP with DeepSpeed ZeRO
• Plan memory budgets for large models

**Key Topics:**
• FSDP sharding strategies
• CPU/NVMe offloading
• Memory budget planning
• FSDP tradeoffs and optimization
• Activation checkpointing integration
• Multi-node FSDP scaling

**Code Examples:**
• FSDP training script with different sharding strategies
• Memory profiling and optimization tools
• CPU offloading configuration examples
• Multi-node FSDP setup scripts
• Comparison benchmarks: FSDP vs DDP vs ZeRO

---

### Chapter 5 — Beyond State Sharding with DeepSpeed and Megatron

**Summary:**
A practical, code-heavy chapter covering DeepSpeed engine and ZeRO optimizations for massive model training. Readers will build production-ready DeepSpeed training jobs.

**Hands-on Exercises:**
• Configure and run DeepSpeed ZeRO Stage 1, 2, and 3
• Build multi-node DeepSpeed training pipeline
• Use DeepSpeed pipeline parallelism
• Implement optimizer and activation partitioning
• Set up CPU/NVMe offloading with DeepSpeed
• Benchmark DeepSpeed performance vs FSDP
• Debug DeepSpeed training issues

**Learning Objectives:**
• Learn ZeRO Stage 1/2/3 implementations
• Use DeepSpeed pipeline & MoE parallelism
• Build multi-node DeepSpeed training
• Optimize memory usage with DeepSpeed

**Key Topics:**
• ZeRO optimizer and activation partitioning
• DeepSpeed configuration files
• Offloading strategies
• Multi-node scaling with DeepSpeed
• Pipeline parallelism with DeepSpeed
• DeepSpeed vs FSDP comparison

**Code Examples:**
• Complete DeepSpeed training script with config
• ZeRO Stage 1/2/3 comparison examples
• DeepSpeed multi-node launch scripts
• Memory optimization examples
• Performance benchmarking scripts
• DeepSpeed checkpointing and resuming


PART II — Distributed Inference and Production Deployment
=========================================================

### Chapter 6 — Distributed Inference Fundamentals and vLLM

**Summary:**
A foundational chapter explaining distributed inference, followed by a deep dive into vLLM. Readers will build and optimize vLLM-based inference systems with hands-on examples.

**Hands-on Exercises:**
• Set up vLLM server with single and multi-GPU configurations
• Implement continuous batching with custom workloads
• Profile PagedAttention memory efficiency
• Build a multi-node vLLM inference cluster
• Optimize vLLM for latency vs throughput
• Benchmark vLLM performance with different models
• Integrate vLLM with custom API endpoints

**Learning Objectives:**
• Understand inference bottlenecks and tradeoffs
• Learn vLLM internal architecture (PagedAttention, KV cache)
• Optimize multi-GPU inference
• Benchmark vLLM workloads effectively

**Key Topics:**
• Latency vs throughput tradeoffs
• Prompt processing vs generation phases
• KV cache management
• vLLM PagedAttention mechanism
• KV block management
• Multi-GPU pipelines
• Engine scheduling and continuous batching

**Code Examples:**
• vLLM server setup and configuration
• Custom vLLM client with batching
• Multi-node vLLM deployment scripts
• Performance profiling and optimization tools
• Benchmarking scripts for throughput/latency
• API integration examples

---

### Chapter 7 — Request-Level Routing and SGLang

**Summary:**
Covers SGLang's lightweight runtime, efficient operator fusion, and multi-node inference. Also explores DeepSeek-style architectures and CPU-based inference options.

**Hands-on Exercises:**
• Set up SGLang inference server
• Build SGLang multi-node inference cluster
• Integrate genai-bench for benchmarking
• Implement router-based inference system
• Build distributed CPU inference with Ray Serve
• Compare SGLang vs vLLM performance
• Implement hybrid CPU/GPU serving

**Learning Objectives:**
• Understand SGLang internals and design
• Build SGLang inference clusters
• Benchmark SGLang workloads with genai-bench
• Understand distributed KV sharing patterns
• Build router-based inference systems
• Implement CPU-based distributed inference

**Key Topics:**
• SGLang design and operator fusion
• genai-bench integration
• Multi-node SGLang inference
• DeepSeek-style chunked prefill
• KV cache sharing strategies
• Router-based scheduling
• CPU autoscaling with Ray Serve
• Hybrid CPU/GPU serving

**Code Examples:**
• SGLang server setup and multi-node configuration
• genai-bench integration examples
• Router-based inference implementation
• Ray Serve CPU inference examples
• Performance comparison scripts
• Hybrid serving architecture examples

---

### Chapter 8 — Kubernetes for AI Workloads

**Summary:**
Hands-on guide to deploying distributed AI systems on Kubernetes, covering GPU scheduling, autoscaling, and production best practices.

**Hands-on Exercises:**
• Set up Kubernetes cluster with GPU support
• Deploy GPU operator and device plugins
• Create GPU-enabled training jobs with proper scheduling
• Implement HPA and KEDA autoscaling for inference workloads
• Configure node labeling, taints, and tolerations
• Set up GPU resource quotas and limits
• Deploy multi-node training jobs on K8s
• Implement queue-based autoscaling

**Learning Objectives:**
• Deploy AI workloads on Kubernetes
• Use GPU operators & device plugins
• Tune scheduling for GPU workloads
• Implement autoscaling strategies

**Key Topics:**
• GPU operator setup
• Node labeling/taints for GPU nodes
• Efficient GPU scheduling
• HPA vs KEDA autoscaling
• Queue-based scaling
• Resource management and quotas

**Code Examples:**
• Complete K8s deployment manifests for training
• GPU operator installation scripts
• Autoscaling configurations (HPA/KEDA)
• Multi-node job deployment examples
• Monitoring and observability setup
• Troubleshooting guides and scripts

---

### Chapter 9 — Production LLM Serving Stack

**Summary:**
A full-stack, hands-on guide to building production LLM serving systems with model runners, API gateways, observability, and reliability features.

**Hands-on Exercises:**
• Build complete serving stack with model runner, API gateway, and tokenizer service
• Implement multi-model routing and load balancing
• Set up A/B testing and canary deployments
• Add distributed tracing and monitoring
• Implement fault tolerance and health checks
• Build request queuing and rate limiting
• Set up cost tracking and optimization
• Deploy end-to-end production system

**Learning Objectives:**
• Build a real production serving stack
• Handle multi-model routing
• Implement A/B deployments and canary rollouts
• Add observability and monitoring

**Key Topics:**
• Model runners (vLLM, SGLang, custom)
• API gateway design
• Multi-model routing
• Canary rollout strategies
• Distributed tracing (OpenTelemetry)
• Monitoring and alerting
• Fault tolerance patterns
• Cost optimization

**Code Examples:**
• Complete serving stack implementation
• API gateway with routing logic
• Canary deployment scripts
• OpenTelemetry tracing setup
• Prometheus/Grafana monitoring
• Health check and fault tolerance
• Cost tracking utilities


PART III — Benchmarking and Specialized Paradigms
==================================================

### Chapter 10 — Distributed Benchmarking and Performance Optimization

**Summary:**
Comprehensive guide to benchmarking distributed training and inference systems, using tools like genai-bench, MLPerf, and custom scripts. Includes optimization techniques.

**Hands-on Exercises:**
• Set up genai-bench for inference benchmarking
• Create custom training benchmarks
• Measure scaling efficiency across nodes
• Identify and fix network bottlenecks
• Benchmark on AWS and OCI cloud platforms
• Implement proper warmup and steady-state testing
• Analyze variance and statistical significance
• Optimize based on benchmark results

**Learning Objectives:**
• Use industry-standard benchmarking tools
• Design correct benchmarking methodology
• Interpret results and identify bottlenecks
• Perform cloud-based benchmarking
• Optimize systems based on metrics

**Key Topics:**
• genai-bench setup and usage
• MLPerf and torchbench
• Throughput/latency measurement
• Scaling efficiency analysis
• Network bottleneck identification
• Warmup/steady state methodology
• Multi-node scaling tests
• Cloud benchmarking (AWS/OCI)
• Statistical analysis of results

**Code Examples:**
• genai-bench integration scripts
• Custom benchmarking framework
• Scaling efficiency measurement tools
• Network profiling scripts
• Cloud benchmarking automation
• Result analysis and visualization
• Optimization scripts based on findings

---

### Chapter 11 — Trends and Future of Distributed AI

**Summary:**
Explores emerging trends and future directions in distributed AI systems. Covers Mixture of Experts (MoE) scaling, hybrid edge-cloud architectures, advanced parallelism strategies, and cost optimization techniques. Provides practical code examples for MoE implementation, expert parallelism, edge-cloud coordination, and emerging technologies.

**Hands-on Exercises:**
• Implement basic MoE layer with expert routing
• Build expert parallelism for distributed MoE training
• Create edge-cloud speculative decoding system
• Implement sequence parallelism for long contexts
• Build dynamic GPU allocation system
• Implement gradient compression for communication efficiency
• Analyze emerging trends and research directions

**Learning Objectives:**
• Understand MoE architectures and expert routing
• Implement advanced parallelism strategies (hierarchical, sequence, expert)
• Build hybrid edge-cloud coordination systems
• Optimize costs with multi-tenant GPU sharing
• Evaluate emerging technologies and research directions
• Prepare for future trends in distributed AI

**Key Topics:**
• Mixture of Experts (MoE) architectures and scaling
• Expert parallelism and load balancing
• Hybrid edge-cloud coordination patterns
• Speculative decoding with edge-cloud collaboration
• Hierarchical parallelism (combining DP, TP, PP)
• Sequence parallelism for long contexts
• Dynamic resource allocation and multi-tenancy
• Gradient compression and communication efficiency
• Emerging research directions and future trends

**Code Examples:**
• MoE layer implementation with expert routing
• Expert parallelism for distributed MoE
• Edge-cloud speculative decoding
• Sequence parallel attention
• Hierarchical parallelism setup
• Dynamic GPU allocator
• Multi-tenant GPU scheduler
• Gradient compression utilities

