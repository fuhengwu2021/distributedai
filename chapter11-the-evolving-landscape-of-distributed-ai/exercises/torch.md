\fancydividerwithicon[center]{hand.png}


## Exercises


### Multiple Choice Questions

1. What is the primary advantage of MoE (Mixture of Experts) architectures?

   a) Faster training convergence
   b) Scales model capacity without proportional compute increase
   c) Eliminates the need for distributed training
   d) Reduces model size

2. In a typical MoE layer with 64 experts and top-2 routing, how many experts process each token?

   a) 1
   b) 2
   c) 32
   d) 64

3. What is the main bottleneck for on-device LLM inference on mobile devices?

   a) CPU compute
   b) Storage space
   c) Memory bandwidth
   d) Battery capacity

4. What does speculative decoding achieve in edge-cloud scenarios?

   a) Reduces model size
   b) Uses a small draft model to propose tokens verified by a larger model
   c) Eliminates network latency
   d) Compresses gradients

5. At 100K GPU scale with 99.9% per-GPU reliability, approximately how many failures occur per day?

   a) 1
   b) 10
   c) 100
   d) 1000

6. What is the purpose of load balancing loss in MoE training?

   a) Reduce memory usage
   b) Encourage uniform expert utilization
   c) Speed up inference
   d) Compress model weights

7. Which communication pattern is most expensive in expert parallelism?

   a) Broadcast
   b) Reduce
   c) All-to-all
   d) Gather

8. What does Ring Attention trade for reduced memory usage?

   a) Accuracy
   b) More communication rounds
   c) Larger batch sizes
   d) Fewer experts


### Short Answer Questions

9. Explain why inference workloads now consume over 55% of AI infrastructure spending, while training was dominant just a few years ago.

10. Describe the key difference between synchronous and asynchronous checkpointing, and why async checkpointing matters at scale.

11. A VLM (Vision-Language Model) uses cross-modal attention. Explain what this means and one challenge in distributing it.

12. Compare top-k gradient sparsification with quantization for gradient compression. What are the trade-offs?


### Hands-On Exercise

__Implement Basic MoE Routing__

Create a simple MoE router that:

- Takes input tokens and routes them to top-k experts
- Computes load balancing loss
- Returns weighted expert outputs

Reference `code/moe_layer.py` for guidance. Test with 4 experts, top-2 routing, and measure expert utilization across a batch of inputs.


## Answer Key

1. b) Scales model capacity without proportional compute increase
2. b) 2
3. c) Memory bandwidth
4. b) Uses a small draft model to propose tokens verified by a larger model
5. c) 100
6. b) Encourage uniform expert utilization
7. c) All-to-all
8. b) More communication rounds

9. Training happens once; inference happens millions of times. As models mature and deployment scales, the economics favor inference. Additionally, infrastructure costs have dropped significantly, making deployment economically viable in more contexts.

10. Synchronous checkpointing blocks training while saving state. Async checkpointing saves in the background, allowing training to continue. At scale, checkpoint time becomes significant (minutes), so blocking would waste substantial GPU time.

11. Cross-modal attention allows text tokens to attend to vision features. Challenge: vision encoder and language model may have different optimal parallelism strategies and batch sizes.

12. Top-k keeps largest gradients (high compression, ~100x) but requires error feedback for convergence. Quantization reduces precision (moderate compression, ~4x) but is simpler and has less overhead.
