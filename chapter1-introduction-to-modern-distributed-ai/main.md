# Chapter 1: Introduction to Modern Distributed AI

**Chapter Length:** 28 pages

## Overview

Modern AI models have grown beyond what single GPUs can handle. Large language models now range from several billion to over a trillion parameters. Training models with tens of billions of parameters on a single GPU would take months, if they even fit in memory. Serving these models at scale requires distributed architectures.

This chapter walks through resource estimation, decision frameworks for choosing between distributed training, fine-tuning, or inference, and practical examples to get you started.

## 1. Why Modern AI Requires Distribution

A few years ago, you could train most models on a single GPU. ResNet-50 on ImageNet took a couple of days. Today, training a 70B parameter language model on a single GPU would take months, if it even fits in memory. The models got bigger, the datasets got bigger, and single-GPU training became impractical.

Looking at recent models, the scale is clear. GPT-4 has over 1 trillion parameters. Training it requires thousands of GPUs working together. Even smaller models like Llama 2 (70B parameters) need multiple GPUs just to fit in memory, let alone train efficiently.

| Model Name | Parameters | Company | Year |
|------------|------------|---------|------|
| ViT-22B | 22B | Google | 2023 |
| Sora | 30B | OpenAI | 2023 |
| Grok-1 | 314B | xAI | 2023 |
| Gemini-1 | 1.6T | Google | 2023 |
| LLaMA-2 | 700B | Meta | 2023 |
| PanGu-Σ | 1.085T | Huawei | 2023 |
| DeepSeek-V1 | 6.7B | DeepSeek | 2023 |
| GPT-4V | 1.8T | OpenAI | 2024 |
| DeepSeek-V2 | 236B MoE (16 experts, 2 active) | DeepSeek | 2024 |
| Grok-4 | ~1.7T (MoE) | xAI | 2025 |
| Qwen-Max | ~1.2T | Alibaba | 2025 |
| GPT-5 | ~2–5T | OpenAI | 2025 |
| DeepSeek-V3 | 671B MoE (64 experts, 8 active) | DeepSeek | 2025 |
| Gemini-3-Pro | ~7.5T | Google | 2025 |

![Model Parameters v.s. Year](code/model_comparison_plot.png)

### The Scale Challenge

Take a 70B parameter model as an example. In full precision (FP32), the model weights alone need 280GB of memory. An A100 GPU can have 80GB memory. You can't even load the model, let alone train it.

Training these models takes thousands of GPU-hours. A single GPU training run would take months. The datasets are massive too - trillions of tokens. Loading and preprocessing this data efficiently requires `distributed pipelines`.

The mismatch is clear: model size and compute requirements have grown exponentially, while single-GPU memory and compute have grown linearly at best.

![Growth Mismatch: Exponential Model Growth vs Linear GPU Growth](code/growth_mismatch.png)

### Estimating Model Resource Requirements

Before you start training or deploying, you need to know how much memory and compute you'll need. Get this wrong, and you'll hit out-of-memory errors or waste money on over-provisioned infrastructure.

The memory footprint depends on what you're storing. For model weights alone, the calculation is straightforward. Each parameter in FP32 takes 4 bytes, FP16/BF16 takes 2 bytes, Int8 takes 1 byte, and Int4 takes 0.5 bytes. For a 7B parameter model, that's 28 GB in FP32, 14 GB in BF16 (or FP16), 7 GB in Int8, and 3.5 GB in Int4.

For training, BF16 (bfloat16) is preferred over FP16 (float16). BF16 has the same exponent range as FP32 (8 bits) but reduced mantissa (7 bits), giving it the same dynamic range as FP32. This makes training more stable—less likely to overflow or underflow. FP16 has a smaller exponent range (5 bits), which can cause numerical issues during training. Modern GPUs (A100, H100) have Tensor Cores optimized for BF16. For inference, both FP16 and BF16 work, but BF16 is still preferred for consistency with training.

But model weights are just the start. During training, you also need space for gradients, optimizer states, and activations. For inference, you need KV cache for attention mechanisms. The KV cache is crucial for LLM inference—it stores the key-value pairs from previous tokens in the sequence, allowing the model to avoid recomputing attention over the entire sequence for each new token. This dramatically speeds up autoregressive generation, but it requires significant memory that scales with batch size and sequence length. The total memory requirement can be several times larger than just the model weights.

#### Training Memory Requirements

Training needs way more memory than inference. You need to store model weights, gradients (one per parameter), optimizer states, and activations from the forward pass.

**Optimizer State**

The optimizer state size depends on which optimizer you use. Take Stochastic Gradient Descent (SGD) as an example, the model weights are updated according to this formula:

$$
w_{t+1} = \boxed{w_t} - \eta  \boxed{g_t}
$$

where:

$w_t$: model parameters at iteration $t$  
$w_{t+1}$: updated parameters  
$g_t$: gradient of loss w.r.t. parameters at iteration $t$ ($g_t = \nabla_w L(w_t)$)  
$\eta$: learning rate (used for both SGD and Adaptive Moment Estimation (Adam))  

The boxed variables are what we need to keep in memory. SGD only needs the learning rate $\eta$ (a scalar) to update parameters. Looking at the formula, you compute $g_t$ during backprop, then subtract $\eta g_t$ from $w_t$. The optimizer state is just $\eta$ - negligible memory. You need to store $w_t$ (model weights) and $g_t$ (gradients), but no additional optimizer tensors.

Adaptive Moment Estimation (Adam)'s formula is more complex:

$$
w_{t+1} = \boxed{w_t} - \eta
\frac{\beta_1 \boxed{m_{t-1}} + (1-\beta_1) \boxed{g_t}}{\sqrt{\beta_2 \boxed{v_{t-1}} + (1-\beta_2) \boxed{g_t}^2} + \epsilon}
\cdot
\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
$$

Variable definitions are as follows:

$\beta_1$: decay rate for the first moment (mean)  
$\beta_2$: decay rate for the second moment (uncentered variance)  
$m_{t-1}$: first-moment estimate from previous step  
$v_{t-1}$: second-moment estimate from previous step  
$\epsilon$: small constant for numerical stability  
$t$: iteration index used for bias correction

Adam needs more. The formula shows it maintains two per-parameter tensors: $m_{t-1}$ (first-moment estimate) and $v_{t-1}$ (second-moment estimate). Both $m_{t-1}$ and $v_{t-1}$ have the same shape as $w_t$ - one value per parameter. So Adam stores $m_{t-1}$ (1× model size) and $v_{t-1}$ (1× model size), totaling 2× model size for optimizer states. 

$\beta_1$ and $\beta_2$ are hyperparameters - scalar constants (typically $\beta_1=0.9$, $\beta_2=0.999$). You store them once as configuration, not per parameter. Same with $\eta$ (learning rate) and $\epsilon$ - they're scalars, negligible memory. Only tensors with the same shape as $w_t$ (one value per parameter) need significant memory: $w_t$, $g_t$, $m_{t-1}$, and $v_{t-1}$.

That's why Adam needs 2× the model size for optimizer states compared to SGD's near-zero overhead.

AdamW (Adam with decoupled weight decay) has the same memory requirements as Adam. The difference is how weight decay is applied - AdamW applies it directly to parameters, while Adam adds it to gradients. But both maintain the same optimizer states: $m_{t-1}$ (1× model size) and $v_{t-1}$ (1× model size), totaling 2× model size. So AdamW also needs 2× the model size for optimizer states.

Here's a summary of optimizer state memory requirements for common optimizers:

| Optimizer | Optimizer States | Memory (relative to model size) |
|-----------|------------------|--------------------------------|
| SGD | Learning rate $\eta$ (scalar) | ~0× |
| RMSprop | $v_{t-1}$ (second moment) | 1× |
| Adagrad | Accumulated gradient squares | 1× |
| Adam | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment) | 2× |
| AdamW | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment) | 2× |

The table shows optimizer states only. You still need to store model weights (1×) and gradients (1×) regardless of which optimizer you use.

**Activation Output**

Activation layers (like ReLU, GELU, sigmoid) don't have parameters - they're just functions applied element-wise. But their outputs (activation outputs, often shortened to "activations") need to be stored in memory during training. 

Consider a simple 3-layer DNN `SimpleDNN` as below:

![](img/simplednn.png)


The data flow is $x \rightarrow z \rightarrow h \rightarrow \hat{y}$ with Linear → Sigmoid → Linear layers, where input $x$ passes through the first linear layer to produce $z$, then through the sigmoid activation to produce $h$, and finally through the second linear layer to produce the prediction $\hat{y}$. The code in PyTorch would be as follows:

```python
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1, bias=False):
        super(SimpleDNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)  # x -> z
        self.activation = nn.Sigmoid()                              # z -> h
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias) # h -> y_hat

    def forward(self, x):
        z = self.linear1(x)      # z = W_1 * x
        h = self.activation(z)   # h = sigmoid(z)
        y_hat = self.linear2(h)  # y_hat = W_2 * h
        return y_hat
```

The forward pass is:

$$
z = W_1 x, \quad h = \sigma(z), \quad \hat{y} = W_2 h, \quad L = \frac{1}{2}(y - \hat{y})^2
$$

To compute gradients using backpropagation, we apply the chain rule.

For $\frac{\partial L}{\partial W_2}$:

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2} = (\hat{y} - y) \cdot h
$$

The gradient depends on $h$ - the activation output from the sigmoid layer. You need $h$ stored in memory to compute this gradient.

For $\frac{\partial L}{\partial W_1}$, the chain is longer:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial W_1}
$$

Expanding each term: $\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$ (needs $\hat{y}$), $\frac{\partial \hat{y}}{\partial h} = W_2$ (needs $W_2$), $\frac{\partial h}{\partial z} = \sigma'(z) = \sigma(z)(1-\sigma(z)) = h(1-h)$ (needs $h$), and $\frac{\partial z}{\partial W_1} = x$ (needs input $x$).

So:

$$
\frac{\partial L}{\partial W_1} = (\hat{y} - y) \cdot W_2 \cdot h(1-h) \cdot x
$$

This gradient requires both $h$ (the activation output from the sigmoid layer) and $x$ (the input data read from the dataloader). Without storing $h$ (activation output) and $x$ (input data) during the forward pass, you can't compute these gradients during backpropagation. That's why activation outputs (activations) and input data must be kept in memory until their gradients are computed.

An interesting observation is that $z$ is not needed. Looking at the gradient computation, we need $\frac{\partial h}{\partial z} = \sigma'(z)$ to compute $\frac{\partial L}{\partial W_1}$. For sigmoid, the derivative is $\sigma'(z) = \sigma(z)(1-\sigma(z)) = h(1-h)$. Since we already have $h$ stored from the forward pass, we can compute the derivative directly from $h$ without needing the original $z$ value. This is a property of sigmoid and some other activation functions—their derivatives can be expressed in terms of their outputs, so you don't need to store the pre-activation values.

However, this isn't true for all activation functions. Some activation functions have derivatives that explicitly depend on the input value $z$, not just the output $h$. This means you cannot compute the derivative from $h$ alone—you need to store the original $z$ value.

For example, consider GELU (Gaussian Error Linear Unit):

$$
h = z \cdot \Phi(z)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution. The derivative is:

$$
\frac{\partial h}{\partial z} = \Phi(z) + z \cdot \phi(z)
$$

where $\phi$ is the probability density function. This derivative explicitly contains $z$ in the term $z \cdot \phi(z)$. Since $\Phi(z)$ and $\phi(z)$ are not easily invertible functions, you cannot recover $z$ from $h$ alone. Therefore, you need to store the original $z$ value to compute the derivative during backpropagation.

Similarly, Swish (also known as SiLU, Sigmoid Linear Unit) has:

$$
h = z \cdot \sigma(z)
$$

with derivative:

$$
\frac{\partial h}{\partial z} = \sigma(z) + z \cdot \sigma(z)(1-\sigma(z))
$$

Again, the derivative contains $z$ explicitly, and you cannot express it purely in terms of $h$. To compute $\frac{\partial h}{\partial z}$, you need both $z$ and $\sigma(z)$, which means storing the original $z$ value.

In practice, frameworks often store both pre-activation values (like $z$) and post-activation values (like $h$) to support all activation functions efficiently. This ensures that backpropagation can compute gradients correctly regardless of which activation function is used, even if it means storing slightly more memory than the theoretical minimum for some functions.

During the forward pass, you compute and store all layer activation outputs (we call these "activations" for short). During backpropagation, you compute gradients layer by layer from the last layer backward.


For each layer, you need its activation outputs to compute gradients. Once you've computed a layer's gradient and updated its parameters, you can free that layer's activation outputs - they're no longer needed. However, during the backward pass, there's overlap: while computing gradients for one layer, you still have activation outputs from earlier layers in memory. The peak memory occurs when you have both activations (activation outputs from forward pass) and gradients (from backward pass) simultaneously.

In practice, activations (activation outputs) and gradients do overlap in memory during backpropagation. The activation memory scales with batch size and sequence length - larger batches or longer sequences mean more activation outputs to store. Techniques like gradient checkpointing trade compute for memory by recomputing activations instead of storing them all.

**Training Stage**

During training, memory usage varies across different stages of the training loop. Understanding when each component is needed helps you estimate peak memory requirements and identify optimization opportunities.

```python
for epoch in range(num_epochs):
    model.train()                         # set to training mode
    for x_batch, y_batch in dataloader:   # iterate over batches
        optimizer.zero_grad()             # 1. clear gradients
        y_hat = model(x_batch)            # 2. forward pass
        loss = criterion(y_hat, y_batch)  # 3. compute loss
        loss.backward()                   # 4. backward pass
        optimizer.step()                  # 5. update parameters
```

Let's break down what happens in each step and what's stored in memory:

Step 2 (Forward pass): The model computes predictions `y_hat` from inputs `x_batch`. During this process, it stores intermediate activations (like $h$ in our earlier example) that will be needed for backpropagation. At this point, memory contains: model weights $w_t$, activations, and optimizer states (like $m_{t-1}$ and $v_{t-1}$ for Adam) from previous iterations.

Step 4 (Backward pass): `loss.backward()` computes gradients $g_t = \nabla_w L(w_t)$ for all parameters. This requires the activations stored during the forward pass. Memory now contains: $w_t$, activations (still needed), gradients $g_t$, and optimizer states.

Step 5 (Optimizer step): `optimizer.step()` updates parameters from $w_t$ to $w_{t+1}$ using the gradients $g_t$ and the optimizer's internal state. For Adam, this means using $m_{t-1}$ and $v_{t-1}$ along with $g_t$ to compute the update. After this step, the optimizer states are updated (e.g., $m_{t-1}$ becomes $m_t$, $v_{t-1}$ becomes $v_t$) and stored for the next iteration. Activations can now be freed since they're no longer needed.

Activations, gradients, and optimizer states don't all exist in memory at the same time. During the forward pass, only activations are actively used—gradients don't exist yet, and optimizer states persist from previous iterations but aren't accessed. During the backward pass, activations and gradients overlap because you need activations to compute gradients. During the optimizer step, gradients and optimizer states overlap as the optimizer uses both to update parameters. Activations are typically freed after the backward pass completes, so they don't overlap with optimizer states.

The peak memory usage occurs during the backward pass when both activations and gradients are in memory simultaneously. After the backward pass, activations can be freed, so the optimizer step only needs gradients and optimizer states.


![Training Memory Timeline](code/training_memory_timeline.png)

The timeline shows memory usage across the training loop. Here's how each stage maps to the code. On line 2 (`y_hat = model(x_batch)`), the forward pass computes and stores activations. Memory usage is weights (14 GB) plus optimizer states (28 GB) plus activations (12 GB), totaling 54 GB. Gradients don't exist yet.

On line 4 (`loss.backward()`), the backward pass is where peak memory occurs. During backpropagation, you need both activations to compute gradients and the gradients being computed. Memory usage is weights (14 GB) plus optimizer states (28 GB) plus activations (12 GB) plus gradients (14 GB), totaling 68 GB. This is the peak because activations and gradients overlap in memory.

On line 5 (`optimizer.step()`), after the backward pass completes, activations can be freed. The optimizer uses gradients and its internal states to update parameters. Memory usage is weights (14 GB) plus optimizer states (28 GB) plus gradients (14 GB), totaling 56 GB. Activations are no longer needed.

The peak memory of 68 GB occurs during `loss.backward()` (line 4) when both activations and gradients are simultaneously in memory. This is why reducing batch size or using gradient checkpointing helps when you hit out-of-memory errors - they reduce activation memory during the backward pass.


Memory breakdown:

For a 7B model with BF16: model weights (14 GB), gradients (14 GB), optimizer states with Adam (28 GB for $m_{t-1}$ and $v_{t-1}$), and activations (8-16 GB depending on batch size and sequence length). That's 64-72 GB total per GPU. With SGD, you'd save 28 GB on optimizer states, but Adam's adaptive learning rates usually converge faster, so the trade-off is worth it for most cases. That's why a 7B model needs at least an A100 (80GB) for training with Adam, even with mixed precision (BF16). Smaller GPUs won't cut it.

![Training Memory Breakdown for 7B Model](code/training_memory_breakdown.png)



#### Inference Memory Requirements

Inference is simpler. You just need the model weights and KV cache for attention. The KV cache is key for LLM inference—it stores computed key-value pairs from previous tokens, allowing the model to generate each new token without recomputing attention over the entire sequence. This makes autoregressive generation much faster, but the cache grows with each generated token and scales with batch size and sequence length.

The KV cache size depends on your batch size, sequence length, and model architecture. For a 70B model with BF16, you're looking at 140 GB for model weights and another 20-40 GB for KV cache with a batch size of 32 and sequence length of 2048. That's 160-180 GB total. That's why inference for large models needs multiple GPUs or model parallelism. A single A100 won't hold it.

#### GPU Requirements Estimation

For training, calculate your memory needs, add 10-20% safety margin for communication buffers and framework overhead, then see if it fits. A 13B model with BF16 needs about 72 GB per GPU. With safety margin, that's 85 GB. An A100 has 80 GB, so you'll need 2 GPUs with model parallelism or FSDP.

For inference, it's similar. Calculate model size plus KV cache. A 70B model in BF16 needs 140 GB just for weights. With KV cache, you're looking at 160-180 GB total. You'll need 2+ A100 GPUs, or use Int8 quantization to get it down to 70 GB for weights, which might fit on one GPU with careful KV cache management.

#### Real-World Considerations

Don't forget the overhead. PyTorch adds 1-2 GB. The OS needs 5-10 GB. Distributed training needs 2-5 GB per GPU for communication buffers. Checkpointing causes temporary spikes. Start conservative and add 20-30% buffer to your estimates. Use mixed precision (BF16 for training, FP16/BF16 for inference) to cut memory in half compared to FP32. Monitor with `nvidia-smi` to see actual usage. For inference, Int8 quantization can halve memory with minimal accuracy loss. Remember that activations scale linearly with batch size—if you hit OOM, reduce batch size first.

Quick Reference Table:

| Model Size | FP32 Weights | BF16 Weights | Training (BF16+Adam) | Inference (BF16) |
|------------|--------------|--------------|----------------------|------------------|
| 1B         | 4 GB         | 2 GB         | ~8 GB                | 2-4 GB           |
| 7B         | 28 GB        | 14 GB        | ~60-70 GB            | 14-20 GB         |
| 13B        | 52 GB        | 26 GB        | ~110-130 GB          | 26-35 GB         |
| 70B        | 280 GB       | 140 GB       | ~600-700 GB          | 140-180 GB       |

*Note: Training estimates assume Adam optimizer and moderate batch size. Actual values vary based on architecture, sequence length, and batch size.*

### The Evolution from Classic ML to Foundation Models

Classic machine learning models were designed to fit on a single machine. Traditional ML models - such as linear regression, logistic regression, decision trees, random forests, support vector machines (SVM), and gradient boosting (XGBoost, LightGBM) - typically had thousands to millions of parameters and were trained on datasets that fit in memory. The deep learning era brought models with hundreds of millions of parameters (e.g., ResNet, BERT), requiring GPUs but still manageable on single devices. Today's foundation model era has models with billions to trillions of parameters (e.g., GPT-4, Gemini, LLaMA), requiring distributed systems from day one.

The shift to distributed AI enabled breakthrough capabilities - models that can understand and generate human-like text, code, and multimodal content. It also drove enterprise adoption, with companies deploying AI at scale for production workloads, and accelerated research through faster iteration cycles enabled by parallel experimentation.

## 2. The Modern AI Model Lifecycle

Building AI models isn't a one-shot process. It's a cycle: you collect data, train a model, deploy it, see how it performs, then go back and improve the data or model. Each stage feeds into the next.

The lifecycle looks like this:

Data Engineering → Model Training → Model Inference → Model Benchmarking → Model Deployment → Data Engineering (repeat)

![Modern AI Model Lifecycle](code/model_lifecycle.png)

You start with data engineering. Collect data, curate it, transform it, validate it, explore it. You're preparing terabytes of data for training. Clean it, deduplicate it, filter for quality, format it. Tools like NeMo Curator handle this at scale.

Then you train the model. Forward pass, backprop, gradient descent. Tune hyperparameters, use parameter-efficient tuning (PEFT), fine-tune, maybe do RLHF. You're learning model parameters from data. This is where distributed training shines - split models and data across GPUs.

Once trained, you run inference. Quantize the model, cache activations, convert to ONNX, fuse operators, optimize CUDA kernels. You're generating predictions. Latency and throughput matter here. Distributed inference handles models too large for a single GPU.

Before deploying, you benchmark. Measure precision and recall, evaluate engineering performance, profile bottlenecks, run stress tests, test different scenarios. You're checking how well the model works. Distributed evaluation speeds up testing on large datasets.

Then you deploy. Set up autoscaling, scheduling, load balancing, observability. Put the model in production. That means API gateways, monitoring, handling thousands of requests per second.

Production feedback tells you what data to collect next, or where the model fails. You loop back to data engineering. The cycle repeats.


This book focuses on the distributed technologies you need for training, inference, benchmarking, and deployment. Data engineering gets a brief overview but isn't the main focus. Distributed data processing is important, but it's a well-established topic. Spark, Dask, and Ray have been around for years. This book covers the basics - what you need to know to prepare data for distributed training - but the real focus is on AI-specific distributed challenges: training large models, serving them at scale, and optimizing inference.

The principles are the same across all stages: parallelism, communication, memory management, fault tolerance. But the techniques differ. Training is iterative with frequent gradient syncs. Inference is latency-sensitive with throughput requirements.

## 3. Training vs Inference vs Serving

Training, inference, and serving are different. Each has different requirements, bottlenecks, and optimization strategies. Know these differences to design effective distributed systems.

### Training: The Learning Phase

Training is about learning model parameters from data. The process follows a pattern: forward pass through the model, loss computation, backward pass to compute gradients, and gradient update to adjust parameters. This happens iteratively over multiple epochs until the model converges.

Training requires storing activations, gradients, and optimizer states in memory. The compute is intensive and iterative. In distributed training, you need frequent gradient synchronization across devices to keep all model copies consistent. The challenges include gradient synchronization overhead, memory constraints for large models, long training times that can span days to weeks, and the need for fault tolerance and checkpointing.

Training a 7B parameter model on 1 trillion tokens typically requires 8 A100 GPUs (80GB each) and about 2 weeks of continuous training. You need careful gradient synchronization to maintain training stability across all GPUs.

### Inference: The Prediction/Generation Phase

Inference is about generating predictions from a trained model. Unlike training, you only need a forward pass—no gradients, no backward pass, no optimizer states. Memory requirements are lower because you only need model weights and KV cache for attention mechanisms. The compute per request is lower, but you need high throughput to serve many requests at once. Communication is minimal, mostly only for distributed inference.

The challenges include latency (sub-second for interactive apps), throughput (thousands of requests per second), efficient memory usage through KV cache management, and effective batching and scheduling. Serving a 70B parameter model for chat requires optimized inference engines like vLLM or SGLang, continuous batching to maximize GPU utilization, and careful KV cache management for variable-length sequences.

### Serving: The Production System

Serving is about providing reliable, scalable access to models. It's not just running inference—it's building a production system with a model runner, API gateway, load balancer, and monitoring. The requirements include high availability, fault tolerance, and observability. At scale, you're dealing with multi-model, multi-tenant systems.

The challenges include system reliability and uptime, multi-model routing and load balancing, cost optimization through GPU utilization and autoscaling, and observability for debugging. A production LLM serving platform might include multiple model variants (different sizes, fine-tuned versions), A/B testing infrastructure, canary deployment pipelines, and distributed tracing and monitoring.

### Comparison Table

| Aspect | Training | Inference | Serving |
|--------|----------|-----------|---------|
| **Primary Goal** | Learn parameters | Generate predictions | Provide access |
| **Memory Usage** | High (activations + gradients) | Medium (weights + KV cache) | Variable |
| **Compute Pattern** | Iterative, intensive | Single forward pass | Request-driven |
| **Communication** | Frequent (gradients) | Minimal | API-level |
| **Latency Requirement** | Hours to days | Milliseconds to seconds | Milliseconds |
| **Throughput Focus** | Samples per second | Tokens per second | Requests per second |

## 4. Decision Framework: When Do You Need Distributed Systems?

Distributed systems add complexity, communication overhead, and cost. Use them when you have to, not when you want to.

### Decision Tree: Quick Reference

The decision framework is summarized in the decision tree below. Start by identifying your use case: training (or fine-tuning) versus inference and serving.

![Decision Framework: When Do You Need Distributed Systems?](img/1.png)

### Understanding the Decision Tree

When training or fine-tuning, start by checking whether your model fits in memory. Calculate model size in BF16 (parameters × 2 bytes). If it exceeds single GPU capacity, you'll need model or parameter parallelism like FSDP or tensor parallelism. A 13B model needs 26GB for weights alone. Add Adam optimizer states and you're looking at 72GB total, which is cutting it close on an 80GB A100.

If the model fits but training drags on for weeks or months, data parallelism can speed things up. Training a 7B model on 1T tokens takes about two weeks on 8 GPUs, compared to months on a single GPU.

When both model size and training time are manageable, your fine-tuning approach matters. Parameter-efficient methods like LoRA and QLoRA change the equation. QLoRA on a 70B model fits in a 48GB GPU because it only trains adapter weights, while full fine-tuning of the same model needs multiple GPUs. But if the base model doesn't fit, you still need model parallelism regardless of which training method you choose.

Large datasets where data loading becomes the bottleneck benefit from distributed data loading. Multi-terabyte datasets are good candidates for data parallelism.

For inference or serving, the logic is similar. If the model exceeds single GPU memory, use model parallelism. A 70B model in BF16 needs 140GB for weights. With KV cache, you're looking at 160-180GB, which requires at least 2 A100 GPUs. If memory is fine but you need high throughput—thousands of requests per second—use multiple GPUs for distributed inference. Real-time services that need sub-second latency at high throughput often require tensor parallelism or multiple inference instances. When both memory and throughput fit within single GPU limits, stick with one GPU and use optimized engines like vLLM or SGLang to maximize efficiency.

## 5. Environment Setup

In this book, we will use PyTorch as our main frame work, and the code is in git repo https://github.com/fuhengwu2021/coderepo.


To run the code, it is the best if you have access to a multiple-GPU machine, such as A10, A100 or H100/200 or even B200. If you don't have access to multiple GPUs locally, Kaggle offers free multi-GPU environments. Log in to Kaggle.com, click Create, and select Notebook.

![](img/1.5.png)

Once you've created a Jupyter notebook, go to Settings → Accelerator and select GPU T4x2.

![](img/3.png)

You should now have 2 T4 GPUs available. To verify your GPU setup, run the code in `code/check_cuda.py`:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024**3)
    print(f"GPU {i}: {props.name} ({vram_gb:.1f} GB)")
```

Running this should show your available GPUs:

![GPU setup](img/2.png)

## 6. Hands-On: Running Distributed Training and Inference

We'll begin with a simple baseline to establish a performance reference point, then move to distributed training to see the speedup in action.

### Single-GPU Baseline

Let's start with a baseline. We train ResNet18 on FashionMNIST, which completes in under 30 seconds on a single GPU—perfect for quick experiments.

```bash
python code/single_gpu_baseline.py
```

Example output after 3 epochs:

```
Epoch 1/3, Loss: 0.4164, Accuracy: 84.94%
Epoch 2/3, Loss: 0.2936, Accuracy: 89.17%
Epoch 3/3, Loss: 0.2531, Accuracy: 90.58%

Total training time: 8.78s
```

Your results will vary depending on your hardware, but you should see similar loss and accuracy values.

This gives us a reference point. The same model architecture is used in the distributed version for a fair comparison.

### Multi-GPU Distributed Training

Now let's run the distributed version. The script uses the same ResNet18 model on FashionMNIST, but splits the work across multiple GPUs.

```bash
torchrun --nproc_per_node=2 code/multi_gpu_ddp.py
```

Or use the launch script:

```bash
bash code/launch_torchrun.sh
```

With 2 GPUs, you'll see similar loss and accuracy values, but the training completes in 5.91 seconds instead of 8.78 seconds—a **1.48× speedup**.

We tested scaling from 1 to 8 GPUs to see how performance improves:

| GPUs | Training Time | Speedup |
|------|---------------|---------|
| 1    | 8.78s         | 1.00×   |
| 2    | 5.91s         | 1.48×   |
| 4    | 3.69s         | 2.38×   |
| 6    | 2.92s         | 3.01×   |
| 8    | 2.44s         | 3.60×   |

![FashionMNIST Scaling Performance](code/fashionmnist_scaling_performance.png)

Training time drops from 8.78 seconds to 2.44 seconds with 8 GPUs, achieving a **3.6× speedup**. Adding more GPUs significantly reduces training time and accelerates development cycles. The speedup becomes even more pronounced with larger workloads, as we'll see next.

### Extended Training: CIFAR-10 with 20 Epochs

For a more realistic workload, we tested ResNet18 on CIFAR-10 with 20 epochs. CIFAR-10 is larger and more complex than FashionMNIST, with 50,000 training images using 3-channel RGB images of 32×32 pixels. This better showcases the benefits of distributed training.

Run the single-GPU baseline:

```bash
python code/single_gpu_extended.py --epochs 20
```

Then test distributed training:

```bash
torchrun --nproc_per_node=2 code/multi_gpu_ddp_extended.py --epochs 20
torchrun --nproc_per_node=4 code/multi_gpu_ddp_extended.py --epochs 20
torchrun --nproc_per_node=6 code/multi_gpu_ddp_extended.py --epochs 20
torchrun --nproc_per_node=8 code/multi_gpu_ddp_extended.py --epochs 20
```

The results show even better scaling:

| GPUs | Training Time | Speedup |
|------|---------------|---------|
| 1    | 73.00s (1.22 min) | 1.00×   |
| 2    | 46.47s (0.77 min) | 1.57×   |
| 4    | 27.72s (0.46 min) | 2.63×   |
| 6    | 21.24s (0.35 min) | 3.44×   |
| 8    | 18.20s (0.30 min) | 4.01×   |

![CIFAR-10 Scaling Performance](code/cifar10_scaling_performance.png)

Training time drops from 73 seconds to 18.2 seconds with 8 GPUs, achieving a **4.01× speedup**—better than the FashionMNIST results. The larger computation workload per epoch means gradient synchronization takes a smaller fraction of total time. With 20 epochs, communication overhead is amortized across more training steps. Each GPU has enough computation work between communication steps to maximize parallel efficiency.

This demonstrates why distributed training is essential for large-scale model training. As computation per step increases, the relative cost of communication decreases, leading to better scaling efficiency. For models with billions of parameters, distributed training scales even more effectively, with speedups approaching linear scaling as computation dominates the training time. The DDP approach shown here uses data parallelism, where each GPU holds a complete copy of the model and processes different data. This is the simplest form of distributed training, suitable for models that fit on a single GPU. For larger models that exceed single-GPU memory, later chapters will explore advanced parallelism strategies such as tensor parallelism (splitting model layers across GPUs), pipeline parallelism (splitting model stages across GPUs), and hybrid approaches that combine multiple parallelism techniques.

### Distributed Inference: Throughput Scaling

While training focuses on reducing time-to-convergence, inference focuses on throughput—how many requests you can process per second. Distributed inference allows multiple GPUs to process different requests simultaneously, dramatically increasing throughput.

We benchmarked ResNet18 inference on FashionMNIST to measure throughput scaling. Unlike training, inference has no gradient synchronization overhead—each GPU simply processes its assigned requests independently. This makes distributed inference highly efficient for serving scenarios.

There are two common approaches to distributing inference requests. The data-split pattern (`multi_gpu_inference.py`) pre-divides requests among GPUs using `DistributedSampler`, where each GPU processes a fixed subset of data. This approach is simple and efficient for batch processing. The request-split pattern (`multi_gpu_inference_queue.py`) assigns requests dynamically in round-robin fashion, where each GPU processes requests as they are assigned, simulating a real queue system. This pattern is better for production serving where requests arrive asynchronously.

Run the benchmarks. Start with the single-GPU baseline:

```bash
python code/single_gpu_inference.py --requests 1000
```

Then test distributed inference with the data-split pattern:

```bash
torchrun --nproc_per_node=2 code/multi_gpu_inference.py --requests 1000
torchrun --nproc_per_node=4 code/multi_gpu_inference.py --requests 1000
torchrun --nproc_per_node=8 code/multi_gpu_inference.py --requests 1000
```

And the request-split pattern:

```bash
torchrun --nproc_per_node=2 code/multi_gpu_inference_queue.py --requests 1000
```

Example results from benchmarking:

| GPUs | Pattern | Time (s) | Throughput (req/s) | Speedup |
|------|---------|----------|---------------------|---------|
| 1    | Baseline| 1.85s    | 541.00 req/s        | 1.00×   |
| 2    | Data-split | 0.98s  | 1025.45 req/s       | 1.89×   |
| 2    | Request-split | 1.22s | 819.09 req/s       | 1.51×   |

With 2 GPUs using the data-split pattern, we achieve **1.89× speedup**, nearly doubling the throughput. The data-split pattern achieves near-linear scaling because each GPU processes independent requests with minimal coordination overhead. The request-split pattern shows slightly lower throughput (1.51×) due to the overhead of round-robin assignment and all GPUs needing to iterate through the dataset, but it's more flexible for dynamic request handling in production environments where requests arrive asynchronously.

Both patterns demonstrate the power of distributed inference: throughput scales almost linearly with the number of GPUs, making it ideal for production serving workloads that require high request rates. The inference patterns shown here use data parallelism, suitable for models that fit on a single GPU. For large language models that exceed single-GPU memory, later chapters will explore advanced techniques such as expert parallelism, sequence parallelism, tensor parallelism, and serving frameworks like vLLM and SGLang.

## 7. PyTorch Distributed Fundamentals

Now that you've seen distributed training and inference in action, let's understand the fundamental concepts and APIs that make it work. The essential building blocks are process groups, ranks, and communication primitives. These form the foundation for all distributed operations, whether you're using DDP or FSDP for data-parallel training, implementing custom parallelism strategies, or building distributed inference systems.

### Process Groups and Ranks

In distributed training, multiple processes work together. Each process runs on a different GPU or node. PyTorch organizes these processes into a process group. Within a process group, each process has a unique rank—an integer identifier starting from 0. The total number of processes is called the world size.

If you have 4 GPUs, you'll have 4 processes. Process 0 runs on GPU 0, process 1 on GPU 1, and so on. The rank tells each process which GPU it should use and which part of the data it should process.

There are two types of ranks: global rank and local rank. The global rank is unique across all processes in the entire distributed job, ranging from 0 to world_size - 1. The local rank is unique only within a single node, starting from 0 on each node. For example, in a 2-node setup with 4 GPUs per node, node 0 has local ranks 0-3 (global ranks 0-3), and node 1 has local ranks 0-3 (global ranks 4-7). The local rank typically corresponds to the GPU index on that node, which is why you often see `torch.cuda.set_device(local_rank)` in distributed code.

### Initializing the Process Group

Before any distributed operations, you need to initialize the process group. This tells PyTorch how processes should communicate. The most common backend for GPU training is NCCL (NVIDIA Collective Communications Library).

The basic initialization pattern looks like this:

```python
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU communication
        rank=rank,       # This process's rank
        world_size=world_size  # Total number of processes
    )
    torch.cuda.set_device(rank)  # Set which GPU this process uses
```

The simplest test to verify your distributed setup works is in `code/distributed_basic_test.py`. It's a basic distributed test that verifies process group initialization and communication. It doesn't use DDP—it just tests that multiple processes can communicate.

Test with multiple GPUs:

```bash
torchrun --nproc_per_node=2 code/distributed_basic_test.py
```

If you only have one GPU but want to test the distributed logic, simulate multiple processes on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/multi_gpu_simulation.py
```

Either way, you'll see "Rank 0 says hello" and "Rank 1 says hello" printed from different processes, confirming that your distributed setup works.

When you start running distributed training or inference with `torchrun`, you might notice a warning about `OMP_NUM_THREADS`. This happens because PyTorch wants to control CPU thread usage to avoid overloading your system. You can silence the warning by setting `OMP_NUM_THREADS=4` (or whatever value fits your system) right before the `torchrun` command:

```bash
OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/distributed_basic_test.py
```

Set it in your shell before `torchrun` starts—setting it inside your Python script won't work because `torchrun` checks the environment before launching your processes.

### DistributedDataParallel (DDP)

This section provides a brief overview of DDP. Chapter 3 covers DDP in depth with detailed implementation, optimization techniques, and best practices.

DDP wraps a model for data-parallel distributed training. When you wrap a model with DDP, PyTorch automatically synchronizes gradients across all processes. Each process computes gradients on its local data, then DDP averages these gradients before updating the model.

DDP assumes each process has a complete copy of the model. Only the data is partitioned—each process trains on a different subset, but all processes maintain identical model parameters after each training step. DDP works well when your model fits on a single GPU. For larger models, FSDP (Fully Sharded Data Parallel) shards the model across GPUs while still using data parallelism. Later chapters will cover FSDP and other parallelism strategies like tensor parallelism and pipeline parallelism.

Wrap a model with DDP:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = YourModel().cuda(rank)
model = DDP(model, device_ids=[rank])
```

After wrapping, use the model exactly as you would in single-GPU training. DDP handles synchronization during `loss.backward()`.

### DistributedSampler

Each process should train on different data, so you need a DistributedSampler. It splits the dataset so each process gets a unique subset. Without it, all processes see the same data, defeating the purpose of distributed training.

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset, 
    num_replicas=world_size,  # Total number of processes
    rank=rank  # This process's rank
)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

Call `sampler.set_epoch(epoch)` at the start of each epoch to ensure data shuffling works correctly across epochs.

### Launching Distributed Jobs

The modern approach uses `torchrun`, which handles process spawning automatically:

```bash
torchrun --nproc_per_node=2 code/multi_gpu_ddp.py
```

This launches 2 processes on the current machine. For multi-node training, specify `--nnodes`, `--node_rank`, and `--master_addr` as well.

This command uses several distributed primitives we've covered: `init_process_group()` to set up communication, `barrier()` to synchronize dataset downloading, `DistributedSampler` to partition data, and `DDP` which internally uses `allreduce()` to synchronize gradients during `loss.backward()`.

This chapter walked through resource estimation, decision frameworks, and practical examples for distributed AI systems. The core idea is simple: estimate your requirements first, then decide whether you actually need distributed systems. Don't assume you need them—calculate memory and compute needs, check if your model fits, and only then consider distribution.

The next chapter explores GPU hardware, networking, and parallelism strategies in more depth.

## Further Reading

PyTorch Distributed Training: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

NVIDIA NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/

GPU Memory Management: https://pytorch.org/docs/stable/notes/cuda.html

Profiling PyTorch Models: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
