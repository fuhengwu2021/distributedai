# Chapter 1: Introduction to Modern Distributed AI

*Building scalable AI systems from single GPU to distributed clusters*

> Don't ask what your system can do, ask what your system can do in parallel.
- Henry Wu | Principal ML Tech Lead, Oracle, 2025

**Code Summary**

- `torch.distributed`: Core module for distributed computing in PyTorch (training and inference)
- `dist.init_process_group()`: Initialize the process group for distributed communication with backends (NCCL, Gloo)
- `dist.all_reduce()`: Apply reduction (sum, max, min, etc.) to tensors across all ranks and stores result in all ranks
- `dist.all_gather()`: Collective operation that gathers tensors from all ranks and concatenates them on all ranks
- `dist.broadcast()`: Broadcasts a tensor from the source rank to all other ranks
- `dist.reduce()`: Reduces tensors from all ranks to a single destination rank (typically rank 0)
- `dist.gather()`: Gathers tensors from all ranks to a single destination rank
- `dist.scatter()`: Scatters a list of tensors from the source rank to all other ranks
- `dist.reduce_scatter()`: Reduces tensors across all ranks and scatters the result to all ranks
- `dist.all_to_all()`: Sends different data chunks from each rank to every other rank


## Overview

Modern AI models have grown beyond what single GPUs can handle. Large language models now range from several billion to over a trillion parameters. Training models with tens of billions of parameters on a single GPU would take months, if they even fit in memory. Serving these models at scale requires distributed architectures.

This chapter walks through resource estimation, decision frameworks for choosing between distributed training, fine-tuning, or inference, and practical examples to get you started.

## 1. Why Modern AI Requires Distribution

A few years ago, you could train most models on a single GPU. ResNet-50 on ImageNet took a couple of days. Today, training a 70B parameter language model on a single GPU would take months, if it even fits in memory. The models got bigger, the datasets got bigger, and single-GPU training became impractical.

Looking at recent models, the scale is clear[^model_size_comp]. GPT-4 has over 1 trillion parameters. Training it requires thousands of GPUs working together[^gpt4_training]. Even smaller models like Llama 2 (70B parameters) need multiple GPUs just to fit in memory, let alone train efficiently. This isn't just a training problem—serving these models at scale for production workloads demands distributed inference architectures that can handle thousands of concurrent requests. The era of single-machine AI is over; modern AI systems are inherently distributed by design. According to PyTorch's distributed training documentation, distributed training involves spreading the training workload across multiple worker nodes, which is particularly beneficial for large models and compute-intensive tasks in deep learning. Additionally, industry reports indicate that training trillion-parameter models requires infrastructure investments of tens of millions of dollars[^training_costs], making distributed computing not just a technical necessity but an economic imperative for modern AI development.

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


[^model_size_comp]: The tilde (~) indicates approximate parameter counts. Many large models are closed-source, so exact parameter counts are not publicly disclosed. These approximations are based on inference from model architecture, training costs, and industry estimates.

[^gpt4_training]: SemiAnalysis, "GPT-4 Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE," 2023; Epoch AI, "Compute Trends Across Three eras of Machine Learning," 2023.

[^training_costs]: Epoch AI, "Trends in GPU price-performance," 2024; SemiAnalysis, "The Cost of Training Large Language Models," 2023; OpenAI, "GPT-4 Technical Report," 2023.


![Model Parameters v.s. Year](code/model_comparison_plot.png)

### The Scale Challenge

Take a 70B parameter model as an example. In full precision (FP32), the model weights alone need 280GB of memory. An A100 GPU can have 80GB memory. You can't even load the model, let alone train it.

Training these models takes thousands of GPU-hours. A single GPU training run would take months. The datasets are massive too - trillions of tokens. Loading and preprocessing this data efficiently requires `distributed pipelines`.

The mismatch is clear: model size and compute requirements have grown `exponentially`, while single-GPU memory and compute have grown `linearly` at best.

![Growth Mismatch: Exponential Model Growth vs Linear GPU Growth](code/growth_mismatch.png)

### Estimating Model Resource Requirements

Before you start training or deploying, you need to know how much memory and compute you'll need. Get this wrong, and you'll hit out-of-memory errors or waste money on over-provisioned infrastructure.

The memory footprint depends on what you're storing. For model weights alone, the calculation is straightforward. Each parameter in FP32 takes 4 bytes, FP16/BF16 takes 2 bytes, Int8 takes 1 byte, and Int4 takes 0.5 bytes. For a 7B parameter model, that's 28GB in FP32, 14GB in BF16 (or FP16), 7GB in Int8, and 3.5GB in Int4.

Here's a quick reference for common precision formats:

| Format | Bytes | Format Details | Primary Usage |
|------|--|----------------|---------------|
| FP32 | 4 | 32-bit floating point (1 sign, 8 exponent, 23 mantissa) | Training, high-precision inference |
| BF16 | 2 | 16-bit bfloat (1 sign, 8 exponent, 7 mantissa) | Training (preferred), inference |
| FP16 | 2 | 16-bit float (1 sign, 5 exponent, 10 mantissa) | Inference |
| FP8 E4M3 | 1 | 8-bit float (1 sign, 4 exponent, 3 mantissa) | Inference (activations, weights) |
| FP8 E5M2 | 1 | 8-bit float (1 sign, 5 exponent, 2 mantissa) | Training (gradient storage) |
| Int8 | 1 | 8-bit integer | Quantized inference |
| Int4 | 0.5 | 4-bit integer | Quantized inference (extreme compression) |

![Float32 vs Float16](img/f32vsf16.png)

Note that FP8 has two formats: E4M3 (higher precision, used for inference activations and weights) and E5M2 (wider dynamic range, used for storage). Both use 1 byte per parameter but serve different purposes.

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

That's why Adam needs 2× the model size for optimizer states compared to SGD's near-zero overhead. AdamW (Adam with decoupled weight decay) has the same memory requirements as Adam—both maintain $m_{t-1}$ and $v_{t-1}$, totaling 2× model size. The only difference is how weight decay is applied (AdamW applies it directly to parameters, while Adam adds it to gradients).

Here's a summary of optimizer state memory requirements for common optimizers:

| Optimizer | Optimizer States | Memory (relative to model size) |
|-----------|------------------|--------------------------------|
| SGD | Learning rate $\eta$ (scalar) | ~0× |
| SGD with Momentum | $v_{t-1}$ (velocity/momentum) | 1× |
| Nesterov | $v_{t-1}$ (velocity/momentum) | 1× |
| Adagrad | Accumulated gradient squares | 1× |
| RMSProp | $v_{t-1}$ (second moment) | 1× |
| Adam | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment) | 2× |
| AdamW | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment) | 2× |
| Adafactor | Row and column statistics (factorized) | ~0.5× |
| LAMB | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment) | 2× |
| Lion | $m_{t-1}$ (first moment only) | 1× |
| Nadam | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment) | 2× |
| AMSGrad | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment) + $v_{\max}$ | 2× |
| SparseAdam | $m_{t-1}$ (first moment) + $v_{t-1}$ (second moment, sparse) | 2× |
| Shampoo | Left and right preconditioner matrices per parameter | >2× (varies) |
| AdaBelief | $m_{t-1}$ (first moment) + $s_{t-1}$ (belief term) | 2× |

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

This gradient requires both $h$ (the activation output from the sigmoid layer) and $x$ (the input data). That's why activation outputs and input data must be kept in memory during the forward pass until their gradients are computed during backpropagation.

An interesting observation is that $z$ is not needed. Looking at the gradient computation, we need $\frac{\partial h}{\partial z} = \sigma'(z)$ to compute $\frac{\partial L}{\partial W_1}$. For sigmoid, the derivative is $\sigma'(z) = \sigma(z)(1-\sigma(z)) = h(1-h)$. Since we already have $h$ stored from the forward pass, we can compute the derivative directly from $h$ without needing the original $z$ value. This is a property of sigmoid and some other activation functions—their derivatives can be expressed in terms of their outputs, so you don't need to store the pre-activation values.

However, this isn't true for all activation functions. Some activation functions have derivatives that explicitly depend on the input value $z$, not just the output $h$. This means you cannot compute the derivative from $h$ alone—you need to store the original $z$ value.

To understand which activation functions require storing the pre-activation value $z$ versus those that can compute derivatives from the post-activation value $h$ alone, let's examine the derivatives of common activation functions:

| Activation | Formula | Derivative |
|------------|---------------------------|-------------------------------------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2 \cdot z) - 1$ | $\tanh'(z) = 1 - \tanh^2(z) = \text{sech}^2(z)$ |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | $\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$ |
| **Leaky ReLU** | $\text{LeakyReLU}(z) = \max(\alpha z, z)$ | $\text{LeakyReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$ |
| **ELU** | $\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$ | $\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z & \text{if } z \leq 0 \end{cases}$ |
| **GELU** | $\text{GELU}(z) = z \cdot \Phi(z)$ | $\text{GELU}'(z) = \Phi(z) + z \cdot \phi(z)$ |
| **Swish** | $\text{Swish}(z) = z \cdot \sigma(z)$ | $\text{Swish}'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z))$ |
| **Mish** | $\text{Mish}(z) = z \cdot \tanh(\text{Softplus}(z)) = z \cdot \tanh(\ln(1 + e^z))$ | $\text{Mish}'(z) = \frac{e^z (4(z+1) + 4e^{2z} + e^{3z} + e^z(4z+6))}{(1 + e^z)^2 (1 + e^{2z})}$ |
| **GEGLU** | $\text{GEGLU}(z) = z \odot \text{GELU}(z)$ | $\text{GEGLU}'(z) = \text{GELU}(z) + z \cdot \text{GELU}'(z) = \text{GELU}(z) + z(\Phi(z) + z \cdot \phi(z))$ |
| **ReGLU** | $\text{ReGLU}(z) = z \odot \text{ReLU}(z)$ | $\text{ReGLU}'(z) = \begin{cases} 2z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$ |
| **SwiGLU** | $\text{SwiGLU}(z) = z \odot \text{Swish}(z) = z^2 \cdot \sigma(z)$ | $\text{SwiGLU}'(z) = 2z \cdot \sigma(z) + z^2 \cdot \sigma(z)(1 - \sigma(z))$ |
| **Softplus** | $\text{Softplus}(z) = \ln(1 + e^z)$ | $\text{Softplus}'(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$ |
| **Softmax** | $\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$ | $\nabla_{\mathbf{z}} \text{Softmax}(\mathbf{z})_{ij} = \begin{cases} \text{Softmax}(\mathbf{z})_i(1 - \text{Softmax}(\mathbf{z})_i) & \text{if } i = j \\ -\text{Softmax}(\mathbf{z})_i \cdot \text{Softmax}(\mathbf{z})_j & \text{if } i \neq j \end{cases}$ |

*Note: All activations above are element-wise (each output depends only on its corresponding input), except Softmax which is vector-valued (takes a vector input and produces a probability distribution that sums to 1). The derivative of Softmax is a Jacobian matrix, denoted by $\nabla_{\mathbf{z}}$. The parameter $\alpha$ in Leaky ReLU and ELU is a constant hyperparameter (typically $\alpha = 0.01$ for Leaky ReLU and $\alpha = 1.0$ for ELU). The GLU (Gated Linear Unit) family (GEGLU, ReGLU, SwiGLU) are gated activations that use element-wise multiplication ($\odot$) to combine two branches: one branch passes through unchanged ($z$) and the other branch applies an activation function. In practice, GLU variants are often implemented with separate linear projections for the two branches, but the simplified form shown here uses the same input $z$ for both branches.*

Looking at the derivatives, we can categorize activation functions based on whether their derivatives can be expressed purely in terms of the output $h$:

- **Derivatives expressible in terms of output only**: Sigmoid, Tanh, and Softplus fall into this category. As we saw with sigmoid, $\sigma'(z) = h(1-h)$ depends only on $h$. Similarly, $\tanh'(z) = 1 - h^2$ and $\text{Softplus}'(z) = \sigma(z)$ can be computed from the output. For these functions, you only need to store $h$ during the forward pass.

- **Derivatives requiring the input $z$**: ReLU, Leaky ReLU, ELU, GELU, Swish, Mish, and the GLU family (GEGLU, ReGLU, SwiGLU) require storing $z$ because their derivatives explicitly contain $z$ or depend on the sign of $z$. For ReLU, you need to know whether $z > 0$ or $z \leq 0$ to compute the derivative. For GELU, Swish, Mish, and GLU variants, the derivative formulas contain $z$ explicitly, making it impossible to recover $z$ from $h$ alone.

Let's examine GELU and Swish in detail to illustrate why they require storing $z$. For GELU (Gaussian Error Linear Unit), the exact form uses the cumulative distribution function:

$$
h = z \cdot \Phi(z)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution. The derivative is:

$$
\frac{\partial h}{\partial z} = \Phi(z) + z \cdot \phi(z)
$$

where $\phi$ is the probability density function (PDF) of standard normal. This derivative explicitly contains $z$ in the term $z \cdot \phi(z)$.

In practice, GELU is often approximated using tanh for computational efficiency:

$$
h = 0.5 \cdot z \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (z + c \cdot z^3)\right)\right)
$$

where $c \approx 0.044715$ is a constant. The derivative of this approximation is:

$$
\frac{\partial h}{\partial z} = 0.5 \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (z + c \cdot z^3)\right)\right) + 0.5 \cdot z \cdot \left(1 - \tanh^2\left(\sqrt{\frac{2}{\pi}} \cdot (z + c \cdot z^3)\right)\right) \cdot \sqrt{\frac{2}{\pi}} \cdot (1 + 3c \cdot z^2)
$$

Even with the tanh approximation, the derivative still explicitly contains $z$ in multiple terms, including $z$ itself and $z^2$ and $z^3$ inside the tanh arguments. Since $\tanh$ is not easily invertible and the expression involves $z$ in multiple places, you cannot recover $z$ from $h$ alone. Therefore, you need to store the original $z$ value to compute the derivative during backpropagation, regardless of whether you use the exact GELU form or the tanh approximation.

Similarly, Swish (also known as SiLU, Sigmoid Linear Unit) has:

$$
h = z \cdot \sigma(z)
$$

with derivative:

$$
\frac{\partial h}{\partial z} = \sigma(z) + z \cdot \sigma(z)(1-\sigma(z))
$$

Again, the derivative contains $z$ explicitly, and you cannot express it purely in terms of $h$. To compute $\frac{\partial h}{\partial z}$, you need both $z$ and $\sigma(z)$, which means storing the original $z$ value.

In practice, frameworks like PyTorch store whatever intermediate values are needed to compute gradients during backpropagation. For activation functions whose derivatives can be expressed in terms of their outputs (like sigmoid), the framework may only store the post-activation values. For functions whose derivatives require the input (like GELU or Swish), the framework stores the pre-activation values. The autograd system automatically determines what needs to be saved based on the operations in the computation graph, ensuring gradients can be computed correctly while minimizing memory usage where possible.

During the forward pass, you compute and store all layer activation outputs (we call these "activations" for short). During backpropagation, you compute gradients layer by layer from the last layer backward. For each layer, you need its activation outputs to compute gradients. Once you've computed a layer's gradient and updated its parameters, you can free that layer's activation outputs—they're no longer needed. However, during the backward pass, there's overlap: while computing gradients for one layer, you still have activation outputs from earlier layers in memory. The peak memory occurs when you have both activations (from forward pass) and gradients (from backward pass) simultaneously.

The activation memory scales with batch size and sequence length—larger batches or longer sequences mean more activation outputs to store. Techniques like gradient checkpointing trade compute for memory by recomputing activations instead of storing them all.

**Training Stage**

During training, memory usage varies across different stages of the training loop. Understanding when each component is needed helps you estimate peak memory requirements and identify optimization opportunities.

```python
#BKG:white;NOLINENUM
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

Inference memory requirements differ from training in that they primarily consist of model weights and the key-value (KV) cache used for attention computation. The KV cache stores precomputed key-value pairs from previous tokens in the sequence, enabling efficient autoregressive generation by avoiding redundant attention computations over the entire sequence history. While this optimization significantly accelerates inference, the KV cache introduces a memory overhead that scales linearly with batch size, sequence length, and model dimensions.

The memory footprint of inference scales with three primary factors: model size, batch size, and sequence length. For a 70B parameter model using BF16 precision, the model weights consume approximately 140 GB. With a batch size of 32 and sequence length of 2048, the KV cache adds an additional 20-40 GB, resulting in a total memory requirement of 160-180 GB. This exceeds the capacity of a single A100 GPU (80 GB), necessitating multi-GPU configurations or model parallelism strategies for inference workloads.

The memory requirements become substantially more demanding for models with ultra-long context windows. The `meta-llama/Llama-4-Scout-17B-16E-Instruct` model, for instance, supports a 10 million token context window. Despite being a 17B parameter Mixture of Experts (MoE) model with 16 experts—where only a subset of experts are activated per token—the KV cache alone can exceed 1 TB of memory for a single sequence at full context length[^vllm-blog-llama4-mem]. This scale of memory requirement renders single-node inference architectures infeasible, making distributed inference systems with dozens of GPUs not merely beneficial but fundamentally necessary. The computational and memory demands of ultra-long context models establish distributed systems as the only viable deployment architecture.

[^vllm-blog-llama4-mem]: Llama 4 in vLLM - https://blog.vllm.ai/2025/04/05/llama4.html

#### GPU Requirements Estimation

Estimating GPU requirements requires accounting for both model memory and operational overhead. For training workloads, the calculation begins with the model's memory footprint, to which a 10-20% safety margin must be added to accommodate communication buffers and framework overhead. Consider a 13B parameter model using BF16 precision: the base memory requirement is approximately 72 GB per GPU. With the safety margin applied, this increases to 85 GB. Since an A100 GPU provides 80 GB of memory, this configuration necessitates 2 GPUs using model parallelism or Fully Sharded Data Parallel (FSDP) strategies.

Inference requirements follow a similar calculation methodology, combining model weights with KV cache memory. A 70B parameter model in BF16 requires 140 GB for weights alone. When accounting for KV cache overhead, the total memory requirement reaches 160-180 GB. This necessitates a minimum of 2 A100 GPUs, or alternatively, Int8 quantization can reduce weight memory to approximately 70 GB, potentially enabling single-GPU deployment with careful KV cache management.

#### Real-World Considerations

Practical deployment introduces additional memory overhead beyond the base model requirements. The PyTorch framework typically consumes 1-2 GB, while the operating system requires 5-10 GB. Distributed training architectures allocate an additional 2-5 GB per GPU for communication buffers. Checkpointing operations create temporary memory spikes that must be accounted for in capacity planning. A conservative approach adds a 20-30% buffer to base estimates to accommodate these overheads and operational variations.

Memory optimization strategies play a critical role in practical deployments. Mixed precision training using BF16 (or FP16/BF16 for inference) reduces memory consumption by approximately 50% compared to FP32, with minimal impact on model accuracy. For inference workloads, Int8 quantization can further halve memory requirements while maintaining acceptable accuracy degradation. Monitoring actual memory usage through tools such as `nvidia-smi` provides empirical validation of theoretical estimates. It is important to recognize that activation memory scales linearly with batch size; when encountering out-of-memory (OOM) errors, reducing batch size represents the most immediate mitigation strategy.

Quick Reference Table[^memory_estimates]:

| Model Size | FP32 Weights | BF16 Weights | Training (BF16+Adam) | Inference (BF16) |
|------------|--------------|--------------|----------------------|------------------|
| 1B         | 4 GB         | 2 GB         | ~8 GB                | 2-4 GB           |
| 7B         | 28 GB        | 14 GB        | ~60-70 GB            | 14-20 GB         |
| 13B        | 52 GB        | 26 GB        | ~110-130 GB          | 26-35 GB         |
| 70B        | 280 GB       | 140 GB       | ~600-700 GB          | 140-180 GB       |

[^memory_estimates]: Training estimates assume Adam optimizer and moderate batch size. Actual values vary based on architecture, sequence length, and batch size.

### The Evolution from Classic ML to Foundation Models

The progression from classic machine learning to modern foundation models represents a fundamental shift in computational requirements and architectural paradigms. Classic machine learning models were designed to operate on single machines, with traditional algorithms—including linear regression, logistic regression, decision trees, random forests, support vector machines (SVM), and gradient boosting methods (XGBoost, LightGBM)—typically comprising thousands to millions of parameters and training on datasets that fit entirely in system memory.

The deep learning era introduced models with hundreds of millions of parameters, exemplified by architectures such as ResNet and BERT. While these models necessitated GPU acceleration, they remained manageable on single-device configurations. The contemporary foundation model era has fundamentally altered this landscape: models now span billions to trillions of parameters (e.g., GPT-4, Gemini, LLaMA), requiring distributed systems as an architectural prerequisite rather than an optimization.

This transition to distributed AI has enabled breakthrough capabilities, including models capable of understanding and generating human-like text, code, and multimodal content. The shift has also catalyzed enterprise adoption, with organizations deploying AI systems at scale for production workloads. Furthermore, distributed architectures have accelerated research progress through faster iteration cycles enabled by parallel experimentation across multiple compute nodes.

## 2. The Modern AI Model Lifecycle

Building AI models isn't a one-shot process. It's a cycle: you collect data, train a model, deploy it, see how it performs, then go back and improve the data or model. Each stage feeds into the next.

The lifecycle looks like this:

![Modern AI Model Lifecycle](img/mdlc.png)

The lifecycle begins with data engineering, where terabytes of data are collected, curated, transformed, validated, cleaned and prepared for training. Training follows, involving forward passes, backpropagation, gradient descent, hyperparameter tuning, and even fine-tuning. Once trained, models undergo inference optimization through quantization, ONNX conversion, operator fusion, and CUDA kernel optimization. Before deployment, comprehensive benchmarking evaluates model performance through precision and recall metrics, engineering performance profiling, bottleneck analysis, and stress testing, with distributed evaluation accelerating testing on large datasets. Production deployment requires autoscaling, scheduling, load balancing, observability, API gateways, and monitoring infrastructure to handle thousands of requests per second. Production feedback identifies data collection priorities and model failure modes, completing the cycle by informing subsequent data engineering efforts and model improvements.


This book focuses on the distributed technologies you need for training, inference, benchmarking, and deployment. Data engineering gets a brief overview but isn't the main focus. Distributed data processing is important, but it's a well-established topic. Spark, Dask, and Ray have been around for years. This book's main focus is on AI-specific distributed challenges: training large models, optimizing inference, and serving them at scale.

#### Training: The Learning Phase

Training is about learning model parameters from data. The process follows a pattern: forward pass through the model, loss computation, backward pass to compute gradients, and gradient update to adjust parameters. This happens iteratively over multiple epochs until the model converges.

Training requires storing activations, gradients, and optimizer states in memory. The compute is intensive and iterative. In distributed training, you need frequent gradient synchronization across devices to keep all model copies consistent. The challenges include gradient synchronization overhead, memory constraints for large models, long training times that can span days to weeks, and the need for fault tolerance and checkpointing.

Training a 7B parameter model on 1 trillion tokens typically requires 8 A100 GPUs (80GB each) and about 2 weeks of continuous training. You need careful gradient synchronization to maintain training stability across all GPUs.

#### Inference: The Prediction/Generation Phase

Inference is about generating predictions from a trained model. Unlike training, you only need a forward pass—no gradients, no backward pass, no optimizer states. Memory requirements are lower because you only need model weights and KV cache for attention mechanisms. The compute per request is lower, but you need high throughput to serve many requests at once. Communication is minimal, mostly only for distributed inference.

The challenges include latency (sub-second for interactive apps), throughput (thousands of requests per second), efficient memory usage through KV cache management, and effective batching and scheduling. Serving a 70B parameter model for chat requires optimized inference engines like vLLM or SGLang, continuous batching to maximize GPU utilization, and careful KV cache management for variable-length sequences.

#### Serving: The Production System

Serving is about providing reliable, scalable access to models. It's not just running inference—it's building a production system with a model runner, API gateway, load balancer, and monitoring. The requirements include high availability, fault tolerance, and observability. At scale, you're dealing with multi-model, multi-tenant systems.

The challenges include system reliability and uptime, multi-model routing and load balancing, cost optimization through GPU utilization and autoscaling, and observability for debugging. A production LLM serving platform might include multiple model variants (different sizes, fine-tuned versions), A/B testing infrastructure, canary deployment pipelines, and distributed tracing and monitoring.

Here is a table of `Training vs Inference vs Serving`:

| Aspect | Training | Inference | Serving |
|--------|----------|-----------|---------|
| **Goal** | Learn parameters | Generate predictions | Provide access |
| **Memory** | High (activations + gradients) | Medium (weights + KV cache) | Variable |
| **Computation** | Iterative, intensive | Single forward pass | Request-driven |
| **Communication** | Frequent (gradients) | Minimal | API-level |
| **Latency** | Hours to days | Milliseconds to seconds | Milliseconds |
| **Throughput** | Samples per second | Tokens per second | Requests per second |


## 3. Decision Framework: When Do You Need Distributed Systems?

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


\fancydividerwithicon{python.png}

## 4. Environment Setup

In this book, we will use PyTorch as our main frame work, and the code can be cloned from git repo.

```bash
git clone https://github.com/fuhengwu2021/coderepo.git
```

To run the code, it is the best if you have access to a multiple-GPU machine, such as A10, A100 or H100/200 or even B200. If you don't have access to multiple GPUs locally, Kaggle offers free multi-GPU environments. Log in to [https://www.kaggle.com](https://www.kaggle.com), click Create, and select Notebook.

![Kaggle Notebook Creation](img/1.5.png)

Once you've created a Jupyter notebook, go to Settings → Accelerator and select GPU T4x2.

![Kaggle GPU Settings](img/3.png)

You should now have 2 T4 GPUs available. To verify your GPU setup, run the code in `code/check_cuda.py`:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}") #HL
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024**3)
    print(f"GPU {i}: {props.name} ({vram_gb:.1f} GB)")
```

\begin{codeexplanation}
\codelineannotation{1}{Checks if CUDA is available on the system}
\codelineannotation{2}{Gets the total number of GPUs}
\codelineannotation{4}{Retrieves properties for the current GPU}
\codelineannotation{5}{Converts memory from bytes to GB}
\end{codeexplanation}

Running this should show your available GPUs:

![GPU setup - 2 Tesla T4 GPUs](img/2.png)

## 5. Hands-On: Running Distributed Training and Inference

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

## 6. Distributed AI Fundamentals with PyTorch

Now that you've seen distributed training and inference in action, let's understand the fundamental concepts and APIs that make it work.

### The Distributed AI Stack

Distributed AI systems are built in layers, from high-level frameworks down to physical hardware. Understanding this stack helps you debug issues, optimize performance, and make informed decisions about which tools to use.

While the stack applies to all distributed frameworks (PyTorch, JAX, TensorFlow), this book uses PyTorch as the primary example. The concepts translate to other frameworks, but the APIs and implementation details differ. We'll focus on PyTorch's distributed APIs throughout.

```
┌─────────────────────────────────────────────────────────────┐
│  Framework Layer                                            │
│  PyTorch, JAX, TensorFlow                                   │
│  High-level APIs for models, optimizers, data loaders       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Messaging Layer                                            │
│  Tensor, Bucket                                             │
│  Organizes data into chunks for efficient communication     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Collective Operations Layer                                │
│  AllReduce, AllGather, Broadcast, Scatter, etc.             │
│  Defines communication patterns between processes           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Data Transfer Layer                                        │
│  NCCL (GPU), GLOO (CPU), MPI                                │
│  Implements collective operations efficiently               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Topology Layer                                             │
│  Ring, Fat-Tree, Mesh, Torus                                │
│  Determines communication paths between devices             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Link Layer                                                 │
│  NVLink, InfiniBand (RDMA), PCIe, Ethernet                  │
│  Physical interconnects between devices                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Physical Layer                                             │
│  GPU, TPU, NPU, CPU                                         │
│  Actual compute and memory hardware                         │
└─────────────────────────────────────────────────────────────┘
```

At the top sits the framework layer. This is where you write your code—PyTorch's `torch.distributed` module, `DDP`, `FSDP`, and the rest. When you define a model and call `loss.backward()`, PyTorch handles the gradient computation and decides when communication needs to happen. You don't think about network packets or hardware links at this level. You just write training loops and let PyTorch orchestrate the distributed operations.

Below the framework, PyTorch organizes your tensors into buckets. Instead of sending each gradient tensor separately, DDP groups multiple tensors together. This reduces communication overhead. If you have thousands of small gradient tensors, sending them one by one would be inefficient. Bucketing batches them together, so you make fewer network calls. This happens automatically—you don't control it directly, but understanding it helps when you're debugging why communication takes longer than expected.

The collective operations layer defines what communication pattern to use. AllReduce sums gradients across all ranks and distributes the result back. AllGather collects data from all ranks. Broadcast sends data from one rank to all others. These are the primitives that higher-level APIs like DDP use. When DDP synchronizes gradients, it's calling AllReduce under the hood. This layer defines what needs to happen, not how it's implemented.

The data transfer layer is where the actual implementation lives. NCCL (NVIDIA Collective Communications Library) is the most common backend for GPU training. It implements AllReduce and other collectives in ways optimized for NVIDIA GPUs. GLOO is a CPU-based backend, useful for testing when you don't have GPUs. PyTorch defaults to NCCL for GPU backends and GLOO for CPU backends, which covers most use cases. MPI is another option, mainly for CPU clusters. When you initialize a process group with `backend="nccl"`, you're choosing which library will handle the actual data movement.

The network topology determines how devices are connected. In a ring topology, GPUs form a ring—each GPU sends data to the next one in the ring. This is simple but can create bottlenecks. Fat-Tree topologies provide multiple paths between devices, reducing congestion. Mesh and Torus topologies offer different trade-offs between complexity and bandwidth. The topology affects how NCCL routes data, which impacts both bandwidth and latency.

Physical links carry the data. NVLink connects GPUs within a single node at high bandwidth—up to 600 GB/s on modern hardware. When you have multiple GPUs in one machine, they communicate over NVLink. InfiniBand with RDMA (Remote Direct Memory Access) enables direct memory access across nodes without involving the CPU. This is crucial for multi-node training. PCIe connects GPUs to CPUs, and Ethernet is slower but more common. The link layer determines the raw bandwidth available.

At the bottom sits the physical hardware—GPUs for parallel compute, TPUs for tensor operations, CPUs for coordination. This layer determines your raw compute capacity and memory limits. No amount of optimization in the layers above can overcome hardware limitations here.

When you call `dist.all_reduce()` in your code, the request flows down this entire stack. PyTorch organizes your tensors into buckets, calls the AllReduce collective operation, NCCL implements it using the network topology it discovers, data moves over NVLink links within a node or InfiniBand links across nodes, and the results end up back in GPU memory. Understanding this flow helps when you're debugging why communication is slow or why a distributed job hangs. Is it a topology issue? A link bandwidth problem? Or something in the collective operation itself? Knowing the stack helps you narrow it down.

Now that we've seen how the layers fit together, let's look at the fundamental concepts you'll work with in PyTorch. The essential building blocks are process groups, ranks, and communication primitives. These form the foundation for all distributed operations, whether you're using DDP or FSDP for data-parallel training, implementing custom parallelism strategies, or building distributed inference systems.

### Process Groups and Ranks

In distributed training, multiple processes work together. Each process runs on a different GPU or node. PyTorch organizes these processes into a process group. Within a process group, each process has a unique rank—an integer identifier starting from 0. The total number of processes is called the world size.

The term "world" refers to the entire distributed job—all processes participating in the training or inference task. The world size is the total number of processes across all nodes and GPUs. For example, if you're training on 2 nodes with 4 GPUs each, your world size is 8.

If you have 4 GPUs, you'll have 4 processes. Process 0 runs on GPU 0, process 1 on GPU 1, and so on. The rank tells each process which GPU it should use and which part of the data it should process.

There are two types of ranks: global rank and local rank. The global rank is unique across all processes in the entire distributed job (the world), ranging from 0 to world_size - 1. The local rank is unique only within a single node, starting from 0 on each node. For example, in a 2-node setup with 4 GPUs per node, node 0 has local ranks 0-3 (global ranks 0-3), and node 1 has local ranks 0-3 (global ranks 4-7). The local rank typically corresponds to the GPU index on that node, which is why you often see `torch.cuda.set_device(local_rank)` in distributed code.

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

### Collective Operations

Collective operations are the building blocks of distributed communication. They define how data moves between processes. Each operation has a specific use case. Understanding them helps you choose the right one for your task and debug communication issues.

PyTorch provides eight main collective operations. Let's walk through each one with code examples.

#### AllReduce

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allreduce.png)

AllReduce is the most common operation in distributed training. It performs a reduction (sum, max, min) across all ranks and stores the result in every rank's buffer.
DDP uses AllReduce to synchronize gradients—each rank computes gradients on its local data, then AllReduce sums them and distributes the averaged result back to all ranks[^allreduce-note].

[^allreduce-note]: AllReduce is the most common collective operation in distributed training.


```python
# Each rank has different input
tensor = torch.tensor([rank + 1, rank + 2, rank + 3], device=device)
# After all_reduce with SUM, all ranks have the same result
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# Result: [sum(1..world_size), sum(2..world_size+1), sum(3..world_size+2)]
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_allreduce.py
```

AllReduce is equivalent to Reduce followed by Broadcast, but it's more efficient because NCCL can optimize the combined operation.

#### AllGather

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allgather.png)

AllGather collects data from all ranks and distributes the concatenated result to every rank. Each rank contributes N values, and every rank receives world_size × N values. The output is ordered by rank index.

Use AllGather when you need every rank to have data from all other ranks. For example, collecting embeddings from all GPUs for a global operation.

```python
# Each rank has different input
input_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], device=device)
# Prepare output list
output_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
dist.all_gather(output_list, input_tensor)
# All ranks now have: [rank0_data, rank1_data, ..., rankN_data]
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_allgather.py
```

Note: ReduceScatter followed by AllGather is equivalent to AllReduce. Some systems use this decomposition for optimization.

#### Broadcast

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/broadcast.png)

Broadcast copies data from a root rank to all other ranks. Only the root rank needs to have the data initially. After the operation, all ranks have identical data.

Use Broadcast to send model weights, hyperparameters, or other shared data from rank 0 to all ranks.

```python
root = 0
if rank == root:
    tensor = torch.tensor([10.0, 20.0, 30.0], device=device)
else:
    tensor = torch.zeros(3, device=device)
# After broadcast, all ranks have [10.0, 20.0, 30.0]
dist.broadcast(tensor, src=root)
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_broadcast.py
```

#### Reduce

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reduce.png)

Reduce performs the same reduction as AllReduce, but only the root rank receives the result. Other ranks' buffers are unchanged.

Use Reduce when only one rank needs the aggregated result, such as collecting metrics to rank 0 for logging.

```python
root = 0
tensor = torch.tensor([rank + 1, rank + 2, rank + 3], device=device)
dist.reduce(tensor, dst=root, op=dist.ReduceOp.SUM)
# Only root has the sum; other ranks unchanged
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_reduce.py
```

Reduce followed by Broadcast is equivalent to AllReduce.

#### Gather

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/gather.png)

Gather collects data from all ranks to the root rank. Each rank sends N values, and the root receives world_size × N values concatenated and ordered by rank index.

Use Gather to collect results from all ranks to a single rank for processing or saving.

```python
root = 0
input_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], device=device)
if rank == root:
    output_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
    dist.gather(input_tensor, output_list, dst=root)
    # Root has all data concatenated
else:
    dist.gather(input_tensor, None, dst=root)
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_gather.py
```

#### Scatter

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/scatter.png)

Scatter is the inverse of Gather. The root rank distributes data to all ranks, with each rank receiving a different chunk.

Use Scatter to distribute different data chunks from one rank to all ranks, such as splitting a large dataset.

```python
root = 0
if rank == root:
    scatter_list = [torch.tensor([r * 10 + 1, r * 10 + 2], device=device) 
                   for r in range(world_size)]
output_tensor = torch.zeros(2, device=device)
if rank == root:
    dist.scatter(output_tensor, scatter_list=scatter_list, src=root)
else:
    dist.scatter(output_tensor, scatter_list=None, src=root)
# Each rank receives its chunk
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_scatter.py
```

#### ReduceScatter

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png)

ReduceScatter combines Reduce and Scatter. It performs a reduction across all ranks, then scatters the result in equal-sized chunks. Each rank receives a different chunk based on its rank index.

Use ReduceScatter in FSDP and other sharded parallelism strategies where each rank needs a shard of the reduced result.

```python
# Each rank has input of size world_size * N
input_list = [torch.tensor([rank * 10 + i, rank * 10 + i + 1], device=device) 
              for i in range(world_size)]
input_tensor = torch.cat(input_list)
# Each rank receives a chunk of the reduced result
output_tensor = torch.zeros(2, device=device)
dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_reducescatter.py
```

#### AlltoAll

![](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/alltoall.png)

AlltoAll is the most general operation. Each rank sends different data to every other rank. Each rank provides world_size chunks, and receives world_size chunks—one from each rank.

Use AlltoAll in tensor parallelism where each rank needs to exchange different parts of tensors with all other ranks.

```python
# Each rank prepares data to send to each other rank
input_list = []
for dst_rank in range(world_size):
    chunk = torch.tensor([rank * 100 + dst_rank * 10 + 1, 
                         rank * 100 + dst_rank * 10 + 2], device=device)
    input_list.append(chunk)
# Each rank receives data from all ranks
output_list = [torch.zeros(2, device=device) for _ in range(world_size)]
dist.all_to_all(output_list, input_list)
```

Run the demo:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_alltoall.py
```

Note: AlltoAll requires NCCL backend. GLOO (CPU backend) doesn't support it. If you see an error with `--use_cpu`, switch to GPU mode.

#### Choosing the Right Operation

So when do you use which operation? For gradient synchronization, AllReduce is the standard choice. DDP uses it automatically when you wrap your model—you don't need to call it yourself. If you need to collect embeddings or metrics from all ranks, AllGather is what you want. It gives every rank a copy of data from all other ranks.

When you have data on one rank that needs to go to everyone else, Broadcast is the simplest option. Think of it as one-to-all communication. The inverse is Gather—when all ranks have data and you need to collect it on one rank, usually rank 0 for logging or saving. Scatter does the opposite, distributing different chunks from one rank to all others.

For sharded parallelism strategies like FSDP, you'll see ReduceScatter and AlltoAll. ReduceScatter combines reduction with scattering, which is efficient when each rank only needs a shard of the result. AlltoAll is the most general case, where every rank sends different data to every other rank. This shows up in tensor parallelism.

Most of the time, you won't call these operations directly. DDP handles AllReduce for you during gradient synchronization. FSDP uses ReduceScatter and AllGather under the hood. But understanding what each operation does helps when you're debugging why communication is slow or when you need to implement custom parallelism strategies that the standard APIs don't cover.


### DistributedDataParallel (DDP)

Among the various distributed training strategies, DDP is the simplest to understand and use. It requires minimal changes to your single-GPU training code, and it's the most common starting point for distributed training. This section provides a brief overview of DDP. Chapter 3 covers DDP in depth with detailed implementation, optimization techniques, and best practices.

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

Now that we understand when and why to use distributed systems, we need to understand the hardware they run on. The next chapter explores GPU hardware, networking topologies, and the fundamental parallelism strategies that make distributed AI possible. Understanding these foundations is crucial for making informed decisions about which distributed approach to use.

## Further Reading

- PyTorch Distributed Training: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- NVIDIA NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- GPU Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- Profiling PyTorch Models: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html



