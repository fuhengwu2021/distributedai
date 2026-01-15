# SGLang Diffusion: Deep Dive into Video and Image Generation Architecture

## Overview

SGLang Diffusion (released November 2025) is a comprehensive framework for accelerating **video and image generation** using diffusion models. It extends SGLang's high-performance serving infrastructure from LLM inference to support generative diffusion-based tasks, with built-in support for **distributed computation** across multiple GPUs.

## Key Supported Models

### Image Generation
- **Qwen-Image** (Alibaba Qwen's image generation model)
- **Z-Image** (Zhejiang University's image model)
- **Flux** (Black Forest Labs' diffusion model)
- **HunyuanVideo** (NVIDIA's video-to-image)

### Video Generation
- **WAN-Video** (Alibaba's WAN video diffusion model)
- **StepVideo** (StepFun's video generation model)
- **CausalWANVideo** (Causal version of WAN)
- **HunyuanVideo** (NVIDIA's video generation model)

---

## Architecture Components

### 1. **Core Entrypoint: DiffGenerator**

**Location:** [`python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py)

```python
class DiffGenerator:
    """
    Unified interface for image/video generation using diffusion models.
    Similar to popular frameworks like HF Diffusers.
    """
    
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.local_scheduler_process: list[mp.Process] | None = None
        self.owns_scheduler_client: bool = False
```

**Key Features:**
- **Process-based Architecture**: Uses multiprocessing for parallel execution
- **Local vs Remote Mode**: Can run locally or connect to remote scheduler
- **Client-Server Model**: DiffGenerator acts as client to Scheduler service
- **LoRA Support**: Supports dynamic LoRA weight merging/unmerging

### 2. **Pipeline Architecture**

#### Core Stages:
```
Input Validation
    ↓
Text Encoding (CLIPTextModel + conditioning)
    ↓
Latent Preparation (VAE encoding if needed)
    ↓
Timestep Embedding
    ↓
DENOISING LOOP (Main computation)
    ├─ Classifier-Free Guidance handling
    ├─ UNet forward pass (with parallelism)
    └─ Iterative noise reduction
    ↓
VAE Decoding (latent → pixels)
    ↓
Output Post-processing
```

#### Pipeline Stage Classes:
- **InputValidationStage**: Validates prompt, image, and video inputs ([`input_validation.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/input_validation.py))
- **TextEncodingStage**: Encodes text prompts using CLIP/T5 ([`text_encoding.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/text_encoding.py))
- **ImageEncodingStage**: Encodes image inputs ([`image_encoding.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py))
- **StepVideoEncodingStage**: Encodes video frames ([`stepvideo_encoding.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/stepvideo_encoding.py))
- **LatentPreparationStage**: Prepares latent space representations ([`latent_preparation.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/latent_preparation.py))
- **TimestepPreparationStage**: Handles timestep scheduling ([`timestep_preparation.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/timestep_preparation.py))
- **DenoisingStage**: Executes the main diffusion loop ([`denoising.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py))
- **CausalDenoisingStage**: Causal version for video generation ([`causal_denoising.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/causal_denoising.py))
- **DecodingStage**: VAE decoding from latents to images ([`decoding.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py))
- **ConditionalStage**: Applies conditioning transformations ([`conditioning.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/conditioning.py))

---

## Distributed Computation Strategy

### 1. **Parallel State Management**

**Location:** [`python/sglang/multimodal_gen/runtime/distributed/parallel_state.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py)

Adapted from Megatron-LM and vLLM, SGLang Diffusion implements sophisticated parallel state management:

```python
# Global parallel groups
_WORLD: GroupCoordinator = None    # World process group
_TP: GroupCoordinator = None       # Tensor Parallel
_SP: SequenceParallelGroupCoordinator = None  # Sequence Parallel
_PP: PipelineGroupCoordinator = None          # Pipeline Parallel
_CFG: GroupCoordinator = None      # Classifier-Free Guidance
_DP: GroupCoordinator = None       # Data Parallel
_DIT: GroupCoordinator = None      # Diffusion Transformer specific
_VAE: GroupCoordinator = None      # VAE-specific parallelism
```

### 2. **Parallelism Types Supported**

#### A. **Tensor Parallelism (TP)**
- Splits model layers across multiple GPUs
- Each GPU processes a subset of attention heads or hidden dimensions
- Implemented in [`layers/linear.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/linear.py) with `ReplicatedLinear`

```python
# From wanvideo.py: Tensor parallelism in attention layers
from sglang.multimodal_gen.runtime.layers.attention import (
    UlyssesAttention_VSA,  # Tensor-parallel sparse attention
    USPAttention,          # Unified Sequence Parallel attention
)
```

#### B. **Sequence Parallelism (SP)**
- Splits sequence/temporal dimension across GPUs
- Particularly useful for video generation (many frames)
- Implemented via `SequenceParallelGroupCoordinator` ([`group_coordinator.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py))

```python
# Getting sequence parallel world size
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size

sp_size = get_sp_world_size()  # Number of GPUs in SP group
```

#### C. **Pipeline Parallelism (PP)**
- Splits model stages across GPUs
- Useful for very large diffusion models
- Implemented via `PipelineGroupCoordinator` ([`group_coordinator.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py))

#### D. **Data Parallelism (DP)**
- Each GPU processes different generation requests
- Batch processing across multiple GPUs
- Implemented via standard `GroupCoordinator`

#### E. **Classifier-Free Guidance (CFG) Specific Parallelism**
- Dedicated parallelism for CFG computation
- Handles both conditional and unconditional predictions
- **Code Locations:** [`parallel_state.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py) + [`denoising.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py)

```python
# CFG-specific group coordination
_CFG: GroupCoordinator = None
get_classifier_free_guidance_rank()  # Identify CFG role
```

#### F. **Model-Specific Parallelism**
- **DIT (Diffusion Transformer) Parallelism**: Specialized for DiT models
- **VAE Parallelism**: Optimized for VAE encoding/decoding

### 3. **Communication Operations**

**Location:** [`python/sglang/multimodal_gen/runtime/distributed/communication_op.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/communication_op.py)

**Key Operations:**
- **All-Reduce**: Synchronize gradients across parallel groups
- **All-Gather**: Collect tensors from all ranks
- **Sequence Parallel All-Gather**: Specialized for sequence dimension

```python
# From denoising.py
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)

# Gather sequence information across parallel ranks
tensor = sequence_model_parallel_all_gather(local_tensor, group=sp_group)
```

---

## Advanced Features

### 1. **Attention Backend Abstraction**

**Location:** [`python/sglang/multimodal_gen/runtime/layers/attention/`](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/attention/)

Multiple attention backends for different hardware:

```python
# From denoising.py
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend

backends = {
    FlashAttentionBackend: "FlashAttention-2 (NVIDIA A100+)",
    SlidingTileAttentionBackend: "Optimized tile-based attention",
    VMOBABackend: "VMOBA attention",
    VideoSparseAttentionBackend: "Sparse attention for video",
}
```

### 2. **Sparse Attention for Video**

**Location:** [`python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn.py)

Video generation can use sparse attention patterns to reduce memory:

```python
# From attention/backends/video_sparse_attn.py
class VideoSparseAttentionBackend:
    """
    Specialized sparse attention for video frames.
    Reduces O(L²) to O(L×M) where M << L
    Useful for long video sequences.
    """
```

### 3. **Dynamic Timestep Embeddings**

**Location:** [`python/sglang/multimodal_gen/runtime/layers/visual_embedding.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/visual_embedding.py)

Efficient timestep handling across parallel dimensions:

```python
# From visual_embedding.py
class TimestepEmbedder(nn.Module):
    def __init__(self, dim, frequency_embedding_size, act_layer="silu"):
        self.time_embedder = nn.Sequential(...)
        # Broadcast across parallel groups
```

### 4. **LoRA Support**

**Location:** [`python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py)

Dynamic Low-Rank Adaptation for model fine-tuning:

```python
# From entrypoints/diffusion_generator.py
class DiffGenerator:
    def _current_lora_path: str | None = None
    def _current_lora_nickname: str | None = None
    def _is_lora_merged: bool = False
    
    # Methods for LoRA management
    def set_lora(self, lora_path)
    def unmerge_lora_weights()
    def merge_lora_weights()
```

---

## Distributed Execution Workflow

### 1. **Local Mode Execution**

```
User Creates DiffGenerator
    ↓
from_pretrained(model_name, tensor_parallel_size=4)
    ↓
Initialize Distributed Environment
    ├─ Create 4 process groups (one per GPU)
    ├─ Initialize PyTorch Distributed Backend (NCCL)
    └─ Set up parallel state groups (TP, SP, PP, etc.)
    ↓
Start Local Scheduler (Multiprocessing)
    ├─ Process 0: Main execution process
    ├─ Process 1-3: Worker processes
    └─ Synchronize via ProcessGroup
    ↓
Prepare Request (prepare_request function)
    ├─ Validate inputs
    ├─ Encode prompts
    └─ Schedule batch
    ↓
Execute Pipeline Stages
    ├─ All GPUs: TextEncodingStage
    ├─ GPU 0: LatentPreparationStage
    ├─ All GPUs (Parallelized): DenoisingStage
    │   ├─ TP: Split UNet across GPUs
    │   ├─ SP: Split temporal dimension
    │   └─ Communicate via All-Reduce
    └─ All GPUs: DecodingStage
    ↓
Return Generated Image/Video
```

### 2. **Remote Mode Execution**

```
User Creates DiffGenerator
    ↓
Connect to Remote Scheduler Service
    ├─ Server listening on specified port
    └─ Client sends requests via sync_scheduler_client
    ↓
Server-side Distributed Setup (same as above)
    ↓
Request Processing (server handles parallelism)
    ↓
Return Results to Client
```

---

## Key Distributed Techniques for Diffusion Models

### 1. **Tensor Parallelism in UNet**

**Location:** [`python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py)

The UNet backbone in diffusion models is split across GPUs:

```python
# From wanvideo.py - Attention layers with TP
class WanVideoBlock(nn.Module):
    def __init__(self, ...):
        # Tensor-parallel attention
        self.attn = UlyssesAttention_VSA(...)  # Ulysses style TP
        # Or
        self.attn = USPAttention(...)          # Unified SP
```

**Ulysses Attention (from xDiT):**
- Splits QKV projections across GPUs
- Each GPU handles subset of attention heads
- Reduces per-GPU memory from $O(S²)$ to $O(S²/N)$ where N is number of GPUs

### 2. **Sequence Parallelism for Video**

**Location:** [`python/sglang/multimodal_gen/runtime/distributed/parallel_state.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py)

Since videos have many frames, sequence parallelism is crucial:

```python
# From parallel_state.py
def get_sp_world_size() -> int:
    """Get sequence parallel group size"""
    if _SP is None:
        return 1
    return _SP.world_size

def get_sp_parallel_rank() -> int:
    """Get rank within sequence parallel group"""
```

**Benefits for Video:**
- 1 frame per GPU (if 4 GPUs, 4 frames processed simultaneously)
- Temporal attention computed locally with all-gather sync
- Reduces peak memory requirement

### 3. **Classifier-Free Guidance Parallelization**

CFG requires computing both conditional and unconditional predictions:

```python
# CFG splits computation across parallel dimension
# GPU 0-1: Process conditional prompts
# GPU 2-3: Process unconditional prompts
# Then merge results
```

**Efficiency Gain:**
- 2× parallelism without sacrificing generation quality
- Shares same model, different guidance conditions

### 4. **VAE Parallelization**

**Location:** [`python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py)

Encoding/decoding can be parallelized:

```python
# From pipelines_core/stages/decoding.py
class DecodingStage(PipelineStage):
    def __call__(self, batch: OutputBatch) -> OutputBatch:
        # VAE decoding parallelized across GPUs
        latents_per_gpu = split_batch_across_devices()
        decoded = vae.decode(latents_per_gpu)
        return gathered_output
```

---

## Performance Optimizations

### 1. **Flash Attention Support**

**Location:** [`python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py)

Efficient attention computation:
```python
FlashAttentionBackend  # For NVIDIA GPUs
```

### 2. **Sparse Attention Patterns**

Reduces attention complexity:
```python
VideoSparseAttentionBackend  # For long video sequences
```

### 3. **Chunked Prefill (from LLM serving)**

Though primarily for LLMs, can apply to prompt encoding:
- Process prompts in chunks
- Amortize encoding cost

### 4. **Overlap Computation and Communication**

Pipeline parallelism overlaps:
- Forward pass on GPU N
- Communication of GPU N-1 result
- Backward pass on GPU N-1

### 5. **Dynamic Scheduling**

**Location:** [`python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py`](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py)

Batch scheduling optimizations:
```python
# From schedule_batch.py
class OutputBatch:
    def __init__(self, requests: list[Req]):
        # Efficient batch scheduling
        self.requests = requests
        # Pad/align for efficient parallel execution
```

---

## Configuration Examples

### Basic Single-GPU Setup
```python
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

generator = DiffGenerator.from_pretrained(
    model="wan-video",
)

output = generator.generate(
    prompt="A serene landscape with mountains",
    num_frames=16,
    height=512,
    width=512,
)
```

### Tensor Parallel Setup (4 GPUs)
```python
generator = DiffGenerator.from_pretrained(
    model="wan-video",
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    sequence_parallel_size=1,
)
```

### Sequence Parallel Setup (Video with many frames)
```python
generator = DiffGenerator.from_pretrained(
    model="wan-video",
    tensor_parallel_size=2,
    sequence_parallel_size=2,  # 2 frames per GPU
)
```

### Combined Parallelism (8 GPUs)
```python
generator = DiffGenerator.from_pretrained(
    model="wan-video",
    tensor_parallel_size=2,      # Split model across 2 GPUs
    pipeline_parallel_size=2,     # Pipeline stages across 2 GPUs
    sequence_parallel_size=2,     # Temporal dimension across 2 GPUs
)
# Total: 2×2×2 = 8 GPUs utilized
```

---

## Directory Structure with Code Links

```
sglang/multimodal_gen/
├── runtime/
│   ├── distributed/                 # Parallelism infrastructure
│   │   ├── parallel_state.py        # Group coordination [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py)
│   │   ├── communication_op.py      # AllReduce, AllGather [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/communication_op.py)
│   │   ├── group_coordinator.py     # Process group management [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py)
│   │   └── device_communicators/    # Backend-specific comms
│   ├── models/dits/                 # Diffusion Transformers
│   │   ├── wanvideo.py             # WAN video model [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py)
│   │   ├── stepvideo.py            # StepVideo model [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/models/dits/stepvideo.py)
│   │   ├── qwen_image.py           # Qwen image model [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py)
│   │   └── hunyuanvideo.py         # HunyuanVideo model [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py)
│   ├── layers/
│   │   ├── attention/               # Attention implementations [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/attention/)
│   │   │   ├── backends/           # Flash, Sparse, etc.
│   │   │   └── selector.py         # Backend selection
│   │   ├── linear.py               # Tensor-parallel linear layers [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/linear.py)
│   │   └── visual_embedding.py     # Timestep, patch embeddings [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/layers/visual_embedding.py)
│   ├── pipelines_core/
│   │   ├── stages/                 # Pipeline stage implementations [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/)
│   │   │   ├── denoising.py        # Main denoising loop [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py)
│   │   │   ├── text_encoding.py    # Prompt encoding [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/text_encoding.py)
│   │   │   ├── image_encoding.py   # Image input processing [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py)
│   │   │   └── decoding.py         # VAE decoding [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py)
│   │   └── schedule_batch.py       # Batch scheduling [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py)
│   ├── entrypoints/
│   │   ├── diffusion_generator.py  # Main user API [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py)
│   │   ├── image_api.py            # OpenAI-compatible API [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py)
│   │   └── video_api.py            # Video generation API [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py)
│   └── launch_server.py            # Server startup [🔗](resources/sglang2025/python/sglang/multimodal_gen/runtime/launch_server.py)
├── configs/
│   ├── pipeline_configs/           # Model-specific configs [🔗](resources/sglang2025/python/sglang/multimodal_gen/configs/pipeline_configs/)
│   │   ├── wan.py [🔗](resources/sglang2025/python/sglang/multimodal_gen/configs/pipeline_configs/wan.py)
│   │   ├── stepvideo.py [🔗](resources/sglang2025/python/sglang/multimodal_gen/configs/pipeline_configs/stepvideo.py)
│   │   ├── qwen_image.py [🔗](resources/sglang2025/python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py)
│   │   └── base.py [🔗](resources/sglang2025/python/sglang/multimodal_gen/configs/pipeline_configs/base.py)
│   └── sample/
│       ├── sampling_params.py      # Generation parameters [🔗](resources/sglang2025/python/sglang/multimodal_gen/configs/sample/sampling_params.py)
│       └── teacache.py             # KV cache optimization [🔗](resources/sglang2025/python/sglang/multimodal_gen/configs/sample/teacache.py)
└── ...
```

---

## Comparison: vLLM vs SGLang Diffusion

| Feature | vLLM | SGLang Diffusion |
|---------|------|------------------|
| **Text Generation** | ✅ LLM inference | ✅ LLM inference |
| **Multimodal Input** | ✅ (analysis) | ✅ (analysis) |
| **Image Generation** | ❌ | ✅ Diffusion-based |
| **Video Generation** | ❌ | ✅ Diffusion-based |
| **Tensor Parallelism** | ✅ | ✅ (advanced) |
| **Pipeline Parallelism** | ✅ | ✅ |
| **Sequence Parallelism** | ❌ | ✅ (for temporal) |
| **Classifier-Free Guidance** | ❌ | ✅ Parallelized |
| **VAE Parallelism** | ❌ | ✅ |
| **Sparse Attention** | ✅ (basic) | ✅ (video-specific) |
| **LoRA Support** | ✅ | ✅ Dynamic merging |
| **OpenAI API** | ✅ | ✅ (image_api, video_api) |

---

## Conclusion

SGLang Diffusion represents a **comprehensive solution** for distributed image and video generation. By leveraging sophisticated parallel state management adapted from Megatron-LM and vLLM, combined with **diffusion-model-specific optimizations** (sparse attention, sequence parallelism for video), it achieves:

1. **Scalability**: From single GPU to 100+ GPU clusters
2. **Efficiency**: Multiple parallelism dimensions for optimal resource utilization
3. **Flexibility**: Supports multiple generation models and backends
4. **Performance**: Optimized attention, LoRA, and batch scheduling
5. **Production-Ready**: OpenAI-compatible APIs for easy integration

The key insight is that **video generation requires different parallelism strategies** than text generation—sequence parallelism becomes as important as tensor parallelism when dealing with 100+ video frames.
