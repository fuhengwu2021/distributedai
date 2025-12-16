# Ollama Model Loading Simulation

This directory contains a simulation of how Ollama loads models and uses PagedAttention for efficient inference.

## Overview

Ollama stores models in a specific format:
- **Blob files**: `/usr/share/ollama/.ollama/models/blobs/` (system-wide) or `~/.ollama/models/blobs/` (user-specific)
- **Manifests**: `/usr/share/ollama/.ollama/models/manifests/` (metadata about models)
- **Format**: GGUF (GPT-Generated Unified Format) - a binary format for storing model weights

This simulation demonstrates:
1. How Ollama locates and loads model files from blob storage
2. How model weights are extracted from GGUF format (simulated)
3. How PagedAttention is integrated for efficient KV cache management
4. How continuous batching works with multiple requests

## Files

- **`model_loader.py`**: Simulates Ollama's model loading process
  - `OllamaModelLoader`: Finds and loads model blobs from Ollama storage
  - `OllamaModelSimulator`: Simulates the complete loading process
  
- **`ollama_inference.py`**: Ollama-style inference engine with PagedAttention
  - `OllamaInferenceEngine`: Main inference engine that uses PagedAttention
  - Demonstrates prefill and decode phases with continuous batching

- **`demo.py`**: Simple demo script

## Model Location

For the model `Qwen3:4b-instruct-2507-fp16`:

- **Manifest**: `/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/Qwen3/4b-instruct-2507-fp16`
- **Blob file**: `/usr/share/ollama/.ollama/models/blobs/sha256-ad7f12578413f17ce506bbf2809c3c7b8fd27bb1cf81acb3ac133a7538fc4259`
- **Size**: ~7.5 GB (FP16 weights)

## Usage

### Basic Usage

```python
from ollama_inference import OllamaInferenceEngine

# Initialize engine (loads model from Ollama blob storage)
engine = OllamaInferenceEngine(
    model_name="Qwen3:4b-instruct-2507-fp16",
    block_size=16,
    device="cuda"
)

# Add a request
seq_id = engine.add_request("What is the capital of France?", max_new_tokens=50)

# Process requests
seq_ids, next_token_ids = engine.step()

# Get results
text = engine.get_sequence_text(seq_id)
print(text)
```

### Running the Demo

```bash
cd /media/wukong/jackie/git.repo/distributed-ai/ollama
python ollama_inference.py
```

## How It Works

### 1. Model Loading

```
┌─────────────────────────────────────────────────────────────┐
│ 1. OllamaModelLoader finds manifest                        │
│    - Searches: /usr/share/ollama/.ollama/models/manifests/ │
│    - Path: registry.ollama.ai/library/Qwen3/4b-instruct... │
│                                                             │
│ 2. Extract blob digest from manifest                       │
│    - Finds layer with mediaType: "application/vnd.ollama...│
│    - Gets digest: sha256:ad7f1257...                      │
│                                                             │
│ 3. Locate blob file                                        │
│    - Path: /usr/share/ollama/.ollama/models/blobs/         │
│    - File: sha256-ad7f1257...                              │
│    - Size: ~7.5 GB                                         │
│                                                             │
│ 4. Parse GGUF format (simulated)                           │
│    - Extract model architecture                            │
│    - Load weight tensors                                   │
│    - Load tokenizer config                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2. PagedAttention Integration

The inference engine uses PagedAttention for efficient KV cache management:

- **Block-based storage**: KV cache stored in fixed-size blocks (16 tokens)
- **No padding**: Only processes tokens that exist
- **Continuous batching**: Multiple sequences processed together
- **Online softmax**: Single-pass attention computation

### 3. Inference Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Prefill Phase:                                              │
│  1. Tokenize prompts                                         │
│  2. Forward pass through model (use_cache=True)            │
│  3. Extract KV cache from past_key_values                   │
│  4. Store KV in PagedAttention blocks                       │
│                                                             │
│ Decode Phase:                                               │
│  1. Embed current tokens (batch)                            │
│  2. For each layer:                                         │
│     - Compute Q, K, V                                       │
│     - Apply RoPE                                            │
│     - Append KV to PagedAttention blocks                   │
│     - Compute attention using PagedAttention                │
│     - MLP and residual                                      │
│  3. LM head and sample next tokens                          │
│  4. Update scheduler                                        │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

### Model Configuration (Qwen3:4b-instruct-2507-fp16)

- **Architecture**: Qwen3
- **Parameters**: 4.0B
- **Quantization**: FP16
- **Context length**: 262,144 tokens
- **Embedding length**: 2,560
- **Attention heads**: 20 (Q), 2 (KV) - GQA
- **Layers**: 28
- **Hidden size**: 2,560
- **Vocab size**: 151,936

### PagedAttention Configuration

- **Block size**: 16 tokens per block
- **Max blocks**: 1000 (pre-allocated)
- **Online softmax**: Enabled (single-pass)
- **GQA support**: Yes (no physical KV repeat)

## Notes

1. **GGUF Parsing**: The actual GGUF format parsing is complex and requires specialized libraries. This simulation uses HuggingFace as a fallback to load actual weights, but demonstrates the loading structure.

2. **Real Ollama**: In production, Ollama uses:
   - `llama.cpp` for GGUF parsing
   - Custom CUDA kernels for PagedAttention
   - Optimized memory management

3. **Model Storage**: Ollama stores models system-wide by default (`/usr/share/ollama/`), but can also use user-specific storage (`~/.ollama/models/`).

## Dependencies

- `torch`
- `transformers`
- PagedAttention module from `resources/coderepo/06/code/pa/`

## References

- [Ollama GitHub](https://github.com/ollama/ollama)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
