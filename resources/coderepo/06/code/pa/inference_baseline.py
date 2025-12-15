"""
Baseline inference script WITHOUT PagedAttention.

This script uses traditional KV cache management (continuous memory with padding)
to serve as a baseline for comparison with PagedAttention implementation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict
import time


class BaselineModelWrapper:
    """
    Baseline model wrapper using traditional KV cache management.
    
    This implementation:
    - Uses continuous memory for KV cache
    - Requires padding to max sequence length in batch
    - Does NOT eliminate padding FLOPs
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda"
    ):
        """
        Initialize the baseline model wrapper.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
        """
        self.device = device
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device
        )
        
        # Get model config
        config = self.model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        
        print(f"Model config: {self.num_heads} Q heads, {self.num_kv_heads} KV heads, "
              f"{self.head_dim} head_dim, {self.num_layers} layers")
        
        # Traditional KV cache: continuous memory per sequence
        # Format: {seq_id: {layer_idx: {'k': tensor, 'v': tensor}}}
        self.kv_caches: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        
        # Track sequences
        self.sequences: Dict[int, Dict] = {}
        self.next_seq_id = 0
    
    def _get_attention_layer(self, layer_idx: int):
        """Get the attention layer from the model."""
        return self.model.model.layers[layer_idx].self_attn
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply Rotary Position Embedding (RoPE) to input tensor.
        Simplified version - rotates pairs of dimensions.
        """
        # x: [batch, num_heads, seq_len, head_dim]
        # cos, sin: [batch, seq_len, head_dim] or [1, seq_len, head_dim]
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Reshape cos and sin to match
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
            sin = sin.unsqueeze(1)
        
        cos = cos[..., 0::2]  # Take even indices
        sin = sin[..., 0::2]
        
        # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Interleave back
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x1_rot
        x_rot[..., 1::2] = x2_rot
        
        return x_rot
    
    def prefill(self, prompt: str, seq_id: Optional[int] = None) -> int:
        """
        Process the prompt and cache KV using traditional continuous memory.
        
        Args:
            prompt: Input prompt text
            seq_id: Optional sequence ID (if None, creates new sequence)
            
        Returns:
            Sequence ID
        """
        if seq_id is None:
            seq_id = self.next_seq_id
            self.next_seq_id += 1
        
        # Apply chat template if available (for Qwen models)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            # Format as chat messages
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        else:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = tokens[0].tolist()
        
        # Store sequence info
        self.sequences[seq_id] = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": [],
            "total_tokens": len(prompt_tokens),
            "max_length": len(prompt_tokens)  # Track max length for this sequence
        }
        
        # Initialize KV cache for this sequence
        self.kv_caches[seq_id] = {}
        
        print(f"\n[Prefill] Sequence {seq_id}: Processing {len(prompt_tokens)} prompt tokens")
        
        # Process prompt tokens using model's forward (includes RoPE automatically)
        with torch.no_grad():
            # Use model's forward with use_cache to get KV cache
            outputs = self.model(input_ids=tokens, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Extract and store KV cache in our format
            for layer_idx in range(self.num_layers):
                k, v = past_key_values[layer_idx]
                # k, v shape: [batch, num_kv_heads, seq_len, head_dim]
                # Convert to [num_heads, seq_len, head_dim] for our cache format
                k_cache = k[0]  # [num_kv_heads, seq_len, head_dim]
                v_cache = v[0]  # [num_kv_heads, seq_len, head_dim]
                
                # Handle GQA: repeat K and V if num_kv_heads < num_heads
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_cache = k_cache.repeat_interleave(repeat_factor, dim=0)  # [num_heads, seq_len, head_dim]
                    v_cache = v_cache.repeat_interleave(repeat_factor, dim=0)  # [num_heads, seq_len, head_dim]
                
                self.kv_caches[seq_id][layer_idx] = {
                    'k': k_cache.clone(),
                    'v': v_cache.clone()
                }
        
        # Check what the first token would be (using model's output)
        logits_check = outputs.logits[:, -1, :]
        first_token_id = torch.argmax(logits_check, dim=-1).item()
        first_token_text = self.tokenizer.decode([first_token_id])
        print(f"[Prefill] Sequence {seq_id}: Cached KV for {len(prompt_tokens)} tokens "
              f"(continuous memory: {self.kv_caches[seq_id][0]['k'].shape})")
        print(f"[Prefill] First token would be: id={first_token_id}, text='{first_token_text}'")
        
        return seq_id
    
    def decode_step(self, seq_id: int) -> Optional[int]:
        """
        Generate one token using traditional KV cache lookup.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            Generated token ID, or None if sequence not found
        """
        if seq_id not in self.sequences:
            return None
        
        seq_info = self.sequences[seq_id]
        
        # Get the last generated token (or last prompt token if no generation yet)
        if seq_info["generated_tokens"]:
            last_token_id = seq_info["generated_tokens"][-1]
        else:
            last_token_id = seq_info["prompt_tokens"][-1]
        
        token_tensor = torch.tensor([[last_token_id]], device=self.device)
        
        with torch.no_grad():
            # Get embedding for the current token
            hidden_states = self.model.model.embed_tokens(token_tensor)
            
            # Process through each layer
            for layer_idx in range(self.num_layers):
                layer = self.model.model.layers[layer_idx]
                attention = self._get_attention_layer(layer_idx)
                
                # Apply layer norm before attention
                hidden_states_norm = layer.input_layernorm(hidden_states)
                
                # Compute Q, K, V for the current token
                q_proj = attention.q_proj(hidden_states_norm)
                k_proj = attention.k_proj(hidden_states_norm)
                v_proj = attention.v_proj(hidden_states_norm)
                
                # Reshape
                q = q_proj.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [1, num_heads, 1, head_dim]
                k = k_proj.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, num_kv_heads, 1, head_dim]
                v = v_proj.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, num_kv_heads, 1, head_dim]
                
                # Get cached K and V (OLD cache, before adding current token)
                # Note: cache already has repeated heads from prefill
                k_cached = self.kv_caches[seq_id][layer_idx]['k']  # [num_heads, cached_len, head_dim]
                v_cached = self.kv_caches[seq_id][layer_idx]['v']  # [num_heads, cached_len, head_dim]
                
                # Handle GQA for current token's K/V before adding to cache
                k_new = k[0]  # [num_kv_heads, 1, head_dim]
                v_new = v[0]  # [num_kv_heads, 1, head_dim]
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_new = k_new.repeat_interleave(repeat_factor, dim=0)  # [num_heads, 1, head_dim]
                    v_new = v_new.repeat_interleave(repeat_factor, dim=0)  # [num_heads, 1, head_dim]
                
                # Compute attention over OLD cached KV (before adding current token's K/V)
                # q: [1, num_heads, 1, head_dim]
                # k_cached: [num_heads, cached_len, head_dim]
                # v_cached: [num_heads, cached_len, head_dim]
                
                # Reshape for batched attention computation
                k_cached_batched = k_cached.unsqueeze(0)  # [1, num_heads, cached_len, head_dim]
                v_cached_batched = v_cached.unsqueeze(0)  # [1, num_heads, cached_len, head_dim]
                
                # Compute attention scores: Q @ K^T
                # q: [1, num_heads, 1, head_dim]
                # k_cached_batched: [1, num_heads, cached_len, head_dim]
                # scores: [1, num_heads, 1, cached_len]
                scale = 1.0 / (self.head_dim ** 0.5)
                scores = torch.matmul(q, k_cached_batched.transpose(-2, -1)) * scale
                attn_weights = F.softmax(scores, dim=-1)  # [1, num_heads, 1, cached_len]
                
                # Compute attention output: attn_weights @ V
                attn_output = torch.matmul(attn_weights, v_cached_batched)  # [1, num_heads, 1, head_dim]
                
                # NOW add new K, V to cache (for next step)
                # k_new and v_new are already repeated for GQA above
                k_cached = torch.cat([k_cached, k_new], dim=1)  # [num_heads, cached_len+1, head_dim]
                v_cached = torch.cat([v_cached, v_new], dim=1)  # [num_heads, cached_len+1, head_dim]
                
                # Update cache
                self.kv_caches[seq_id][layer_idx]['k'] = k_cached
                self.kv_caches[seq_id][layer_idx]['v'] = v_cached
                
                # Reshape for output projection
                attn_output = attn_output.transpose(1, 2).contiguous()  # [1, 1, num_heads, head_dim]
                attn_output = attn_output.view(1, 1, -1)
                attn_output = attention.o_proj(attn_output)
                
                # Residual connection
                hidden_states = hidden_states + attn_output
                
                # Feedforward
                hidden_states_norm = layer.post_attention_layernorm(hidden_states)
                mlp_output = layer.mlp(hidden_states_norm)
                hidden_states = hidden_states + mlp_output
            
            # Apply final layer norm (like HuggingFace does)
            hidden_states = self.model.model.norm(hidden_states)
            
            # Get logits and sample next token
            logits = self.model.lm_head(hidden_states)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Debug: print first few tokens
            if len(seq_info["generated_tokens"]) < 5:
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                print(f"  Debug token {len(seq_info['generated_tokens'])+1}: id={next_token_id}, text='{token_text}', logit_max={next_token_logits.max().item():.2f}")
            
            # Update sequence info
            seq_info["generated_tokens"].append(next_token_id)
            seq_info["total_tokens"] += 1
        
        return next_token_id
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text using traditional KV cache.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        # Prefill
        seq_id = self.prefill(prompt)
        
        # Decode
        print(f"\n[Decode] Sequence {seq_id}: Generating up to {max_new_tokens} tokens")
        generated_tokens = []
        
        for step in range(max_new_tokens):
            token_id = self.decode_step(seq_id)
            if token_id is None:
                break
            
            generated_tokens.append(token_id)
            
            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break
            
            if (step + 1) % 10 == 0:
                cached_len = self.kv_caches[seq_id][0]['k'].shape[1]
                print(f"  Step {step + 1}: {cached_len} tokens in cache")
        
        # Decode tokens to text
        full_tokens = self.sequences[seq_id]["prompt_tokens"] + generated_tokens
        generated_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
        
        # Clean up
        del self.kv_caches[seq_id]
        del self.sequences[seq_id]
        
        return generated_text
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics."""
        total_kv_elements = 0
        total_sequences = len(self.kv_caches)
        
        for seq_id, layer_caches in self.kv_caches.items():
            for layer_idx, cache in layer_caches.items():
                k_shape = cache['k'].shape
                v_shape = cache['v'].shape
                # Each element is float16 (2 bytes)
                total_kv_elements += k_shape[0] * k_shape[1] * k_shape[2]  # K cache
                total_kv_elements += v_shape[0] * v_shape[1] * v_shape[2]  # V cache
        
        total_memory_mb = (total_kv_elements * 2) / (1024 * 1024)  # 2 bytes per float16
        
        return {
            "total_sequences": total_sequences,
            "total_kv_elements": total_kv_elements,
            "total_memory_mb": total_memory_mb
        }


def main():
    """Main function to run baseline inference."""
    print("=" * 60)
    print("Baseline Inference (Traditional KV Cache)")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize model wrapper
    model_wrapper = BaselineModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device
    )
    
    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i + 1}: {prompt}")
        print(f"{'=' * 60}")
        
        start_time = time.time()
        generated = model_wrapper.generate(prompt, max_new_tokens=50)
        elapsed_time = time.time() - start_time
        
        print(f"\nGenerated text:")
        print(generated)
        print(f"\nTime taken: {elapsed_time:.2f} seconds")
        print()
    
    # Print final stats
    print(f"\n{'=' * 60}")
    print("Final Memory Stats:")
    print(f"{'=' * 60}")
    stats = model_wrapper.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
