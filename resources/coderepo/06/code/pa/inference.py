""" 
Inference script using custom PagedAttention for KV cache management.

This script demonstrates how to use the custom PagedAttention implementation
for KV cache management and (minimally) integrate it into the attention compute
path during decode for Qwen/Qwen2.5-0.5B-Instruct.

Design (intentionally minimal):
- Prefill: use HuggingFace forward(use_cache=True) and write the returned KV
  (already RoPE'd) into PagedAttention blocks.
- Decode: run a minimal manual per-layer forward for ONE token, calling
  PagedAttention.compute_attention(...) so PA participates in the compute path.
"""

import os
import sys
import time
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

# Add parent directory to path to import pa module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pa import PagedAttention


class PagedAttentionModelWrapper:
    """
    Wrapper around a HuggingFace model to use custom PagedAttention for KV cache.
    
    This is a simplified implementation that demonstrates the concept.
    In production systems like vLLM, this is integrated at the kernel level.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        block_size: int = 16,
        device: str = "cuda",
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: HuggingFace model name
            block_size: Block size for PagedAttention
            device: Device to use
        """
        self.device = device
        self.block_size = block_size
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        
        # Get model config
        config = self.model.config
        self.num_heads = int(config.num_attention_heads)
        # Qwen2.5 uses GQA, so K/V might have fewer heads
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.num_layers = int(config.num_hidden_layers)
        
        print(f"Model config: {self.num_heads} Q heads, {self.num_kv_heads} KV heads, {self.head_dim} head_dim, {self.num_layers} layers")
        
        # Initialize PagedAttention for each layer
        self.paged_attentions = [
            PagedAttention(
                block_size=block_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                device=device
            )
            for _ in range(self.num_layers)
        ]
        
        # Track sequences
        self.sequences: dict[int, dict] = {}
        self.next_seq_id = 0
    
    def _apply_rope(
        self,
        attn_module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE with fallback mechanisms for different transformers versions.
        
        Args:
            attn_module: Attention module (may have rotary_emb)
            query_states: Query tensor of shape [B, Hq, q_len, D]
            key_states: Key tensor of shape [B, Hkv, q_len, D]
            position_ids: Position IDs of shape [B, q_len]
            kv_seq_len: Total KV sequence length
            
        Returns:
            Tuple of (query_states_rope, key_states_rope) with RoPE applied
        """
        rotary_emb = getattr(attn_module, "rotary_emb", None)
        if rotary_emb is None:
            rotary_emb = getattr(self.model.model, "rotary_emb", None)
        if rotary_emb is None:
            raise RuntimeError("Could not find rotary_emb on attention module or model.")

        cos_sin = None
        for call in (
            lambda: rotary_emb(key_states, position_ids),
            lambda: rotary_emb(key_states, seq_len=kv_seq_len),
            lambda: rotary_emb(key_states, kv_seq_len),
        ):
            try:
                cos_sin = call()
                break
            except TypeError:
                continue
        if cos_sin is None:
            raise RuntimeError("Failed to call rotary_emb with supported signatures.")

        cos, sin = cos_sin
        try:
            q, k = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        except TypeError:
            q, k = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return q, k
    
    def prefill(self, prompt: str, seq_id: Optional[int] = None) -> int:
        """
        Process the prompt and cache KV using PagedAttention.
        
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
            "total_tokens": len(prompt_tokens),  # number of tokens already cached
            "next_token_id": None,               # one-token lookahead buffer
        }
        
        print(f"\n[Prefill] Sequence {seq_id}: Processing {len(prompt_tokens)} prompt tokens")
        
        # Process prompt tokens using model's forward (includes RoPE automatically)
        with torch.no_grad():
            # Use model's forward with use_cache to get KV cache with RoPE applied
            outputs = self.model(input_ids=tokens, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Extract KV cache from past_key_values and store in PagedAttention blocks
            for layer_idx in range(self.num_layers):
                k, v = past_key_values[layer_idx]
                # k, v shape: [batch, num_kv_heads, seq_len, head_dim]
                k_cache = k[0]  # [num_kv_heads, seq_len, head_dim]
                v_cache = v[0]  # [num_kv_heads, seq_len, head_dim]
                
                # Store each token's K/V in PagedAttention blocks
                for token_idx in range(k_cache.shape[1]):  # seq_len
                    k_token = k_cache[:, token_idx, :]  # [num_kv_heads, head_dim]
                    v_token = v_cache[:, token_idx, :]  # [num_kv_heads, head_dim]
                    
                    # Handle GQA: repeat KV heads to match Q heads
                    if self.num_kv_heads < self.num_heads:
                        repeat_factor = self.num_heads // self.num_kv_heads
                        k_token = k_token.repeat_interleave(repeat_factor, dim=0)  # [num_heads, head_dim]
                        v_token = v_token.repeat_interleave(repeat_factor, dim=0)  # [num_heads, head_dim]
                    
                    self.paged_attentions[layer_idx].append_kv(
                        seq_id, k_token, v_token, token_idx
                    )
        
        # Get first token from prefill logits
        logits_check = outputs.logits[:, -1, :]
        first_token_id = int(torch.argmax(logits_check, dim=-1).item())
        self.sequences[seq_id]["next_token_id"] = first_token_id
        first_token_text = self.tokenizer.decode([first_token_id])
        print(f"[Prefill] Sequence {seq_id}: Cached KV for {len(prompt_tokens)} tokens")
        stats = self.paged_attentions[0].get_stats()
        print(f"[Prefill] Block stats: {stats}")
        print(f"[Prefill] First token would be: id={first_token_id}, text='{first_token_text}'")
        
        return seq_id
    
    def decode_step(self, seq_id: int) -> Optional[int]:
        """Generate one token, using PagedAttention for attention computation."""
        if seq_id not in self.sequences:
            return None

        seq_info = self.sequences[seq_id]

        # one-token lookahead:
        # - emit seq_info['next_token_id'] now (and cache it)
        # - compute next token id and store back to seq_info['next_token_id']
        token_to_emit = seq_info.get("next_token_id")
        if token_to_emit is None:
            token_to_emit = seq_info["prompt_tokens"][-1]

        token_tensor = torch.tensor([[int(token_to_emit)]], device=self.device)
        position = int(seq_info["total_tokens"])
        position_ids = torch.tensor([[position]], device=self.device, dtype=torch.long)
        kv_seq_len = position + 1

        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(token_tensor)  # [1,1,H]

            for layer_idx in range(self.num_layers):
                layer = self.model.model.layers[layer_idx]
                attn = layer.self_attn

                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                q = attn.q_proj(hidden_states)
                k = attn.k_proj(hidden_states)
                v = attn.v_proj(hidden_states)

                q = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)      # [1,Hq,1,D]
                k = k.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [1,Hkv,1,D]
                v = v.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [1,Hkv,1,D]

                q, k = self._apply_rope(attn, q, k, position_ids, kv_seq_len)

                q_tok = q[0, :, 0, :]   # [Hq,D]
                k_tok = k[0, :, 0, :]   # [Hkv,D]
                v_tok = v[0, :, 0, :]   # [Hkv,D]

                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_tok = k_tok.repeat_interleave(repeat_factor, dim=0)  # [Hq,D]
                    v_tok = v_tok.repeat_interleave(repeat_factor, dim=0)

                # IMPORTANT: append KV first so attention includes self (causal allows self)
                self.paged_attentions[layer_idx].append_kv(seq_id, k_tok, v_tok, position)

                # Attention via PagedAttention (no padding; iterates only allocated blocks)
                ctx = self.paged_attentions[layer_idx].compute_attention(seq_id, q_tok)

                ctx = ctx.reshape(1, 1, self.num_heads * self.head_dim)
                attn_out = attn.o_proj(ctx)
                hidden_states = residual + attn_out

                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                mlp_out = layer.mlp(hidden_states)
                hidden_states = residual + mlp_out

            hidden_states = self.model.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)  # [1,1,vocab]
            next_token_id = int(torch.argmax(logits[0, -1, :]).item())

        seq_info["generated_tokens"].append(int(token_to_emit))
        seq_info["total_tokens"] += 1
        seq_info["next_token_id"] = next_token_id

        return int(token_to_emit)
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text using PagedAttention.
        
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
                stats = self.paged_attentions[0].get_stats()
                print(f"  Step {step + 1}: {stats['total_tokens']} total tokens, "
                      f"{stats['allocated_blocks']} blocks allocated")
        
        # Decode tokens to text
        full_tokens = self.sequences[seq_id]["prompt_tokens"] + generated_tokens
        generated_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
        
        # Clean up
        for layer_idx in range(self.num_layers):
            self.paged_attentions[layer_idx].free_sequence(seq_id)
        del self.sequences[seq_id]
        
        return generated_text


def main():
    """Main function to run inference with PagedAttention."""
    print("=" * 60)
    print("PagedAttention Inference Demo")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize model wrapper
    model_wrapper = PagedAttentionModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        block_size=16,
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
    print("Final Block Manager Stats:")
    print(f"{'=' * 60}")
    stats = model_wrapper.paged_attentions[0].get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
