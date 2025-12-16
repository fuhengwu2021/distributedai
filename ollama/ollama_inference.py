"""
Ollama Inference with PagedAttention - Simulates Ollama's inference process.

This module demonstrates how Ollama would use PagedAttention for efficient
inference with the Qwen3:4b-instruct-2507-fp16 model.
"""

import os
import sys
import torch
from typing import List, Optional, Tuple

# Add PagedAttention to path
pa_code_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "coderepo", "06", "code"
)
sys.path.insert(0, pa_code_path)

# Import from pa package
from pa import PagedAttentionV2, BlockManager
from pa.scheduler import ContinuousBatchScheduler, SequenceState

# Import model loader
from model_loader import OllamaModelSimulator
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class OllamaInferenceEngine:
    """
    Ollama-style inference engine using PagedAttention.
    
    This simulates how Ollama would:
    1. Load models from blob storage
    2. Use PagedAttention for efficient KV cache management
    3. Process multiple requests with continuous batching
    """
    
    def __init__(
        self,
        model_name: str = "Qwen3:4b-instruct-2507-fp16",
        block_size: int = 16,
        device: str = "cuda",
        use_online_softmax: bool = True,
        max_batch_size: int = 32
    ):
        """
        Initialize the Ollama inference engine.
        
        Args:
            model_name: Model name (e.g., "Qwen3:4b-instruct-2507-fp16")
            block_size: Block size for PagedAttention
            device: Device to use
            use_online_softmax: Use online softmax for attention
            max_batch_size: Maximum batch size
        """
        self.device = device
        self.block_size = block_size
        self.use_online_softmax = use_online_softmax
        self.max_batch_size = max_batch_size
        
        print("=" * 70)
        print("Ollama Inference Engine with PagedAttention")
        print("=" * 70)
        
        # Step 1: Load model from Ollama blob storage (simulated)
        print(f"\n[Step 1] Loading model: {model_name}")
        self.simulator = OllamaModelSimulator(
            model_name=model_name,
            device=device,
            use_hf_fallback=True  # Use HF for actual weights
        )
        
        self.model = self.simulator.get_model()
        self.tokenizer = self.simulator.get_tokenizer()
        self.config = self.simulator.get_config()
        
        # Step 2: Initialize PagedAttention for each layer
        print(f"\n[Step 2] Initializing PagedAttention (block_size={block_size})")
        self.num_heads = int(self.model.config.num_attention_heads)
        self.num_kv_heads = int(getattr(self.model.config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(self.model.config.hidden_size // self.model.config.num_attention_heads)
        self.num_layers = int(self.model.config.num_hidden_layers)
        
        print(f"  - Num heads (Q): {self.num_heads}")
        print(f"  - Num KV heads: {self.num_kv_heads}")
        print(f"  - Head dim: {self.head_dim}")
        print(f"  - Num layers: {self.num_layers}")
        
        # Initialize PagedAttention for each layer
        self.paged_attentions = [
            PagedAttentionV2(
                block_size=block_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                device=device,
                use_online_softmax=use_online_softmax,
                num_kv_heads=self.num_kv_heads
            )
            for _ in range(self.num_layers)
        ]
        
        # Step 3: Initialize scheduler
        print(f"\n[Step 3] Initializing continuous batch scheduler")
        self.scheduler = ContinuousBatchScheduler(max_batch_size=max_batch_size)
        
        print("\n[Ready] Ollama inference engine initialized")
        print("=" * 70)
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        kv_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using HuggingFace's implementation."""
        try:
            if hasattr(self.model.model, 'rotary_emb'):
                rotary_emb = self.model.model.rotary_emb
                cos, sin = rotary_emb(k, position_ids)
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                return q_rope, k_rope
        except Exception as e:
            print(f"Warning: RoPE application failed: {e}, using identity")
            return q, k
        
        return q, k
    
    def add_request(self, prompt: str, max_new_tokens: int = 50) -> int:
        """
        Add a new request to the scheduler.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Sequence ID
        """
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
        else:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        prompt_tokens = tokens[0].tolist()
        
        # Add to scheduler
        seq_id = self.scheduler.add_sequence(
            prompt_tokens,
            max_new_tokens,
            immediate_prefill=False
        )
        
        return seq_id
    
    def prefill_batch(self) -> int:
        """
        Process a batch of sequences in prefill phase.
        
        Returns:
            Number of sequences processed
        """
        seq_ids, prompt_token_lists, positions_start, seq_lengths = \
            self.scheduler.get_prefill_batch(max_batch_size=self.max_batch_size)
        
        if not seq_ids:
            return 0
        
        print(f"\n[Prefill] Processing {len(seq_ids)} sequences")
        print(f"  - Total tokens: {sum(seq_lengths)}")
        print(f"  - Sequence lengths: {seq_lengths}")
        
        with torch.no_grad():
            # Process each sequence
            for seq_id, prompt_tokens in zip(seq_ids, prompt_token_lists):
                seq_tokens = torch.tensor([prompt_tokens], device=self.device)
                
                # Forward pass to get KV cache
                outputs = self.model(input_ids=seq_tokens, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Extract and store KV cache in PagedAttention blocks
                for layer_idx in range(self.num_layers):
                    k, v = past_key_values[layer_idx]
                    # k, v shape: [1, num_kv_heads, L, head_dim]
                    k_cache = k[0]  # [num_kv_heads, L, head_dim]
                    v_cache = v[0]  # [num_kv_heads, L, head_dim]
                    
                    # Store each token's K/V
                    for token_pos in range(len(prompt_tokens)):
                        k_tok = k_cache[:, token_pos, :]  # [num_kv_heads, head_dim]
                        v_tok = v_cache[:, token_pos, :]  # [num_kv_heads, head_dim]
                        self.paged_attentions[layer_idx].append_kv(
                            seq_id, k_tok, v_tok, token_pos
                        )
        
        # Update scheduler
        for seq_id in seq_ids:
            seq_info = self.scheduler.sequences[seq_id]
            seq_info.state = SequenceState.DECODE
            seq_info.position = len(seq_info.prompt_tokens)
        
        return len(seq_ids)
    
    def decode_batch(self) -> Tuple[List[int], List[int]]:
        """
        Process a batch of sequences in decode phase using PagedAttention.
        
        Returns:
            Tuple of (seq_ids, next_token_ids)
        """
        seq_ids, positions, token_ids = self.scheduler.get_batch(
            include_prefill=False,
            include_decode=True
        )
        
        if not seq_ids:
            return [], []
        
        num_seqs = len(seq_ids)
        
        with torch.no_grad():
            # Batch embedding
            token_tensor = torch.tensor([token_ids], device=self.device)  # [1, num_seqs]
            hidden_states = self.model.model.embed_tokens(token_tensor)  # [1, num_seqs, H]
            
            # Process through each layer
            for layer_idx in range(self.num_layers):
                layer = self.model.model.layers[layer_idx]
                attn = layer.self_attn
                
                # Residual connection
                residual = hidden_states.clone()
                
                # Layer norm
                hidden_states = layer.input_layernorm(hidden_states)
                
                # Q, K, V projection
                q = attn.q_proj(hidden_states)  # [1, num_seqs, Hq*D]
                k = attn.k_proj(hidden_states)  # [1, num_seqs, Hkv*D]
                v = attn.v_proj(hidden_states)  # [1, num_seqs, Hkv*D]
                
                # Reshape for attention
                q = q.view(1, num_seqs, self.num_heads, self.head_dim).transpose(1, 2)  # [1, Hq, num_seqs, D]
                k = k.view(1, num_seqs, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, Hkv, num_seqs, D]
                v = v.view(1, num_seqs, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, Hkv, num_seqs, D]
                
                # Apply RoPE per sequence
                for i, seq_id in enumerate(seq_ids):
                    seq_position = torch.tensor([[positions[i]]], device=self.device, dtype=torch.long)
                    q_seq = q[:, :, i:i+1, :]  # [1, Hq, 1, D]
                    k_seq = k[:, :, i:i+1, :]  # [1, Hkv, 1, D]
                    q_seq, k_seq = self._apply_rope(q_seq, k_seq, seq_position, positions[i] + 1)
                    q[:, :, i:i+1, :] = q_seq
                    k[:, :, i:i+1, :] = k_seq
                
                # Append KV cache
                k_batch = k[0].transpose(0, 1)  # [num_seqs, Hkv, D]
                v_batch = v[0].transpose(0, 1)  # [num_seqs, Hkv, D]
                
                for i, seq_id in enumerate(seq_ids):
                    k_tok = k_batch[i]  # [Hkv, D]
                    v_tok = v_batch[i]  # [Hkv, D]
                    self.paged_attentions[layer_idx].append_kv(
                        seq_id, k_tok, v_tok, positions[i]
                    )
                
                # Compute attention using PagedAttention
                q_batch = q[0].transpose(0, 1)  # [num_seqs, Hq, D]
                attn_outputs = []
                
                for i, seq_id in enumerate(seq_ids):
                    q_tok = q_batch[i]  # [Hq, D]
                    attn_output = self.paged_attentions[layer_idx].compute_attention(seq_id, q_tok)
                    attn_outputs.append(attn_output)
                
                # Stack and reshape
                attn_output_tensor = torch.stack(attn_outputs, dim=0)  # [num_seqs, Hq, D]
                attn_output_tensor = attn_output_tensor.view(
                    num_seqs, self.num_heads * self.head_dim
                ).unsqueeze(0)  # [1, num_seqs, Hq*D]
                
                # Output projection
                attn_output = attn.o_proj(attn_output_tensor)  # [1, num_seqs, H]
                hidden_states = residual + attn_output
                
                # MLP
                hidden_states_norm = layer.post_attention_layernorm(hidden_states)
                mlp_output = layer.mlp(hidden_states_norm)
                hidden_states = hidden_states + mlp_output
            
            # Final layer norm and LM head
            hidden_states = self.model.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)  # [1, num_seqs, vocab_size]
            
            # Sample next tokens
            next_token_logits = logits[0, :, :]  # [num_seqs, vocab_size]
            next_token_ids = [
                int(torch.argmax(next_token_logits[i]).item())
                for i in range(num_seqs)
            ]
            
            # Update scheduler
            self.scheduler.update_sequences(
                seq_ids,
                next_token_ids,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            return seq_ids, next_token_ids
    
    def step(self) -> Tuple[List[int], List[int]]:
        """Process one step: prefill and decode."""
        # Process prefill
        prefill_count = self.prefill_batch()
        if prefill_count > 0:
            print(f"  [Step] Processed {prefill_count} sequences in prefill")
        
        # Process decode
        return self.decode_batch()
    
    def get_sequence_text(self, seq_id: int) -> Optional[str]:
        """Get generated text for a sequence."""
        if seq_id not in self.scheduler.sequences:
            return None
        
        seq_info = self.scheduler.sequences[seq_id]
        full_tokens = seq_info.prompt_tokens + seq_info.generated_tokens
        return self.tokenizer.decode(full_tokens, skip_special_tokens=True)
    
    def cleanup_finished(self):
        """Clean up finished sequences."""
        finished_ids = self.scheduler.get_finished_sequences()
        for seq_id in finished_ids:
            for layer_idx in range(self.num_layers):
                self.paged_attentions[layer_idx].block_manager.free_sequence(seq_id)
            self.scheduler.remove_sequence(seq_id)
        return len(finished_ids)
    
    def get_stats(self) -> dict:
        """Get inference statistics."""
        scheduler_stats = self.scheduler.get_stats()
        block_stats = self.paged_attentions[0].block_manager.get_stats() if self.paged_attentions else {}
        
        return {
            "scheduler": scheduler_stats,
            "blocks": block_stats
        }


def main():
    """Main function to demonstrate Ollama-style inference."""
    print("\n" + "=" * 70)
    print("Ollama Model Loading and Inference Simulation")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize Ollama inference engine
    engine = OllamaInferenceEngine(
        model_name="Qwen3:4b-instruct-2507-fp16",
        block_size=16,
        device=device,
        use_online_softmax=True,
        max_batch_size=32
    )
    
    # Add requests
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about AI.",
    ]
    
    print(f"\n{'=' * 70}")
    print("Adding Requests")
    print(f"{'=' * 70}")
    seq_ids = []
    seq_results = {}
    
    for i, prompt in enumerate(prompts):
        seq_id = engine.add_request(prompt, max_new_tokens=50)
        seq_ids.append(seq_id)
        print(f"  Request {i+1}: seq_id={seq_id}, prompt='{prompt[:50]}...'")
    
    # Process requests
    print(f"\n{'=' * 70}")
    print("Processing Requests with PagedAttention")
    print(f"{'=' * 70}")
    step = 0
    max_steps = 200
    
    while step < max_steps:
        processed_seq_ids, next_token_ids = engine.step()
        
        if not processed_seq_ids and engine.scheduler.get_prefill_batch()[0] == []:
            break
        
        step += 1
        
        # Store results for finished sequences
        finished_ids = engine.scheduler.get_finished_sequences()
        for seq_id in finished_ids:
            if seq_id not in seq_results:
                text = engine.get_sequence_text(seq_id)
                seq_results[seq_id] = text
        
        # Print progress
        if step % 10 == 0:
            stats = engine.get_stats()
            scheduler_stats = stats["scheduler"]
            print(f"  Step {step}: {scheduler_stats['active']} active sequences "
                  f"(prefill: {scheduler_stats['prefill']}, "
                  f"decode: {scheduler_stats['decode']}, "
                  f"finished: {scheduler_stats['finished']})")
        
        # Cleanup
        finished_count = engine.cleanup_finished()
        if finished_count > 0:
            print(f"  Step {step}: {finished_count} sequence(s) finished")
        
        if engine.scheduler.get_active_count() == 0:
            break
    
    # Print results
    print(f"\n{'=' * 70}")
    print("Generation Results")
    print(f"{'=' * 70}")
    for i, seq_id in enumerate(seq_ids):
        text = seq_results.get(seq_id, "N/A")
        if text and text != "N/A":
            print(f"\nRequest {i+1} (seq_id={seq_id}):")
            print(f"  {text}")
    
    # Print final stats
    print(f"\n{'=' * 70}")
    print("Final Statistics")
    print(f"{'=' * 70}")
    stats = engine.get_stats()
    print("\nScheduler Stats:")
    for key, value in stats["scheduler"].items():
        print(f"  {key}: {value}")
    
    print("\nBlock Manager Stats:")
    for key, value in stats["blocks"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
