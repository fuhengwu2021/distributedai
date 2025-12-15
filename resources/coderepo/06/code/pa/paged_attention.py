"""
PagedAttention implementation.

Implements attention computation that iterates over blocks instead of padded sequences,
eliminating padding FLOPs.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from .block_manager import BlockManager, BlockTable, Block


class PagedAttention:
    """
    PagedAttention: Block-based attention computation.
    
    Instead of padding sequences to max length, attention iterates over
    the actual blocks allocated to each sequence, eliminating padding FLOPs.
    """
    
    def __init__(
        self,
        block_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        max_blocks: int = 1000,
        device: str = "cuda"
    ):
        """
        Initialize PagedAttention.
        
        Args:
            block_size: Number of tokens per block
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_blocks: Maximum number of blocks to pre-allocate
            device: Device to use ('cuda' or 'cpu')
        """
        self.block_manager = BlockManager(
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_blocks=max_blocks,
            device=device
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
    
    def append_kv(self, seq_id: int, k: torch.Tensor, v: torch.Tensor, token_idx: int):
        """
        Append KV cache for a token.
        
        Args:
            seq_id: Sequence ID
            k: Key tensor of shape [num_heads, head_dim]
            v: Value tensor of shape [num_heads, head_dim]
            token_idx: Logical token index in the sequence
        """
        return self.block_manager.append_kv(seq_id, k, v, token_idx)
    
    def compute_attention(
        self,
        seq_id: int,
        q: torch.Tensor,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention for a query token, iterating only over blocks
        that actually exist for this sequence (no padding).
        
        Args:
            seq_id: Sequence ID
            q: Query tensor of shape [num_heads, head_dim]
            scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            
        Returns:
            Attention output of shape [num_heads, head_dim]
        """
        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)
        
        block_table = self.block_manager.get_block_table(seq_id)
        if block_table is None or len(block_table.blocks) == 0:
            # No cached KV, return zeros
            return torch.zeros_like(q)
        
        # Collect all K and V from blocks
        k_list = []
        v_list = []
        
        for block in block_table.blocks:
            # Only use valid tokens in the block
            num_valid = block.num_tokens
            if num_valid > 0:
                k_list.append(block.k_cache[:num_valid])  # [num_valid, num_heads, head_dim]
                v_list.append(block.v_cache[:num_valid])
        
        if not k_list:
            return torch.zeros_like(q)
        
        # Concatenate all K and V
        # Shape: [total_tokens, num_heads, head_dim]
        k_cached = torch.cat(k_list, dim=0)
        v_cached = torch.cat(v_list, dim=0)
        
        # Compute attention scores
        # q: [num_heads, head_dim]
        # k_cached: [total_tokens, num_heads, head_dim]
        # We need: Q @ K^T -> [num_heads, total_tokens]
        
        # Reshape for matrix multiplication
        # q: [num_heads, head_dim] -> [num_heads, 1, head_dim]
        # k_cached: [total_tokens, num_heads, head_dim] -> [num_heads, total_tokens, head_dim]
        q_expanded = q.unsqueeze(1)  # [num_heads, 1, head_dim]
        k_cached_transposed = k_cached.transpose(0, 1)  # [num_heads, total_tokens, head_dim]
        
        # Compute scores: [num_heads, 1, head_dim] @ [num_heads, total_tokens, head_dim]^T
        # = [num_heads, 1, total_tokens]
        scores = torch.matmul(q_expanded, k_cached_transposed.transpose(-2, -1)) * scale
        scores = scores.squeeze(1)  # [num_heads, total_tokens]
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [num_heads, total_tokens]
        
        # Compute weighted sum of values
        # attn_weights: [num_heads, total_tokens]
        # v_cached: [total_tokens, num_heads, head_dim] -> [num_heads, total_tokens, head_dim]
        v_cached_transposed = v_cached.transpose(0, 1)  # [num_heads, total_tokens, head_dim]
        # output: [num_heads, 1, total_tokens] @ [num_heads, total_tokens, head_dim] = [num_heads, 1, head_dim]
        output = torch.matmul(attn_weights.unsqueeze(1), v_cached_transposed)
        output = output.squeeze(1)  # [num_heads, head_dim]
        
        return output
    
    def compute_attention_batch(
        self,
        seq_ids: List[int],
        q_batch: torch.Tensor,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention for a batch of queries.
        
        This is where PagedAttention shines: each sequence only iterates
        over its own blocks, not padded to max length.
        
        Args:
            seq_ids: List of sequence IDs
            q_batch: Query tensor of shape [batch_size, num_heads, head_dim]
            scale: Scaling factor for attention scores
            
        Returns:
            Attention outputs of shape [batch_size, num_heads, head_dim]
        """
        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)
        
        batch_size = len(seq_ids)
        outputs = []
        
        for i in range(batch_size):
            output = self.compute_attention(seq_ids[i], q_batch[i], scale)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)  # [batch_size, num_heads, head_dim]
    
    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence."""
        self.block_manager.free_sequence(seq_id)
    
    def get_stats(self) -> dict:
        """Get statistics about block usage."""
        return self.block_manager.get_stats()
