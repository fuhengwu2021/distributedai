"""
Block Manager for PagedAttention.

Manages fixed-size blocks for KV cache storage, similar to OS page management.
"""

import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Block:
    """A fixed-size block storing KV cache for a fixed number of tokens."""
    block_id: int
    k_cache: torch.Tensor  # Shape: [block_size, num_heads, head_dim]
    v_cache: torch.Tensor  # Shape: [block_size, num_heads, head_dim]
    block_size: int
    num_tokens: int = 0  # Number of valid tokens in this block (0 <= num_tokens <= block_size)
    
    def is_full(self) -> bool:
        """Check if the block is full."""
        return self.num_tokens >= self.block_size
    
    def has_space(self) -> bool:
        """Check if the block has space for more tokens."""
        return self.num_tokens < self.block_size


class BlockTable:
    """Block table for a sequence, mapping logical token positions to physical blocks."""
    
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.blocks: List[Block] = []  # Ordered list of blocks for this sequence
    
    def append_block(self, block: Block):
        """Append a block to this sequence's block table."""
        self.blocks.append(block)
    
    def get_num_blocks(self) -> int:
        """Get the number of blocks allocated to this sequence."""
        return len(self.blocks)
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens stored across all blocks."""
        return sum(block.num_tokens for block in self.blocks)
    
    def get_last_block(self) -> Optional[Block]:
        """Get the last block, which might have space for more tokens."""
        return self.blocks[-1] if self.blocks else None


class BlockManager:
    """
    Manages allocation and deallocation of KV cache blocks.
    
    Similar to OS memory management, this allocator:
    - Maintains a pool of fixed-size blocks
    - Allocates blocks to sequences as needed
    - Frees blocks when sequences complete
    - Reuses freed blocks to reduce fragmentation
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
        Initialize the block manager.
        
        Args:
            block_size: Number of tokens per block (e.g., 16)
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_blocks: Maximum number of blocks to pre-allocate
            device: Device to store blocks on ('cuda' or 'cpu')
        """
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_blocks = max_blocks
        self.device = device
        
        # Pre-allocate a pool of blocks
        self.free_blocks: List[Block] = []
        self.allocated_blocks: dict[int, Block] = {}  # block_id -> Block
        self.next_block_id = 0
        
        # Sequence management
        self.sequence_tables: dict[int, BlockTable] = {}  # seq_id -> BlockTable
        
        # Pre-allocate initial blocks
        self._preallocate_blocks(max_blocks)
    
    def _preallocate_blocks(self, num_blocks: int):
        """Pre-allocate a pool of blocks."""
        for _ in range(num_blocks):
            block = self._create_block()
            self.free_blocks.append(block)
    
    def _create_block(self) -> Block:
        """Create a new block with allocated memory."""
        block_id = self.next_block_id
        self.next_block_id += 1
        
        # Allocate memory for K and V caches
        k_cache = torch.zeros(
            (self.block_size, self.num_heads, self.head_dim),
            dtype=(torch.float16 if str(self.device).startswith('cuda') else torch.float32),
            device=self.device
        )
        v_cache = torch.zeros(
            (self.block_size, self.num_heads, self.head_dim),
            dtype=(torch.float16 if str(self.device).startswith('cuda') else torch.float32),
            device=self.device
        )
        
        block = Block(
            block_id=block_id,
            k_cache=k_cache,
            v_cache=v_cache,
            block_size=self.block_size,
            num_tokens=0
        )
        
        self.allocated_blocks[block_id] = block
        return block
    
    def allocate_block(self, seq_id: int) -> Block:
        """
        Allocate a block to a sequence.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            Allocated block
        """
        # Try to reuse a free block first
        if self.free_blocks:
            block = self.free_blocks.pop()
            block.num_tokens = 0  # Reset
        else:
            # Create a new block if pool is exhausted
            block = self._create_block()
        
        # Add to sequence's block table
        if seq_id not in self.sequence_tables:
            self.sequence_tables[seq_id] = BlockTable(seq_id)
        
        self.sequence_tables[seq_id].append_block(block)
        return block
    
    def get_block_table(self, seq_id: int) -> Optional[BlockTable]:
        """Get the block table for a sequence."""
        return self.sequence_tables.get(seq_id)
    
    def append_kv(
        self,
        seq_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        token_idx: int
    ) -> Tuple[Block, int]:
        """
        Append KV cache for a token to a sequence.
        
        Args:
            seq_id: Sequence ID
            k: Key tensor of shape [num_heads, head_dim]
            v: Value tensor of shape [num_heads, head_dim]
            token_idx: Logical token index in the sequence
            
        Returns:
            Tuple of (block, position_in_block) where the KV was stored
        """
        if seq_id not in self.sequence_tables:
            self.sequence_tables[seq_id] = BlockTable(seq_id)
        
        block_table = self.sequence_tables[seq_id]

        # Enforce sequential appends for this minimal demo.
        # In a full PA implementation, token_idx is used to map logical positions
        # to physical blocks (supporting out-of-order writes).
        expected_idx = block_table.get_total_tokens()
        if token_idx != expected_idx:
            raise ValueError(
                f"Non-sequential token_idx for seq_id={seq_id}: got {token_idx}, expected {expected_idx}."
            )
        
        # Check if last block has space
        last_block = block_table.get_last_block()
        if last_block and last_block.has_space():
            block = last_block
            pos_in_block = block.num_tokens
        else:
            # Allocate a new block
            block = self.allocate_block(seq_id)
            pos_in_block = 0
        
        # Store K and V in the block
        block.k_cache[pos_in_block] = k
        block.v_cache[pos_in_block] = v
        block.num_tokens += 1
        
        return block, pos_in_block
    
    def free_sequence(self, seq_id: int):
        """
        Free all blocks allocated to a sequence.
        
        Args:
            seq_id: Sequence ID to free
        """
        if seq_id not in self.sequence_tables:
            return
        
        block_table = self.sequence_tables[seq_id]
        
        # Return all blocks to the free pool
        for block in block_table.blocks:
            block.num_tokens = 0  # Reset
            self.free_blocks.append(block)
        
        # Remove sequence table
        del self.sequence_tables[seq_id]
    
    def get_stats(self) -> dict:
        """Get statistics about block usage."""
        total_blocks = len(self.allocated_blocks)
        free_blocks = len(self.free_blocks)
        allocated_blocks = total_blocks - free_blocks
        
        total_sequences = len(self.sequence_tables)
        total_tokens = sum(
            table.get_total_tokens()
            for table in self.sequence_tables.values()
        )
        
        return {
            "total_blocks": total_blocks,
            "free_blocks": free_blocks,
            "allocated_blocks": allocated_blocks,
            "total_sequences": total_sequences,
            "total_tokens": total_tokens,
            "block_size": self.block_size
        }
