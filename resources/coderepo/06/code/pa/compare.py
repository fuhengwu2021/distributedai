"""
Comparison script between Baseline and PagedAttention inference.

This script runs both implementations and compares their performance.
"""

import torch
import time
from inference_baseline import BaselineModelWrapper
from inference import PagedAttentionModelWrapper


def compare_inference(prompt: str, max_new_tokens: int = 50):
    """
    Compare baseline and PagedAttention inference.
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print(f"Comparing Inference Methods")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Max new tokens: {max_new_tokens}")
    print()
    
    # Test Baseline
    print("-" * 80)
    print("BASELINE (Traditional KV Cache)")
    print("-" * 80)
    baseline_model = BaselineModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device
    )
    
    start_time = time.time()
    baseline_output = baseline_model.generate(prompt, max_new_tokens=max_new_tokens)
    baseline_time = time.time() - start_time
    
    baseline_stats = baseline_model.get_memory_stats()
    
    print(f"\nTime: {baseline_time:.2f} seconds")
    print(f"Memory: {baseline_stats['total_memory_mb']:.2f} MB")
    print(f"Generated text (first 200 chars): {baseline_output[:200]}...")
    
    # Clean up
    del baseline_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Test PagedAttention
    print("\n" + "-" * 80)
    print("PAGEDATTENTION (Block-based KV Cache)")
    print("-" * 80)
    pa_model = PagedAttentionModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        block_size=16,
        device=device
    )
    
    start_time = time.time()
    pa_output = pa_model.generate(prompt, max_new_tokens=max_new_tokens)
    pa_time = time.time() - start_time
    
    pa_stats = pa_model.paged_attentions[0].get_stats()
    
    print(f"\nTime: {pa_time:.2f} seconds")
    print(f"Blocks allocated: {pa_stats['allocated_blocks']}")
    print(f"Total tokens: {pa_stats['total_tokens']}")
    print(f"Generated text (first 200 chars): {pa_output[:200]}...")
    
    # Clean up
    del pa_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Baseline time:     {baseline_time:.2f} seconds")
    print(f"PagedAttention time: {pa_time:.2f} seconds")
    if baseline_time > 0:
        speedup = baseline_time / pa_time
        print(f"Speedup:           {speedup:.2f}x")
    print(f"\nBaseline memory:    {baseline_stats['total_memory_mb']:.2f} MB")
    print(f"PagedAttention:    Block-based (fragmentation eliminated)")
    print()


def main():
    """Main comparison function."""
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    for prompt in prompts:
        compare_inference(prompt, max_new_tokens=30)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
