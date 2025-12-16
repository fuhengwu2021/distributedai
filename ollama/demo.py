"""
Simple demo script for Ollama model loading simulation.
"""

import torch
from ollama_inference import OllamaInferenceEngine


def main():
    """Run a simple demo."""
    print("Ollama Model Loading Simulation Demo")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Initialize engine
    engine = OllamaInferenceEngine(
        model_name="Qwen3:4b-instruct-2507-fp16",
        block_size=16,
        device=device,
        use_online_softmax=True,
        max_batch_size=32
    )
    
    # Single request
    print("\n" + "=" * 60)
    print("Single Request Demo")
    print("=" * 60)
    
    prompt = "What is the capital of France?"
    seq_id = engine.add_request(prompt, max_new_tokens=30)
    print(f"Added request: '{prompt}' (seq_id={seq_id})")
    
    # Process
    step = 0
    while step < 50:
        seq_ids, next_token_ids = engine.step()
        if not seq_ids:
            break
        
        step += 1
        if step % 5 == 0:
            stats = engine.get_stats()
            print(f"  Step {step}: {stats['scheduler']['active']} active sequences")
        
        # Check if finished
        if seq_id in engine.scheduler.get_finished_sequences():
            break
    
    # Get result
    result = engine.get_sequence_text(seq_id)
    print(f"\nResult:")
    print(f"  {result}")
    
    # Print stats
    stats = engine.get_stats()
    print(f"\nFinal Stats:")
    print(f"  Blocks used: {stats['blocks'].get('allocated_blocks', 0)}")
    print(f"  Total tokens: {stats['blocks'].get('total_tokens', 0)}")


if __name__ == "__main__":
    main()
