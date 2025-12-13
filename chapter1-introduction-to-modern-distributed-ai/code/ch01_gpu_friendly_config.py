# GPU-friendly configuration example
config = {
    'batch_size': 32,  # Fits in GPU memory
    'sequence_length': 2048,  # Reasonable for most GPUs
    'precision': 'bf16',  # Better than FP32, more stable than FP16
    'gradient_checkpointing': True,  # Save memory
    'gradient_accumulation_steps': 4  # Effective batch size = 128
}

def print_config():
    for k, v in config.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    print_config()
