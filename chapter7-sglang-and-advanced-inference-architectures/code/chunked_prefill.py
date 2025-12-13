#!/usr/bin/env python3
"""Simple, runnable chunked prefill stub.

This script simulates chunked prefill behavior with a dummy tokenizer and model.
It's not a real model runtime — replace `DummyModel` with your real model object's
`forward_prefill` and manage device placement accordingly.
"""
from typing import List


class DummyTokenizer:
    def __call__(self, text: str) -> List[int]:
        # naive tokenization: split on spaces and return token lengths
        return [len(t) for t in text.split()]


class DummyModel:
    def forward_prefill(self, tokens):
        # pretend to compute and build KV cache
        print(f"Prefilling chunk with {len(tokens)} tokens -> KV entries updated")


def chunked_prefill(model, tokenizer, prompt: str, chunk_size: int = 512):
    tokens = tokenizer(prompt)
    # simulate token ids as integers; chunk on token count
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)} (size={len(chunk)})")
        model.forward_prefill(chunk)
        # In production: optionally offload older KV pages to CPU/NVMe here
    print("Prefill complete — ready to decode with hot KV cache")


def main():
    prompt = "This is a sample prompt repeated " * 200  # make a long prompt
    tokenizer = DummyTokenizer()
    model = DummyModel()
    chunked_prefill(model, tokenizer, prompt, chunk_size=50)


if __name__ == "__main__":
    main()
