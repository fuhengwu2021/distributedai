#!/usr/bin/env python3
"""Microbenchmark helper to find hot operators.

This script simulates operator timings and prints the top N hot operators
for a fusion prioritization exercise. Replace with real operator timing
data collected from `torch.profiler` or kernel-level traces.
"""
import random


def simulate_operator_timings(n_ops=20):
    ops = {}
    for i in range(n_ops):
        name = f"op_{i:02d}"
        # simulate latency in microseconds
        ops[name] = random.uniform(1.0, 500.0)
    return ops


def top_n(ops, n=5):
    return sorted(ops.items(), key=lambda kv: kv[1], reverse=True)[:n]


def main():
    ops = simulate_operator_timings(50)
    print("Simulated operator latencies (us). Top hot ops:")
    for name, lat in top_n(ops, 8):
        print(f"  {name}: {lat:.2f} us")
    print("Pick the top ops for fusion and re-run measurements.")


if __name__ == '__main__':
    main()
