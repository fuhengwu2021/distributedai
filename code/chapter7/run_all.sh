#!/usr/bin/env bash
# Run quick checks for Chapter 7 examples: syntax check and dry-run safe scripts
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Running syntax checks for Python scripts in $ROOT_DIR"

py_files=(chunked_prefill.py router_example.py operator_fusion_profiling.py)
for f in "${py_files[@]}"; do
  echo "Checking $f"
  python3 -m py_compile "$ROOT_DIR/$f"
done

echo "All Python files compiled (syntax OK)."

echo "Making shell helper executable (genai_bench_run.sh)"
chmod +x "$ROOT_DIR/genai_bench_run.sh" || true

echo "Dry-run genai-bench wrapper (no external call)"
bash -n "$ROOT_DIR/genai_bench_run.sh" || true

echo "Done. To actually run the examples, install dependencies from requirements.txt and run the desired script." 
echo "Install: python3 -m pip install -r requirements.txt"
