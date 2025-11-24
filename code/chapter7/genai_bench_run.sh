#!/usr/bin/env bash
# Wrapper to run a conceptual genai-bench command shown in Chapter 7.
set -euo pipefail
WORKLOAD=${1:-my_sglang_workload.yaml}
OUT=${2:-results.json}
echo "Running genai-bench with workload=$WORKLOAD -> output=$OUT"
# Uncomment and adapt the following line if genai-bench is installed in your PATH
# genai-bench run --workload "$WORKLOAD" --output "$OUT"
echo "(dry-run) genai-bench run --workload $WORKLOAD --output $OUT"
