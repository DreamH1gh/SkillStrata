#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATASET="${DATASET:-all}"
MODEL="${MODEL:-all}"

"$PYTHON_BIN" run.py \
  --config experiment_config.json \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --device "$DEVICE"
