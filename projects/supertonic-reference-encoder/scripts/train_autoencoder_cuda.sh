#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="${MANIFEST:-data/public-autoencoder-sample/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/autoencoder-public-cuda}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SECONDS="${MAX_SECONDS:-3.0}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
MIXED_PRECISION="${MIXED_PRECISION:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

args=(
  uv run supertonic-autoencoder-train
  --manifest "$MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device cuda
  --max-seconds "$MAX_SECONDS"
  --learning-rate "$LEARNING_RATE"
  --log-every-steps "$LOG_EVERY_STEPS"
  --amp-dtype "$AMP_DTYPE"
)

if [[ "$MIXED_PRECISION" == "0" ]]; then
  args+=(--no-mixed-precision)
fi

if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi

"${args[@]}"
