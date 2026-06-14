#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="${MANIFEST:-data/autoencoder-public-plus-target/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/autoencoder-public-plus-target-ft}"
RESUME="${RESUME:-checkpoints/autoencoder-public.pt}"
VALIDATION_AUDIO="${VALIDATION_AUDIO:-$HOME/Downloads/Voicy_Jarvis Start Up.mp3}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SECONDS="${MAX_SECONDS:-3.0}"
LEARNING_RATE="${LEARNING_RATE:-0.00003}"

uv run supertonic-autoencoder-train \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --resume "$RESUME" \
  --validation-audio "$VALIDATION_AUDIO" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda \
  --max-seconds "$MAX_SECONDS" \
  --learning-rate "$LEARNING_RATE"
