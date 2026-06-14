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

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install ffmpeg before training with MP3 validation audio." >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  echo "Run scripts/prepare_target_autoencoder_ft.sh first, or set MANIFEST=/path/to/manifest.jsonl." >&2
  exit 1
fi

if [[ ! -f "$RESUME" ]]; then
  echo "resume checkpoint not found: $RESUME" >&2
  echo "Set RESUME=/path/to/best.pt or copy checkpoints/autoencoder-public.pt into place." >&2
  exit 1
fi

if [[ ! -f "$VALIDATION_AUDIO" ]]; then
  echo "validation audio not found: $VALIDATION_AUDIO" >&2
  echo "Set VALIDATION_AUDIO=/path/to/reference.wav-or-mp3." >&2
  exit 1
fi

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
