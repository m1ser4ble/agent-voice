#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_AUDIO="${TARGET_AUDIO:-$HOME/Downloads/Voicy_Jarvis Start Up.mp3}"
PUBLIC_MANIFEST="${PUBLIC_MANIFEST:-data/public-autoencoder-sample/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/autoencoder-public-plus-target}"
TARGET_REPEAT="${TARGET_REPEAT:-10}"
SAMPLE_RATE="${SAMPLE_RATE:-44100}"
WINDOW_SECONDS="${WINDOW_SECONDS:-3.0}"
HOP_SECONDS="${HOP_SECONDS:-3.0}"
MIN_WINDOW_SECONDS="${MIN_WINDOW_SECONDS:-1.0}"

args=(
  uv run supertonic-target-audio-dataset
  --target-audio "$TARGET_AUDIO"
  --output-dir "$OUTPUT_DIR"
  --target-repeat "$TARGET_REPEAT"
  --sample-rate "$SAMPLE_RATE"
  --window-seconds "$WINDOW_SECONDS"
  --hop-seconds "$HOP_SECONDS"
  --min-window-seconds "$MIN_WINDOW_SECONDS"
)

if [[ -f "$PUBLIC_MANIFEST" ]]; then
  args+=(--public-manifest "$PUBLIC_MANIFEST")
else
  echo "public manifest not found, preparing target-only manifest: $PUBLIC_MANIFEST" >&2
fi

"${args[@]}"
