#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_AUDIO="${TARGET_AUDIO:-$HOME/Downloads/Voicy_Jarvis Start Up.mp3}"
PUBLIC_MANIFEST="${PUBLIC_MANIFEST:-data/public-autoencoder-sample/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/autoencoder-jarvis-intelligible}"
TARGET_REPEAT="${TARGET_REPEAT:-20}"
SAMPLE_RATE="${SAMPLE_RATE:-44100}"
WINDOW_SECONDS="${WINDOW_SECONDS:-3.0}"
HOP_SECONDS="${HOP_SECONDS:-3.0}"
MIN_WINDOW_SECONDS="${MIN_WINDOW_SECONDS:-1.0}"
DEFAULT_AUGMENTATIONS=("original" "bandpass" "compressed")

if [[ -n "${AUGMENTATIONS:-}" ]]; then
  read -r -a AUGMENTATION_ARGS <<< "$AUGMENTATIONS"
else
  AUGMENTATION_ARGS=("${DEFAULT_AUGMENTATIONS[@]}")
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install ffmpeg before preparing MP3 target audio." >&2
  exit 1
fi

if [[ ! -f "$TARGET_AUDIO" ]]; then
  echo "target audio not found: $TARGET_AUDIO" >&2
  echo "Set TARGET_AUDIO=/path/to/reference.wav-or-mp3." >&2
  exit 1
fi

args=(
  uv run supertonic-target-audio-dataset
  --target-audio "$TARGET_AUDIO"
  --output-dir "$OUTPUT_DIR"
  --target-repeat "$TARGET_REPEAT"
  --sample-rate "$SAMPLE_RATE"
  --window-seconds "$WINDOW_SECONDS"
  --hop-seconds "$HOP_SECONDS"
  --min-window-seconds "$MIN_WINDOW_SECONDS"
  --augmentations "${AUGMENTATION_ARGS[@]}"
)

if [[ -f "$PUBLIC_MANIFEST" ]]; then
  uv run supertonic-check-audio-manifest --manifest "$PUBLIC_MANIFEST"
  args+=(--public-manifest "$PUBLIC_MANIFEST")
else
  echo "public manifest not found, preparing target-only manifest: $PUBLIC_MANIFEST" >&2
fi

echo "prepare_target_autoencoder_intelligible_ft:"
echo "  output_dir=$OUTPUT_DIR"
echo "  target_repeat=$TARGET_REPEAT"
echo "  augmentations=${AUGMENTATION_ARGS[*]}"

"${args[@]}"
