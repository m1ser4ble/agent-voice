#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-data/public-autoencoder-sample}"
COPY_MODE="${COPY_MODE:-symlink}"

LIBRITTS_ROOT="${LIBRITTS_ROOT:-/datasets/LibriTTS/dev-clean}"
VCTK_ROOT="${VCTK_ROOT:-/datasets/VCTK-Corpus/wav48_silence_trimmed}"
ZEROTH_ROOT="${ZEROTH_ROOT:-/datasets/zeroth_korean}"
FLEURS_KO_ROOT="${FLEURS_KO_ROOT:-/datasets/fleurs/ko_kr}"
COMMON_VOICE_KO_ROOT="${COMMON_VOICE_KO_ROOT:-/datasets/common_voice_ko}"

uv run supertonic-public-audio-dataset \
  --output-dir "$OUTPUT_DIR" \
  --copy-mode "$COPY_MODE" \
  --source "libritts=$LIBRITTS_ROOT=$SAMPLES_PER_SOURCE" \
  --source "vctk=$VCTK_ROOT=$SAMPLES_PER_SOURCE" \
  --source "zeroth-ko=$ZEROTH_ROOT=$SAMPLES_PER_SOURCE" \
  --source "fleurs-ko=$FLEURS_KO_ROOT=$SAMPLES_PER_SOURCE" \
  --source "common-voice-ko=$COMMON_VOICE_KO_ROOT=$SAMPLES_PER_SOURCE"
