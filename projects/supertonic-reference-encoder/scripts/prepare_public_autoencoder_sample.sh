#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-data/public-autoencoder-sample}"
COPY_MODE="${COPY_MODE:-symlink}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/datasets}"

resolve_first_dir() {
  local fallback=""
  for candidate in "$@"; do
    if [[ -n "$candidate" && -z "$fallback" ]]; then
      fallback="$candidate"
    fi
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$fallback"
}

LIBRITTS_ROOT="$(resolve_first_dir "${LIBRITTS_ROOT:-}" "$DOWNLOAD_ROOT/LibriTTS/dev-clean")"
VCTK_ROOT="$(resolve_first_dir "${VCTK_ROOT:-}" "$DOWNLOAD_ROOT/VCTK-Corpus-0.92/wav48_silence_trimmed" "$DOWNLOAD_ROOT/VCTK-Corpus/wav48_silence_trimmed" "$DOWNLOAD_ROOT/wav48_silence_trimmed" "$DOWNLOAD_ROOT/wav48_slience_trimmed" "$DOWNLOAD_ROOT/VCTK-Corpus-0.92")"
ZEROTH_ROOT="$(resolve_first_dir "${ZEROTH_ROOT:-}" "$DOWNLOAD_ROOT/zeroth_korean" "$DOWNLOAD_ROOT/zeroth_korean/train_data_01")"
FLEURS_KO_ROOT="$(resolve_first_dir "${FLEURS_KO_ROOT:-}" "$DOWNLOAD_ROOT/fleurs/ko_kr" "$DOWNLOAD_ROOT/fleurs/ko_kr/train")"
COMMON_VOICE_KO_ROOT="$(resolve_first_dir "${COMMON_VOICE_KO_ROOT:-}" "$DOWNLOAD_ROOT/common_voice_ko")"

sources=()

add_source_if_present() {
  local name="$1"
  local root="$2"
  if [[ -d "$root" ]]; then
    sources+=(--source "$name=$root=$SAMPLES_PER_SOURCE")
  else
    echo "skipping missing corpus: $name root=$root" >&2
  fi
}

add_source_if_present "libritts" "$LIBRITTS_ROOT"
add_source_if_present "vctk" "$VCTK_ROOT"
add_source_if_present "zeroth-ko" "$ZEROTH_ROOT"
add_source_if_present "fleurs-ko" "$FLEURS_KO_ROOT"
add_source_if_present "common-voice-ko" "$COMMON_VOICE_KO_ROOT"

if [[ "${#sources[@]}" -eq 0 ]]; then
  cat >&2 <<'EOF'
No corpus roots were found. Run scripts/download_public_corpora.sh first, or set
LIBRITTS_ROOT/VCTK_ROOT/ZEROTH_ROOT/FLEURS_KO_ROOT/COMMON_VOICE_KO_ROOT to the
directories where the corpora are extracted.
EOF
  exit 1
fi

uv run supertonic-public-audio-dataset \
  --output-dir "$OUTPUT_DIR" \
  --copy-mode "$COPY_MODE" \
  "${sources[@]}"
