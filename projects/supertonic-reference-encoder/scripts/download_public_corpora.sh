#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/datasets}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$DOWNLOAD_ROOT/archives}"

INCLUDE_LIBRITTS="${INCLUDE_LIBRITTS:-1}"
INCLUDE_VCTK="${INCLUDE_VCTK:-1}"
INCLUDE_ZEROTH="${INCLUDE_ZEROTH:-1}"
INCLUDE_FLEURS_KO="${INCLUDE_FLEURS_KO:-1}"
INCLUDE_COMMON_VOICE_KO="${INCLUDE_COMMON_VOICE_KO:-0}"

LIBRITTS_URL="${LIBRITTS_URL:-https://www.openslr.org/resources/60/dev-clean.tar.gz}"
VCTK_URL="${VCTK_URL:-https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip}"
ZEROTH_URL="${ZEROTH_URL:-https://openslr.trmal.net/resources/40/zeroth_korean.tar.gz}"
FLEURS_SPLIT="${FLEURS_SPLIT:-train}"

mkdir -p "$DOWNLOAD_ROOT" "$ARCHIVE_DIR"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    echo "archive exists: $output"
    return
  fi
  echo "downloading: $url"
  curl -L --fail --continue-at - --output "$output" "$url"
}

extract_tar_gz() {
  local archive="$1"
  local marker="$2"
  if [[ -e "$marker" ]]; then
    echo "already extracted: $marker"
    return
  fi
  echo "extracting: $archive"
  tar -xzf "$archive" -C "$DOWNLOAD_ROOT"
}

extract_zip() {
  local archive="$1"
  local marker="$2"
  if [[ -e "$marker" ]]; then
    echo "already extracted: $marker"
    return
  fi
  echo "extracting: $archive"
  unzip -q "$archive" -d "$DOWNLOAD_ROOT"
}

if [[ "$INCLUDE_LIBRITTS" == "1" ]]; then
  archive="$ARCHIVE_DIR/libritts-dev-clean.tar.gz"
  marker="$DOWNLOAD_ROOT/LibriTTS/dev-clean"
  if [[ -e "$marker" ]]; then
    echo "already extracted: $marker"
  else
    download_file "$LIBRITTS_URL" "$archive"
    extract_tar_gz "$archive" "$marker"
  fi
fi

if [[ "$INCLUDE_VCTK" == "1" ]]; then
  archive="$ARCHIVE_DIR/VCTK-Corpus-0.92.zip"
  marker="$DOWNLOAD_ROOT/VCTK-Corpus-0.92/wav48_silence_trimmed"
  if [[ -e "$DOWNLOAD_ROOT/wav48_silence_trimmed" ]]; then
    marker="$DOWNLOAD_ROOT/wav48_silence_trimmed"
  elif [[ -e "$DOWNLOAD_ROOT/wav48_slience_trimmed" ]]; then
    marker="$DOWNLOAD_ROOT/wav48_slience_trimmed"
  fi
  if [[ -e "$marker" ]]; then
    echo "already extracted: $marker"
  else
    download_file "$VCTK_URL" "$archive"
    extract_zip "$archive" "$marker"
  fi
fi

if [[ "$INCLUDE_ZEROTH" == "1" ]]; then
  archive="$ARCHIVE_DIR/zeroth_korean.tar.gz"
  marker="$DOWNLOAD_ROOT/zeroth_korean"
  if [[ -e "$marker" ]]; then
    echo "already extracted: $marker"
  else
    download_file "$ZEROTH_URL" "$archive"
    extract_tar_gz "$archive" "$marker"
  fi
fi

if [[ "$INCLUDE_FLEURS_KO" == "1" ]]; then
  output_dir="$DOWNLOAD_ROOT/fleurs/ko_kr"
  if [[ -d "$output_dir/$FLEURS_SPLIT" ]] && find "$output_dir/$FLEURS_SPLIT" -type f -name '*.wav' | grep -q .; then
    echo "already exported: $output_dir/$FLEURS_SPLIT"
  else
    mkdir -p "$output_dir"
    echo "downloading/exporting FLEURS ko_kr split=$FLEURS_SPLIT"
    FLEURS_OUTPUT_DIR="$output_dir" FLEURS_SPLIT="$FLEURS_SPLIT" \
      uv run --with datasets python - <<'PY'
import os
import shutil
from pathlib import Path

from datasets import Audio, load_dataset

output = Path(os.environ["FLEURS_OUTPUT_DIR"])
split = os.environ["FLEURS_SPLIT"]
split_dir = output / split
split_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("google/fleurs", "ko_kr", split=split).cast_column(
    "audio",
    Audio(decode=False),
)
for index, row in enumerate(dataset):
    audio = row["audio"]
    source_path = Path(audio["path"]) if audio.get("path") else None
    if source_path and source_path.exists():
        target = split_dir / f"{index:08d}{source_path.suffix.lower() or '.wav'}"
        shutil.copy2(source_path, target)
    elif audio.get("bytes") is not None:
        target = split_dir / f"{index:08d}.wav"
        target.write_bytes(audio["bytes"])
    else:
        raise RuntimeError(f"FLEURS row {index} has neither a local path nor bytes")
print(f"exported={len(dataset)} path={split_dir}")
PY
  fi
fi

if [[ "$INCLUDE_COMMON_VOICE_KO" == "1" ]]; then
  cat >&2 <<'EOF'
Common Voice Korean is not auto-downloaded by this script because Mozilla's
Data Collective flow may require dataset selection, terms, and authenticated
browser download. Download the Korean validated clips manually, extract them
under /datasets/common_voice_ko, then run prepare_public_autoencoder_sample.sh.
EOF
fi
