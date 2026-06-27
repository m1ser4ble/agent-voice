#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-checkpoints/autoencoder-public.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/autoencoder-cer-loop-3000}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-1000}"
EVAL_LIMIT="${EVAL_LIMIT:-1000}"
MIN_CER_IMPROVEMENT="${MIN_CER_IMPROVEMENT:-0.005}"

MANIFEST="${MANIFEST:-data/public-autoencoder-sample/manifest.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SECONDS="${MAX_SECONDS:-3.0}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
MIXED_PRECISION="${MIXED_PRECISION:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

WHISPER_MODEL="${WHISPER_MODEL:-medium}"
STT_DEVICE="${STT_DEVICE:-cuda}"
STT_COMPUTE_TYPE="${STT_COMPUTE_TYPE:-float16}"

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "initial checkpoint not found: $INITIAL_CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  echo "Set MANIFEST=/path/to/manifest.jsonl or prepare data/public-autoencoder-sample first." >&2
  exit 1
fi

current_checkpoint="$INITIAL_CHECKPOINT"

for round in $(seq 1 "$MAX_ROUNDS"); do
  round_dir="$OUTPUT_ROOT/round-$(printf '%03d' "$round")"
  train_dir="$round_dir/train"
  eval_dir="$round_dir/eval"
  previous_checkpoint="$current_checkpoint"

  echo "round=$round train resume=$previous_checkpoint output=$train_dir"
  RESUME="$previous_checkpoint" \
  OUTPUT_DIR="$train_dir" \
  EPOCHS="$EPOCHS_PER_ROUND" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  MAX_SECONDS="$MAX_SECONDS" \
  LEARNING_RATE="$LEARNING_RATE" \
  LOG_EVERY_STEPS="$LOG_EVERY_STEPS" \
  MIXED_PRECISION="$MIXED_PRECISION" \
  AMP_DTYPE="$AMP_DTYPE" \
    scripts/train_autoencoder_cuda.sh

  candidate_checkpoint="$train_dir/best.pt"
  if [[ ! -f "$candidate_checkpoint" ]]; then
    candidate_checkpoint="$train_dir/latest.pt"
  fi
  if [[ ! -f "$candidate_checkpoint" ]]; then
    echo "training did not produce best.pt or latest.pt in $train_dir" >&2
    exit 1
  fi

  echo "round=$round eval prev=$previous_checkpoint current=$candidate_checkpoint output=$eval_dir"
  uv run --with datasets --with faster-whisper \
    python scripts/eval_fleurs_autoencoder_cer.py \
      --checkpoint "prev=$previous_checkpoint" \
      --checkpoint "current=$candidate_checkpoint" \
      --output-dir "$eval_dir" \
      --limit "$EVAL_LIMIT" \
      --whisper-model "$WHISPER_MODEL" \
      --stt-device "$STT_DEVICE" \
      --stt-compute-type "$STT_COMPUTE_TYPE"

  decision_json="$round_dir/decision.json"
  python - "$eval_dir/summary.json" "$MIN_CER_IMPROVEMENT" "$decision_json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
min_improvement = float(sys.argv[2])
decision_path = Path(sys.argv[3])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
by_label = {item["label"]: item for item in summary["checkpoints"]}
previous_cer = by_label["prev"]["generated_mean_cer"]
current_cer = by_label["current"]["generated_mean_cer"]
cer_improvement = previous_cer - current_cer
continue_training = cer_improvement >= min_improvement
decision = {
    "previous_generated_mean_cer": previous_cer,
    "current_generated_mean_cer": current_cer,
    "cer_improvement": cer_improvement,
    "min_cer_improvement": min_improvement,
    "continue_training": continue_training,
}
decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(decision, ensure_ascii=False))
PY

  if python - "$decision_json" <<'PY'
import json
import sys
from pathlib import Path

decision = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if decision["continue_training"] else 1)
PY
  then
    current_checkpoint="$candidate_checkpoint"
    ln -sfn "$(realpath "$current_checkpoint")" "$OUTPUT_ROOT/accepted.pt"
    echo "continuing: CER improved enough; accepted=$current_checkpoint"
  else
    echo "stopping: CER improvement below threshold; keeping previous checkpoint=$previous_checkpoint"
    echo "$previous_checkpoint" > "$OUTPUT_ROOT/final_checkpoint.txt"
    exit 0
  fi
done

echo "$current_checkpoint" > "$OUTPUT_ROOT/final_checkpoint.txt"
echo "finished: reached MAX_ROUNDS=$MAX_ROUNDS final_checkpoint=$current_checkpoint"
