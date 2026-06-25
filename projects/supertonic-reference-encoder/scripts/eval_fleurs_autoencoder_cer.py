from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel

from supertonic_reference_encoder.audio import load_audio
from supertonic_reference_encoder.evaluation import (
    EvaluationRow,
    character_error_rate,
    summarize_evaluation_rows,
)
from supertonic_reference_encoder.speech_autoencoder import (
    MultiResolutionMelLoss,
    SpeechAutoencoder,
    load_autoencoder_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate speech-autoencoder reconstruction quality on FLEURS Korean "
            "using Whisper transcription CER."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help=(
            "Checkpoint as LABEL=PATH_OR_GIT_COMMIT. Repeat to compare checkpoints. "
            "A bare git commit is resolved through the repository's LFS object store."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--stt-device", default="cpu")
    parser.add_argument("--stt-compute-type", default="int8")
    parser.add_argument("--keep-audio", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [_parse_checkpoint(value) for value in args.checkpoint]
    dataset_split = f"{args.split}[:{args.limit}]"
    dataset = load_dataset("google/fleurs", "ko_kr", split=dataset_split).cast_column(
        "audio",
        Audio(decode=False),
    )
    stt = WhisperModel(
        args.whisper_model,
        device=args.stt_device,
        compute_type=args.stt_compute_type,
    )

    with tempfile.TemporaryDirectory(prefix="fleurs-cer-") as temp_dir:
        working_dir = output_dir if args.keep_audio else Path(temp_dir)
        reference_rows = _transcribe_references(
            dataset,
            stt=stt,
            working_dir=working_dir / "reference",
        )
        reference_summary = summarize_evaluation_rows(reference_rows)
        checkpoint_summaries = []
        for label, checkpoint in checkpoints:
            summary = _evaluate_checkpoint(
                label=label,
                checkpoint=checkpoint,
                dataset=dataset,
                reference_rows=reference_rows,
                stt=stt,
                output_dir=output_dir,
                working_dir=working_dir / label,
                sample_rate=args.sample_rate,
                keep_audio=args.keep_audio,
            )
            checkpoint_summaries.append(summary)
            print(json.dumps(_compact_checkpoint_summary(summary), ensure_ascii=False), flush=True)

    series_summary = {
        "dataset": "google/fleurs ko_kr",
        "split": dataset_split,
        "limit": len(dataset),
        "reference_summary": reference_summary,
        "checkpoints": [_compact_checkpoint_summary(summary) for summary in checkpoint_summaries],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(series_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("SUMMARY")
    print(json.dumps(series_summary, ensure_ascii=False, indent=2))
    return 0


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", maxsplit=1)
    else:
        raw_path = value
        label = Path(value).stem
    checkpoint = _resolve_checkpoint(raw_path)
    return label, checkpoint


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    repo_root = _repo_root()
    pointer = subprocess.check_output(
        [
            "git",
            "show",
            f"{value}:projects/supertonic-reference-encoder/checkpoints/autoencoder-public.pt",
        ],
        cwd=repo_root,
        text=True,
    )
    match = re.search(r"oid sha256:([0-9a-f]+)", pointer)
    if match is None:
        raise RuntimeError(f"could not parse LFS pointer for checkpoint commit: {value}")
    oid = match.group(1)
    lfs_object = repo_root / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not lfs_object.exists():
        raise FileNotFoundError(
            "checkpoint LFS object is not available locally. "
            f"Run `git lfs fetch origin {value}` first. missing={lfs_object}"
        )
    return lfs_object


def _repo_root() -> Path:
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return Path(root)


def _transcribe_references(
    dataset,
    *,
    stt: WhisperModel,
    working_dir: Path,
) -> list[EvaluationRow]:
    working_dir.mkdir(parents=True, exist_ok=True)
    rows: list[EvaluationRow] = []
    for index, row in enumerate(dataset):
        text = _reference_text(row)
        reference_audio = _write_dataset_audio(row, working_dir / f"{index:05d}.wav")
        transcript = _transcribe(stt, reference_audio)
        rows.append(
            EvaluationRow(
                index=index,
                audio=str(reference_audio),
                text=text,
                transcript=transcript,
                cer=character_error_rate(text, transcript),
            )
        )
    return rows


def _evaluate_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    dataset,
    reference_rows: list[EvaluationRow],
    stt: WhisperModel,
    output_dir: Path,
    working_dir: Path,
    sample_rate: int,
    keep_audio: bool,
) -> dict:
    working_dir.mkdir(parents=True, exist_ok=True)
    model = SpeechAutoencoder(sample_rate=sample_rate)
    load_autoencoder_checkpoint(model, checkpoint)
    model.eval()
    loss_fn = MultiResolutionMelLoss(sample_rate=sample_rate)

    generated_rows: list[EvaluationRow] = []
    row_payloads = []
    mel_losses: list[float] = []
    for index, row in enumerate(dataset):
        reference_audio = _write_dataset_audio(row, working_dir / f"{index:05d}.reference.wav")
        waveform = load_audio(reference_audio, sample_rate=sample_rate)
        with torch.no_grad():
            generated = model(waveform.unsqueeze(0)).waveform.squeeze(0).cpu()
            min_len = min(waveform.numel(), generated.numel())
            waveform = waveform[:min_len].cpu()
            generated = generated[:min_len]
            mel_loss = float(loss_fn(generated.unsqueeze(0), waveform.unsqueeze(0)).detach().cpu())
            mel_losses.append(mel_loss)
        generated_audio = working_dir / f"{index:05d}.generated.wav"
        sf.write(generated_audio, _peak_normalize(generated).numpy(), sample_rate)
        generated_transcript = _transcribe(stt, generated_audio)
        reference = reference_rows[index]
        generated_row = EvaluationRow(
            index=index,
            audio=str(generated_audio),
            text=reference.text,
            transcript=generated_transcript,
            cer=character_error_rate(reference.text, generated_transcript),
        )
        generated_rows.append(generated_row)
        row_payloads.append(
            {
                "index": index,
                "text": reference.text,
                "reference_cer": reference.cer,
                "generated_cer": generated_row.cer,
                "cer_delta": generated_row.cer - reference.cer,
                "mel_loss": mel_loss,
                "reference_transcript": reference.transcript,
                "generated_transcript": generated_transcript,
                "reference_audio": reference.audio if keep_audio else None,
                "generated_audio": str(generated_audio) if keep_audio else None,
            }
        )

    summary = {
        "label": label,
        "checkpoint": str(checkpoint),
        "reference_summary": summarize_evaluation_rows(reference_rows),
        "generated_summary": summarize_evaluation_rows(generated_rows),
        "mean_cer_delta": sum(
            generated.cer - reference.cer
            for reference, generated in zip(reference_rows, generated_rows)
        )
        / len(generated_rows),
        "mean_mel_loss": sum(mel_losses) / len(mel_losses),
        "rows": row_payloads,
    }
    checkpoint_dir = output_dir / label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _compact_checkpoint_summary(summary: dict) -> dict:
    return {
        "label": summary["label"],
        "checkpoint": summary["checkpoint"],
        "mean_mel_loss": summary["mean_mel_loss"],
        "generated_mean_cer": summary["generated_summary"]["mean_cer"],
        "generated_median_cer": summary["generated_summary"]["median_cer"],
        "generated_max_cer": summary["generated_summary"]["max_cer"],
        "mean_cer_delta": summary["mean_cer_delta"],
    }


def _reference_text(row) -> str:
    return row["transcription"] or row["raw_transcription"]


def _write_dataset_audio(row, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    audio = row["audio"]
    if audio.get("bytes") is not None:
        target.write_bytes(audio["bytes"])
        return target
    shutil.copy2(Path(audio["path"]), target)
    return target


def _transcribe(stt: WhisperModel, audio: Path) -> str:
    segments, _ = stt.transcribe(str(audio), language="ko", beam_size=5, vad_filter=False)
    return "".join(segment.text for segment in segments).strip()


def _peak_normalize(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max().clamp_min(1e-6)
    return (waveform / peak * 0.95).contiguous()


if __name__ == "__main__":
    raise SystemExit(main())
