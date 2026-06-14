from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import soundfile as sf
import torch
import torchaudio

from supertonic_reference_encoder.audio import load_audio
from supertonic_reference_encoder.data import _resolve


AugmentationName = Literal[
    "original",
    "bandpass",
    "compressed",
    "distorted",
    "codec",
]
DEFAULT_AUGMENTATIONS: tuple[AugmentationName, ...] = (
    "original",
    "bandpass",
    "compressed",
    "distorted",
    "codec",
)


@dataclass(frozen=True)
class TargetAudioDatasetConfig:
    target_audio: Path
    output_dir: Path
    public_manifest: Path | None = None
    target_repeat: int = 10
    sample_rate: int = 44_100
    window_seconds: float = 3.0
    hop_seconds: float = 3.0
    min_window_seconds: float = 1.0
    augmentations: tuple[AugmentationName, ...] = DEFAULT_AUGMENTATIONS


@dataclass(frozen=True)
class TargetAudioDatasetResult:
    target_manifest: Path
    mixed_manifest: Path
    target_sample_count: int
    mixed_sample_count: int


def prepare_target_audio_dataset(config: TargetAudioDatasetConfig) -> TargetAudioDatasetResult:
    if config.target_repeat < 1:
        raise ValueError("target_repeat must be >= 1")
    target_dir = config.output_dir / "target"
    audio_dir = target_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    target_manifest = target_dir / "manifest.jsonl"
    mixed_manifest = config.output_dir / "manifest.jsonl"
    normalized_audio = normalize_target_audio(
        config.target_audio,
        output_dir=target_dir,
        sample_rate=config.sample_rate,
    )

    waveform = load_audio(
        normalized_audio,
        sample_rate=config.sample_rate,
        max_seconds=None,
    )
    windows = _window_audio(
        waveform,
        sample_rate=config.sample_rate,
        window_seconds=config.window_seconds,
        hop_seconds=config.hop_seconds,
        min_window_seconds=config.min_window_seconds,
    )
    if not windows:
        raise ValueError(f"target audio is too short: {config.target_audio}")

    target_records: list[dict[str, object]] = []
    for window_index, window in enumerate(windows):
        for augmentation in config.augmentations:
            augmented = apply_augmentation(window, augmentation, sample_rate=config.sample_rate)
            audio_path = audio_dir / f"target_{window_index:04d}_{augmentation}.wav"
            sf.write(audio_path, augmented.numpy(), config.sample_rate, format="WAV")
            target_records.append(
                {
                    "audio": str(audio_path.relative_to(config.output_dir)),
                    "dataset": "target-jarvis",
                    "source_audio": str(config.target_audio),
                    "normalized_audio": str(normalized_audio),
                    "window_index": window_index,
                    "augmentation": augmentation,
                }
            )
    _write_jsonl(target_manifest, target_records)

    mixed_records: list[dict[str, object]] = []
    if config.public_manifest is not None:
        mixed_records.extend(_read_manifest_with_absolute_audio(config.public_manifest))
    for _ in range(config.target_repeat):
        mixed_records.extend(target_records)
    _write_jsonl(mixed_manifest, mixed_records)

    return TargetAudioDatasetResult(
        target_manifest=target_manifest,
        mixed_manifest=mixed_manifest,
        target_sample_count=len(target_records),
        mixed_sample_count=len(mixed_records),
    )


def normalize_target_audio(
    source: Path,
    *,
    output_dir: Path,
    sample_rate: int,
    run_command: Callable[..., object] = subprocess.run,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "target_normalized.wav"
    if source.suffix.lower() == ".wav":
        waveform = load_audio(source, sample_rate=sample_rate, max_seconds=None)
        sf.write(target, waveform.numpy(), sample_rate, format="WAV")
        return target

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(target),
    ]
    try:
        run_command(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to normalize non-WAV target audio. "
            "Install ffmpeg or pass a WAV file as TARGET_AUDIO."
        ) from exc
    return target


def apply_augmentation(
    waveform: torch.Tensor,
    augmentation: AugmentationName,
    *,
    sample_rate: int,
) -> torch.Tensor:
    if augmentation == "original":
        return waveform.contiguous()
    if augmentation == "bandpass":
        return _fft_bandpass(waveform, sample_rate=sample_rate, low_hz=300.0, high_hz=3400.0)
    if augmentation == "compressed":
        return _soft_clip(waveform, drive=2.0)
    if augmentation == "distorted":
        return _soft_clip(waveform, drive=4.0)
    if augmentation == "codec":
        low_rate = min(16_000, sample_rate)
        downsampled = torchaudio.functional.resample(waveform, sample_rate, low_rate)
        restored = torchaudio.functional.resample(downsampled, low_rate, sample_rate)
        restored = restored[: waveform.numel()]
        if restored.numel() < waveform.numel():
            restored = torch.nn.functional.pad(restored, (0, waveform.numel() - restored.numel()))
        quantized = torch.round(torch.clamp(restored, -1.0, 1.0) * 127.0) / 127.0
        return quantized.contiguous()
    raise ValueError(f"unsupported augmentation: {augmentation}")


def _window_audio(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    min_window_seconds: float,
) -> list[torch.Tensor]:
    window_samples = int(sample_rate * window_seconds)
    hop_samples = int(sample_rate * hop_seconds)
    min_samples = int(sample_rate * min_window_seconds)
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    windows = []
    for start in range(0, waveform.numel(), hop_samples):
        window = waveform[start : start + window_samples]
        if window.numel() < min_samples:
            break
        if window.numel() < window_samples:
            window = torch.nn.functional.pad(window, (0, window_samples - window.numel()))
        windows.append(window.contiguous())
    return windows


def _fft_bandpass(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> torch.Tensor:
    spectrum = torch.fft.rfft(waveform)
    freqs = torch.fft.rfftfreq(waveform.numel(), d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return torch.fft.irfft(spectrum * mask, n=waveform.numel()).float().contiguous()


def _soft_clip(waveform: torch.Tensor, *, drive: float) -> torch.Tensor:
    driven = waveform * drive
    return (torch.tanh(driven) / torch.tanh(torch.tensor(drive))).float().contiguous()


def _read_manifest_with_absolute_audio(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    root = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        audio_path = _resolve(root, str(record["audio"]))
        source_audio = record.get("source_audio")
        if not audio_path.exists() and isinstance(source_audio, str):
            source_audio_path = Path(source_audio)
            if source_audio_path.exists():
                audio_path = source_audio_path
        record["audio"] = str(audio_path)
        records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Jarvis-like target audio windows and augmentations.",
    )
    parser.add_argument("--target-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument("--target-repeat", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=3.0)
    parser.add_argument("--min-window-seconds", type=float, default=1.0)
    parser.add_argument("--augmentations", nargs="+", default=list(DEFAULT_AUGMENTATIONS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_target_audio_dataset(
        TargetAudioDatasetConfig(
            target_audio=args.target_audio,
            output_dir=args.output_dir,
            public_manifest=args.public_manifest,
            target_repeat=args.target_repeat,
            sample_rate=args.sample_rate,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            min_window_seconds=args.min_window_seconds,
            augmentations=tuple(args.augmentations),
        )
    )
    print(result.target_manifest)
    print(f"target_samples={result.target_sample_count}")
    print(result.mixed_manifest)
    print(f"mixed_samples={result.mixed_sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
