from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from supertonic_reference_encoder.audio import LogMelExtractor, load_audio
from supertonic_reference_encoder.model import AudioToStyleEncoder
from supertonic_reference_encoder.styles import save_style_json
from supertonic_reference_encoder.train import _resolve_device


@dataclass(frozen=True)
class PredictConfig:
    checkpoint: Path
    audio: Path
    output: Path
    device: str = "auto"
    sample_rate: int = 44_100
    n_mels: int = 228
    max_seconds: float = 12.0


@dataclass(frozen=True)
class PredictResult:
    output: Path


def predict_style(config: PredictConfig) -> PredictResult:
    device = _resolve_device(config.device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    model = AudioToStyleEncoder()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    waveform = load_audio(
        config.audio,
        sample_rate=config.sample_rate,
        max_seconds=config.max_seconds,
    )
    mel = LogMelExtractor(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
    )(waveform)
    with torch.no_grad():
        style = model(mel.unsqueeze(0).to(device))

    save_style_json(
        config.output,
        style_ttl=style.style_ttl.squeeze(0).cpu(),
        style_dp=style.style_dp.squeeze(0).cpu(),
        metadata={
            "source_audio": str(config.audio),
            "checkpoint": str(config.checkpoint),
        },
    )
    return PredictResult(output=config.output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Supertonic style JSON from audio.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--n-mels", type=int, default=228)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = predict_style(
        PredictConfig(
            checkpoint=args.checkpoint,
            audio=args.audio,
            output=args.output,
            device=args.device,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            max_seconds=args.max_seconds,
        )
    )
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
