from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from supertonic_reference_encoder.styles import load_style_json


SynthesizeFn = Callable[..., None]
DEFAULT_PRESETS = ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")


@dataclass(frozen=True)
class GeneratePresetDatasetConfig:
    style_dir: Path
    texts_path: Path
    output_dir: Path
    presets: list[str]
    sample_rate: int = 44_100
    lang: str = "en"
    speed: float = 0.7


@dataclass(frozen=True)
class GeneratePresetDatasetResult:
    manifest_path: Path
    sample_count: int


def generate_preset_dataset(
    config: GeneratePresetDatasetConfig,
    *,
    synthesize_fn: SynthesizeFn | None = None,
) -> GeneratePresetDatasetResult:
    texts = _read_texts(config.texts_path)
    audio_dir = config.output_dir / "audio"
    dataset_style_dir = config.output_dir / "styles"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dataset_style_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "manifest.jsonl"
    synthesize = synthesize_fn or synthesize_supertonic_style

    sample_count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for preset in config.presets:
            style_json = config.style_dir / f"{preset}.json"
            if not style_json.exists():
                raise FileNotFoundError(f"missing preset style JSON: {style_json}")
            dataset_style_json = dataset_style_dir / style_json.name
            shutil.copy2(style_json, dataset_style_json)
            for text_index, text in enumerate(texts):
                audio_path = audio_dir / f"{preset}_{text_index:04d}.wav"
                synthesize(
                    style_json=dataset_style_json,
                    text=text,
                    output_path=audio_path,
                    sample_rate=config.sample_rate,
                    lang=config.lang,
                    speed=config.speed,
                )
                manifest.write(
                    json.dumps(
                        {
                            "audio": str(audio_path.relative_to(config.output_dir)),
                            "style_json": str(
                                dataset_style_json.relative_to(config.output_dir)
                            ),
                            "speaker_id": preset,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sample_count += 1
    return GeneratePresetDatasetResult(
        manifest_path=manifest_path,
        sample_count=sample_count,
    )


def synthesize_supertonic_style(
    *,
    style_json: Path,
    text: str,
    output_path: Path,
    sample_rate: int,
    lang: str,
    speed: float,
) -> None:
    from supertonic import TTS
    from supertonic.pipeline import Style

    style_tensors = load_style_json(style_json)
    style = Style(
        style_tensors.style_ttl.unsqueeze(0).numpy().astype(np.float32),
        style_tensors.style_dp.unsqueeze(0).numpy().astype(np.float32),
    )
    tts = TTS(auto_download=True)
    audio, _ = tts.synthesize(
        text,
        voice_style=style,
        speed=speed,
        lang=lang,
        total_steps=12,
        silence_duration=0.25,
    )
    waveform = np.asarray(audio, dtype=np.float32).squeeze()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        waveform,
        int(getattr(tts, "sample_rate", sample_rate) or sample_rate),
        format="WAV",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supervised audio/style pairs from public Supertonic presets.",
    )
    parser.add_argument(
        "--style-dir",
        type=Path,
        default=Path.home() / ".cache" / "supertonic3" / "voice_styles",
    )
    parser.add_argument("--texts", dest="texts_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--speed", type=float, default=0.7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_preset_dataset(
        GeneratePresetDatasetConfig(
            style_dir=args.style_dir,
            texts_path=args.texts_path,
            output_dir=args.output_dir,
            presets=args.presets,
            sample_rate=args.sample_rate,
            lang=args.lang,
            speed=args.speed,
        )
    )
    print(result.manifest_path)
    print(f"samples={result.sample_count}")
    return 0


def _read_texts(path: Path) -> list[str]:
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [line for line in texts if line]
    if not texts:
        raise ValueError(f"text prompt file is empty: {path}")
    return texts


if __name__ == "__main__":
    raise SystemExit(main())
