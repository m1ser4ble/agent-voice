from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordingConfig:
    output: Path
    seconds: float = 5.0
    sample_rate: int = 16_000
    input_device: int | str | None = None
    text: str | None = None
    manifest: Path | None = None
    language: str = "ko"
    source: str = "local-mic"


def record_wav(
    config: RecordingConfig,
    *,
    sounddevice: Any | None = None,
    soundfile: Any | None = None,
) -> Path:
    if config.seconds <= 0:
        raise ValueError("seconds must be greater than 0")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")

    if sounddevice is None:
        import sounddevice
    if soundfile is None:
        import soundfile

    sd = sounddevice
    sf = soundfile

    output = config.output
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = int(config.seconds * config.sample_rate)
    audio = sd.rec(
        frames,
        samplerate=config.sample_rate,
        channels=1,
        dtype="float32",
        device=config.input_device,
    )
    sd.wait()
    sf.write(
        str(output),
        audio,
        config.sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    if config.manifest is not None:
        _append_manifest(config, output)

    return output


def _append_manifest(config: RecordingConfig, output: Path) -> None:
    manifest = config.manifest
    if manifest is None:
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "audio": output.name,
        "text": config.text or "",
        "language": config.language,
        "sample_rate": config.sample_rate,
        "source": config.source,
    }
    with manifest.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
