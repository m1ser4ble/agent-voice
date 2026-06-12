import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from supertonic_reference_encoder.model import AudioToStyleEncoder
from supertonic_reference_encoder.predict import PredictConfig, predict_style


def _write_wav(path: Path) -> None:
    sample_rate = 16_000
    t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def _write_checkpoint(path: Path) -> None:
    model = AudioToStyleEncoder()
    torch.save({"model": model.state_dict()}, path)


def test_predict_style_writes_supertonic_json(tmp_path):
    audio = tmp_path / "reference.wav"
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "voice-style.json"
    _write_wav(audio)
    _write_checkpoint(checkpoint)

    result = predict_style(
        PredictConfig(
            checkpoint=checkpoint,
            audio=audio,
            output=output,
            device="cpu",
            sample_rate=16_000,
        )
    )

    payload = json.loads(result.output.read_text(encoding="utf-8"))

    assert result.output == output
    assert payload["style_ttl"]["dims"] == [1, 50, 256]
    assert payload["style_dp"]["dims"] == [1, 8, 16]
    assert payload["metadata"]["source_audio"] == str(audio)
