import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from supertonic_reference_encoder.data import (
    ReferenceStyleDataset,
    collate_reference_styles,
)
from supertonic_reference_encoder.styles import load_style_json, save_style_json
from supertonic_reference_encoder.train import train_one_step


def _write_wav(path: Path, frequency: float = 220.0) -> None:
    sample_rate = 16_000
    t = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def _write_style(path: Path, value: float = 0.1) -> None:
    save_style_json(
        path,
        style_ttl=torch.full((50, 256), value),
        style_dp=torch.full((8, 16), value * 2),
        metadata={"speaker": "test"},
    )


def test_style_json_round_trips_supertonic_shapes(tmp_path):
    path = tmp_path / "style.json"
    _write_style(path, value=0.25)

    style = load_style_json(path)

    assert style.style_ttl.shape == (50, 256)
    assert style.style_dp.shape == (8, 16)
    assert torch.isclose(style.style_ttl.mean(), torch.tensor(0.25))
    assert torch.isclose(style.style_dp.mean(), torch.tensor(0.5))


def test_reference_style_dataset_reads_manifest_audio_and_style(tmp_path):
    audio = tmp_path / "voice.wav"
    style = tmp_path / "voice-style.json"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    _write_style(style)
    manifest.write_text(
        json.dumps({"audio": str(audio), "style_json": str(style), "speaker_id": "s1"})
        + "\n",
        encoding="utf-8",
    )

    dataset = ReferenceStyleDataset(manifest, sample_rate=16_000, n_mels=228)
    item = dataset[0]

    assert item.mel.shape[0] == 228
    assert item.target.style_ttl.shape == (50, 256)
    assert item.target.style_dp.shape == (8, 16)
    assert item.metadata["speaker_id"] == "s1"


def test_collate_reference_styles_pads_variable_length_mels(tmp_path):
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    style_a = tmp_path / "a.json"
    style_b = tmp_path / "b.json"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio_a, 220)
    _write_wav(audio_b, 440)
    _write_style(style_a, 0.1)
    _write_style(style_b, 0.2)
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"audio": str(audio_a), "style_json": str(style_a)}),
                json.dumps({"audio": str(audio_b), "style_json": str(style_b)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ReferenceStyleDataset(manifest, sample_rate=16_000, n_mels=228)

    batch = collate_reference_styles([dataset[0], dataset[1]])

    assert batch.mel.shape[0] == 2
    assert batch.mel.shape[1] == 228
    assert batch.target.style_ttl.shape == (2, 50, 256)
    assert batch.target.style_dp.shape == (2, 8, 16)
    assert batch.lengths.shape == (2,)


def test_train_one_step_updates_model_parameters(tmp_path):
    audio = tmp_path / "voice.wav"
    style = tmp_path / "voice-style.json"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    _write_style(style)
    manifest.write_text(
        json.dumps({"audio": str(audio), "style_json": str(style)}) + "\n",
        encoding="utf-8",
    )
    dataset = ReferenceStyleDataset(manifest, sample_rate=16_000, n_mels=228)
    batch = collate_reference_styles([dataset[0]])

    metrics = train_one_step(batch, device=torch.device("cpu"))

    assert metrics["loss"] > 0.0
    assert metrics["ttl_loss"] >= 0.0
    assert metrics["dp_loss"] >= 0.0
