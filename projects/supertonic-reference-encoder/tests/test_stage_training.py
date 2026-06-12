import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from supertonic_reference_encoder.stage_training import (
    StageDataset,
    collate_stage_items,
    tokenize_text,
    train_duration_one_step,
    train_text_to_latent_one_step,
)
from supertonic_reference_encoder.styles import save_style_json


def _write_wav(path: Path) -> None:
    sample_rate = 16_000
    t = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def _write_style(path: Path) -> None:
    save_style_json(
        path,
        style_ttl=torch.zeros(50, 256),
        style_dp=torch.zeros(8, 16),
    )


def _write_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "voice.wav"
    style = tmp_path / "style.json"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    _write_style(style)
    manifest.write_text(
        json.dumps({"audio": str(audio), "style_json": str(style), "text": "hello"})
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_tokenize_text_returns_fixed_length_ids():
    tokens = tokenize_text("hello", max_length=8)

    assert tokens.shape == (8,)
    assert tokens[0].item() > 0
    assert tokens[-1].item() == 0


def test_stage_dataset_reads_audio_style_and_text(tmp_path):
    manifest = _write_manifest(tmp_path)

    dataset = StageDataset(manifest, sample_rate=16_000, n_mels=228, max_text_length=16)
    item = dataset[0]

    assert item.mel.shape[0] == 228
    assert item.text_tokens.shape == (16,)
    assert item.target.style_ttl.shape == (50, 256)


def test_stage_training_steps_run(tmp_path):
    manifest = _write_manifest(tmp_path)
    dataset = StageDataset(manifest, sample_rate=16_000, n_mels=228, max_text_length=16)
    batch = collate_stage_items([dataset[0]])

    flow_metrics = train_text_to_latent_one_step(batch, device=torch.device("cpu"))
    duration_metrics = train_duration_one_step(batch, device=torch.device("cpu"))

    assert flow_metrics["loss"] > 0.0
    assert duration_metrics["loss"] > 0.0
