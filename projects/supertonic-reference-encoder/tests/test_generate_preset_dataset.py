import json
from pathlib import Path

import numpy as np
import soundfile as sf

from supertonic_reference_encoder.generate_preset_dataset import (
    GeneratePresetDatasetConfig,
    generate_preset_dataset,
)
from supertonic_reference_encoder.styles import save_style_json


def _write_style(path: Path, value: float) -> None:
    import torch

    save_style_json(
        path,
        style_ttl=torch.full((50, 256), value),
        style_dp=torch.full((8, 16), value),
        metadata={"preset": path.stem},
    )


def test_generate_preset_dataset_writes_wavs_and_manifest(tmp_path):
    style_dir = tmp_path / "styles"
    style_dir.mkdir()
    _write_style(style_dir / "M2.json", 0.1)
    _write_style(style_dir / "F1.json", 0.2)
    texts = tmp_path / "texts.txt"
    texts.write_text("hello world\nsystems online\n", encoding="utf-8")

    def synthesize(*, style_json, text, output_path, sample_rate, lang, speed):
        t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
        audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(output_path, audio, sample_rate)

    result = generate_preset_dataset(
        GeneratePresetDatasetConfig(
            style_dir=style_dir,
            texts_path=texts,
            output_dir=tmp_path / "dataset",
            presets=["M2", "F1"],
            sample_rate=16_000,
            lang="en",
            speed=0.7,
        ),
        synthesize_fn=synthesize,
    )

    records = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 4
    assert result.manifest_path == tmp_path / "dataset" / "manifest.jsonl"
    assert all((tmp_path / "dataset" / record["audio"]).exists() for record in records)
    assert {record["speaker_id"] for record in records} == {"M2", "F1"}
    assert {record["style_json"] for record in records} == {
        "styles/M2.json",
        "styles/F1.json",
    }
    assert (tmp_path / "dataset" / "styles" / "M2.json").exists()
    assert (tmp_path / "dataset" / "styles" / "F1.json").exists()
