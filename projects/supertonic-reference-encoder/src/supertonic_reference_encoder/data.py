from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from supertonic_reference_encoder.audio import LogMelExtractor, load_audio
from supertonic_reference_encoder.model import StyleTensors
from supertonic_reference_encoder.styles import load_style_json


@dataclass(frozen=True)
class ReferenceStyleItem:
    mel: torch.Tensor
    target: StyleTensors
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReferenceStyleBatch:
    mel: torch.Tensor
    lengths: torch.Tensor
    target: StyleTensors
    metadata: list[dict[str, Any]]


class ReferenceStyleDataset(Dataset[ReferenceStyleItem]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        sample_rate: int = 44_100,
        n_mels: int = 228,
        max_seconds: float | None = 12.0,
    ) -> None:
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.mel_extractor = LogMelExtractor(sample_rate=sample_rate, n_mels=n_mels)
        self.records = _read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ReferenceStyleItem:
        record = self.records[index]
        audio_path = _resolve(self.root, record["audio"])
        style_path = _resolve(self.root, record["style_json"])
        waveform = load_audio(
            audio_path,
            sample_rate=self.sample_rate,
            max_seconds=self.max_seconds,
        )
        return ReferenceStyleItem(
            mel=self.mel_extractor(waveform),
            target=load_style_json(style_path),
            metadata={k: v for k, v in record.items() if k not in {"audio", "style_json"}},
        )


def collate_reference_styles(items: list[ReferenceStyleItem]) -> ReferenceStyleBatch:
    if not items:
        raise ValueError("cannot collate an empty batch")

    lengths = torch.as_tensor([item.mel.shape[1] for item in items], dtype=torch.long)
    max_length = int(lengths.max().item())
    n_mels = items[0].mel.shape[0]
    mel = torch.zeros(len(items), n_mels, max_length, dtype=torch.float32)
    for index, item in enumerate(items):
        mel[index, :, : item.mel.shape[1]] = item.mel

    return ReferenceStyleBatch(
        mel=mel,
        lengths=lengths,
        target=StyleTensors(
            style_ttl=torch.stack([item.target.style_ttl for item in items]),
            style_dp=torch.stack([item.target.style_dp for item in items]),
        ),
        metadata=[item.metadata for item in items],
    )


def _read_manifest(
    path: Path,
    *,
    required: tuple[str, ...] = ("audio", "style_json"),
) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"{path}:{line_number} missing required keys: {missing}")
        records.append(record)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
