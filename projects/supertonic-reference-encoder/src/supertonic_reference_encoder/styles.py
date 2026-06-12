from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from supertonic_reference_encoder.model import StyleTensors


def load_style_json(path: Path) -> StyleTensors:
    data = json.loads(path.read_text(encoding="utf-8"))
    ttl = torch.as_tensor(data["style_ttl"]["data"], dtype=torch.float32)
    dp = torch.as_tensor(data["style_dp"]["data"], dtype=torch.float32)
    ttl = _squeeze_batch(ttl, expected_shape=(50, 256), path=path, key="style_ttl")
    dp = _squeeze_batch(dp, expected_shape=(8, 16), path=path, key="style_dp")
    return StyleTensors(style_ttl=ttl, style_dp=dp)


def save_style_json(
    path: Path,
    *,
    style_ttl: torch.Tensor,
    style_dp: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> None:
    ttl = _squeeze_batch(style_ttl.detach().cpu().float(), expected_shape=(50, 256), path=path, key="style_ttl")
    dp = _squeeze_batch(style_dp.detach().cpu().float(), expected_shape=(8, 16), path=path, key="style_dp")
    payload = {
        "style_ttl": {
            "dims": [1, 50, 256],
            "data": ttl.unsqueeze(0).tolist(),
        },
        "style_dp": {
            "dims": [1, 8, 16],
            "data": dp.unsqueeze(0).tolist(),
        },
        "metadata": {
            "generator": "supertonic-reference-encoder",
            **(metadata or {}),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _squeeze_batch(
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, int],
    path: Path,
    key: str,
) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{path} {key} must be {expected_shape} or (1, {expected_shape[0]}, {expected_shape[1]}), got {tuple(value.shape)}")
    return value.contiguous()
