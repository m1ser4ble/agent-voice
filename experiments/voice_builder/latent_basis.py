from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.voice_builder.features import ReferenceFeatures


TTL_DIMS = [1, 50, 256]
DP_DIMS = [1, 8, 16]


@dataclass(frozen=True)
class StyleArrays:
    ttl: np.ndarray
    dp: np.ndarray


def load_default_calibration_styles() -> dict[str, StyleArrays]:
    style_dir = Path.home() / ".cache" / "supertonic3" / "voice_styles"
    out: dict[str, StyleArrays] = {}
    for label in ("M2", "F1", "F2"):
        path = style_dir / f"{label}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[label] = StyleArrays(
                ttl=np.asarray(data["style_ttl"]["data"], dtype=np.float32),
                dp=np.asarray(data["style_dp"]["data"], dtype=np.float32),
            )
        except (KeyError, json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
    return out


def direction_alpha(features: ReferenceFeatures) -> float:
    f0_score = _clamp((features.f0 - 170.0) / 110.0)
    brightness_score = _clamp((features.centroid - 2800.0) / 1800.0)
    return 0.25 + (0.90 * f0_score) + (0.45 * brightness_score)


def build_directional_style(
    *,
    projected_ttl: np.ndarray,
    calibration_styles: dict[str, StyleArrays],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not {"M2", "F1", "F2"}.issubset(calibration_styles):
        return projected_ttl.astype(np.float32), np.zeros(DP_DIMS, dtype=np.float32)

    m2 = calibration_styles["M2"]
    f1 = calibration_styles["F1"]
    f2 = calibration_styles["F2"]
    ttl = f2.ttl + (alpha * (f1.ttl - m2.ttl)) + projected_ttl
    dp = f2.dp + (alpha * (f1.dp - m2.dp))
    return ttl.astype(np.float32), dp.astype(np.float32)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))
