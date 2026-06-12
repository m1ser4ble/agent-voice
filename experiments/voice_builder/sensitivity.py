from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np

from experiments.voice_builder.evaluate import score_features
from experiments.voice_builder.features import ReferenceFeatures
from experiments.voice_builder.latent_basis import DP_DIMS, TTL_DIMS, StyleArrays


StyleKind = Literal["ttl", "dp"]


@dataclass(frozen=True)
class StyleBlock:
    kind: StyleKind
    token_start: int
    token_end: int
    channel_start: int
    channel_end: int
    delta: float

    @property
    def label(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return (
            f"{self.kind}:"
            f"t{self.token_start:02d}-{self.token_end:02d}:"
            f"c{self.channel_start:03d}-{self.channel_end:03d}:"
            f"{sign}{self.delta:.3f}"
        )


@dataclass(frozen=True)
class EffectMetrics:
    label: str
    features: ReferenceFeatures
    score: float = 0.0
    improvement: float = 0.0


def apply_block_delta(style: StyleArrays, block: StyleBlock) -> StyleArrays:
    ttl = np.array(style.ttl, dtype=np.float32, copy=True).reshape(TTL_DIMS)
    dp = np.array(style.dp, dtype=np.float32, copy=True).reshape(DP_DIMS)

    if block.kind == "ttl":
        _apply_delta(ttl, block)
    elif block.kind == "dp":
        _apply_delta(dp, block)
    else:
        raise ValueError(f"unsupported style block kind: {block.kind}")

    return StyleArrays(ttl=ttl, dp=dp)


def block_grid(
    *,
    ttl_token_bins: int,
    ttl_channel_bins: int,
    dp_query_bins: int,
    dp_channel_bins: int,
    delta: float,
) -> list[StyleBlock]:
    blocks: list[StyleBlock] = []
    blocks.extend(
        _blocks_for(
            kind="ttl",
            token_count=TTL_DIMS[1],
            channel_count=TTL_DIMS[2],
            token_bins=ttl_token_bins,
            channel_bins=ttl_channel_bins,
            delta=delta,
        )
    )
    blocks.extend(
        _blocks_for(
            kind="dp",
            token_count=DP_DIMS[1],
            channel_count=DP_DIMS[2],
            token_bins=dp_query_bins,
            channel_bins=dp_channel_bins,
            delta=delta,
        )
    )
    return blocks


def rank_effects(
    *,
    reference: ReferenceFeatures,
    baseline: ReferenceFeatures,
    candidates: list[EffectMetrics],
) -> list[EffectMetrics]:
    baseline_score = score_features(reference, baseline)
    ranked = [
        EffectMetrics(
            label=item.label,
            features=item.features,
            score=score_features(reference, item.features),
            improvement=baseline_score - score_features(reference, item.features),
        )
        for item in candidates
    ]
    return sorted(ranked, key=lambda item: item.score)


def style_payload(
    style: StyleArrays,
    *,
    label: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "style_ttl": {
            "dims": TTL_DIMS,
            "data": np.asarray(style.ttl, dtype=np.float32).reshape(TTL_DIMS).tolist(),
        },
        "style_dp": {
            "dims": DP_DIMS,
            "data": np.asarray(style.dp, dtype=np.float32).reshape(DP_DIMS).tolist(),
        },
        "metadata": {
            "generator": "agent-voice voice_builder.sensitivity",
            "label": label,
            **(metadata or {}),
        },
    }


def write_effects_jsonl(path: Path, effects: list[EffectMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for effect in effects:
            file.write(json.dumps(_effect_record(effect), ensure_ascii=False) + "\n")


def _apply_delta(values: np.ndarray, block: StyleBlock) -> None:
    if block.token_start < 0 or block.channel_start < 0:
        raise ValueError(f"block indices must be non-negative: {block}")
    if block.token_end <= block.token_start or block.channel_end <= block.channel_start:
        raise ValueError(f"block ranges must be non-empty: {block}")
    if block.token_end > values.shape[1] or block.channel_end > values.shape[2]:
        raise ValueError(f"block exceeds style shape {values.shape}: {block}")
    values[:, block.token_start : block.token_end, block.channel_start : block.channel_end] += (
        block.delta
    )


def _blocks_for(
    *,
    kind: StyleKind,
    token_count: int,
    channel_count: int,
    token_bins: int,
    channel_bins: int,
    delta: float,
) -> list[StyleBlock]:
    if token_bins <= 0 or channel_bins <= 0:
        raise ValueError("bin counts must be positive")

    blocks: list[StyleBlock] = []
    token_edges = _edges(token_count, token_bins)
    channel_edges = _edges(channel_count, channel_bins)
    for token_start, token_end in zip(token_edges, token_edges[1:]):
        for channel_start, channel_end in zip(channel_edges, channel_edges[1:]):
            blocks.append(
                StyleBlock(
                    kind=kind,
                    token_start=token_start,
                    token_end=token_end,
                    channel_start=channel_start,
                    channel_end=channel_end,
                    delta=delta,
                )
            )
    return blocks


def _edges(count: int, bins: int) -> list[int]:
    return [int(round(value)) for value in np.linspace(0, count, bins + 1)]


def _effect_record(effect: EffectMetrics) -> dict[str, Any]:
    return {
        "label": effect.label,
        "score": effect.score,
        "improvement": effect.improvement,
        "features": {
            "duration": effect.features.duration,
            "f0": effect.features.f0,
            "centroid": effect.features.centroid,
            "rolloff": effect.features.rolloff,
            "rms": effect.features.rms,
        },
    }
