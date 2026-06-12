from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Coefficients:
    alpha: float
    ttl_projection_scale: float


def coefficient_grid(
    *,
    seed_alpha: float,
    seed_projection_scale: float,
) -> list[Coefficients]:
    alpha_values = [
        max(0.0, seed_alpha - 0.5),
        seed_alpha,
        seed_alpha + 0.5,
    ]
    projection_values = [
        max(0.0, seed_projection_scale - 0.05),
        seed_projection_scale,
        seed_projection_scale + 0.05,
    ]
    return [
        Coefficients(alpha=round(alpha, 6), ttl_projection_scale=round(projection, 6))
        for alpha in alpha_values
        for projection in projection_values
    ]


def select_best_result(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return min(results, key=lambda row: float(row["score"]))
