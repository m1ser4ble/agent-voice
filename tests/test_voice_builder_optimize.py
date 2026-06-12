import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.voice_builder.optimize import (
    Coefficients,
    coefficient_grid,
    select_best_result,
)


def test_coefficient_grid_expands_around_seed_alpha_and_projection_scale():
    grid = coefficient_grid(seed_alpha=1.0, seed_projection_scale=0.1)

    assert Coefficients(alpha=0.5, ttl_projection_scale=0.05) in grid
    assert Coefficients(alpha=1.0, ttl_projection_scale=0.1) in grid
    assert Coefficients(alpha=1.5, ttl_projection_scale=0.15) in grid


def test_select_best_result_returns_lowest_score():
    results = [
        {"score": 0.8, "coefficients": Coefficients(alpha=0.5, ttl_projection_scale=0.1)},
        {"score": 0.2, "coefficients": Coefficients(alpha=1.5, ttl_projection_scale=0.05)},
    ]

    assert select_best_result(results)["coefficients"].alpha == 1.5
