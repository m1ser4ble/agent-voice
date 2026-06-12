import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.voice_builder.features import ReferenceFeatures
from experiments.voice_builder.latent_basis import (
    DP_DIMS,
    TTL_DIMS,
    StyleArrays,
    build_directional_style,
    direction_alpha,
)


def _style(ttl_value: float, dp_value: float) -> StyleArrays:
    return StyleArrays(
        ttl=np.full(TTL_DIMS, ttl_value, dtype=np.float32),
        dp=np.full(DP_DIMS, dp_value, dtype=np.float32),
    )


def test_direction_alpha_increases_for_high_pitch_bright_reference():
    low = ReferenceFeatures(
        duration=1.0,
        f0=170.0,
        centroid=2800.0,
        rolloff=4000.0,
        rms=0.1,
    )
    high = ReferenceFeatures(
        duration=1.0,
        f0=280.0,
        centroid=4600.0,
        rolloff=7800.0,
        rms=0.1,
    )

    assert direction_alpha(high) > direction_alpha(low)


def test_build_directional_style_extrapolates_f2_along_f1_minus_m2():
    calibration = {
        "M2": _style(0.0, 0.0),
        "F1": _style(0.4, 0.2),
        "F2": _style(0.2, 0.1),
    }
    projected = np.full(TTL_DIMS, 0.01, dtype=np.float32)

    ttl, dp = build_directional_style(
        projected_ttl=projected,
        calibration_styles=calibration,
        alpha=1.5,
    )

    assert np.isclose(ttl.mean(), 0.2 + (1.5 * 0.4) + 0.01)
    assert np.isclose(dp.mean(), 0.1 + (1.5 * 0.2))
