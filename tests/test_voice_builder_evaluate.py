import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.voice_builder.evaluate import score_features
from experiments.voice_builder.features import ReferenceFeatures


def test_score_features_prefers_candidate_close_to_reference():
    reference = ReferenceFeatures(
        duration=16.0,
        f0=270.0,
        centroid=4500.0,
        rolloff=7800.0,
        rms=0.1,
    )
    close = ReferenceFeatures(
        duration=15.5,
        f0=260.0,
        centroid=4400.0,
        rolloff=7700.0,
        rms=0.11,
    )
    far = ReferenceFeatures(
        duration=8.0,
        f0=110.0,
        centroid=3000.0,
        rolloff=5200.0,
        rms=0.2,
    )

    assert score_features(reference, close) < score_features(reference, far)
