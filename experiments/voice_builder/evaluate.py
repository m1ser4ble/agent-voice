from __future__ import annotations

from experiments.voice_builder.features import ReferenceFeatures


def score_features(reference: ReferenceFeatures, candidate: ReferenceFeatures) -> float:
    weights = {
        "duration": 0.30,
        "f0": 1.40,
        "centroid": 1.00,
        "rolloff": 1.00,
        "rms": 0.10,
    }
    score = 0.0
    for field, weight in weights.items():
        ref = max(float(getattr(reference, field)), 1e-6)
        cand = float(getattr(candidate, field))
        score += weight * abs(cand - ref) / max(abs(ref), 1.0)
    return score
