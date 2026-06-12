import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.voice_builder.features import ReferenceFeatures
from experiments.voice_builder.latent_basis import DP_DIMS, TTL_DIMS, StyleArrays
from experiments.voice_builder.sensitivity import (
    EffectMetrics,
    StyleBlock,
    apply_block_delta,
    block_grid,
    rank_effects,
    style_payload,
    write_effects_jsonl,
)


def _style() -> StyleArrays:
    return StyleArrays(
        ttl=np.zeros(TTL_DIMS, dtype=np.float32),
        dp=np.zeros(DP_DIMS, dtype=np.float32),
    )


def test_apply_block_delta_changes_only_selected_ttl_region():
    style = _style()
    block = StyleBlock(
        kind="ttl",
        token_start=10,
        token_end=20,
        channel_start=64,
        channel_end=128,
        delta=0.25,
    )

    changed = apply_block_delta(style, block)

    assert np.allclose(changed.dp, 0.0)
    assert np.allclose(changed.ttl[:, 10:20, 64:128], 0.25)
    assert np.allclose(changed.ttl[:, :10, :], 0.0)
    assert np.allclose(changed.ttl[:, 20:, :], 0.0)
    assert np.allclose(changed.ttl[:, 10:20, :64], 0.0)
    assert np.allclose(changed.ttl[:, 10:20, 128:], 0.0)


def test_block_grid_covers_ttl_and_dp_blocks_with_labels():
    blocks = block_grid(
        ttl_token_bins=5,
        ttl_channel_bins=4,
        dp_query_bins=4,
        dp_channel_bins=2,
        delta=0.1,
    )

    assert len(blocks) == (5 * 4) + (4 * 2)
    assert blocks[0].label == "ttl:t00-10:c000-064:+0.100"
    assert blocks[-1].label == "dp:t06-08:c008-016:+0.100"


def test_rank_effects_prefers_candidates_closer_to_reference():
    reference = ReferenceFeatures(
        duration=16.0,
        f0=270.0,
        centroid=4500.0,
        rolloff=7800.0,
        rms=0.1,
    )
    baseline = ReferenceFeatures(
        duration=8.0,
        f0=110.0,
        centroid=2200.0,
        rolloff=3600.0,
        rms=0.1,
    )
    dull = ReferenceFeatures(
        duration=9.0,
        f0=120.0,
        centroid=2400.0,
        rolloff=3800.0,
        rms=0.1,
    )
    closer = ReferenceFeatures(
        duration=14.0,
        f0=230.0,
        centroid=4100.0,
        rolloff=7100.0,
        rms=0.1,
    )

    ranked = rank_effects(
        reference=reference,
        baseline=baseline,
        candidates=[
            EffectMetrics(label="dull", features=dull),
            EffectMetrics(label="closer", features=closer),
        ],
    )

    assert [item.label for item in ranked] == ["closer", "dull"]
    assert ranked[0].improvement > ranked[1].improvement
    assert ranked[0].improvement > 0


def test_style_payload_serializes_style_arrays_with_metadata():
    payload = style_payload(_style(), label="baseline", metadata={"source": "test"})

    assert payload["style_ttl"]["dims"] == TTL_DIMS
    assert payload["style_dp"]["dims"] == DP_DIMS
    assert payload["metadata"]["label"] == "baseline"
    assert payload["metadata"]["source"] == "test"


def test_write_effects_jsonl_records_ranked_candidates(tmp_path):
    reference = ReferenceFeatures(
        duration=1.0,
        f0=100.0,
        centroid=1000.0,
        rolloff=2000.0,
        rms=0.1,
    )
    effect = EffectMetrics(
        label="ttl:t00-10:c000-064:+0.100",
        features=reference,
        score=0.25,
        improvement=0.5,
    )

    path = tmp_path / "effects.jsonl"
    write_effects_jsonl(path, [effect])

    text = path.read_text(encoding="utf-8")
    assert '"label": "ttl:t00-10:c000-064:+0.100"' in text
    assert '"improvement": 0.5' in text
