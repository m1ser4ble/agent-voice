import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.voice_builder.mp3_to_json import (
    build_voice_style_json,
    build_voice_style_payload,
    main,
    parse_args,
)


def _write_tone(path: Path, frequency: float) -> None:
    sample_rate = 16000
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_build_voice_style_payload_derives_supertonic_json_from_audio_file(tmp_path):
    source = tmp_path / "voice.wav"
    _write_tone(source, 440)

    payload = build_voice_style_payload(source)

    assert payload["style_ttl"]["dims"] == [1, 50, 256]
    assert payload["style_dp"]["dims"] == [1, 8, 16]
    assert np.asarray(payload["style_ttl"]["data"]).shape == (1, 50, 256)
    assert np.asarray(payload["style_dp"]["data"]).shape == (1, 8, 16)
    assert payload["metadata"]["generator"] == "agent-voice voice_builder.mp3_to_json"
    assert payload["metadata"]["source_audio"] == str(source)
    assert payload["metadata"]["method"] == "direct_projection_v1"
    assert payload["metadata"]["ttl_projection_scale"] == 0.5
    assert payload["metadata"]["dp_projection_scale"] == 0.5
    assert payload["metadata"]["calibration_mode"] == "none"
    assert payload["metadata"]["brightness_gain"] >= 1.0


def test_different_audio_produces_different_voice_style_json(tmp_path):
    low = tmp_path / "low.wav"
    high = tmp_path / "high.wav"
    _write_tone(low, 220)
    _write_tone(high, 880)

    low_payload = build_voice_style_payload(low)
    high_payload = build_voice_style_payload(high)

    assert low_payload["style_ttl"]["data"] != high_payload["style_ttl"]["data"]
    assert low_payload["style_dp"]["data"] != high_payload["style_dp"]["data"]


def test_voice_style_payload_can_directly_project_duration_style(tmp_path):
    source = tmp_path / "voice.wav"
    _write_tone(source, 440)

    payload = build_voice_style_payload(source)
    dp = np.asarray(payload["style_dp"]["data"])

    assert not np.allclose(dp, 0.0)


def test_voice_style_payload_can_opt_into_calibration_prior(tmp_path):
    source = tmp_path / "voice.wav"
    _write_tone(source, 880)
    m2_ttl = np.zeros((1, 50, 256), dtype=np.float32)
    m2_dp = np.zeros((1, 8, 16), dtype=np.float32)
    f1_ttl = np.full((1, 50, 256), 0.4, dtype=np.float32)
    f1_dp = np.full((1, 8, 16), 0.4, dtype=np.float32)
    f2_ttl = np.full((1, 50, 256), 0.2, dtype=np.float32)
    f2_dp = np.full((1, 8, 16), 0.2, dtype=np.float32)

    payload = build_voice_style_payload(
        source,
        calibration_styles={
            "M2": (m2_ttl, m2_dp),
            "F1": (f1_ttl, f1_dp),
            "F2": (f2_ttl, f2_dp),
        },
    )
    ttl = np.asarray(payload["style_ttl"]["data"])
    alpha = payload["metadata"]["direction_alpha"]

    assert alpha > 0
    assert ttl.mean() > 0.2
    assert payload["metadata"]["calibration_mode"] == "f2_plus_f1_minus_m2_direction"


def test_build_voice_style_json_writes_json_and_preview_from_input_audio(tmp_path):
    source = tmp_path / "voice.wav"
    _write_tone(source, 660)

    calls = []

    def synthesize_preview(payload, *, text, output_path, speed, lang):
        calls.append((payload, text, output_path, speed, lang))
        output_path.write_bytes(b"preview")

    result = build_voice_style_json(
        source,
        out_dir=tmp_path / "built",
        text="안녕하세요",
        speed=0.7,
        lang="ko",
        synthesize_preview_fn=synthesize_preview,
    )

    payload = json.loads(result.style_path.read_text())

    assert result.style_path == tmp_path / "built" / "voice-style.json"
    assert result.preview_path == tmp_path / "built" / "preview.wav"
    assert result.style_path.exists()
    assert result.preview_path.read_bytes() == b"preview"
    assert payload["metadata"]["source_audio"] == str(source)
    assert calls == [(payload, "안녕하세요", result.preview_path, 0.7, "ko")]


def test_build_voice_style_json_can_override_coefficients(tmp_path):
    source = tmp_path / "voice.wav"
    _write_tone(source, 660)

    result = build_voice_style_json(
        source,
        out_dir=tmp_path / "built",
        text="안녕하세요",
        alpha=1.25,
        ttl_projection_scale=0.2,
        synthesize_preview_fn=lambda payload, *, text, output_path, speed, lang: output_path.write_bytes(
            b"preview"
        ),
    )

    payload = json.loads(result.style_path.read_text())

    assert payload["metadata"]["direction_alpha"] == 1.25
    assert payload["metadata"]["ttl_projection_scale"] == 0.2


def test_cli_is_mp3_to_json_not_builtin_voice_search(tmp_path):
    source = tmp_path / "voice.wav"
    _write_tone(source, 440)

    args = parse_args(
        [
            "--input",
            str(source),
            "--out-dir",
            str(tmp_path / "out"),
            "--speed",
            "0.7",
            "--lang",
            "en",
            "--optimize",
        ]
    )

    assert args.input == source
    assert args.speed == 0.7
    assert args.lang == "en"
    assert args.optimize is True
    assert not hasattr(args, "voices")
    with pytest.raises(SystemExit):
        parse_args(["--input", str(source), "--voices", "M2"])


def test_main_uses_optimizer_when_requested(tmp_path):
    source = tmp_path / "voice.wav"
    _write_tone(source, 440)
    calls = []

    def optimizer(**kwargs):
        calls.append(kwargs)
        return {"alpha": 1.5, "ttl_projection_scale": 0.05, "score": 0.1}

    exit_code = main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(tmp_path / "out"),
            "--optimize",
        ],
        synthesize_preview_fn=lambda payload, *, text, output_path, speed, lang: output_path.write_bytes(
            b"preview"
        ),
        optimizer_fn=optimizer,
    )

    payload = json.loads((tmp_path / "out" / "voice-style.json").read_text())

    assert exit_code == 0
    assert calls
    assert payload["metadata"]["direction_alpha"] == 1.5
    assert payload["metadata"]["ttl_projection_scale"] == 0.05
