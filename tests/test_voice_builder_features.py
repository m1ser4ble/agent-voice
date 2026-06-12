import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.voice_builder.features import extract_reference_features, load_mono_audio


def _write_tone(path: Path, frequency: float) -> None:
    sample_rate = 16000
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_load_mono_audio_normalizes_stereo_audio(tmp_path):
    path = tmp_path / "stereo.wav"
    sample_rate = 16000
    audio = np.asarray([[0.5, -0.5], [0.25, -0.25], [1.0, -1.0]], dtype=np.float32)
    sf.write(path, audio, sample_rate)

    mono, loaded_rate = load_mono_audio(path)

    assert loaded_rate == sample_rate
    assert mono.ndim == 1
    assert np.max(np.abs(mono)) <= 1.0


def test_extract_reference_features_detects_higher_pitch_and_brightness(tmp_path):
    low = tmp_path / "low.wav"
    high = tmp_path / "high.wav"
    _write_tone(low, 220)
    _write_tone(high, 880)

    low_features = extract_reference_features(low)
    high_features = extract_reference_features(high)

    assert high_features.f0 > low_features.f0
    assert high_features.centroid > low_features.centroid
    assert high_features.duration == low_features.duration
