from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
import torch

from supertonic_reference_encoder.audio import load_audio


def test_load_audio_reads_wav(tmp_path):
    path = tmp_path / "voice.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16_000)

    waveform = load_audio(path, sample_rate=16_000)

    assert waveform.shape == (1600,)


def test_load_audio_reports_missing_file_before_decoder_fallback(tmp_path):
    missing = tmp_path / "missing.mp3"

    try:
        load_audio(missing, sample_rate=16_000)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

    assert str(missing) in message


def test_load_audio_falls_back_to_ffmpeg_when_soundfile_fails(tmp_path, monkeypatch):
    source = tmp_path / "voice.mp3"
    source.write_bytes(b"not really mp3")
    calls = []

    def fake_sf_read(path, dtype, always_2d):
        if Path(path) == source:
            raise sf.LibsndfileError(1)
        return np.zeros((800, 1), dtype=np.float32), 16_000

    def fake_run(command, check, stdout, stderr, text):
        calls.append(command)
        Path(command[-1]).write_bytes(b"wav")

    monkeypatch.setattr("supertonic_reference_encoder.audio.sf.read", fake_sf_read)
    monkeypatch.setattr("supertonic_reference_encoder.audio.subprocess.run", fake_run)

    waveform = load_audio(source, sample_rate=16_000)

    assert torch.equal(waveform, torch.zeros(800))
    assert calls
    assert calls[0][0] == "ffmpeg"
    assert str(source) in calls[0]


def test_load_audio_reports_ffmpeg_failure_path_and_stderr(tmp_path, monkeypatch):
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"not really mp3")

    def fake_sf_read(path, dtype, always_2d):
        raise sf.LibsndfileError(1)

    def fake_run(command, check, stdout, stderr, text):
        raise subprocess.CalledProcessError(
            returncode=254,
            cmd=command,
            stderr="Invalid data found when processing input",
        )

    monkeypatch.setattr("supertonic_reference_encoder.audio.sf.read", fake_sf_read)
    monkeypatch.setattr("supertonic_reference_encoder.audio.subprocess.run", fake_run)

    try:
        load_audio(source, sample_rate=16_000)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert str(source) in message
    assert "returncode=254" in message
    assert "Invalid data found" in message
