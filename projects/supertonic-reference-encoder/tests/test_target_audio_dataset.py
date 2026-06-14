import json
from pathlib import Path

import numpy as np
import soundfile as sf

from supertonic_reference_encoder.target_audio_dataset import (
    TargetAudioDatasetConfig,
    normalize_target_audio,
    prepare_target_audio_dataset,
)


def _write_wav(path: Path, *, seconds: float = 2.2, sample_rate: int = 16_000) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_target_audio_dataset_windows_and_augments_target_audio(tmp_path):
    target = tmp_path / "jarvis.wav"
    _write_wav(target)

    result = prepare_target_audio_dataset(
        TargetAudioDatasetConfig(
            target_audio=target,
            output_dir=tmp_path / "target",
            sample_rate=16_000,
            window_seconds=1.0,
            hop_seconds=1.0,
            augmentations=("original", "bandpass", "codec"),
            min_window_seconds=0.1,
        )
    )

    assert result.target_sample_count == 9
    records = _read_jsonl(result.target_manifest)
    assert {record["augmentation"] for record in records} == {"original", "bandpass", "codec"}
    assert all(Path(tmp_path / "target" / str(record["audio"])).exists() for record in records)


def test_prepare_target_audio_dataset_builds_mixed_manifest_with_repeated_target(tmp_path):
    public_audio = tmp_path / "public.wav"
    target = tmp_path / "jarvis.wav"
    _write_wav(public_audio, seconds=1.0)
    _write_wav(target, seconds=1.0)
    public_manifest = tmp_path / "public_manifest.jsonl"
    public_manifest.write_text(
        json.dumps({"audio": str(public_audio), "dataset": "public"}) + "\n",
        encoding="utf-8",
    )

    result = prepare_target_audio_dataset(
        TargetAudioDatasetConfig(
            target_audio=target,
            output_dir=tmp_path / "mixed",
            public_manifest=public_manifest,
            target_repeat=3,
            sample_rate=16_000,
            window_seconds=1.0,
            hop_seconds=1.0,
            augmentations=("original",),
        )
    )

    records = _read_jsonl(result.mixed_manifest)
    assert result.mixed_sample_count == 4
    assert records[0]["dataset"] == "public"
    assert [record["dataset"] for record in records[1:]] == ["target-jarvis"] * 3


def test_prepare_target_audio_dataset_falls_back_to_source_audio_for_broken_symlink(tmp_path):
    public_audio = tmp_path / "public.wav"
    target = tmp_path / "jarvis.wav"
    broken_link = tmp_path / "public-link.wav"
    _write_wav(public_audio, seconds=1.0)
    _write_wav(target, seconds=1.0)
    broken_link.symlink_to(tmp_path / "missing-public.wav")
    public_manifest = tmp_path / "public_manifest.jsonl"
    public_manifest.write_text(
        json.dumps(
            {
                "audio": "public-link.wav",
                "source_audio": str(public_audio),
                "dataset": "public",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = prepare_target_audio_dataset(
        TargetAudioDatasetConfig(
            target_audio=target,
            output_dir=tmp_path / "mixed",
            public_manifest=public_manifest,
            target_repeat=1,
            sample_rate=16_000,
            window_seconds=1.0,
            hop_seconds=1.0,
            augmentations=("original",),
        )
    )

    records = _read_jsonl(result.mixed_manifest)
    assert records[0]["audio"] == str(public_audio)


def test_normalize_target_audio_uses_ffmpeg_for_non_wav_sources(tmp_path):
    source = tmp_path / "voice.mp3"
    source.write_bytes(b"fake mp3")
    calls = []

    def fake_run(command, check):
        calls.append(command)
        Path(command[-1]).write_bytes(b"wav")

    normalized = normalize_target_audio(
        source,
        output_dir=tmp_path / "normalized",
        sample_rate=16_000,
        run_command=fake_run,
    )

    assert normalized.name == "target_normalized.wav"
    assert normalized.exists()
    assert calls
    assert calls[0][0] == "ffmpeg"
    assert str(source) in calls[0]
