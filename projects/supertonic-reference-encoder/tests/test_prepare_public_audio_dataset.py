import json
from pathlib import Path

from supertonic_reference_encoder.prepare_public_audio_dataset import (
    PublicAudioSource,
    prepare_public_audio_dataset,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"wav")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_public_audio_dataset_copies_samples_and_writes_metadata(tmp_path):
    libri = tmp_path / "libritts"
    fleurs = tmp_path / "fleurs"
    _touch(libri / "LibriTTS" / "dev-clean" / "19" / "198" / "19_198_000000_000000.wav")
    _touch(libri / "LibriTTS" / "dev-clean" / "20" / "205" / "20_205_000000_000000.wav")
    _touch(fleurs / "ko_kr" / "train" / "1000.wav")

    result = prepare_public_audio_dataset(
        [
            PublicAudioSource(name="libritts-dev-clean", root=libri, max_samples=1),
            PublicAudioSource(name="fleurs-ko", root=fleurs, max_samples=2),
        ],
        output_dir=tmp_path / "public-autoencoder-sample",
        copy_mode="copy",
    )

    assert result.sample_count == 2
    assert result.source_counts == {"libritts-dev-clean": 1, "fleurs-ko": 1}
    records = _read_jsonl(result.manifest_path)
    assert [record["dataset"] for record in records] == ["fleurs-ko", "libritts-dev-clean"]
    assert all(Path(tmp_path / "public-autoencoder-sample" / record["audio"]).exists() for record in records)


def test_prepare_public_audio_dataset_can_symlink_instead_of_copying(tmp_path):
    source_root = tmp_path / "zeroth"
    source_audio = source_root / "train_data_01" / "speaker" / "utt.flac"
    _touch(source_audio)

    result = prepare_public_audio_dataset(
        [PublicAudioSource(name="zeroth-ko", root=source_root)],
        output_dir=tmp_path / "dataset",
        copy_mode="symlink",
    )

    records = _read_jsonl(result.manifest_path)
    linked = tmp_path / "dataset" / str(records[0]["audio"])
    assert result.sample_count == 1
    assert linked.is_symlink()
    assert linked.resolve() == source_audio
