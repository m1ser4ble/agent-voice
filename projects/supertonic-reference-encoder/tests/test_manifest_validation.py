import json

from supertonic_reference_encoder.manifest_validation import validate_audio_manifest


def test_validate_audio_manifest_reports_missing_relative_audio(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio": "missing.wav"}) + "\n",
        encoding="utf-8",
    )

    result = validate_audio_manifest(manifest)

    assert result.total_count == 1
    assert result.existing_count == 0
    assert len(result.missing) == 1
    assert result.missing[0].line_number == 1
    assert str(tmp_path / "missing.wav") == str(result.missing[0].path)


def test_validate_audio_manifest_accepts_existing_absolute_audio(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio": str(audio)}) + "\n",
        encoding="utf-8",
    )

    result = validate_audio_manifest(manifest)

    assert result.total_count == 1
    assert result.existing_count == 1
    assert result.missing == []
