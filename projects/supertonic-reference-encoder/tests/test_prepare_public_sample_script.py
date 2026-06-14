import json
import os
import subprocess
from pathlib import Path


def test_prepare_public_sample_script_discovers_nested_common_voice(tmp_path):
    download_root = tmp_path / "datasets"
    common_voice = download_root / "mozilla" / "cv-corpus-22.0-2025-06-20" / "ko"
    clip = common_voice / "clips" / "common_voice_ko_00000001.mp3"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"mp3")
    (common_voice / "validated.tsv").write_text(
        "client_id\tpath\tsentence\nspeaker\tcommon_voice_ko_00000001.mp3\t안녕하세요\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "public-autoencoder-sample"
    env = {
        **os.environ,
        "DOWNLOAD_ROOT": str(download_root),
        "OUTPUT_DIR": str(output_dir),
        "COPY_MODE": "copy",
    }
    subprocess.run(
        ["bash", "scripts/prepare_public_autoencoder_sample.sh"],
        cwd=Path(__file__).parents[1],
        env=env,
        check=True,
    )

    records = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["dataset"] for record in records] == ["common-voice-ko"]
