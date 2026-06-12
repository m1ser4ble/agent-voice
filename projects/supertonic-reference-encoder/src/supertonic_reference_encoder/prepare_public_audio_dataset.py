from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
DEFAULT_SOURCE_NAMES = (
    "libritts",
    "vctk",
    "fleurs-ko",
    "zeroth-ko",
    "common-voice-ko",
)


@dataclass(frozen=True)
class PublicAudioSource:
    name: str
    root: Path
    max_samples: int | None = None


@dataclass(frozen=True)
class PublicAudioDatasetResult:
    manifest_path: Path
    sample_count: int
    source_counts: dict[str, int]


def prepare_public_audio_dataset(
    sources: list[PublicAudioSource],
    *,
    output_dir: Path,
    copy_mode: Literal["copy", "symlink"] = "symlink",
) -> PublicAudioDatasetResult:
    if not sources:
        raise ValueError("at least one source is required")
    if copy_mode not in {"copy", "symlink"}:
        raise ValueError(f"unsupported copy mode: {copy_mode}")

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    sample_count = 0
    source_counts: dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for source in sorted(sources, key=lambda item: item.name):
            paths = list_audio_files(source.root)
            if source.max_samples is not None:
                paths = paths[: source.max_samples]
            source_counts[source.name] = len(paths)
            for source_index, source_audio in enumerate(paths):
                extension = source_audio.suffix.lower()
                target = audio_dir / source.name / f"{source_index:08d}{extension}"
                _materialize_audio(source_audio, target, copy_mode=copy_mode)
                manifest.write(
                    json.dumps(
                        {
                            "audio": str(target.relative_to(output_dir)),
                            "dataset": source.name,
                            "source_audio": str(source_audio),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sample_count += 1

    if sample_count == 0:
        raise ValueError("no audio files found in selected sources")
    return PublicAudioDatasetResult(
        manifest_path=manifest_path,
        sample_count=sample_count,
        source_counts=source_counts,
    )


def list_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"source root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def parse_sources(values: list[str]) -> list[PublicAudioSource]:
    sources = []
    for value in values:
        parts = value.split("=", maxsplit=2)
        if len(parts) not in {2, 3}:
            raise ValueError(
                "--source must be NAME=ROOT or NAME=ROOT=MAX_SAMPLES, "
                f"got: {value}"
            )
        name, root = parts[0], parts[1]
        max_samples = int(parts[2]) if len(parts) == 3 else None
        sources.append(PublicAudioSource(name=name, root=Path(root), max_samples=max_samples))
    return sources


def _materialize_audio(
    source: Path,
    target: Path,
    *,
    copy_mode: Literal["copy", "symlink"],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy_mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an audio-only manifest for autoencoder pretraining from "
            "downloaded public speech corpora."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source as NAME=ROOT or NAME=ROOT=MAX_SAMPLES. Repeat for each corpus.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_public_audio_dataset(
        parse_sources(args.source),
        output_dir=args.output_dir,
        copy_mode=args.copy_mode,
    )
    print(result.manifest_path)
    print(f"samples={result.sample_count}")
    for source_name, count in sorted(result.source_counts.items()):
        print(f"{source_name}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
