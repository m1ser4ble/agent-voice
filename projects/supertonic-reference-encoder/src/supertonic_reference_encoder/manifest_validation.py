from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from supertonic_reference_encoder.data import _resolve


@dataclass(frozen=True)
class MissingAudio:
    line_number: int
    path: Path


@dataclass(frozen=True)
class ManifestValidationResult:
    manifest: Path
    total_count: int
    existing_count: int
    missing: list[MissingAudio]


def validate_audio_manifest(manifest: Path) -> ManifestValidationResult:
    root = manifest.parent
    total_count = 0
    existing_count = 0
    missing: list[MissingAudio] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total_count += 1
        record = json.loads(line)
        if "audio" not in record:
            raise ValueError(f"{manifest}:{line_number} missing required key: audio")
        audio_path = _resolve(root, str(record["audio"]))
        if audio_path.exists():
            existing_count += 1
        else:
            missing.append(MissingAudio(line_number=line_number, path=audio_path))
    return ManifestValidationResult(
        manifest=manifest,
        total_count=total_count,
        existing_count=existing_count,
        missing=missing,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate audio paths in a JSONL manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-missing", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_audio_manifest(args.manifest)
    print(f"manifest={result.manifest}")
    print(f"existing={result.existing_count}")
    print(f"total={result.total_count}")
    if result.missing:
        print(f"missing={len(result.missing)}")
        for item in result.missing[: args.max_missing]:
            print(f"missing line={item.line_number} path={item.path}")
        return 1
    print("missing=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
