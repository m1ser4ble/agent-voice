from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from agent_voice.recording import RecordingConfig, record_wav


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record a local microphone WAV fixture for STT evaluation."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/stt-fixtures/ko/local/local_0001.wav"),
        help="Output WAV path.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="WAV sample rate.",
    )
    parser.add_argument(
        "--input-device",
        help="sounddevice input device index or name.",
    )
    parser.add_argument(
        "--text",
        default="",
        help="Reference transcript to append to the manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/stt-fixtures/ko/local/manifest.jsonl"),
        help="Manifest JSONL path.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not append a manifest row.",
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=1.0,
        help="Seconds to wait before recording.",
    )
    args = parser.parse_args(argv)

    if args.countdown > 0:
        print(f"Recording starts in {args.countdown:g}s...", flush=True)
        time.sleep(args.countdown)

    print(
        f"Recording {args.seconds:g}s -> {args.out} "
        f"({args.sample_rate} Hz mono WAV)",
        flush=True,
    )
    output = record_wav(
        RecordingConfig(
            output=args.out,
            seconds=args.seconds,
            sample_rate=args.sample_rate,
            input_device=args.input_device,
            text=args.text,
            manifest=None if args.no_manifest else args.manifest,
        )
    )
    print(f"Saved {output}", flush=True)
    if not args.no_manifest:
        print(f"Updated {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
