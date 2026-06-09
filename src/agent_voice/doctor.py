from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO


@dataclass(frozen=True)
class DoctorOptions:
    agent: str = "codex"
    cache_dir: Path = Path(".cache/agent-voice")
    list_devices: bool = False
    deep: bool = False
    whisper_model: str = "tiny"


class DoctorProbe(Protocol):
    def find_command(self, command: str) -> str | None:
        """Return the executable path for a command, if available."""

    def has_package(self, package: str) -> bool:
        """Return whether an import package is available in this environment."""

    def query_audio_devices(self) -> Sequence[Mapping[str, Any]]:
        """Return sounddevice-style audio device dictionaries."""

    def path_exists(self, path: Path) -> bool:
        """Return whether a runtime asset path exists."""


class SystemDoctorProbe:
    def find_command(self, command: str) -> str | None:
        return shutil.which(command)

    def has_package(self, package: str) -> bool:
        return importlib.util.find_spec(package) is not None

    def query_audio_devices(self) -> Sequence[Mapping[str, Any]]:
        import sounddevice as sd

        devices = sd.query_devices()
        return list(devices)

    def path_exists(self, path: Path) -> bool:
        return path.exists()


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str


REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("pexpect", "pexpect"),
    ("faster_whisper", "faster-whisper"),
    ("kokoro_onnx", "kokoro-onnx"),
    ("pipecat", "pipecat-ai"),
    ("sounddevice", "sounddevice"),
)

KOKORO_MODEL_FILES: tuple[str, ...] = (
    "kokoro-v1.0.onnx",
    "voices-v1.0.bin",
)


def build_doctor_parser(
    *,
    default_cache_dir: Path | str = Path(".cache/agent-voice"),
    default_whisper_model: str = "tiny",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-voice doctor",
        description="Check local voice runtime prerequisites.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--agent",
        choices=("codex", "pi", "none"),
        default="codex",
        help="Agent command to check. Use 'none' to skip the agent CLI check.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(default_cache_dir),
        help="Directory where local voice model assets are cached.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print discovered audio devices.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Include slower runtime checks when they are available.",
    )
    parser.add_argument(
        "--whisper-model",
        default=default_whisper_model,
        help="Whisper model name or path to report for deep checks.",
    )
    return parser


def run_doctor(
    options: DoctorOptions,
    *,
    probe: DoctorProbe | None = None,
    output: TextIO | None = None,
) -> int:
    probe = probe or SystemDoctorProbe()
    output = output or sys.stdout

    results: list[CheckResult] = []
    results.extend(_check_packages(probe))
    results.extend(_check_agent(options, probe))
    devices, audio_results = _check_audio_devices(probe)
    results.extend(audio_results)
    results.append(_check_kokoro_cache(options, probe))
    if options.deep:
        results.append(_check_deep_readiness(options))

    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}", file=output)

    if options.list_devices:
        _print_audio_devices(devices, output)

    ok_count = sum(1 for result in results if result.status == "ok")
    warn_count = sum(1 for result in results if result.status == "warn")
    fail_count = sum(1 for result in results if result.status == "fail")
    print(
        f"Summary: {ok_count} ok, {warn_count} warn, {fail_count} fail",
        file=output,
    )
    return 1 if fail_count else 0


def _check_packages(probe: DoctorProbe) -> list[CheckResult]:
    results: list[CheckResult] = []
    for import_name, distribution_name in REQUIRED_PACKAGES:
        if probe.has_package(import_name):
            results.append(
                CheckResult(
                    "ok",
                    f"python package {distribution_name}",
                    "available",
                )
            )
        else:
            results.append(
                CheckResult(
                    "fail",
                    f"python package {distribution_name}",
                    f"missing import '{import_name}'",
                )
            )
    return results


def _check_agent(options: DoctorOptions, probe: DoctorProbe) -> list[CheckResult]:
    if options.agent == "none":
        return [CheckResult("warn", "agent command", "skipped")]

    command_path = probe.find_command(options.agent)
    if command_path:
        return [CheckResult("ok", f"command {options.agent}", command_path)]
    return [
        CheckResult(
            "fail",
            f"command {options.agent}",
            "not found on PATH",
        )
    ]


def _check_audio_devices(
    probe: DoctorProbe,
) -> tuple[Sequence[Mapping[str, Any]], list[CheckResult]]:
    try:
        devices = probe.query_audio_devices()
    except Exception as error:  # pragma: no cover - exercised by real machines.
        detail = f"sounddevice query failed: {error}"
        return (), [
            CheckResult("fail", "audio input devices", detail),
            CheckResult("fail", "audio output devices", detail),
        ]

    input_count = sum(
        _device_channels(device, "max_input_channels") > 0 for device in devices
    )
    output_count = sum(
        _device_channels(device, "max_output_channels") > 0 for device in devices
    )
    results = [
        _device_result("audio input devices", input_count, "input"),
        _device_result("audio output devices", output_count, "output"),
    ]
    return devices, results


def _device_result(name: str, count: int, label: str) -> CheckResult:
    if count:
        return CheckResult("ok", name, f"{count} {label} device(s) found")
    return CheckResult("fail", name, f"no {label} devices found")


def _device_channels(device: Mapping[str, Any], key: str) -> int:
    try:
        return int(device.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _check_kokoro_cache(options: DoctorOptions, probe: DoctorProbe) -> CheckResult:
    cache_dir = options.cache_dir / "kokoro"
    missing = [
        filename
        for filename in KOKORO_MODEL_FILES
        if not probe.path_exists(cache_dir / filename)
    ]
    if not missing:
        return CheckResult("ok", "kokoro model cache", str(cache_dir))
    return CheckResult(
        "warn",
        "kokoro model cache",
        "missing "
        + ", ".join(missing)
        + "; runtime will download these assets on first use",
    )


def _check_deep_readiness(options: DoctorOptions) -> CheckResult:
    return CheckResult(
        "warn",
        "deep runtime checks",
        f"model loading/playback checks are not automated yet; whisper={options.whisper_model}",
    )


def _print_audio_devices(
    devices: Sequence[Mapping[str, Any]],
    output: TextIO,
) -> None:
    print("Audio devices:", file=output)
    if not devices:
        print("  (none)", file=output)
        return

    for index, device in enumerate(devices):
        name = device.get("name", f"device {index}")
        input_channels = _device_channels(device, "max_input_channels")
        output_channels = _device_channels(device, "max_output_channels")
        print(
            f"  {index}: {name} "
            f"(in={input_channels}, out={output_channels})",
            file=output,
        )
