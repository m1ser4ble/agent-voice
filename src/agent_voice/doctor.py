from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from agent_voice.providers import KOKORO_MODEL_MIN_BYTES, KOKORO_VOICES_MIN_BYTES


@dataclass(frozen=True)
class DoctorOptions:
    agent: str = "codex"
    cache_dir: Path = Path(".cache/agent-voice")
    list_devices: bool = False
    deep: bool = False
    whisper_model: str = "tiny"
    companion_codex: bool = False


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class DoctorProbe(Protocol):
    def find_command(self, command: str) -> str | None:
        """Return the executable path for a command, if available."""

    def has_package(self, package: str) -> bool:
        """Return whether an import package is available in this environment."""

    def query_audio_devices(self) -> Sequence[Mapping[str, Any]]:
        """Return sounddevice-style audio device dictionaries."""

    def path_exists(self, path: Path) -> bool:
        """Return whether a runtime asset path exists."""

    def path_size(self, path: Path) -> int:
        """Return the runtime asset size in bytes, or 0 when unavailable."""

    def default_audio_device(self, kind: str) -> int | None:
        """Return the default sounddevice index for 'input' or 'output'."""

    def run_command(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> CommandResult:
        """Run a short read-only command for capability checks."""


class SystemDoctorProbe:
    def find_command(self, command: str) -> str | None:
        return shutil.which(command)

    def has_package(self, package: str) -> bool:
        return importlib.util.find_spec(package) is not None

    def query_audio_devices(self) -> Sequence[Mapping[str, Any]]:
        import sounddevice as sd

        devices = sd.query_devices()
        return list(devices)

    def default_audio_device(self, kind: str) -> int | None:
        import sounddevice as sd

        try:
            default_device = sd.query_devices(kind=kind)
            devices = list(sd.query_devices())
        except Exception:
            return None
        return _match_audio_device_index(devices, default_device, kind)

    def path_exists(self, path: Path) -> bool:
        return path.exists()

    def path_size(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def run_command(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> CommandResult:
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return CommandResult(
                124,
                stdout=error.stdout or "",
                stderr=error.stderr or "command timed out",
            )
        except OSError as error:
            return CommandResult(127, stderr=str(error))
        return CommandResult(
            completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


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
    ("supertonic", "supertonic"),
    ("livekit", "livekit"),
)

KOKORO_MODEL_FILES: tuple[tuple[str, int], ...] = (
    ("kokoro-v1.0.onnx", KOKORO_MODEL_MIN_BYTES),
    ("voices-v1.0.bin", KOKORO_VOICES_MIN_BYTES),
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
    parser.add_argument(
        "--companion-codex",
        action="store_true",
        help=(
            "Check Codex app-server and remote TUI capabilities needed by "
            "`agent-voice companion codex`."
        ),
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
    if options.companion_codex:
        results.extend(_check_codex_companion(options, probe))
    devices, audio_results = _check_audio_devices(probe)
    results.extend(audio_results)
    results.append(_check_kokoro_cache(options, probe))
    if options.deep:
        results.append(_check_deep_readiness(options))

    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}", file=output)

    if options.list_devices:
        _print_audio_devices(
            devices,
            output,
            default_input=probe.default_audio_device("input"),
            default_output=probe.default_audio_device("output"),
        )

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


def _check_codex_companion(
    options: DoctorOptions,
    probe: DoctorProbe,
) -> list[CheckResult]:
    if options.agent not in {"codex", "none"}:
        return [
            CheckResult(
                "fail",
                "companion codex",
                "--companion-codex requires --agent codex or --agent none",
            )
        ]

    if probe.find_command("codex") is None:
        return [
            CheckResult(
                "fail",
                "companion codex",
                "codex command not found on PATH",
            )
        ]

    return [
        _check_codex_app_server_help(probe),
        _check_codex_remote_resume_help(probe),
    ]


def _check_codex_app_server_help(probe: DoctorProbe) -> CheckResult:
    result = probe.run_command(
        ("codex", "app-server", "--help"),
        timeout_seconds=5.0,
    )
    if result.returncode == 0:
        return CheckResult(
            "ok",
            "codex app-server",
            "`codex app-server --help` succeeded",
        )
    return CheckResult(
        "fail",
        "codex app-server",
        _command_failure_detail("codex app-server --help", result),
    )


def _check_codex_remote_resume_help(probe: DoctorProbe) -> CheckResult:
    result = probe.run_command(
        ("codex", "resume", "--help"),
        timeout_seconds=5.0,
    )
    if result.returncode != 0:
        return CheckResult(
            "fail",
            "codex remote resume",
            _command_failure_detail("codex resume --help", result),
        )

    help_text = f"{result.stdout}\n{result.stderr}"
    missing = [
        flag
        for flag in ("--remote", "--no-alt-screen")
        if flag not in help_text
    ]
    if not missing:
        return CheckResult(
            "ok",
            "codex remote resume",
            "`codex resume --help` exposes --remote and --no-alt-screen",
        )
    return CheckResult(
        "fail",
        "codex remote resume",
        "missing expected option(s): " + ", ".join(missing),
    )


def _command_failure_detail(command: str, result: CommandResult) -> str:
    detail = (result.stderr or result.stdout).strip()
    if detail:
        first_line = detail.splitlines()[0]
        return f"`{command}` exited {result.returncode}: {first_line}"
    return f"`{command}` exited {result.returncode}"


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
    problems: list[str] = []
    for filename, min_bytes in KOKORO_MODEL_FILES:
        path = cache_dir / filename
        if not probe.path_exists(path):
            problems.append(f"missing {filename}")
            continue
        size = probe.path_size(path)
        if size < min_bytes:
            problems.append(
                f"{filename} is too small ({size} bytes; expected at least {min_bytes})"
            )

    if not problems:
        return CheckResult("ok", "kokoro model cache", str(cache_dir))
    return CheckResult(
        "warn",
        "kokoro model cache",
        "; ".join(problems) + "; runtime will re-download these assets on first use",
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
    *,
    default_input: int | None = None,
    default_output: int | None = None,
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
            f"(in={input_channels}, out={output_channels})"
            f"{_default_device_label(index, default_input, default_output)}",
            file=output,
        )


def _default_device_label(
    index: int,
    default_input: int | None,
    default_output: int | None,
) -> str:
    labels: list[str] = []
    if index == default_input:
        labels.append("default input")
    if index == default_output:
        labels.append("default output")
    if not labels:
        return ""
    return " [" + ", ".join(labels) + "]"


def _match_audio_device_index(
    devices: Sequence[Mapping[str, Any]],
    default_device: Mapping[str, Any],
    kind: str,
) -> int | None:
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    default_name = default_device.get("name")
    for index, device in enumerate(devices):
        if device.get("name") != default_name:
            continue
        if _device_channels(device, channel_key) > 0:
            return index
    return None
