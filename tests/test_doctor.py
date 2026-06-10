from pathlib import Path

from agent_voice.doctor import CommandResult, DoctorOptions, DoctorProbe, run_doctor
from agent_voice.providers import KOKORO_MODEL_MIN_BYTES, KOKORO_VOICES_MIN_BYTES


class FakeProbe(DoctorProbe):
    def __init__(
        self,
        *,
        commands=None,
        packages=None,
        devices=None,
        existing_paths=None,
        path_sizes=None,
        default_devices=None,
        command_results=None,
    ):
        self.commands = commands or {}
        self.packages = packages or set()
        self.devices = devices or []
        self.existing_paths = {Path(path) for path in (existing_paths or set())}
        self.path_sizes = {Path(path): size for path, size in (path_sizes or {}).items()}
        self.default_devices = default_devices or {}
        self.command_results = {
            tuple(command): result
            for command, result in (command_results or {}).items()
        }

    def find_command(self, command):
        return self.commands.get(command)

    def has_package(self, package):
        return package in self.packages

    def query_audio_devices(self):
        return self.devices

    def path_exists(self, path):
        return Path(path) in self.existing_paths

    def path_size(self, path):
        return self.path_sizes.get(Path(path), 0)

    def default_audio_device(self, kind):
        return self.default_devices.get(kind)

    def run_command(self, command, *, timeout_seconds):
        return self.command_results.get(tuple(command), CommandResult(127))


def required_packages():
    return {
        "pexpect",
        "faster_whisper",
        "kokoro_onnx",
        "pipecat",
        "sounddevice",
        "supertonic",
        "livekit",
    }


def test_doctor_returns_success_when_required_runtime_checks_pass(capsys):
    cache_dir = Path("/tmp/cache")
    model_path = cache_dir / "kokoro" / "kokoro-v1.0.onnx"
    voices_path = cache_dir / "kokoro" / "voices-v1.0.bin"
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages=required_packages(),
        devices=[
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ],
        existing_paths={model_path, voices_path},
        path_sizes={
            model_path: KOKORO_MODEL_MIN_BYTES,
            voices_path: KOKORO_VOICES_MIN_BYTES,
        },
    )

    exit_code = run_doctor(
        DoctorOptions(agent="codex", cache_dir=cache_dir),
        probe=probe,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[ok] command codex" in output
    assert "[ok] audio input devices" in output
    assert "[ok] kokoro model cache" in output
    assert "Summary:" in output


def test_doctor_returns_failure_when_agent_or_audio_is_missing(capsys):
    probe = FakeProbe(
        commands={},
        packages=required_packages(),
        devices=[],
    )

    exit_code = run_doctor(
        DoctorOptions(agent="pi", cache_dir=Path("/tmp/cache")),
        probe=probe,
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "[fail] command pi" in output
    assert "[fail] audio input devices" in output
    assert "[fail] audio output devices" in output
    assert "[warn] kokoro model cache" in output


def test_doctor_can_list_audio_devices(capsys):
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages=required_packages(),
        devices=[
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ],
    )

    exit_code = run_doctor(
        DoctorOptions(agent="codex", list_devices=True),
        probe=probe,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Audio devices:" in output
    assert "Mic" in output
    assert "Speaker" in output


def test_doctor_marks_default_audio_devices(capsys):
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages=required_packages(),
        devices=[
            {"name": "iPhone Microphone", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "WH-1000XM5", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "WH-1000XM5", "max_input_channels": 0, "max_output_channels": 2},
        ],
        default_devices={"input": 1, "output": 2},
    )

    exit_code = run_doctor(
        DoctorOptions(agent="codex", list_devices=True),
        probe=probe,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "1: WH-1000XM5 (in=1, out=0) [default input]" in output
    assert "2: WH-1000XM5 (in=0, out=2) [default output]" in output


def test_doctor_warns_when_kokoro_cache_files_are_too_small(capsys):
    cache_dir = Path("/tmp/cache")
    model_path = cache_dir / "kokoro" / "kokoro-v1.0.onnx"
    voices_path = cache_dir / "kokoro" / "voices-v1.0.bin"
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages=required_packages(),
        devices=[
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ],
        existing_paths={model_path, voices_path},
        path_sizes={
            model_path: 2048,
            voices_path: 2048,
        },
    )

    exit_code = run_doctor(
        DoctorOptions(agent="codex", cache_dir=cache_dir),
        probe=probe,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[warn] kokoro model cache" in output
    assert "too small" in output


def test_doctor_checks_codex_companion_capabilities(capsys):
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages=required_packages(),
        devices=[
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ],
        command_results={
            ("codex", "app-server", "--help"): CommandResult(
                0,
                stdout="Usage: codex app-server --listen <url>",
            ),
            ("codex", "resume", "--help"): CommandResult(
                0,
                stdout="Usage: codex resume <id> --remote <url> --no-alt-screen",
            ),
        },
    )

    exit_code = run_doctor(
        DoctorOptions(agent="codex", companion_codex=True),
        probe=probe,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[ok] codex app-server" in output
    assert "[ok] codex remote resume" in output


def test_doctor_fails_companion_check_when_remote_resume_is_too_old(capsys):
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages=required_packages(),
        devices=[
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ],
        command_results={
            ("codex", "app-server", "--help"): CommandResult(0),
            ("codex", "resume", "--help"): CommandResult(
                0,
                stdout="Usage: codex resume <id>",
            ),
        },
    )

    exit_code = run_doctor(
        DoctorOptions(agent="codex", companion_codex=True),
        probe=probe,
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "[fail] codex remote resume" in output
    assert "missing expected option(s): --remote, --no-alt-screen" in output
