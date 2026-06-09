from pathlib import Path

from agent_voice.doctor import DoctorOptions, DoctorProbe, run_doctor
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
    ):
        self.commands = commands or {}
        self.packages = packages or set()
        self.devices = devices or []
        self.existing_paths = {Path(path) for path in (existing_paths or set())}
        self.path_sizes = {Path(path): size for path, size in (path_sizes or {}).items()}

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


def test_doctor_returns_success_when_required_runtime_checks_pass(capsys):
    cache_dir = Path("/tmp/cache")
    model_path = cache_dir / "kokoro" / "kokoro-v1.0.onnx"
    voices_path = cache_dir / "kokoro" / "voices-v1.0.bin"
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages={"pexpect", "faster_whisper", "kokoro_onnx", "pipecat", "sounddevice"},
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
        packages={"pexpect", "faster_whisper", "kokoro_onnx", "pipecat", "sounddevice"},
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
        packages={"pexpect", "faster_whisper", "kokoro_onnx", "pipecat", "sounddevice"},
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


def test_doctor_warns_when_kokoro_cache_files_are_too_small(capsys):
    cache_dir = Path("/tmp/cache")
    model_path = cache_dir / "kokoro" / "kokoro-v1.0.onnx"
    voices_path = cache_dir / "kokoro" / "voices-v1.0.bin"
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages={"pexpect", "faster_whisper", "kokoro_onnx", "pipecat", "sounddevice"},
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
