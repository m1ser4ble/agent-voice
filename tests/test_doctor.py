from pathlib import Path

from agent_voice.doctor import DoctorOptions, DoctorProbe, run_doctor


class FakeProbe(DoctorProbe):
    def __init__(
        self,
        *,
        commands=None,
        packages=None,
        devices=None,
        existing_paths=None,
    ):
        self.commands = commands or {}
        self.packages = packages or set()
        self.devices = devices or []
        self.existing_paths = {Path(path) for path in (existing_paths or set())}

    def find_command(self, command):
        return self.commands.get(command)

    def has_package(self, package):
        return package in self.packages

    def query_audio_devices(self):
        return self.devices

    def path_exists(self, path):
        return Path(path) in self.existing_paths


def test_doctor_returns_success_when_required_runtime_checks_pass(capsys):
    cache_dir = Path("/tmp/cache")
    probe = FakeProbe(
        commands={"codex": "/usr/bin/codex"},
        packages={"pexpect", "faster_whisper", "kokoro_onnx", "pipecat", "sounddevice"},
        devices=[
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ],
        existing_paths={
            cache_dir / "kokoro" / "kokoro-v1.0.onnx",
            cache_dir / "kokoro" / "voices-v1.0.bin",
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
