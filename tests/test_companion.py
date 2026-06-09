from __future__ import annotations

from pathlib import Path

from agent_voice.companion import CodexTuiCompanionConfig, run_codex_tui_companion


class FakeProcess:
    def __init__(self, command):
        self.command = command
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def test_companion_runs_codex_tui_foreground_and_hides_voice_logs(tmp_path):
    started = []
    foreground = []
    waited_ports = []

    def fake_process_factory(command, **kwargs):
        process = FakeProcess(command)
        started.append((command, kwargs, process))
        return process

    def fake_foreground_runner(command, **kwargs):
        foreground.append((command, kwargs))
        return 7

    exit_code = run_codex_tui_companion(
        CodexTuiCompanionConfig(
            port=4567,
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            voice_args=("--whisper-model", "base"),
            codex_args=("--model", "gpt-5.4"),
        ),
        process_factory=fake_process_factory,
        foreground_runner=fake_foreground_runner,
        wait_ready=lambda port: waited_ports.append(port),
        create_thread=lambda url, cwd: "thread-123",
    )

    assert exit_code == 7
    assert waited_ports == [4567]
    assert started[0][0] == [
        "codex",
        "app-server",
        "--listen",
        "ws://127.0.0.1:4567",
    ]
    assert started[1][0][:4] == [
        started[1][0][0],
        "-m",
        "agent_voice.cli",
        "--agent-backend",
    ]
    assert "--no-keyboard" in started[1][0]
    assert "--codex-thread-id" in started[1][0]
    assert "thread-123" in started[1][0]
    assert "--whisper-model" in started[1][0]
    assert foreground == [
        (
            [
                "codex",
                "resume",
                "thread-123",
                "--remote",
                "ws://127.0.0.1:4567",
                "--no-alt-screen",
                "--model",
                "gpt-5.4",
            ],
            {"cwd": str(tmp_path)},
        )
    ]
    assert (tmp_path / "logs" / "codex-app-server.log").exists()
    assert (tmp_path / "logs" / "agent-voice.log").exists()
    assert all(process.terminated for *_rest, process in started)


def test_companion_uses_existing_thread_without_creating_starter(tmp_path):
    started = []
    created = []

    def fake_process_factory(command, **kwargs):
        process = FakeProcess(command)
        started.append((command, kwargs, process))
        return process

    exit_code = run_codex_tui_companion(
        CodexTuiCompanionConfig(
            port=4568,
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            thread_id="existing-thread",
        ),
        process_factory=fake_process_factory,
        foreground_runner=lambda *_args, **_kwargs: 0,
        wait_ready=lambda _port: None,
        create_thread=lambda *_args: created.append("created") or "new-thread",
    )

    assert exit_code == 0
    assert created == []
    assert "existing-thread" in started[1][0]


def test_companion_defaults_log_dir_under_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    started = []

    def fake_process_factory(command, **kwargs):
        process = FakeProcess(command)
        started.append((command, kwargs, process))
        return process

    run_codex_tui_companion(
        CodexTuiCompanionConfig(port=4569),
        process_factory=fake_process_factory,
        foreground_runner=lambda *_args, **_kwargs: 0,
        wait_ready=lambda _port: None,
        create_thread=lambda *_args: "thread-123",
    )

    assert Path(".cache/agent-voice/companion/codex-app-server.log").exists()
    assert Path(".cache/agent-voice/companion/agent-voice.log").exists()
