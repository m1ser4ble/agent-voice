from io import StringIO
from pathlib import Path

from agent_voice.doctor import CommandResult, DoctorProbe
from agent_voice.cli import _collect_agent_output, _parse_audio_device, main


class FakeAgent:
    def __init__(self):
        self.submitted = []
        self.chunks = ["Modified:\n- auth.py\n\nTests:\n1 passed\n", "", "", "", ""]
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1
        return None

    def submit(self, text):
        self.submitted.append(text)

    def read_available(self):
        if not self.chunks:
            return ""
        return self.chunks.pop(0)

    def stop(self):
        self.stops += 1
        return None


class FakeVoiceLoop:
    def __init__(self):
        self.runs = 0

    def run_forever(self):
        self.runs += 1
        return 0


class FakeDoctorProbe(DoctorProbe):
    def __init__(self):
        self.commands = {"codex": "/usr/bin/codex", "pi": "/usr/bin/pi"}

    def find_command(self, command):
        return self.commands.get(command)

    def has_package(self, package):
        return package in {
            "pexpect",
            "faster_whisper",
            "kokoro_onnx",
            "pipecat",
            "sounddevice",
            "supertonic",
            "livekit",
        }

    def query_audio_devices(self):
        return [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]

    def path_exists(self, path):
        return Path(path).name in {"kokoro-v1.0.onnx", "voices-v1.0.bin"}

    def path_size(self, path):
        if Path(path).name == "kokoro-v1.0.onnx":
            return 50 * 1024 * 1024
        if Path(path).name == "voices-v1.0.bin":
            return 1 * 1024 * 1024
        return 0

    def default_audio_device(self, kind):
        return None

    def run_command(self, command, *, timeout_seconds):
        if tuple(command) == ("codex", "app-server", "--help"):
            return CommandResult(0, stdout="Usage: codex app-server --listen <url>")
        if tuple(command) == ("codex", "resume", "--help"):
            return CommandResult(
                0,
                stdout="Usage: codex resume <id> --remote <url> --no-alt-screen",
            )
        return CommandResult(0)


def test_cli_codex_once_sends_command_and_prints_voice_summary():
    agent = FakeAgent()
    output = StringIO()

    exit_code = main(
        ["--text", "--once", "auth 버그 고쳐", "--poll-interval", "0", "codex"],
        agent_factory=lambda _: agent,
        output=output,
    )

    assert exit_code == 0
    assert agent.submitted == ["auth 버그 고쳐"]
    assert "파일 1개를 수정했고, 테스트 1개는 모두 통과했습니다." in output.getvalue()


def test_cli_codex_text_loop_keeps_one_agent_session_for_multiple_commands(monkeypatch):
    agent = FakeAgent()
    agent.chunks = [
        "Modified:\n- auth.py\n\nTests:\n1 passed\n",
        "",
        "Modified:\n- login.py\n\nTests:\n2 passed\n",
        "",
    ]
    output = StringIO()
    commands = iter(["auth 버그 고쳐", "테스트는?"])

    def fake_input(_prompt):
        try:
            return next(commands)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    exit_code = main(
        ["--text", "--idle-reads", "1", "--poll-interval", "0", "codex"],
        agent_factory=lambda _: agent,
        output=output,
    )

    assert exit_code == 0
    assert agent.starts == 1
    assert agent.stops == 1
    assert agent.submitted == ["auth 버그 고쳐", "테스트는?"]
    assert "테스트 1개는 모두 통과했습니다." in output.getvalue()
    assert "테스트 2개는 모두 통과했습니다." in output.getvalue()


def test_cli_codex_defaults_to_voice_mode_with_passthrough_agent_args():
    agent = FakeAgent()
    voice_loop = FakeVoiceLoop()
    output = StringIO()
    captured_commands = []

    exit_code = main(
        ["codex", "resume", "--model", "gpt-5"],
        agent_factory=lambda _: agent,
        voice_loop_factory=lambda command, _: (
            captured_commands.append(command) or voice_loop
        ),
        output=output,
    )

    assert exit_code == 0
    assert agent.starts == 0
    assert voice_loop.runs == 1
    assert captured_commands == [("codex", "resume", "--model", "gpt-5")]


def test_cli_voice_mode_applies_default_voice_preset():
    voice_loop = FakeVoiceLoop()
    captured_settings = []

    exit_code = main(
        ["codex"],
        voice_loop_factory=lambda _, args: (
            captured_settings.append(
                (
                    args.tts_voice,
                    args.supertonic_voice,
                    args.tts_lang,
                    args.tts_speed,
                    args.stt_language,
                )
            )
            or voice_loop
        ),
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured_settings == [("am_michael", "M3", "ko", 0.9, "ko")]


def test_cli_voice_mode_accepts_audio_device_selection():
    voice_loop = FakeVoiceLoop()
    captured_devices = []

    exit_code = main(
        ["--input-device", "2", "--output-device", "USB Speaker", "pi"],
        voice_loop_factory=lambda _, args: (
            captured_devices.append((args.input_device, args.output_device)) or voice_loop
        ),
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured_devices == [("2", "USB Speaker")]


def test_cli_voice_mode_accepts_keyboard_and_transparency_controls():
    voice_loop = FakeVoiceLoop()
    captured_flags = []

    exit_code = main(
        ["--no-keyboard", "--quiet-agent-io", "--no-aec", "--aec-delay-ms", "120", "codex"],
        voice_loop_factory=lambda _, args: (
            captured_flags.append(
                (
                    args.no_keyboard,
                    args.quiet_agent_io,
                    args.no_aec,
                    args.aec_delay_ms,
                )
            )
            or voice_loop
        ),
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured_flags == [(True, True, True, 120)]


def test_cli_voice_mode_accepts_tts_backend_controls():
    voice_loop = FakeVoiceLoop()
    captured_settings = []

    exit_code = main(
        [
            "--tts-backend",
            "macos-say",
            "--macos-say-voice",
            "Yuna",
            "--macos-say-rate",
            "185",
            "--supertonic-voice",
            "F2",
            "pi",
        ],
        voice_loop_factory=lambda _, args: (
            captured_settings.append(
                (
                    args.tts_backend,
                    args.macos_say_voice,
                    args.macos_say_rate,
                    args.supertonic_voice,
                )
            )
            or voice_loop
        ),
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured_settings == [("macos-say", "Yuna", 185, "F2")]


def test_audio_device_parser_accepts_index_or_name():
    assert _parse_audio_device("2") == 2
    assert _parse_audio_device("USB Microphone") == "USB Microphone"
    assert _parse_audio_device(None) is None


def test_cli_tts_overrides_take_precedence_over_voice_preset():
    voice_loop = FakeVoiceLoop()
    captured_settings = []

    exit_code = main(
        [
            "--voice-preset",
            "high_quality",
            "--tts-voice",
            "af_bella",
            "--tts-lang",
            "en-us",
            "--tts-speed",
            "1.05",
            "codex",
        ],
        voice_loop_factory=lambda _, args: (
            captured_settings.append((args.tts_voice, args.tts_lang, args.tts_speed))
            or voice_loop
        ),
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured_settings == [("af_bella", "en-us", 1.05)]


def test_cli_voice_config_file_can_change_default_preset(tmp_path):
    config_path = tmp_path / "voice-presets.toml"
    config_path.write_text(
        """
        [defaults]
        preset = "workstation"

        [presets.workstation]
        voice = "bm_george"
        lang = "en-gb"
        speed = 0.9
        """,
        encoding="utf-8",
    )
    voice_loop = FakeVoiceLoop()
    captured_settings = []

    exit_code = main(
        ["--voice-config", str(config_path), "codex"],
        voice_loop_factory=lambda _, args: (
            captured_settings.append((args.tts_voice, args.tts_lang, args.tts_speed))
            or voice_loop
        ),
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured_settings == [("bm_george", "en-gb", 0.9)]


def test_cli_unknown_voice_preset_fails_before_starting_voice_loop():
    voice_loop = FakeVoiceLoop()
    output = StringIO()

    exit_code = main(
        ["--voice-preset", "missing", "codex"],
        voice_loop_factory=lambda *_: voice_loop,
        output=output,
    )

    assert exit_code == 2
    assert voice_loop.runs == 0
    assert "unknown voice preset 'missing'" in output.getvalue()


def test_cli_doctor_runs_runtime_checks_instead_of_agent_target():
    agent = FakeAgent()
    voice_loop = FakeVoiceLoop()
    output = StringIO()

    exit_code = main(
        ["doctor", "--agent", "pi", "--list-devices"],
        agent_factory=lambda _: agent,
        voice_loop_factory=lambda *_: voice_loop,
        doctor_probe=FakeDoctorProbe(),
        output=output,
    )

    assert exit_code == 0
    assert agent.starts == 0
    assert voice_loop.runs == 0
    assert "[ok] command pi" in output.getvalue()
    assert "Audio devices:" in output.getvalue()


def test_cli_doctor_accepts_companion_codex_check():
    output = StringIO()

    exit_code = main(
        ["doctor", "--agent", "codex", "--companion-codex"],
        doctor_probe=FakeDoctorProbe(),
        output=output,
    )

    assert exit_code == 0
    assert "[ok] codex app-server" in output.getvalue()
    assert "[ok] codex remote resume" in output.getvalue()


def test_cli_companion_codex_runs_foreground_tui_wrapper(tmp_path):
    captured = []

    exit_code = main(
        [
            "--whisper-model",
            "base",
            "--codex-app-server-port",
            "4567",
            "--codex-thread-id",
            "thread-123",
            "--companion-log-dir",
            str(tmp_path / "logs"),
            "companion",
            "codex",
            "--",
            "--model",
            "gpt-5.4",
        ],
        companion_runner=lambda config: captured.append(config) or 0,
        output=StringIO(),
    )

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0].port == 4567
    assert captured[0].thread_id == "thread-123"
    assert captured[0].log_dir == tmp_path / "logs"
    assert captured[0].voice_args == ("--whisper-model", "base")
    assert captured[0].codex_args == ("--model", "gpt-5.4")


def test_cli_companion_forwards_whisper_cpp_voice_args(tmp_path):
    captured = []

    exit_code = main(
        [
            "--stt-backend",
            "whisper-cpp",
            "--whisper-model",
            "/models/ggml-large-v3-q5_0.bin",
            "--whisper-cpp-executable",
            "/opt/homebrew/bin/whisper-cli",
            "--codex-app-server-port",
            "4567",
            "companion",
            "codex",
        ],
        companion_runner=lambda config: captured.append(config) or 0,
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured[0].voice_args == (
        "--whisper-model",
        "/models/ggml-large-v3-q5_0.bin",
        "--stt-backend",
        "whisper-cpp",
        "--whisper-cpp-executable",
        "/opt/homebrew/bin/whisper-cli",
    )


def test_cli_companion_forwards_debug_audio_recording_args(tmp_path):
    captured = []
    debug_dir = tmp_path / "debug-audio"

    exit_code = main(
        [
            "--record-debug-audio",
            "--debug-audio-dir",
            str(debug_dir),
            "companion",
            "codex",
        ],
        companion_runner=lambda config: captured.append(config) or 0,
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured[0].voice_args == (
        "--record-debug-audio",
        "--debug-audio-dir",
        str(debug_dir),
    )


def test_cli_companion_forwards_non_default_assistant_style():
    captured = []

    exit_code = main(
        [
            "--assistant-style",
            "none",
            "companion",
            "codex",
        ],
        companion_runner=lambda config: captured.append(config) or 0,
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured[0].voice_args == ("--assistant-style", "none")


def test_cli_companion_codex_resume_uses_thread_id_as_user_expects():
    captured = []

    exit_code = main(
        [
            "--whisper-model",
            "base",
            "companion",
            "codex",
            "resume",
            "thread-123",
            "--",
            "--model",
            "gpt-5.4",
        ],
        companion_runner=lambda config: captured.append(config) or 0,
        output=StringIO(),
    )

    assert exit_code == 0
    assert captured[0].thread_id == "thread-123"
    assert captured[0].voice_args == ("--whisper-model", "base")
    assert captured[0].codex_args == ("--model", "gpt-5.4")


def test_cli_companion_rejects_conflicting_resume_thread_ids():
    output = StringIO()

    exit_code = main(
        [
            "--codex-thread-id",
            "thread-a",
            "companion",
            "codex",
            "resume",
            "thread-b",
        ],
        companion_runner=lambda _config: 0,
        output=output,
    )

    assert exit_code == 2
    assert "conflicting Codex thread ids" in output.getvalue()


def test_cli_companion_requires_codex_subtarget():
    output = StringIO()

    exit_code = main(
        ["companion", "pi"],
        companion_runner=lambda _config: 0,
        output=output,
    )

    assert exit_code == 2
    assert "usage: agent-voice companion codex" in output.getvalue()


def test_cli_codex_once_requires_text_mode():
    agent = FakeAgent()
    output = StringIO()

    exit_code = main(
        ["--once", "auth 버그 고쳐", "codex"],
        agent_factory=lambda _: agent,
        output=output,
    )

    assert exit_code == 2
    assert agent.starts == 0
    assert "use --text --once" in output.getvalue()


def test_cli_text_once_uses_voice_loop_exit_semantics():
    agent = FakeAgent()

    exit_code = main(
        ["--text", "--once", "종료", "codex"],
        agent_factory=lambda _: agent,
        output=StringIO(),
    )

    assert exit_code == 0
    assert agent.submitted == []


def test_cli_pi_text_once_uses_pi_target_with_passthrough_args():
    agent = FakeAgent()
    output = StringIO()
    captured_commands = []

    exit_code = main(
        ["--text", "--once", "테스트는?", "pi", "-c"],
        agent_factory=lambda command: captured_commands.append(command) or agent,
        output=output,
    )

    assert exit_code == 0
    assert captured_commands == [("pi", "-c")]
    assert agent.submitted == ["테스트는?"]


def test_cli_codex_app_server_backend_uses_structured_codex_agent(monkeypatch):
    created = []

    class FakeCodexAppServerAgent(FakeAgent):
        def __init__(self, command, **kwargs):
            super().__init__()
            created.append((command, kwargs))

    monkeypatch.setattr(
        "agent_voice.cli.CodexAppServerAgent",
        FakeCodexAppServerAgent,
    )

    exit_code = main(
        [
            "--agent-backend",
            "codex-app-server",
            "--text",
            "--once",
            "테스트는?",
            "codex",
        ],
        output=StringIO(),
    )

    assert exit_code == 0
    assert created[0][0] == ("codex",)
    assert "sir" in created[0][1]["developer_instructions"].casefold()


def test_cli_assistant_style_none_disables_developer_instructions(monkeypatch):
    created = []

    class FakeCodexRemoteAppServerAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(kwargs)

    monkeypatch.setattr(
        "agent_voice.cli.CodexRemoteAppServerAgent",
        FakeCodexRemoteAppServerAgent,
    )

    exit_code = main(
        [
            "--agent-backend",
            "codex-remote-app-server",
            "--codex-app-server-port",
            "4500",
            "--assistant-style",
            "none",
            "--text",
            "--once",
            "테스트는?",
            "codex",
        ],
        output=StringIO(),
    )

    assert exit_code == 0
    assert "developer_instructions" not in created[0]


def test_cli_codex_remote_backend_requires_url_or_port():
    output = StringIO()

    exit_code = main(
        [
            "--agent-backend",
            "codex-remote-app-server",
            "--text",
            "--once",
            "테스트는?",
            "codex",
        ],
        output=output,
    )

    assert exit_code == 2
    assert "requires --codex-app-server-url or --codex-app-server-port" in output.getvalue()


def test_cli_codex_remote_backend_uses_configured_url_and_thread(monkeypatch):
    created = []

    class FakeCodexRemoteAppServerAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(kwargs)

    monkeypatch.setattr(
        "agent_voice.cli.CodexRemoteAppServerAgent",
        FakeCodexRemoteAppServerAgent,
    )

    exit_code = main(
        [
            "--agent-backend",
            "codex-remote-app-server",
            "--codex-app-server-url",
            "ws://127.0.0.1:4500",
            "--codex-thread-id",
            "thread-123",
            "--text",
            "--once",
            "테스트는?",
            "codex",
        ],
        output=StringIO(),
    )

    assert exit_code == 0
    assert created[0]["url"] == "ws://127.0.0.1:4500"
    assert created[0]["thread_id"] == "thread-123"
    assert created[0]["cwd"] is None
    assert "sir" in created[0]["developer_instructions"].casefold()


def test_cli_codex_remote_backend_uses_debug_agent_event_log(monkeypatch, tmp_path):
    created = []
    log_path = tmp_path / "events.jsonl"

    class FakeCodexRemoteAppServerAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(kwargs)

    monkeypatch.setattr(
        "agent_voice.cli.CodexRemoteAppServerAgent",
        FakeCodexRemoteAppServerAgent,
    )

    exit_code = main(
        [
            "--agent-backend",
            "codex-remote-app-server",
            "--codex-app-server-port",
            "4500",
            "--debug-agent-events",
            str(log_path),
            "--text",
            "--once",
            "테스트는?",
            "codex",
        ],
        output=StringIO(),
    )

    assert exit_code == 0
    assert created[0]["event_logger"].path == log_path


def test_cli_codex_remote_backend_builds_url_from_port(monkeypatch):
    created = []

    class FakeCodexRemoteAppServerAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(kwargs)

    monkeypatch.setattr(
        "agent_voice.cli.CodexRemoteAppServerAgent",
        FakeCodexRemoteAppServerAgent,
    )

    exit_code = main(
        [
            "--agent-backend",
            "codex-remote-app-server",
            "--codex-app-server-port",
            "4501",
            "--text",
            "--once",
            "테스트는?",
            "codex",
        ],
        output=StringIO(),
    )

    assert exit_code == 0
    assert created[0]["url"] == "ws://127.0.0.1:4501"


def test_cli_agent_options_after_target_are_not_parsed_as_product_options():
    voice_loop = FakeVoiceLoop()
    captured_commands = []

    exit_code = main(
        ["codex", "--text", "--once", "agent side option"],
        voice_loop_factory=lambda command, _: (
            captured_commands.append(command) or voice_loop
        ),
        output=StringIO(),
    )

    assert exit_code == 0
    assert voice_loop.runs == 1
    assert captured_commands == [("codex", "--text", "--once", "agent side option")]


class StreamingFakeAgent:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def read_available(self):
        if not self.chunks:
            return ""
        return self.chunks.pop(0)


def test_collect_agent_output_waits_for_first_output_then_stops_after_idle_reads():
    agent = StreamingFakeAgent(
        [
            "",
            "Modified:\n- auth.py\n",
            "Tests:\n1 passed\n",
            "",
            "",
            "late output that should not be read",
        ]
    )

    output = _collect_agent_output(
        agent,
        idle_reads=2,
        max_reads=10,
        poll_interval=0,
    )

    assert output == "Modified:\n- auth.py\nTests:\n1 passed\n"


class ActiveTurnFakeAgent:
    def __init__(self):
        self.reads = 0
        self.chunks = [
            "Modified:\n- auth.py\n",
            "",
            "",
            "Tests:\n1 passed\n",
            "",
            "",
        ]

    def read_available(self):
        self.reads += 1
        if not self.chunks:
            return ""
        return self.chunks.pop(0)

    def is_turn_active(self):
        return self.reads < 4


def test_collect_agent_output_waits_through_idle_reads_while_turn_is_active():
    agent = ActiveTurnFakeAgent()

    output = _collect_agent_output(
        agent,
        idle_reads=2,
        max_reads=10,
        poll_interval=0,
    )

    assert output == "Modified:\n- auth.py\nTests:\n1 passed\n"
