from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TextIO

from agent_voice.adapter import (
    Agent,
    CodexAppServerAgent,
    CodexRemoteAppServerAgent,
    JsonlEventLogger,
    PexpectAgent,
)
from agent_voice.assistant_style import (
    ASSISTANT_STYLE_CHOICES,
    DEFAULT_ASSISTANT_STYLE,
    resolve_developer_instructions,
)
from agent_voice.companion import CodexTuiCompanionConfig, run_codex_tui_companion
from agent_voice.doctor import DoctorProbe, build_doctor_parser, run_doctor
from agent_voice.interrupt import VoiceSession
from agent_voice.loop import VoiceLoop
from agent_voice.presenter import VoicePresenter
from agent_voice.voice_config import VoicePresetError, resolve_voice_settings


AgentFactory = Callable[[tuple[str, ...]], Agent]


class VoiceLoopRunner(Protocol):
    def run_forever(self) -> int:
        """Run the voice loop until the user exits."""


VoiceLoopFactory = Callable[[tuple[str, ...], argparse.Namespace], VoiceLoopRunner]
CompanionRunner = Callable[[CodexTuiCompanionConfig], int]


class VoiceModeUnavailableError(RuntimeError):
    pass


class SingleTranscriptSource:
    def __init__(self, transcript: str):
        self._transcript = transcript

    def next_transcript(self) -> str | None:
        transcript = self._transcript
        self._transcript = ""
        return transcript or None


class TextSpeaker:
    def __init__(self, output: TextIO):
        self.output = output

    def say(self, text: str) -> None:
        print(text, file=self.output)

    def stop(self) -> None:
        return None


def main(
    argv: Sequence[str] | None = None,
    *,
    agent_factory: AgentFactory | None = None,
    voice_loop_factory: VoiceLoopFactory | None = None,
    companion_runner: CompanionRunner | None = None,
    doctor_probe: DoctorProbe | None = None,
    output: TextIO | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    output = output or sys.stdout

    if args.target == "doctor":
        return _run_doctor(args, probe=doctor_probe, output=output)

    if args.target == "companion":
        return _run_companion(args, runner=companion_runner, output=output)

    command = _build_agent_command(args.target, args.agent_args)
    return _run_target(
        args,
        command=command,
        agent_factory=agent_factory,
        voice_loop_factory=voice_loop_factory,
        output=output,
    )


def _run_doctor(
    args: argparse.Namespace,
    *,
    probe: DoctorProbe | None,
    output: TextIO,
) -> int:
    parser = build_doctor_parser(
        default_cache_dir=Path(args.cache_dir),
        default_whisper_model=args.whisper_model,
    )
    options = parser.parse_args(args.agent_args)
    return run_doctor(options, probe=probe, output=output)


def _run_companion(
    args: argparse.Namespace,
    *,
    runner: CompanionRunner | None,
    output: TextIO,
) -> int:
    if not args.agent_args or args.agent_args[0] != "codex":
        print(
            "usage: agent-voice companion codex [resume <thread-id>] [-- <codex args>]",
            file=output,
        )
        return 2

    try:
        thread_id, codex_args = _parse_companion_codex_args(
            tuple(args.agent_args[1:]),
            configured_thread_id=args.codex_thread_id,
        )
    except ValueError as error:
        print(str(error), file=output)
        return 2

    config = CodexTuiCompanionConfig(
        port=args.codex_app_server_port,
        url=args.codex_app_server_url,
        thread_id=thread_id,
        cwd=Path.cwd(),
        log_dir=Path(args.companion_log_dir) if args.companion_log_dir else None,
        voice_args=tuple(_companion_voice_args(args)),
        codex_args=codex_args,
    )
    return (runner or run_codex_tui_companion)(config)


def _parse_companion_codex_args(
    codex_args: tuple[str, ...],
    *,
    configured_thread_id: str | None,
) -> tuple[str | None, tuple[str, ...]]:
    thread_id = configured_thread_id
    if codex_args[:1] == ("resume",):
        if len(codex_args) < 2 or codex_args[1] == "--":
            raise ValueError(
                "usage: agent-voice companion codex resume <thread-id> "
                "[-- <codex args>]"
            )
        resume_thread_id = codex_args[1]
        if thread_id is not None and thread_id != resume_thread_id:
            raise ValueError(
                "conflicting Codex thread ids: "
                f"--codex-thread-id {thread_id} and resume {resume_thread_id}"
            )
        thread_id = resume_thread_id
        codex_args = codex_args[2:]

    if codex_args and codex_args[0] == "--":
        codex_args = codex_args[1:]

    return thread_id, codex_args


def _run_target(
    args: argparse.Namespace,
    *,
    command: tuple[str, ...],
    agent_factory: AgentFactory | None,
    voice_loop_factory: VoiceLoopFactory | None,
    output: TextIO,
) -> int:
    if args.once is not None and not args.text:
        print(
            "agent-voice --once is a text-mode smoke path; "
            "use --text --once before the target.",
            file=output,
        )
        return 2

    if not args.text:
        return _run_voice_target(
            args,
            command=command,
            voice_loop_factory=voice_loop_factory,
            output=output,
        )

    factory = agent_factory or (lambda command: _build_agent(command, args))
    try:
        agent = factory(command)
    except ValueError as error:
        print(str(error), file=output)
        return 2
    presenter = VoicePresenter(language=args.language)
    session = VoiceSession()

    agent.start()
    try:
        if args.once is not None:
            _submit_once(
                args.once,
                agent=agent,
                presenter=presenter,
                session=session,
                output=output,
                idle_reads=args.idle_reads,
                max_reads=args.max_reads,
                poll_interval=args.poll_interval,
            )
            return 0
        return _interactive_loop(
            target=args.target,
            agent=agent,
            presenter=presenter,
            session=session,
            output=output,
            idle_reads=args.idle_reads,
            max_reads=args.max_reads,
            poll_interval=args.poll_interval,
        )
    finally:
        agent.stop()


def _run_voice_target(
    args: argparse.Namespace,
    *,
    command: tuple[str, ...],
    voice_loop_factory: VoiceLoopFactory | None,
    output: TextIO,
) -> int:
    try:
        _apply_voice_settings(args)
    except VoicePresetError as error:
        print(str(error), file=output)
        return 2

    factory = voice_loop_factory or _build_default_voice_loop
    args._agent_voice_output = output
    try:
        loop = factory(command, args)
    except (ValueError, VoiceModeUnavailableError) as error:
        print(str(error), file=output)
        return 2
    return loop.run_forever()


def _build_default_voice_loop(
    command: tuple[str, ...],
    args: argparse.Namespace,
) -> VoiceLoopRunner:
    try:
        from agent_voice.providers import build_local_voice_loop

        return build_local_voice_loop(
            command=command,
            language=args.language,
            cache_dir=Path(args.cache_dir),
            whisper_model=args.whisper_model,
            whisper_language=args.stt_language,
            stt_backend=args.stt_backend,
            whisper_cpp_executable=args.whisper_cpp_executable,
            tts_voice=args.tts_voice,
            tts_lang=args.tts_lang,
            tts_speed=args.tts_speed,
            sample_rate=args.sample_rate,
            vad_threshold=args.vad_threshold,
            input_device=_parse_audio_device(args.input_device),
            output_device=_parse_audio_device(args.output_device),
            keyboard_input=not args.no_keyboard,
            terminal_output=getattr(args, "_agent_voice_output", sys.stdout),
            transparent_io=not args.quiet_agent_io,
            output_idle_reads=args.idle_reads,
            output_max_reads=args.max_reads,
            output_poll_interval_seconds=args.poll_interval,
            record_debug_audio=args.record_debug_audio,
            debug_audio_dir=Path(args.debug_audio_dir),
            tts_backend=args.tts_backend,
            supertonic_voice=args.supertonic_voice,
            macos_say_voice=args.macos_say_voice,
            macos_say_rate=args.macos_say_rate,
            agent=_build_agent(command, args),
            aec_enabled=not args.no_aec,
            aec_delay_ms=args.aec_delay_ms,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise VoiceModeUnavailableError(
            "agent-voice voice mode requires local audio dependencies. "
            "Run `uv sync` and ensure PortAudio/sounddevice can access your "
            f"mic and speaker. Missing dependency: {error.name}."
        ) from error


def _apply_voice_settings(args: argparse.Namespace) -> None:
    settings = resolve_voice_settings(
        config_path=Path(args.voice_config) if args.voice_config else None,
        preset_name=args.voice_preset,
        voice_override=args.tts_voice,
        supertonic_voice_override=args.supertonic_voice,
        lang_override=args.tts_lang,
        speed_override=args.tts_speed,
    )
    args.voice_preset = settings.preset
    args.tts_voice = settings.kokoro_voice
    args.supertonic_voice = settings.supertonic_voice
    if args.macos_say_voice is None and settings.macos_say_voice is not None:
        args.macos_say_voice = settings.macos_say_voice
    args.tts_lang = settings.lang
    args.tts_speed = settings.speed


def _parse_audio_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _build_agent_command(target: str, agent_args: Sequence[str]) -> tuple[str, ...]:
    command = (target, *tuple(agent_args))
    if not command[0]:
        raise ValueError("agent command must not be empty")
    return command


def _build_agent(command: tuple[str, ...], args: argparse.Namespace) -> Agent:
    if args.agent_backend == "pexpect":
        return PexpectAgent(command=command)
    event_logger = _build_agent_event_logger(args)
    developer_instructions = resolve_developer_instructions(args.assistant_style)
    if args.agent_backend == "codex-app-server":
        if command[0] != "codex":
            raise ValueError("codex-app-server backend can only be used with codex")
        kwargs = {"event_logger": event_logger} if event_logger is not None else {}
        if developer_instructions is not None:
            kwargs["developer_instructions"] = developer_instructions
        return CodexAppServerAgent(command=command, **kwargs)
    if args.agent_backend == "codex-remote-app-server":
        if command[0] != "codex":
            raise ValueError(
                "codex-remote-app-server backend can only be used with codex"
            )
        kwargs = {
            "url": _codex_remote_app_server_url(args),
            "thread_id": args.codex_thread_id,
            "cwd": None,
        }
        if event_logger is not None:
            kwargs["event_logger"] = event_logger
        if developer_instructions is not None:
            kwargs["developer_instructions"] = developer_instructions
        return CodexRemoteAppServerAgent(**kwargs)
    raise ValueError(f"unknown agent backend: {args.agent_backend}")


def _build_agent_event_logger(args: argparse.Namespace) -> JsonlEventLogger | None:
    if args.debug_agent_events is None:
        return None
    return JsonlEventLogger(Path(args.debug_agent_events))


def _codex_remote_app_server_url(args: argparse.Namespace) -> str:
    if args.codex_app_server_url:
        return args.codex_app_server_url
    if args.codex_app_server_port is not None:
        return f"ws://127.0.0.1:{args.codex_app_server_port}"
    raise ValueError(
        "codex-remote-app-server requires --codex-app-server-url "
        "or --codex-app-server-port"
    )


def _submit_once(
    text: str,
    *,
    agent: Agent,
    presenter: VoicePresenter,
    session: VoiceSession,
    output: TextIO,
    idle_reads: int = 4,
    max_reads: int = 600,
    poll_interval: float = 0.2,
) -> None:
    loop = VoiceLoop(
        transcript_source=SingleTranscriptSource(text),
        agent=agent,
        presenter=presenter,
        speaker=TextSpeaker(output),
        session=session,
        collect_output=lambda agent: _collect_agent_output(
            agent,
            idle_reads=idle_reads,
            max_reads=max_reads,
            poll_interval=poll_interval,
        ),
    )
    loop.run_once()


def _interactive_loop(
    *,
    target: str,
    agent: Agent,
    presenter: VoicePresenter,
    session: VoiceSession,
    output: TextIO,
    idle_reads: int = 4,
    max_reads: int = 600,
    poll_interval: float = 0.2,
) -> int:
    print(f"agent-voice {target} session started. Press Ctrl-D to exit.", file=output)
    while True:
        try:
            text = input("> ")
        except EOFError:
            print("", file=output)
            return 0
        if not text.strip():
            continue
        _submit_once(
            text,
            agent=agent,
            presenter=presenter,
            session=session,
            output=output,
            idle_reads=idle_reads,
            max_reads=max_reads,
            poll_interval=poll_interval,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-voice",
        description="Local-first voice layer for terminal coding agents.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use the current keyboard-driven text loop instead of voice mode.",
    )
    parser.add_argument(
        "--once",
        help="Text-mode smoke path: send one command and summarize current output.",
    )
    parser.add_argument(
        "--language",
        choices=("ko", "en"),
        default="ko",
        help="Voice summary language.",
    )
    parser.add_argument(
        "--idle-reads",
        type=int,
        default=4,
        help="Stop reading after this many empty polls once output has started.",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=600,
        help="Maximum output polls before giving up.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="Seconds to wait between output polls.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/agent-voice",
        help="Directory for downloaded local voice model assets.",
    )
    parser.add_argument(
        "--whisper-model",
        default="tiny",
        help=(
            "faster-whisper model name/path, or whisper.cpp GGML model path. "
            "With --stt-backend whisper-cpp, the default resolves to "
            "large-v3-q5_0."
        ),
    )
    parser.add_argument(
        "--stt-backend",
        choices=("faster-whisper", "whisper-cpp"),
        default="faster-whisper",
        help="Speech-to-text backend.",
    )
    parser.add_argument(
        "--whisper-cpp-executable",
        default="whisper-cli",
        help="whisper.cpp CLI executable used by --stt-backend whisper-cpp.",
    )
    parser.add_argument(
        "--stt-language",
        default="ko",
        help="Whisper language hint such as 'ko' or 'en'. Defaults to 'ko'.",
    )
    parser.add_argument(
        "--voice-config",
        default=None,
        help="Optional TOML file that adds or overrides voice presets.",
    )
    parser.add_argument(
        "--voice-preset",
        default=None,
        help="Voice preset name from the bundled or user voice config.",
    )
    parser.add_argument(
        "--tts-voice",
        default=None,
        help="Override the Kokoro voice name from the selected voice preset.",
    )
    parser.add_argument(
        "--tts-lang",
        default=None,
        help="Override the TTS language/accent code from the selected voice preset.",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=None,
        help="Override the TTS speech speed from the selected voice preset.",
    )
    parser.add_argument(
        "--tts-backend",
        choices=("auto", "kokoro", "macos-say", "supertonic"),
        default="auto",
        help=(
            "TTS backend. 'auto' uses Supertonic for Korean, otherwise Kokoro."
        ),
    )
    parser.add_argument(
        "--supertonic-voice",
        default=None,
        help="Voice style for the Supertonic backend, for example M2 or F2.",
    )
    parser.add_argument(
        "--macos-say-voice",
        default=None,
        help="Voice name for the macOS say backend, for example Yuna.",
    )
    parser.add_argument(
        "--macos-say-rate",
        type=int,
        default=None,
        help="Speech rate for the macOS say backend.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone capture sample rate.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.01,
        help="Simple energy threshold used before Smart Turn.",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="sounddevice input device index or name for microphone capture.",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="sounddevice output device index or name for speaker playback.",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable typed line input while voice mode is running.",
    )
    parser.add_argument(
        "--quiet-agent-io",
        action="store_true",
        help="Do not print transcript, agent input/output, and summary events.",
    )
    parser.add_argument(
        "--record-debug-audio",
        action="store_true",
        help=(
            "Record finalized microphone utterances and Smart Turn rejects "
            "for STT debugging."
        ),
    )
    parser.add_argument(
        "--debug-audio-dir",
        default=".cache/agent-voice/debug-audio",
        help="Directory for --record-debug-audio WAV files and manifest.jsonl.",
    )
    parser.add_argument(
        "--debug-agent-events",
        default=None,
        help=(
            "Write raw Codex app-server send/recv JSON-RPC events to this "
            "JSONL file for debugging."
        ),
    )
    parser.add_argument(
        "--assistant-style",
        choices=ASSISTANT_STYLE_CHOICES,
        default=DEFAULT_ASSISTANT_STYLE,
        help=(
            "Voice companion response style. 'jarvis-lite' is a restrained "
            "executive-assistant style and does not imitate a character voice."
        ),
    )
    parser.add_argument(
        "--no-aec",
        action="store_true",
        help="Disable LiveKit/WebRTC echo cancellation for local TTS playback.",
    )
    parser.add_argument(
        "--aec-delay-ms",
        type=int,
        default=120,
        help=(
            "Estimated speaker-to-microphone delay for LiveKit/WebRTC AEC. "
            "Tune this if the assistant still hears itself."
        ),
    )
    parser.add_argument(
        "--agent-backend",
        choices=("pexpect", "codex-app-server", "codex-remote-app-server"),
        default="pexpect",
        help=(
            "Agent control backend. 'codex-app-server' starts a private "
            "Codex JSON-RPC server; 'codex-remote-app-server' connects to a "
            "shared Codex app-server so a Codex TUI can use --remote."
        ),
    )
    parser.add_argument(
        "--codex-app-server-url",
        default=None,
        help=(
            "WebSocket URL for a shared Codex app-server, for example "
            "ws://127.0.0.1:4500. Used with --agent-backend "
            "codex-remote-app-server."
        ),
    )
    parser.add_argument(
        "--codex-app-server-port",
        type=int,
        default=None,
        help=(
            "Localhost port for a shared Codex app-server. Equivalent to "
            "--codex-app-server-url ws://127.0.0.1:PORT."
        ),
    )
    parser.add_argument(
        "--codex-thread-id",
        default=None,
        help=(
            "Existing Codex thread id to resume on the shared app-server. "
            "If omitted, agent-voice starts a new thread."
        ),
    )
    parser.add_argument(
        "--companion-log-dir",
        default=None,
        help=(
            "Directory for hidden companion logs when running "
            "`agent-voice companion codex`."
        ),
    )

    parser.add_argument(
        "target",
        choices=("doctor", "codex", "pi", "companion"),
        help="Command to run. Use 'doctor', 'codex', 'pi', or 'companion'.",
    )
    parser.add_argument(
        "agent_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the target agent CLI.",
    )

    return parser


def _companion_voice_args(args: argparse.Namespace) -> list[str]:
    voice_args: list[str] = []
    _append_changed(voice_args, "--language", args.language, "ko")
    _append_changed(voice_args, "--idle-reads", args.idle_reads, 4)
    _append_changed(voice_args, "--max-reads", args.max_reads, 600)
    _append_changed(voice_args, "--poll-interval", args.poll_interval, 0.2)
    _append_changed(voice_args, "--cache-dir", args.cache_dir, ".cache/agent-voice")
    _append_changed(voice_args, "--whisper-model", args.whisper_model, "tiny")
    _append_changed(voice_args, "--stt-backend", args.stt_backend, "faster-whisper")
    _append_changed(
        voice_args,
        "--whisper-cpp-executable",
        args.whisper_cpp_executable,
        "whisper-cli",
    )
    _append_changed(voice_args, "--stt-language", args.stt_language, "ko")
    _append_optional(voice_args, "--voice-config", args.voice_config)
    _append_optional(voice_args, "--voice-preset", args.voice_preset)
    _append_optional(voice_args, "--tts-voice", args.tts_voice)
    _append_optional(voice_args, "--tts-lang", args.tts_lang)
    _append_optional(voice_args, "--tts-speed", args.tts_speed)
    _append_changed(voice_args, "--tts-backend", args.tts_backend, "auto")
    _append_optional(voice_args, "--supertonic-voice", args.supertonic_voice)
    _append_optional(voice_args, "--macos-say-voice", args.macos_say_voice)
    _append_optional(voice_args, "--macos-say-rate", args.macos_say_rate)
    _append_changed(voice_args, "--sample-rate", args.sample_rate, 16000)
    _append_changed(voice_args, "--vad-threshold", args.vad_threshold, 0.01)
    _append_optional(voice_args, "--input-device", args.input_device)
    _append_optional(voice_args, "--output-device", args.output_device)
    if args.quiet_agent_io:
        voice_args.append("--quiet-agent-io")
    if args.record_debug_audio:
        voice_args.append("--record-debug-audio")
    _append_changed(
        voice_args,
        "--debug-audio-dir",
        args.debug_audio_dir,
        ".cache/agent-voice/debug-audio",
    )
    _append_optional(voice_args, "--debug-agent-events", args.debug_agent_events)
    _append_changed(
        voice_args,
        "--assistant-style",
        args.assistant_style,
        DEFAULT_ASSISTANT_STYLE,
    )
    if args.no_aec:
        voice_args.append("--no-aec")
    _append_changed(voice_args, "--aec-delay-ms", args.aec_delay_ms, 120)
    return voice_args


def _append_changed(
    values: list[str],
    flag: str,
    value: object,
    default: object,
) -> None:
    if value != default:
        values.extend([flag, str(value)])


def _append_optional(values: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        values.extend([flag, str(value)])


def _collect_agent_output(
    agent: Agent,
    *,
    idle_reads: int = 4,
    max_reads: int = 600,
    poll_interval: float = 0.2,
) -> str:
    chunks: list[str] = []
    empty_reads = 0

    for _ in range(max_reads):
        chunk = agent.read_available()
        if chunk:
            chunks.append(chunk)
            empty_reads = 0
        else:
            empty_reads += 1
            turn_active = getattr(agent, "is_turn_active", lambda: False)()
            if chunks and empty_reads >= idle_reads and not turn_active:
                break

        if poll_interval > 0:
            time.sleep(poll_interval)

    return "".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
