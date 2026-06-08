from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TextIO

from agent_voice.adapter import Agent, PexpectAgent
from agent_voice.interrupt import VoiceSession
from agent_voice.loop import VoiceLoop
from agent_voice.presenter import VoicePresenter


AgentFactory = Callable[[tuple[str, ...]], Agent]


class VoiceLoopRunner(Protocol):
    def run_forever(self) -> int:
        """Run the voice loop until the user exits."""


VoiceLoopFactory = Callable[[tuple[str, ...], argparse.Namespace], VoiceLoopRunner]


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
    output: TextIO | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    output = output or sys.stdout

    command = _build_agent_command(args.target, args.agent_args)
    return _run_target(
        args,
        command=command,
        agent_factory=agent_factory,
        voice_loop_factory=voice_loop_factory,
        output=output,
    )


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

    factory = agent_factory or (lambda command: PexpectAgent(command=command))
    agent = factory(command)
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
    factory = voice_loop_factory or _build_default_voice_loop
    try:
        loop = factory(command, args)
    except VoiceModeUnavailableError as error:
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
            collect_output=lambda agent: _collect_agent_output(
                agent,
                idle_reads=args.idle_reads,
                max_reads=args.max_reads,
                poll_interval=args.poll_interval,
            ),
            cache_dir=Path(args.cache_dir),
            whisper_model=args.whisper_model,
            whisper_language=args.stt_language,
            tts_voice=args.tts_voice,
            tts_lang=args.tts_lang,
            sample_rate=args.sample_rate,
            vad_threshold=args.vad_threshold,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise VoiceModeUnavailableError(
            "agent-voice voice mode requires local audio dependencies. "
            "Run `uv sync` and ensure PortAudio/sounddevice can access your "
            f"mic and speaker. Missing dependency: {error.name}."
        ) from error


def _build_agent_command(target: str, agent_args: Sequence[str]) -> tuple[str, ...]:
    command = (target, *tuple(agent_args))
    if not command[0]:
        raise ValueError("agent command must not be empty")
    return command


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
        help="faster-whisper model name or path.",
    )
    parser.add_argument(
        "--stt-language",
        default=None,
        help="Optional Whisper language hint such as 'ko' or 'en'.",
    )
    parser.add_argument(
        "--tts-voice",
        default="af_sarah",
        help="Kokoro voice name.",
    )
    parser.add_argument(
        "--tts-lang",
        default="en-us",
        help="Kokoro language/accent code.",
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
        "target",
        choices=("codex", "pi"),
        help="Agent CLI to wrap. Use 'codex' or 'pi'.",
    )
    parser.add_argument(
        "agent_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the target agent CLI.",
    )

    return parser


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
            if chunks and empty_reads >= idle_reads:
                break

        if poll_interval > 0:
            time.sleep(poll_interval)

    return "".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
