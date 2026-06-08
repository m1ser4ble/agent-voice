from __future__ import annotations

import argparse
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from typing import TextIO

from agent_voice.adapter import Agent, PexpectAgent
from agent_voice.interrupt import VoiceSession
from agent_voice.presenter import VoicePresenter


AgentFactory = Callable[[tuple[str, ...]], Agent]


def main(
    argv: Sequence[str] | None = None,
    *,
    agent_factory: AgentFactory | None = None,
    output: TextIO | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    output = output or sys.stdout

    if args.agent == "codex":
        return _run_codex(args, agent_factory=agent_factory, output=output)

    raise ValueError(f"unsupported agent: {args.agent}")


def _run_codex(
    args: argparse.Namespace,
    *,
    agent_factory: AgentFactory | None,
    output: TextIO,
) -> int:
    if args.once is not None and not args.text:
        print(
            "agent-voice codex --once is a text-mode smoke path; "
            "use --text --once.",
            file=output,
        )
        return 2

    if not args.text:
        print(
            "agent-voice codex voice mode is not implemented yet. "
            "use --text for the current keyboard-driven development loop.",
            file=output,
        )
        return 2

    command = tuple(shlex.split(args.agent_command))
    if not command:
        raise ValueError("--agent-command must not be empty")

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
    session.heard_command()
    agent.submit(text)
    raw_output = _collect_agent_output(
        agent,
        idle_reads=idle_reads,
        max_reads=max_reads,
        poll_interval=poll_interval,
    )
    session.agent_responded()
    summary = presenter.summarize(raw_output)
    if summary:
        print(summary, file=output)


def _interactive_loop(
    *,
    agent: Agent,
    presenter: VoicePresenter,
    session: VoiceSession,
    output: TextIO,
    idle_reads: int = 4,
    max_reads: int = 600,
    poll_interval: float = 0.2,
) -> int:
    print("agent-voice codex session started. Press Ctrl-D to exit.", file=output)
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
    )
    subparsers = parser.add_subparsers(dest="agent", required=True)

    codex = subparsers.add_parser("codex", help="Wrap Codex CLI.")
    codex.add_argument(
        "--text",
        action="store_true",
        help="Use the current keyboard-driven text loop instead of voice mode.",
    )
    codex.add_argument(
        "--once",
        help="Text-mode smoke path: send one command and summarize current output.",
    )
    codex.add_argument(
        "--agent-command",
        default="codex",
        help="Terminal command to spawn. Defaults to 'codex'.",
    )
    codex.add_argument(
        "--language",
        choices=("ko", "en"),
        default="ko",
        help="Voice summary language.",
    )
    codex.add_argument(
        "--idle-reads",
        type=int,
        default=4,
        help="Stop reading after this many empty polls once output has started.",
    )
    codex.add_argument(
        "--max-reads",
        type=int,
        default=600,
        help="Maximum output polls before giving up.",
    )
    codex.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="Seconds to wait between output polls.",
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
