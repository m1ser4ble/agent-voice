from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from agent_voice.adapter import Agent
from agent_voice.interrupt import InterruptManager, VoiceSession
from agent_voice.presenter import VoicePresenter


class TranscriptSource(Protocol):
    def next_transcript(self) -> str | None:
        """Return the next completed transcript, or None if no input is ready."""


class Speaker(Protocol):
    def say(self, text: str) -> None:
        """Speak text to the user."""

    def stop(self) -> None:
        """Stop current speech playback."""


CollectOutput = Callable[[Agent], str]

DEFAULT_EXIT_PHRASES = (
    "이제 그만",
    "그만",
    "종료",
    "끝내",
    "꺼줘",
    "exit",
    "quit",
    "stop agent voice",
)


@dataclass
class VoiceLoop:
    transcript_source: TranscriptSource
    agent: Agent
    presenter: VoicePresenter
    speaker: Speaker
    session: VoiceSession = field(default_factory=VoiceSession)
    interrupt: InterruptManager = field(default_factory=InterruptManager)
    exit_phrases: tuple[str, ...] = DEFAULT_EXIT_PHRASES
    collect_output: CollectOutput | None = None
    should_exit: bool = field(default=False, init=False)

    def run_forever(
        self,
        *,
        max_polls: int | None = None,
        idle_sleep_seconds: float = 0.05,
    ) -> int:
        handled_count = 0
        poll_count = 0

        while not self.should_exit and (max_polls is None or poll_count < max_polls):
            poll_count += 1
            if self.run_once():
                handled_count += 1
                continue
            if idle_sleep_seconds > 0:
                time.sleep(idle_sleep_seconds)

        return handled_count

    def run_until_idle(self, *, max_turns: int | None = None) -> int:
        handled_count = 0

        while max_turns is None or handled_count < max_turns:
            if not self.run_once():
                break
            handled_count += 1

        return handled_count

    def run_once(self) -> bool:
        if self.should_exit:
            return False

        transcript = self.transcript_source.next_transcript()
        if transcript is None:
            return False

        transcript = transcript.strip()
        if not transcript:
            return False

        if self._should_exit(transcript):
            self.speaker.stop()
            self.agent.stop()
            self.should_exit = True
            return True

        if self.interrupt.should_interrupt(transcript, self.session.state):
            self.speaker.stop()
            self.session.interrupt()
            self.session.resume_listening()
            return True

        self.session.heard_command()
        self.agent.submit(transcript)
        raw_output = self._collect_output()
        self.session.agent_responded()

        summary = self.presenter.summarize(raw_output)
        if summary:
            self.speaker.say(summary)
        self.session.tts_finished()
        return True

    def _collect_output(self) -> str:
        if self.collect_output is not None:
            return self.collect_output(self.agent)
        return self.agent.read_available()

    def _should_exit(self, transcript: str) -> bool:
        normalized = transcript.casefold()
        return any(phrase.casefold() in normalized for phrase in self.exit_phrases)
