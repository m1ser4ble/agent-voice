from __future__ import annotations

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


@dataclass
class VoiceLoop:
    transcript_source: TranscriptSource
    agent: Agent
    presenter: VoicePresenter
    speaker: Speaker
    session: VoiceSession = field(default_factory=VoiceSession)
    interrupt: InterruptManager = field(default_factory=InterruptManager)
    collect_output: CollectOutput | None = None

    def run_until_idle(self, *, max_turns: int | None = None) -> int:
        handled_count = 0

        while max_turns is None or handled_count < max_turns:
            if not self.run_once():
                break
            handled_count += 1

        return handled_count

    def run_once(self) -> bool:
        transcript = self.transcript_source.next_transcript()
        if transcript is None:
            return False

        transcript = transcript.strip()
        if not transcript:
            return False

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
