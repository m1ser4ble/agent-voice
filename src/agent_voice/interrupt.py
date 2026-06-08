from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SessionState(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class InterruptManager:
    stop_phrases: tuple[str, ...] = ("잠깐", "멈춰", "stop", "pause")

    def should_interrupt(self, transcript: str, state: SessionState) -> bool:
        if state is not SessionState.SPEAKING:
            return False
        normalized = transcript.casefold()
        return any(phrase.casefold() in normalized for phrase in self.stop_phrases)


@dataclass
class VoiceSession:
    state: SessionState = SessionState.LISTENING
    history: list[SessionState] = field(
        default_factory=lambda: [SessionState.LISTENING]
    )

    def heard_command(self) -> None:
        self._transition(SessionState.THINKING)

    def agent_responded(self) -> None:
        self._transition(SessionState.SPEAKING)

    def tts_finished(self) -> None:
        self._transition(SessionState.LISTENING)

    def interrupt(self) -> bool:
        if self.state is not SessionState.SPEAKING:
            return False
        self._transition(SessionState.INTERRUPTED)
        return True

    def resume_listening(self) -> None:
        self._transition(SessionState.LISTENING)

    def _transition(self, state: SessionState) -> None:
        self.state = state
        self.history.append(state)
