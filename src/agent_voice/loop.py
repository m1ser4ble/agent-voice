from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from agent_voice.adapter import Agent
from agent_voice.interrupt import InterruptManager, SessionState, VoiceSession
from agent_voice.presenter import VoicePresenter


@dataclass(frozen=True)
class Transcript:
    text: str
    source: str = "unknown"


TranscriptInput = str | Transcript


class TranscriptSource(Protocol):
    def next_transcript(self) -> TranscriptInput | None:
        """Return the next completed transcript, or None if no input is ready."""


class Speaker(Protocol):
    def say(self, text: str) -> None:
        """Speak text to the user."""

    def stop(self) -> None:
        """Stop current speech playback."""


CollectOutput = Callable[[Agent], str]

VoiceLoopEventKind = Literal[
    "transcript",
    "agent_input",
    "agent_output",
    "speech_summary",
    "interrupt",
    "exit",
    "queued_transcript",
    "ignored_transcript",
]


@dataclass(frozen=True)
class VoiceLoopEvent:
    kind: VoiceLoopEventKind
    text: str
    source: str = "unknown"


VoiceLoopObserver = Callable[[VoiceLoopEvent], None]

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
    observer: VoiceLoopObserver | None = None
    speech_poll_interval_seconds: float = 0.05
    speech_join_timeout_seconds: float = 1.0
    should_exit: bool = field(default=False, init=False)
    _pending_transcripts: deque[Transcript] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )

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

        transcript = self._next_transcript()
        if transcript is None:
            return False

        transcript = self._clean_transcript(transcript)
        if transcript is None:
            return False

        self._emit("transcript", transcript.text, source=transcript.source)

        if self._should_exit(transcript.text):
            self._emit("exit", transcript.text, source=transcript.source)
            self.speaker.stop()
            self.agent.stop()
            self.should_exit = True
            return True

        if self.interrupt.should_interrupt(transcript.text, self.session.state):
            self._emit("interrupt", transcript.text, source=transcript.source)
            self.speaker.stop()
            self.session.interrupt()
            self.session.resume_listening()
            return True

        self.session.heard_command()
        self._emit("agent_input", transcript.text, source=transcript.source)
        self.agent.submit(transcript.text)
        raw_output = self._collect_output()
        if raw_output:
            self._emit("agent_output", raw_output)
        self.session.agent_responded()

        summary = self.presenter.summarize(raw_output)
        if summary:
            self._emit("speech_summary", summary)
            self._speak_interruptibly(summary)
        elif self.session.state is SessionState.SPEAKING:
            self.session.tts_finished()
        return True

    def _speak_interruptibly(self, text: str) -> None:
        errors: list[BaseException] = []

        def speak() -> None:
            try:
                self.speaker.say(text)
            except BaseException as error:  # pragma: no cover - re-raised below
                errors.append(error)

        speech_thread = threading.Thread(target=speak, daemon=True)
        speech_thread.start()

        while speech_thread.is_alive() and not self.should_exit:
            transcript = self._next_transcript(include_pending=False)
            if transcript is not None and self._handle_speaking_transcript(transcript):
                break
            if self.speech_poll_interval_seconds > 0:
                time.sleep(self.speech_poll_interval_seconds)

        speech_thread.join(timeout=self.speech_join_timeout_seconds)
        if errors:
            raise errors[0]
        if self.session.state is SessionState.SPEAKING:
            self.session.tts_finished()

    def _handle_speaking_transcript(self, transcript: TranscriptInput) -> bool:
        transcript = self._clean_transcript(transcript)
        if transcript is None:
            return False

        self._emit("transcript", transcript.text, source=transcript.source)

        if self._should_exit(transcript.text):
            self._emit("exit", transcript.text, source=transcript.source)
            self.speaker.stop()
            self.agent.stop()
            self.should_exit = True
            self.session.interrupt()
            self.session.resume_listening()
            return True

        if self.interrupt.should_interrupt(transcript.text, self.session.state):
            self._emit("interrupt", transcript.text, source=transcript.source)
            self.speaker.stop()
            self.session.interrupt()
            self.session.resume_listening()
            return True

        if transcript.source == "keyboard":
            self._pending_transcripts.append(transcript)
            self._emit("queued_transcript", transcript.text, source=transcript.source)
        else:
            self._emit("ignored_transcript", transcript.text, source=transcript.source)
        return False

    def _collect_output(self) -> str:
        if self.collect_output is not None:
            return self.collect_output(self.agent)
        return self.agent.read_available()

    def _should_exit(self, transcript: str) -> bool:
        normalized = transcript.casefold()
        return any(phrase.casefold() in normalized for phrase in self.exit_phrases)

    def _next_transcript(
        self,
        *,
        include_pending: bool = True,
    ) -> TranscriptInput | None:
        if include_pending and self._pending_transcripts:
            return self._pending_transcripts.popleft()
        return self.transcript_source.next_transcript()

    def _clean_transcript(self, transcript: TranscriptInput) -> Transcript | None:
        if isinstance(transcript, Transcript):
            cleaned = transcript.text.strip()
            source = transcript.source or "unknown"
        else:
            cleaned = transcript.strip()
            source = "unknown"

        if not cleaned:
            return None
        return Transcript(text=cleaned, source=source)

    def _emit(
        self,
        kind: VoiceLoopEventKind,
        text: str,
        *,
        source: str = "unknown",
    ) -> None:
        if self.observer is None:
            return
        self.observer(VoiceLoopEvent(kind=kind, text=text, source=source))
