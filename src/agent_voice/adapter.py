from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Agent(Protocol):
    def start(self) -> None:
        """Start the wrapped terminal agent."""

    def submit(self, text: str) -> None:
        """Submit one user utterance as terminal input."""

    def read_available(self) -> str:
        """Read currently available output without blocking indefinitely."""

    def stop(self) -> None:
        """Stop the wrapped terminal agent."""


@dataclass
class PexpectAgent:
    command: Sequence[str] = ("codex",)
    cwd: str | Path | None = None
    env: Mapping[str, str] | None = None
    child_factory: Callable[["PexpectAgent"], Any] | None = None
    _child: Any = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self._child is not None:
            return

        if not self.command:
            raise ValueError("agent command must not be empty")

        if self.child_factory is not None:
            self._child = self.child_factory(self)
            return

        import pexpect

        command = list(self.command)
        self._child = pexpect.spawn(
            command[0],
            command[1:],
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=dict(self.env) if self.env is not None else None,
            encoding="utf-8",
            echo=False,
        )

    def submit(self, text: str) -> None:
        child = self._require_child()
        child.send(text)
        child.send("\r")

    def read_available(self) -> str:
        child = self._require_child()
        if not hasattr(child, "read_nonblocking"):
            return ""

        import pexpect

        try:
            return child.read_nonblocking(size=8192, timeout=0.2)
        except (pexpect.TIMEOUT, pexpect.EOF):
            return ""

    def stop(self) -> None:
        if self._child is None:
            return
        close = getattr(self._child, "close", None)
        if close is not None:
            close()
        self._child = None

    def _require_child(self) -> Any:
        if self._child is None:
            raise RuntimeError("agent has not been started")
        return self._child
