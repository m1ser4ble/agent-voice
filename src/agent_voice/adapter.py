from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
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


ProcessFactory = Callable[..., Any]


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


@dataclass
class CodexAppServerAgent:
    command: Sequence[str] = ("codex",)
    cwd: str | Path | None = None
    env: Mapping[str, str] | None = None
    request_timeout_seconds: float = 5.0
    read_grace_seconds: float = 0.05
    process_factory: ProcessFactory | None = None
    _process: Any = field(default=None, init=False, repr=False)
    _reader: threading.Thread | None = field(default=None, init=False, repr=False)
    _messages: queue.Queue[dict[str, Any]] = field(
        default_factory=queue.Queue,
        init=False,
        repr=False,
    )
    _thread_id: str | None = field(default=None, init=False, repr=False)
    _turn_active: bool = field(default=False, init=False, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)
    _agent_message_deltas: dict[str, list[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _unknown_phase_agent_messages: list[str] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _has_final_agent_message: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.command:
            raise ValueError("agent command must not be empty")

        self._process = self._spawn_process()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

        self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "agent-voice",
                    "title": "agent-voice",
                    "version": "0.1.0",
                }
            },
        )
        self._notify("initialized", {})
        thread = self._request("thread/start", {})["thread"]
        self._thread_id = thread["id"]

    def submit(self, text: str) -> None:
        if self._thread_id is None:
            raise RuntimeError("agent has not been started")
        self._request(
            "turn/start",
            {
                "threadId": self._thread_id,
                "input": [{"type": "text", "text": text}],
            },
            wait=False,
        )
        self._reset_turn_output()
        self._turn_active = True

    def read_available(self) -> str:
        rendered: list[str] = []
        if self._messages.empty() and self.read_grace_seconds > 0:
            time.sleep(self.read_grace_seconds)
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                break
            text = self._handle_notification(message)
            if message.get("method") == "turn/completed":
                self._turn_active = False
            if text:
                rendered.append(text)
        return "".join(rendered)

    def is_turn_active(self) -> bool:
        return self._turn_active

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        if self._reader is not None:
            self._reader.join(timeout=0.5)
            self._reader = None

    def _spawn_process(self) -> Any:
        command = (self.command[0], "app-server", *tuple(self.command[1:]))
        factory = self.process_factory or subprocess.Popen
        return factory(
            command,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=dict(self.env) if self.env is not None else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def _read_stdout(self) -> None:
        process = self._process
        stdout = getattr(process, "stdout", None)
        if stdout is None:
            return
        while self._process is process:
            line = stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self._messages.put(message)

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        wait: bool = True,
    ) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"method": method, "id": request_id, "params": params})
        if not wait:
            return {}
        return self._wait_for_response(request_id)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        stdin = getattr(process, "stdin", None)
        if stdin is None:
            raise RuntimeError("codex app-server stdin is unavailable")
        stdin.write(json.dumps(message) + "\n")
        stdin.flush()

    def _wait_for_response(self, request_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.request_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for response {request_id}")
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(
                    f"timed out waiting for response {request_id}"
                ) from error

            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"]
                raise RuntimeError(f"codex app-server error: {error}")
            result = message.get("result", {})
            if not isinstance(result, dict):
                return {}
            return result

    def _reset_turn_output(self) -> None:
        self._agent_message_deltas.clear()
        self._unknown_phase_agent_messages.clear()
        self._has_final_agent_message = False

    def _handle_notification(self, message: Mapping[str, Any]) -> str:
        method = message.get("method")
        params = _mapping(message.get("params"))

        if method == "item/agentMessage/delta":
            return self._handle_agent_message_delta(params)
        if method == "item/completed":
            return self._handle_item_completed(_mapping(params.get("item")))
        if method == "turn/completed":
            return self._handle_turn_completed()
        return _render_codex_app_server_message(message)

    def _handle_agent_message_delta(self, params: Mapping[str, Any]) -> str:
        text = _first_string(params, ("delta", "text", "content"))
        if not text:
            return ""

        item_id = _first_string(params, ("itemId", "item_id", "id"))
        if not item_id:
            return text

        self._agent_message_deltas.setdefault(item_id, []).append(text)
        return ""

    def _handle_item_completed(self, item: Mapping[str, Any]) -> str:
        item_type = _first_string(item, ("type", "kind"))
        if item_type not in {"agentMessage", "agent_message"}:
            return _render_codex_item_completed(item)

        item_id = _first_string(item, ("id", "itemId", "item_id"))
        delta_text = "".join(self._agent_message_deltas.pop(item_id, []))
        text = _first_string(item, ("text", "content")) or delta_text
        phase = _first_string(item, ("phase",))

        if phase == "final_answer":
            self._has_final_agent_message = True
            return text
        if phase in {"commentary", "interim"}:
            return ""

        if text:
            self._unknown_phase_agent_messages.append(text)
        return ""

    def _handle_turn_completed(self) -> str:
        if self._has_final_agent_message:
            self._agent_message_deltas.clear()
            self._unknown_phase_agent_messages.clear()
            return ""

        buffered_delta_text = "".join(
            chunk
            for chunks in self._agent_message_deltas.values()
            for chunk in chunks
        )
        self._agent_message_deltas.clear()
        if buffered_delta_text:
            self._unknown_phase_agent_messages.append(buffered_delta_text)

        text = "".join(self._unknown_phase_agent_messages)
        self._unknown_phase_agent_messages.clear()
        return text


def _render_codex_app_server_message(message: Mapping[str, Any]) -> str:
    method = message.get("method")
    params = _mapping(message.get("params"))

    if method == "item/agentMessage/delta":
        return _first_string(params, ("delta", "text", "content"))
    if method == "item/started":
        return _render_codex_item_started(_mapping(params.get("item")))
    if method == "item/completed":
        return _render_codex_item_completed(_mapping(params.get("item")))
    if method == "turn/completed":
        return ""
    return ""


def _render_codex_item_started(item: Mapping[str, Any]) -> str:
    return ""


def _render_codex_item_completed(item: Mapping[str, Any]) -> str:
    item_type = _first_string(item, ("type", "kind"))
    if item_type in {"fileChange", "file_change"}:
        path = _first_string(item, ("path", "file", "filename"))
        if path:
            return f"Modified:\n- {path}\n"
    if item_type in {"commandExecution", "command_execution"}:
        return ""
    return ""


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _first_string(mapping: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str):
            return value
    return ""
