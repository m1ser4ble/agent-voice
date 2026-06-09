from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_voice.adapter import _WebSocketJsonRpcConnection


@dataclass(frozen=True)
class CodexTuiCompanionConfig:
    port: int | None = None
    url: str | None = None
    thread_id: str | None = None
    cwd: Path = field(default_factory=Path.cwd)
    log_dir: Path | None = None
    voice_args: tuple[str, ...] = ()
    codex_args: tuple[str, ...] = ()


ProcessFactory = Callable[..., Any]
ForegroundRunner = Callable[..., int]
WaitReady = Callable[[int], None]
CreateThread = Callable[[str, Path], str]


def run_codex_tui_companion(
    config: CodexTuiCompanionConfig,
    *,
    process_factory: ProcessFactory = subprocess.Popen,
    foreground_runner: ForegroundRunner = subprocess.call,
    wait_ready: WaitReady | None = None,
    create_thread: CreateThread | None = None,
) -> int:
    cwd = config.cwd.resolve()
    port = config.port or _port_from_url(config.url) or _free_port()
    url = config.url or f"ws://127.0.0.1:{port}"
    log_dir = (config.log_dir or Path(".cache/agent-voice/companion")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    wait_ready = wait_ready or _wait_ready
    create_thread = create_thread or create_starter_thread

    app_server_process = None
    voice_process = None
    app_log_path = log_dir / "codex-app-server.log"
    voice_log_path = log_dir / "agent-voice.log"

    with app_log_path.open("a", encoding="utf-8", buffering=1) as app_log:
        app_server_process = process_factory(
            ["codex", "app-server", "--listen", url],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=app_log,
            stderr=app_log,
            text=True,
        )
        try:
            wait_ready(port)
            thread_id = config.thread_id or create_thread(url, cwd)

            with voice_log_path.open("a", encoding="utf-8", buffering=1) as voice_log:
                voice_process = process_factory(
                    _voice_worker_command(url, thread_id, config.voice_args),
                    cwd=str(cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=voice_log,
                    stderr=voice_log,
                    text=True,
                )
                return foreground_runner(
                    [
                        "codex",
                        "resume",
                        thread_id,
                        "--remote",
                        url,
                        "--no-alt-screen",
                        *config.codex_args,
                    ],
                    cwd=str(cwd),
                )
        finally:
            if voice_process is not None:
                _terminate_process(voice_process)
            if app_server_process is not None:
                _terminate_process(app_server_process)


def create_starter_thread(url: str, cwd: Path) -> str:
    connection = _WebSocketJsonRpcConnection.connect(url)
    try:
        rpc = _JsonRpcClient(connection)
        rpc.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "agent-voice-companion",
                    "title": "agent-voice Codex TUI companion",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        rpc.notify("initialized", {})
        result = rpc.request(
            "thread/start",
            {"cwd": str(cwd), "threadSource": "user"},
        )
        thread_id = str(result["thread"]["id"])
        rpc.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [
                    {
                        "type": "text",
                        "text": (
                            "agent-voice companion 초기화입니다. "
                            "도구를 쓰지 말고 init-ok 라고만 답하세요."
                        ),
                    }
                ],
            },
        )
        rpc.wait_for_notification("turn/completed", timeout=90)
        return thread_id
    finally:
        connection.close()


def _voice_worker_command(
    url: str,
    thread_id: str,
    voice_args: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "agent_voice.cli",
        "--agent-backend",
        "codex-remote-app-server",
        "--codex-app-server-url",
        url,
        "--codex-thread-id",
        thread_id,
        "--no-keyboard",
        *voice_args,
        "codex",
    ]


class _JsonRpcClient:
    def __init__(self, connection: _WebSocketJsonRpcConnection) -> None:
        self.connection = connection
        self.next_id = 1

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        self.connection.send_json(
            {"method": method, "id": request_id, "params": params}
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                message = self.connection.recv_json()
            except TimeoutError:
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method} failed: {message['error']}")
            result = message.get("result", {})
            if isinstance(result, dict):
                return result
            return {}
        raise TimeoutError(f"timed out waiting for {method}")

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.connection.send_json({"method": method, "params": params})

    def wait_for_notification(self, method: str, *, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = self.connection.recv_json()
            except TimeoutError:
                continue
            if message.get("method") == method:
                return message
        raise TimeoutError(f"timed out waiting for {method}")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _wait_ready(port: int) -> None:
    deadline = time.monotonic() + 10
    url = f"http://127.0.0.1:{port}/readyz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError("codex app-server did not become ready")


def _port_from_url(url: str | None) -> int | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "ws":
        raise ValueError("companion mode only supports ws:// app-server URLs")
    return parsed.port


def _terminate_process(process: Any) -> None:
    poll = getattr(process, "poll", None)
    if poll is not None and poll() is not None:
        return
    terminate = getattr(process, "terminate", None)
    if terminate is not None:
        terminate()
    wait = getattr(process, "wait", None)
    if wait is None:
        return
    try:
        wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        kill = getattr(process, "kill", None)
        if kill is not None:
            kill()
        wait(timeout=1.0)
