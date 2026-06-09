from __future__ import annotations

import argparse
import json
import shutil
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from agent_voice.adapter import _WebSocketJsonRpcConnection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch Codex TUI and agent-voice against one shared app-server."
    )
    parser.add_argument("--port", type=int, default=0, help="Local app-server port.")
    parser.add_argument(
        "--session",
        default="agent-voice-codex",
        help="tmux session name.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Existing Codex thread id to resume. If omitted, a starter thread is created.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Working directory for Codex and agent-voice.",
    )
    parser.add_argument(
        "--no-attach",
        action="store_true",
        help="Create the tmux session but do not attach to it.",
    )
    parser.add_argument(
        "agent_voice_args",
        nargs=argparse.REMAINDER,
        help="Extra agent-voice options before the codex target. Prefix with --.",
    )
    args = parser.parse_args(argv)

    if shutil.which("tmux") is None:
        print("tmux is required for the companion launcher.", file=sys.stderr)
        return 2

    cwd = args.cwd.resolve()
    port = args.port or _free_port()
    url = f"ws://127.0.0.1:{port}"

    if _tmux_session_exists(args.session):
        print(
            f"tmux session already exists: {args.session}\n"
            f"Attach with: tmux attach -t {args.session}",
            file=sys.stderr,
        )
        return 2

    app_server_command = f"cd {shlex.quote(str(cwd))} && codex app-server --listen {shlex.quote(url)}"
    _run(["tmux", "new-session", "-d", "-s", args.session, "-c", str(cwd), app_server_command])

    try:
        _wait_ready(port)
        thread_id = args.thread_id or _create_starter_thread(url, cwd)
        tui_command = (
            f"cd {shlex.quote(str(cwd))} && "
            f"codex resume {shlex.quote(thread_id)} --remote {shlex.quote(url)} --no-alt-screen"
        )
        voice_command = _agent_voice_command(url, thread_id, cwd, args.agent_voice_args)

        _run(["tmux", "split-window", "-h", "-t", args.session, "-c", str(cwd), tui_command])
        _run(["tmux", "split-window", "-v", "-t", f"{args.session}:0.1", "-c", str(cwd), voice_command])
        _run(["tmux", "select-layout", "-t", args.session, "tiled"])

        print(f"codex app-server: {url}")
        print(f"codex thread id: {thread_id}")
        print(f"tmux session: {args.session}")
        if args.no_attach:
            print(f"Attach with: tmux attach -t {args.session}")
            return 0

        return subprocess.call(["tmux", "attach", "-t", args.session])
    except Exception:
        subprocess.run(
            ["tmux", "kill-session", "-t", args.session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raise


def _agent_voice_command(
    url: str,
    thread_id: str,
    cwd: Path,
    extra_args: list[str],
) -> str:
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command = [
        "uv",
        "run",
        "agent-voice",
        "--agent-backend",
        "codex-remote-app-server",
        "--codex-app-server-url",
        url,
        "--codex-thread-id",
        thread_id,
        *extra_args,
        "codex",
    ]
    return f"cd {shlex.quote(str(cwd))} && {shlex.join(command)}"


def _create_starter_thread(url: str, cwd: Path) -> str:
    connection = _WebSocketJsonRpcConnection.connect(url)
    try:
        rpc = _JsonRpcClient(connection)
        rpc.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "agent-voice-companion-launcher",
                    "title": "agent-voice companion launcher",
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
        thread_id = result["thread"]["id"]
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
        return str(thread_id)
    finally:
        connection.close()


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


def _tmux_session_exists(session: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
