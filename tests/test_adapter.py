import json
import queue

from agent_voice.adapter import CodexAppServerAgent, JsonlEventLogger, PexpectAgent


class FakeChild:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, text):
        self.sent.append(text)

    def close(self):
        self.closed = True


def test_pexpect_agent_submits_text_as_terminal_input():
    child = FakeChild()
    agent = PexpectAgent(command=("codex",), child_factory=lambda _: child)

    agent.start()
    agent.submit("auth 버그 고쳐")

    assert child.sent == ["auth 버그 고쳐", "\r"]


def test_pexpect_agent_closes_child_on_stop():
    child = FakeChild()
    agent = PexpectAgent(command=("codex",), child_factory=lambda _: child)

    agent.start()
    agent.stop()

    assert child.closed is True


class FakeJsonStdin:
    def __init__(self):
        self.messages = []

    def write(self, line):
        self.messages.append(json.loads(line))

    def flush(self):
        return None


class FakeJsonStdout:
    def __init__(self, lines):
        self._lines = queue.Queue()
        self.closed = False
        for line in lines:
            self.put(line)

    def put(self, message):
        self._lines.put(json.dumps(message) + "\n")

    def readline(self):
        if self.closed:
            return ""
        try:
            return self._lines.get(timeout=0.2)
        except queue.Empty:
            return ""

    def close(self):
        self.closed = True


class FakeAppServerProcess:
    def __init__(self, lines):
        self.stdin = FakeJsonStdin()
        self.stdout = FakeJsonStdout(lines)
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True
        self.stdout.close()

    def kill(self):
        self.killed = True
        self.stdout.close()

    def wait(self, timeout=None):
        return 0


def test_codex_app_server_agent_starts_thread_with_json_rpc_handshake():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {"serverInfo": {"name": "codex"}}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(process_factory=lambda *_args, **_kwargs: process)

    agent.start()

    assert process.stdin.messages == [
        {
            "method": "initialize",
            "id": 1,
            "params": {
                "clientInfo": {
                    "name": "agent-voice",
                    "title": "agent-voice",
                    "version": "0.1.0",
                }
            },
        },
        {"method": "initialized", "params": {}},
        {"method": "thread/start", "id": 2, "params": {}},
    ]


def test_codex_app_server_agent_starts_thread_with_developer_instructions():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(
        process_factory=lambda *_args, **_kwargs: process,
        developer_instructions="Speak like a calm executive assistant.",
    )

    agent.start()

    assert process.stdin.messages[2] == {
        "method": "thread/start",
        "id": 2,
        "params": {
            "developerInstructions": "Speak like a calm executive assistant.",
        },
    }


def test_codex_app_server_agent_submits_turn_and_renders_agent_message_events():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(process_factory=lambda *_args, **_kwargs: process)
    agent.start()

    agent.submit("테스트 요약해줘")
    process.stdout.put(
        {
            "method": "item/agentMessage/delta",
            "params": {"delta": "테스트는 모두 통과했습니다."},
        }
    )

    assert process.stdin.messages[-1] == {
        "method": "turn/start",
        "id": 3,
        "params": {
            "threadId": "thread-1",
            "input": [{"type": "text", "text": "테스트 요약해줘"}],
        },
    }
    assert agent.read_available() == "테스트는 모두 통과했습니다."


def test_codex_app_server_agent_speaks_final_answer_instead_of_commentary():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(process_factory=lambda *_args, **_kwargs: process)
    agent.start()
    agent.submit("상태 요약해줘")

    process.stdout.put(
        {
            "method": "item/agentMessage/delta",
            "params": {
                "itemId": "msg-1",
                "delta": "진행 중입니다. 내부 확인을 계속합니다.",
            },
        }
    )
    process.stdout.put(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "id": "msg-1",
                    "type": "agentMessage",
                    "text": "진행 중입니다. 내부 확인을 계속합니다.",
                    "phase": "commentary",
                }
            },
        }
    )
    process.stdout.put(
        {
            "method": "item/agentMessage/delta",
            "params": {
                "itemId": "msg-2",
                "delta": "최종 답변입니다. 요청한 상태를 확인했습니다.",
            },
        }
    )
    process.stdout.put(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "id": "msg-2",
                    "type": "agentMessage",
                    "text": "최종 답변입니다. 요청한 상태를 확인했습니다.",
                    "phase": "final_answer",
                }
            },
        }
    )
    process.stdout.put({"method": "turn/completed", "params": {"status": "completed"}})

    assert agent.read_available() == "최종 답변입니다. 요청한 상태를 확인했습니다."


def test_codex_app_server_agent_exposes_commentary_as_progress_not_final_output():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(process_factory=lambda *_args, **_kwargs: process)
    agent.start()
    agent.submit("프로젝트 파악해봐")

    process.stdout.put(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "id": "msg-1",
                    "type": "agentMessage",
                    "text": "$analyze로 읽기 전용 프로젝트 파악을 진행하겠습니다.",
                    "phase": "commentary",
                }
            },
        }
    )

    assert agent.read_available() == ""
    assert (
        agent.read_progress_available()
        == "$analyze로 읽기 전용 프로젝트 파악을 진행하겠습니다."
    )
    assert agent.read_progress_available() == ""


def test_codex_app_server_agent_logs_raw_events_to_jsonl(tmp_path):
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    log_path = tmp_path / "events.jsonl"
    agent = CodexAppServerAgent(
        process_factory=lambda *_args, **_kwargs: process,
        event_logger=JsonlEventLogger(log_path),
    )
    agent.start()
    agent.submit("프로젝트 파악해봐")
    process.stdout.put(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "id": "msg-1",
                    "type": "agentMessage",
                    "phase": "commentary",
                    "text": "프로젝트 구조를 훑겠습니다.",
                }
            },
        }
    )
    process.stdout.put({"method": "turn/completed", "params": {"status": "completed"}})
    agent.read_available()

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(
        row["direction"] == "send"
        and row["message"].get("method") == "turn/start"
        for row in rows
    )
    assert any(
        row["direction"] == "recv"
        and row["message"].get("method") == "item/completed"
        and row["message"]["params"]["item"]["phase"] == "commentary"
        for row in rows
    )


def test_codex_app_server_agent_keeps_unknown_phase_agent_message_compatibility():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(process_factory=lambda *_args, **_kwargs: process)
    agent.start()
    agent.submit("요약해줘")

    process.stdout.put(
        {
            "method": "item/agentMessage/delta",
            "params": {
                "itemId": "msg-1",
                "delta": "phase 없는 provider 응답입니다.",
            },
        }
    )
    process.stdout.put(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "id": "msg-1",
                    "type": "agentMessage",
                    "text": "phase 없는 provider 응답입니다.",
                    "phase": None,
                }
            },
        }
    )
    process.stdout.put({"method": "turn/completed", "params": {"status": "completed"}})

    assert agent.read_available() == "phase 없는 provider 응답입니다."


def test_codex_app_server_agent_renders_structured_work_events_without_raw_json():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(process_factory=lambda *_args, **_kwargs: process)
    agent.start()
    agent.submit("수정해줘")

    process.stdout.put(
        {
            "method": "item/started",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "command": "uv run pytest -q",
                }
            },
        }
    )
    process.stdout.put(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "fileChange",
                    "path": "src/agent_voice/adapter.py",
                }
            },
        }
    )
    process.stdout.put({"method": "turn/completed", "params": {"status": "completed"}})

    assert agent.read_available() == "Modified:\n- src/agent_voice/adapter.py\n"


def test_codex_app_server_agent_stops_process():
    process = FakeAppServerProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        ]
    )
    agent = CodexAppServerAgent(process_factory=lambda *_args, **_kwargs: process)

    agent.start()
    agent.stop()

    assert process.terminated is True
