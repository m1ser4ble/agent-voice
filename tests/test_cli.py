from io import StringIO

from agent_voice.cli import _collect_agent_output, main


class FakeAgent:
    def __init__(self):
        self.submitted = []
        self.chunks = ["Modified:\n- auth.py\n\nTests:\n1 passed\n", "", "", "", ""]
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1
        return None

    def submit(self, text):
        self.submitted.append(text)

    def read_available(self):
        if not self.chunks:
            return ""
        return self.chunks.pop(0)

    def stop(self):
        self.stops += 1
        return None


def test_cli_codex_once_sends_command_and_prints_voice_summary():
    agent = FakeAgent()
    output = StringIO()

    exit_code = main(
        ["codex", "--text", "--once", "auth 버그 고쳐", "--poll-interval", "0"],
        agent_factory=lambda _: agent,
        output=output,
    )

    assert exit_code == 0
    assert agent.submitted == ["auth 버그 고쳐"]
    assert "파일 1개를 수정했고, 테스트 1개는 모두 통과했습니다." in output.getvalue()


def test_cli_codex_text_loop_keeps_one_agent_session_for_multiple_commands(monkeypatch):
    agent = FakeAgent()
    agent.chunks = [
        "Modified:\n- auth.py\n\nTests:\n1 passed\n",
        "",
        "Modified:\n- login.py\n\nTests:\n2 passed\n",
        "",
    ]
    output = StringIO()
    commands = iter(["auth 버그 고쳐", "테스트는?"])

    def fake_input(_prompt):
        try:
            return next(commands)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    exit_code = main(
        ["codex", "--text", "--idle-reads", "1", "--poll-interval", "0"],
        agent_factory=lambda _: agent,
        output=output,
    )

    assert exit_code == 0
    assert agent.starts == 1
    assert agent.stops == 1
    assert agent.submitted == ["auth 버그 고쳐", "테스트는?"]
    assert "테스트 1개는 모두 통과했습니다." in output.getvalue()
    assert "테스트 2개는 모두 통과했습니다." in output.getvalue()


def test_cli_codex_defaults_to_voice_mode_and_does_not_start_text_agent():
    agent = FakeAgent()
    output = StringIO()

    exit_code = main(
        ["codex"],
        agent_factory=lambda _: agent,
        output=output,
    )

    assert exit_code == 2
    assert agent.starts == 0
    assert "voice mode is not implemented yet" in output.getvalue()
    assert "use --text" in output.getvalue()


def test_cli_codex_once_requires_text_mode():
    agent = FakeAgent()
    output = StringIO()

    exit_code = main(
        ["codex", "--once", "auth 버그 고쳐"],
        agent_factory=lambda _: agent,
        output=output,
    )

    assert exit_code == 2
    assert agent.starts == 0
    assert "use --text --once" in output.getvalue()


class StreamingFakeAgent:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def read_available(self):
        if not self.chunks:
            return ""
        return self.chunks.pop(0)


def test_collect_agent_output_waits_for_first_output_then_stops_after_idle_reads():
    agent = StreamingFakeAgent(
        [
            "",
            "Modified:\n- auth.py\n",
            "Tests:\n1 passed\n",
            "",
            "",
            "late output that should not be read",
        ]
    )

    output = _collect_agent_output(
        agent,
        idle_reads=2,
        max_reads=10,
        poll_interval=0,
    )

    assert output == "Modified:\n- auth.py\nTests:\n1 passed\n"
