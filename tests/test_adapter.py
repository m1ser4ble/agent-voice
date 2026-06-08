from agent_voice.adapter import PexpectAgent


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
