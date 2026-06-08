import threading

from agent_voice.interrupt import InterruptManager, SessionState, VoiceSession
from agent_voice.loop import VoiceLoop
from agent_voice.presenter import VoicePresenter


class FakeTranscriptSource:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)

    def next_transcript(self):
        if not self.transcripts:
            return None
        return self.transcripts.pop(0)


class PollingTranscriptSource:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.polls = 0

    def next_transcript(self):
        self.polls += 1
        if not self.transcripts:
            return None
        return self.transcripts.pop(0)


class InterruptDuringSpeechSource:
    def __init__(self, first_transcript, interrupt_transcript, speaking_started):
        self.first_transcript = first_transcript
        self.interrupt_transcript = interrupt_transcript
        self.speaking_started = speaking_started
        self.interrupt_sent = False

    def next_transcript(self):
        if self.first_transcript is not None:
            transcript = self.first_transcript
            self.first_transcript = None
            return transcript
        if not self.speaking_started.is_set() or self.interrupt_sent:
            return None
        self.interrupt_sent = True
        return self.interrupt_transcript


class FakeAgent:
    def __init__(self, output):
        self.outputs = output if isinstance(output, list) else [output]
        self.submitted = []
        self.stops = 0

    def start(self):
        return None

    def submit(self, text):
        self.submitted.append(text)

    def read_available(self):
        if len(self.outputs) == 1:
            return self.outputs[0]
        return self.outputs.pop(0)

    def stop(self):
        self.stops += 1
        return None


class FakeSpeaker:
    def __init__(self):
        self.said = []
        self.stops = 0

    def say(self, text):
        self.said.append(text)

    def stop(self):
        self.stops += 1


class BlockingSpeaker:
    def __init__(self, *, timeout=0.1):
        self.said = []
        self.stops = 0
        self.timeout = timeout
        self.started = threading.Event()
        self.stopped = threading.Event()

    def say(self, text):
        self.said.append(text)
        self.started.set()
        self.stopped.wait(timeout=self.timeout)

    def stop(self):
        self.stops += 1
        self.stopped.set()


def test_voice_loop_sends_transcript_to_agent_and_speaks_presented_summary():
    source = FakeTranscriptSource(["auth 버그 고쳐"])
    agent = FakeAgent("Modified:\n- auth.py\n\nTests:\n1 passed\n")
    speaker = FakeSpeaker()
    session = VoiceSession()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(language="ko"),
        speaker=speaker,
        session=session,
        collect_output=lambda agent: agent.read_available(),
    )

    handled = loop.run_once()

    assert handled is True
    assert agent.submitted == ["auth 버그 고쳐"]
    assert speaker.said == ["파일 1개를 수정했고, 테스트 1개는 모두 통과했습니다."]
    assert speaker.stops == 0
    assert session.state is SessionState.LISTENING
    assert session.history == [
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.LISTENING,
    ]


def test_voice_loop_returns_false_when_no_transcript_is_available():
    source = FakeTranscriptSource([])
    agent = FakeAgent("Tests:\n1 passed\n")
    speaker = FakeSpeaker()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(),
        speaker=speaker,
        collect_output=lambda agent: agent.read_available(),
    )

    handled = loop.run_once()

    assert handled is False
    assert agent.submitted == []
    assert speaker.said == []


def test_voice_loop_runs_until_transcript_source_is_idle():
    source = FakeTranscriptSource(["auth 버그 고쳐", "테스트는?"])
    agent = FakeAgent(
        [
            "Modified:\n- auth.py\n\nTests:\n1 passed\n",
            "Tests:\n2 passed\n",
        ]
    )
    speaker = FakeSpeaker()
    session = VoiceSession()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(language="ko"),
        speaker=speaker,
        session=session,
        collect_output=lambda agent: agent.read_available(),
    )

    handled_count = loop.run_until_idle()

    assert handled_count == 2
    assert agent.submitted == ["auth 버그 고쳐", "테스트는?"]
    assert speaker.said == [
        "파일 1개를 수정했고, 테스트 1개는 모두 통과했습니다.",
        "테스트 2개는 모두 통과했습니다.",
    ]
    assert session.state is SessionState.LISTENING


def test_voice_loop_exit_command_stops_runtime_without_sending_to_agent():
    source = FakeTranscriptSource(["auth 버그 고쳐", "이제 그만", "테스트는?"])
    agent = FakeAgent(
        [
            "Modified:\n- auth.py\n\nTests:\n1 passed\n",
            "Tests:\n2 passed\n",
        ]
    )
    speaker = FakeSpeaker()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(language="ko"),
        speaker=speaker,
        collect_output=lambda agent: agent.read_available(),
    )

    handled_count = loop.run_until_idle()

    assert handled_count == 2
    assert loop.should_exit is True
    assert agent.submitted == ["auth 버그 고쳐"]
    assert agent.stops == 1
    assert speaker.stops == 1
    assert speaker.said == [
        "파일 1개를 수정했고, 테스트 1개는 모두 통과했습니다.",
    ]


def test_voice_loop_forever_keeps_polling_when_user_is_silent():
    source = PollingTranscriptSource([None, None, "auth 버그 고쳐", None])
    agent = FakeAgent("Modified:\n- auth.py\n\nTests:\n1 passed\n")
    speaker = FakeSpeaker()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(language="ko"),
        speaker=speaker,
        collect_output=lambda agent: agent.read_available(),
    )

    handled_count = loop.run_forever(max_polls=4, idle_sleep_seconds=0)

    assert handled_count == 1
    assert source.polls == 4
    assert agent.submitted == ["auth 버그 고쳐"]
    assert speaker.said == ["파일 1개를 수정했고, 테스트 1개는 모두 통과했습니다."]


def test_voice_loop_forever_exits_only_on_exit_intent_not_idle():
    source = PollingTranscriptSource([None, None, "종료", "auth 버그 고쳐"])
    agent = FakeAgent("Tests:\n1 passed\n")
    speaker = FakeSpeaker()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(language="ko"),
        speaker=speaker,
        collect_output=lambda agent: agent.read_available(),
    )

    handled_count = loop.run_forever(max_polls=10, idle_sleep_seconds=0)

    assert handled_count == 1
    assert source.polls == 3
    assert loop.should_exit is True
    assert agent.submitted == []
    assert agent.stops == 1


def test_voice_loop_stops_speaker_when_interrupt_arrives_while_speaking():
    source = FakeTranscriptSource(["잠깐"])
    agent = FakeAgent("Tests:\n1 passed\n")
    speaker = FakeSpeaker()
    session = VoiceSession()
    session.agent_responded()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(),
        speaker=speaker,
        session=session,
        interrupt=InterruptManager(stop_phrases=("잠깐",)),
        collect_output=lambda agent: agent.read_available(),
    )

    handled = loop.run_once()

    assert handled is True
    assert agent.submitted == []
    assert speaker.said == []
    assert speaker.stops == 1
    assert session.state is SessionState.LISTENING
    assert session.history == [
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.INTERRUPTED,
        SessionState.LISTENING,
    ]


def test_voice_loop_polls_for_interrupts_while_speech_is_playing():
    speaker = BlockingSpeaker()
    source = InterruptDuringSpeechSource(
        first_transcript="auth 버그 고쳐",
        interrupt_transcript="잠깐",
        speaking_started=speaker.started,
    )
    agent = FakeAgent("Modified:\n- auth.py\n\nTests:\n1 passed\n")
    session = VoiceSession()
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(language="ko"),
        speaker=speaker,
        session=session,
        interrupt=InterruptManager(stop_phrases=("잠깐",)),
        collect_output=lambda agent: agent.read_available(),
    )

    handled = loop.run_once()

    assert handled is True
    assert agent.submitted == ["auth 버그 고쳐"]
    assert speaker.said == ["파일 1개를 수정했고, 테스트 1개는 모두 통과했습니다."]
    assert speaker.stops == 1
    assert session.state is SessionState.LISTENING
    assert session.history == [
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.INTERRUPTED,
        SessionState.LISTENING,
    ]


def test_voice_loop_ignores_non_interrupt_transcripts_while_speaking():
    speaker = BlockingSpeaker(timeout=0.01)
    source = InterruptDuringSpeechSource(
        first_transcript="auth 버그 고쳐",
        interrupt_transcript="테스트는?",
        speaking_started=speaker.started,
    )
    agent = FakeAgent(
        [
            "Modified:\n- auth.py\n\nTests:\n1 passed\n",
            "Tests:\n2 passed\n",
        ]
    )
    loop = VoiceLoop(
        transcript_source=source,
        agent=agent,
        presenter=VoicePresenter(language="ko"),
        speaker=speaker,
        interrupt=InterruptManager(stop_phrases=("잠깐",)),
        collect_output=lambda agent: agent.read_available(),
    )

    first_handled = loop.run_once()
    second_handled = loop.run_once()

    assert first_handled is True
    assert second_handled is False
    assert agent.submitted == ["auth 버그 고쳐"]
