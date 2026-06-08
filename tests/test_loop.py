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


class FakeAgent:
    def __init__(self, output):
        self.output = output
        self.submitted = []

    def start(self):
        return None

    def submit(self, text):
        self.submitted.append(text)

    def read_available(self):
        return self.output

    def stop(self):
        return None


class FakeSpeaker:
    def __init__(self):
        self.said = []
        self.stops = 0

    def say(self, text):
        self.said.append(text)

    def stop(self):
        self.stops += 1


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
