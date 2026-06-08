from agent_voice.interrupt import InterruptManager, SessionState, VoiceSession


def test_interrupt_only_triggers_while_speaking():
    manager = InterruptManager(stop_phrases=("잠깐", "stop"))

    assert manager.should_interrupt("잠깐", SessionState.SPEAKING)
    assert not manager.should_interrupt("잠깐", SessionState.LISTENING)
    assert not manager.should_interrupt("auth 버그 고쳐", SessionState.SPEAKING)


def test_voice_session_moves_from_speaking_to_interrupted_to_listening():
    session = VoiceSession()

    session.heard_command()
    session.agent_responded()
    interrupted = session.interrupt()
    session.resume_listening()

    assert interrupted is True
    assert session.state is SessionState.LISTENING
    assert session.history == [
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.INTERRUPTED,
        SessionState.LISTENING,
    ]
