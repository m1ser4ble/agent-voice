from agent_voice.presenter import VoicePresenter


def test_presenter_summarizes_modified_files_and_passing_tests_in_korean():
    output = """
Modified:
- auth.py
- login.py

Tests:
18 passed
"""

    summary = VoicePresenter(language="ko").summarize(output)

    assert summary == "파일 2개를 수정했고, 테스트 18개는 모두 통과했습니다."


def test_presenter_returns_first_useful_line_when_no_known_pattern_matches():
    output = "\x1b[32mDone\x1b[0m\n\nRun pytest before committing."

    summary = VoicePresenter(language="en").summarize(output)

    assert summary == "Done"


def test_presenter_skips_echoed_prompt_when_falling_back_to_agent_text():
    output = """
 안녕!


 안녕하세요! 무엇을 도와드릴까요?

────────────────────────────────────────────────────────────────────────────────
"""

    summary = VoicePresenter(language="ko").summarize(output, prompt="안녕!")

    assert summary == "안녕하세요! 무엇을 도와드릴까요?"


def test_presenter_skips_polite_echoed_prompt_when_falling_back_to_agent_text():
    output = """
 수고하셨습니다.


 감사합니다! 필요하시면 언제든 다시 불러주세요.

────────────────────────────────────────────────────────────────────────────────
"""

    summary = VoicePresenter(language="ko").summarize(
        output,
        prompt="수고하셨습니다.",
    )

    assert summary == "감사합니다! 필요하시면 언제든 다시 불러주세요."


def test_presenter_skips_pi_startup_and_terminal_status_noise():
    output = """
 pi v0.78.1
 escape interrupt · ctrl+c/ctrl+d clear/exit · / commands · ! bash · ctrl+o
 more
 Press ctrl+o to show full startup help and loaded resources.

 Pi can explain its own features and look up its docs. Ask it how to use or
 extend Pi.

 [Extensions]
   pi-chrome:chrome-profile-bridge



 그거라!


 좋아요! 😄
 이어서 뭘 해드릴까요?

────────────────────────────────────────────────────────────────────────────────
~/Workspace/agent-voice (main)
↑1.0k ↓62 $0.007 (sub) 0.4%/272k (auto)          (openai-codex) gpt-5.5 • medium
"""

    summary = VoicePresenter(language="ko").summarize(output, prompt="그거라!")

    assert summary == "좋아요! 이어서 뭘 해드릴까요?"
