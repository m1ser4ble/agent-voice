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
