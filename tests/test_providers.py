import time
from io import StringIO
from pathlib import Path

import pytest

from agent_voice.providers import (
    KeyboardTranscriptSource,
    KokoroSpeaker,
    MacOSSaySpeaker,
    ManagedVoiceLoop,
    MergedTranscriptSource,
    MicrophoneWhisperTranscriptSource,
    StderrDownloadReporter,
    _build_speaker,
    _download_if_missing,
)


class FakeKokoro:
    def __init__(self):
        self.calls = []

    def create(self, text, *, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        return [0.1, -0.1], 24000


class FakeAudioPlayer:
    def __init__(self):
        self.plays = []
        self.stops = 0

    def play(self, audio, sample_rate):
        self.plays.append((audio, sample_rate))

    def stop(self):
        self.stops += 1


class FakeProcess:
    def __init__(self):
        self.terminated = 0
        self.killed = 0
        self.waits = 0
        self.returncode = None

    def wait(self, timeout=None):
        self.waits += 1
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = 0

    def kill(self):
        self.killed += 1
        self.returncode = 0


class FakeLoop:
    def __init__(self):
        self.runs = 0

    def run_forever(self):
        self.runs += 1
        return 7


class FakeAgent:
    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1


class StartFailAgent(FakeAgent):
    def start(self):
        super().start()
        raise RuntimeError("start failed")


class FakeCloseable:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


class FakeTranscriptSource:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.closes = 0

    def next_transcript(self):
        if not self.transcripts:
            return None
        return self.transcripts.pop(0)

    def close(self):
        self.closes += 1


def test_kokoro_speaker_synthesizes_and_plays_audio():
    kokoro = FakeKokoro()
    player = FakeAudioPlayer()
    speaker = KokoroSpeaker(
        kokoro=kokoro,
        player=player,
        voice="af_sarah",
        speed=1.1,
        lang="en-us",
    )

    speaker.say("Tests passed.")
    speaker.stop()

    assert kokoro.calls == [("Tests passed.", "af_sarah", 1.1, "en-us")]
    assert player.plays == [([0.1, -0.1], 24000)]
    assert player.stops == 1


def test_sounddevice_player_uses_selected_output_device(monkeypatch):
    from agent_voice.providers import SoundDevicePlayer

    calls = []

    class FakeSoundDevice:
        def play(self, audio, sample_rate, *, blocking, device):
            calls.append((audio, sample_rate, blocking, device))

        def stop(self):
            return None

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSoundDevice())

    SoundDevicePlayer(output_device="USB Speaker").play([0.1], 24000)

    assert calls == [([0.1], 24000, True, "USB Speaker")]


def test_macos_say_speaker_invokes_say_command(monkeypatch):
    calls = []

    def fake_popen(command):
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr("agent_voice.providers.subprocess.Popen", fake_popen)

    MacOSSaySpeaker(voice="Yuna", rate=185).say("감사합니다.")

    assert calls == [["say", "-v", "Yuna", "-r", "185", "감사합니다."]]


def test_build_speaker_auto_prefers_macos_say_for_korean_on_macos(monkeypatch):
    def fake_run(command, *, capture_output, text, check):
        class Result:
            stdout = "Yuna            ko_KR    # 안녕하세요?\\n"

        return Result()

    monkeypatch.setattr("agent_voice.providers.platform.system", lambda: "Darwin")
    monkeypatch.setattr("agent_voice.providers.shutil.which", lambda command: "/bin/say")
    monkeypatch.setattr("agent_voice.providers.subprocess.run", fake_run)

    speaker, backend = _build_speaker(
        backend="auto",
        cache_dir=Path(".cache/test"),
        tts_voice="am_michael",
        tts_lang="ko",
        tts_speed=0.94,
        output_device=None,
        macos_say_voice=None,
        macos_say_rate=None,
    )

    assert backend == "macos-say"
    assert isinstance(speaker, MacOSSaySpeaker)
    assert speaker.voice == "Yuna"


def test_build_speaker_auto_keeps_kokoro_off_macos(monkeypatch):
    calls = []

    def fake_from_cache(**kwargs):
        calls.append(kwargs)
        return "kokoro-speaker"

    monkeypatch.setattr("agent_voice.providers.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "agent_voice.providers.KokoroSpeaker.from_cache",
        fake_from_cache,
    )

    speaker, backend = _build_speaker(
        backend="auto",
        cache_dir=Path(".cache/test"),
        tts_voice="am_michael",
        tts_lang="ko",
        tts_speed=0.94,
        output_device="Speaker",
        macos_say_voice=None,
        macos_say_rate=None,
    )

    assert backend == "kokoro"
    assert speaker == "kokoro-speaker"
    assert calls[0]["voice"] == "am_michael"
    assert calls[0]["lang"] == "ko"
    assert calls[0]["output_device"] == "Speaker"


def test_keyboard_transcript_source_reads_typed_lines():
    source = KeyboardTranscriptSource(input_stream=StringIO("auth 버그 고쳐\n\n종료\n"))

    try:
        transcripts = []
        for _ in range(20):
            transcript = source.next_transcript()
            if transcript is not None:
                transcripts.append(transcript)
            if len(transcripts) == 2:
                break
            time.sleep(0.01)
    finally:
        source.close()

    assert [(item.text, item.source) for item in transcripts] == [
        ("auth 버그 고쳐", "keyboard"),
        ("종료", "keyboard"),
    ]


def test_merged_transcript_source_polls_sources_in_order_and_closes_them():
    first = FakeTranscriptSource([None, "typed"])
    second = FakeTranscriptSource(["spoken"])
    source = MergedTranscriptSource([first, second])

    first_transcript = source.next_transcript()
    second_transcript = source.next_transcript()
    source.close()

    assert first_transcript == "spoken"
    assert second_transcript == "typed"
    assert first.closes == 1
    assert second.closes == 1


def test_microphone_source_uses_selected_input_device(monkeypatch):
    import sys
    from types import SimpleNamespace

    captured_stream_kwargs = []

    class FakeInputStream:
        def __init__(self, **kwargs):
            captured_stream_kwargs.append(kwargs)

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            return None

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream),
    )
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    source = MicrophoneWhisperTranscriptSource(input_device=2)
    source.close()

    assert captured_stream_kwargs[0]["device"] == 2


def test_managed_voice_loop_starts_agent_and_closes_resources():
    loop = FakeLoop()
    agent = FakeAgent()
    source = FakeCloseable()
    runner = ManagedVoiceLoop(loop=loop, agent=agent, closeables=(source,))

    exit_code = runner.run_forever()

    assert exit_code == 7
    assert loop.runs == 1
    assert agent.starts == 1
    assert agent.stops == 1
    assert source.closes == 1


def test_managed_voice_loop_closes_resources_when_agent_start_fails():
    loop = FakeLoop()
    agent = StartFailAgent()
    source = FakeCloseable()
    runner = ManagedVoiceLoop(loop=loop, agent=agent, closeables=(source,))

    with pytest.raises(RuntimeError, match="start failed"):
        runner.run_forever()

    assert loop.runs == 0
    assert agent.starts == 1
    assert agent.stops == 1
    assert source.closes == 1


def test_download_if_missing_replaces_too_small_existing_asset(tmp_path, monkeypatch):
    path = tmp_path / "kokoro-v1.0.onnx"
    path.write_bytes(b"<html>rate limited</html>")

    def fake_urlretrieve(url, target, reporthook=None):
        if reporthook is not None:
            reporthook(1, 1024, 2048)
            reporthook(2, 1024, 2048)
        target.write_bytes(b"x" * 2048)

    monkeypatch.setattr(
        "agent_voice.providers.urllib.request.urlretrieve",
        fake_urlretrieve,
    )

    _download_if_missing("https://example.invalid/model.onnx", path, min_bytes=1024)

    assert path.read_bytes() == b"x" * 2048


def test_download_if_missing_rejects_too_small_download(tmp_path, monkeypatch):
    path = tmp_path / "kokoro-v1.0.onnx"

    def fake_urlretrieve(url, target, reporthook=None):
        target.write_bytes(b"<html>not a model</html>")

    monkeypatch.setattr(
        "agent_voice.providers.urllib.request.urlretrieve",
        fake_urlretrieve,
    )

    with pytest.raises(RuntimeError, match="Downloaded asset kokoro-v1.0.onnx"):
        _download_if_missing("https://example.invalid/model.onnx", path, min_bytes=1024)

    assert not path.exists()
    assert not (tmp_path / "kokoro-v1.0.onnx.download").exists()


def test_download_if_missing_reports_download_progress(tmp_path, monkeypatch, capsys):
    path = tmp_path / "kokoro-v1.0.onnx"

    def fake_urlretrieve(url, target, reporthook=None):
        if reporthook is not None:
            reporthook(1, 1024, 2048)
            reporthook(2, 1024, 2048)
        target.write_bytes(b"x" * 2048)

    monkeypatch.setattr(
        "agent_voice.providers.urllib.request.urlretrieve",
        fake_urlretrieve,
    )

    _download_if_missing(
        "https://example.invalid/model.onnx",
        path,
        min_bytes=1024,
        reporter=StderrDownloadReporter(),
    )

    stderr = capsys.readouterr().err
    assert "Downloading kokoro-v1.0.onnx" in stderr
    assert "kokoro-v1.0.onnx: 100%" in stderr
    assert "Downloaded kokoro-v1.0.onnx" in stderr
