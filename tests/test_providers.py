import json
import time
from io import StringIO
from pathlib import Path

import pytest

from agent_voice.providers import (
    AecAudioPlayer,
    DebugAudioRecorder,
    KeyboardTranscriptSource,
    KokoroSpeaker,
    LiveKitEchoCanceller,
    MacOSSaySpeaker,
    ManagedVoiceLoop,
    MergedTranscriptSource,
    MicrophoneWhisperTranscriptSource,
    StderrDownloadReporter,
    SupertonicSpeaker,
    WhisperCppTranscriber,
    _build_speaker,
    _download_if_missing,
)


class FakeKokoro:
    def __init__(self):
        self.calls = []

    def create(self, text, *, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        return [0.1, -0.1], 24000


class FakeSupertonic:
    sample_rate = 44100

    def __init__(self):
        self.styles = []
        self.calls = []

    def get_voice_style(self, *, voice_name):
        self.styles.append(voice_name)
        return {"voice": voice_name}

    def synthesize(self, text, *, voice_style, speed, lang):
        self.calls.append((text, voice_style, speed, lang))
        return [[0.1, -0.1, 0.2]], [0.1]


class FakeAudioPlayer:
    def __init__(self):
        self.plays = []
        self.stops = 0

    def play(self, audio, sample_rate):
        self.plays.append((audio, sample_rate))

    def stop(self):
        self.stops += 1


class FakeTranscriber:
    def __init__(self, text="clean speech"):
        self.text = text
        self.calls = []

    def transcribe(self, audio, *, sample_rate):
        self.calls.append((audio.copy(), sample_rate))
        return self.text


class FakeChunkedAudioPlayer(FakeAudioPlayer):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def play_chunks(self, audio, sample_rate, *, chunk_size, before_play):
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            before_play(chunk)
            self.events.append(("play", len(chunk)))
            self.plays.append((chunk, sample_rate))


class FakeLiveKitFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self.data = memoryview(data).cast("B").cast("h")
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class FakeLiveKitApm:
    def __init__(self):
        self.reverse_frames = []
        self.capture_frames = []
        self.delays = []

    def set_stream_delay_ms(self, delay_ms):
        self.delays.append(delay_ms)

    def process_reverse_stream(self, frame):
        self.reverse_frames.append(list(frame.data))

    def process_stream(self, frame):
        self.capture_frames.append(list(frame.data))
        for index in range(len(frame.data)):
            frame.data[index] = int(frame.data[index] / 2)


class FakeEchoCanceller:
    def __init__(self, events=None):
        self.events = events
        self.render_frames = []
        self.capture_frames = []

    def analyze_render(self, frame):
        if self.events is not None:
            self.events.append(("render", len(frame)))
        self.render_frames.append(frame)

    def process_capture(self, frame):
        self.capture_frames.append(frame)
        return frame * 0.5


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


class FakeStatusAgent(FakeAgent):
    def status_lines(self):
        return ("agent status: ready",)


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


def test_supertonic_speaker_synthesizes_and_plays_mono_audio():
    tts = FakeSupertonic()
    player = FakeAudioPlayer()
    speaker = SupertonicSpeaker(
        tts=tts,
        player=player,
        voice="M2",
        speed=0.94,
        lang="ko",
        sample_rate=44100,
    )

    speaker.say("테스트는 모두 통과했습니다.")
    speaker.stop()

    assert tts.styles == ["M2"]
    assert tts.calls == [
        (
            "테스트는 모두 통과했습니다.",
            {"voice": "M2"},
            0.94,
            "ko",
        )
    ]
    audio, sample_rate = player.plays[0]
    assert list(audio) == [0.1, -0.1, 0.2]
    assert sample_rate == 44100
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


def test_livekit_echo_canceller_feeds_render_and_cleans_capture_frames():
    apm = FakeLiveKitApm()
    canceller = LiveKitEchoCanceller(
        apm=apm,
        frame_factory=FakeLiveKitFrame,
        sample_rate=16000,
        stream_delay_ms=42,
    )

    canceller.analyze_render([0.25] * 160)
    cleaned = canceller.process_capture([0.5] * 160)

    assert apm.delays == [42]
    assert len(apm.reverse_frames) == 1
    assert len(apm.capture_frames) == 1
    assert apm.reverse_frames[0][0] == pytest.approx(int(0.25 * 32767), abs=1)
    assert cleaned.tolist() == pytest.approx([0.25] * 160, abs=1e-3)


def test_aec_audio_player_feeds_resampled_tts_pcm_to_livekit_echo_canceller():
    player = FakeAudioPlayer()
    apm = FakeLiveKitApm()
    canceller = LiveKitEchoCanceller(
        apm=apm,
        frame_factory=FakeLiveKitFrame,
        sample_rate=16000,
    )
    aec_player = AecAudioPlayer(
        player=player,
        echo_canceller=canceller,
        target_sample_rate=16000,
    )

    aec_player.play([0.1, -0.1, 0.2], 16000)
    aec_player.stop()

    assert len(player.plays) == 1
    assert player.plays[0][0].tolist() == pytest.approx([0.1, -0.1, 0.2])
    assert player.plays[0][1] == 16000
    assert player.stops == 1
    assert len(apm.reverse_frames) == 1


def test_aec_audio_player_feeds_render_frames_in_playback_order():
    events = []
    player = FakeChunkedAudioPlayer(events)
    canceller = FakeEchoCanceller(events)
    aec_player = AecAudioPlayer(
        player=player,
        echo_canceller=canceller,
        target_sample_rate=16000,
    )

    aec_player.play([0.1] * 320, 16000)

    assert events == [
        ("render", 160),
        ("play", 160),
        ("render", 160),
        ("play", 160),
    ]


def test_macos_say_speaker_invokes_say_command(monkeypatch):
    calls = []

    def fake_popen(command):
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr("agent_voice.providers.subprocess.Popen", fake_popen)

    MacOSSaySpeaker(voice="Yuna", rate=185).say("감사합니다.")

    assert calls == [["say", "-v", "Yuna", "-r", "185", "감사합니다."]]


def test_build_speaker_auto_prefers_supertonic_for_korean(monkeypatch):
    calls = []

    def fake_from_cache(**kwargs):
        calls.append(kwargs)
        return "supertonic-speaker"

    monkeypatch.setattr(
        "agent_voice.providers.SupertonicSpeaker.from_cache",
        fake_from_cache,
    )

    speaker, backend = _build_speaker(
        backend="auto",
        cache_dir=Path(".cache/test"),
        tts_voice="am_michael",
        supertonic_voice="M2",
        tts_lang="ko",
        tts_speed=0.94,
        output_device="Speaker",
        macos_say_voice=None,
        macos_say_rate=None,
        echo_canceller=None,
        aec_sample_rate=16000,
    )

    assert backend == "supertonic"
    assert speaker == "supertonic-speaker"
    assert calls[0]["voice"] == "M2"
    assert calls[0]["lang"] == "ko"
    assert calls[0]["output_device"] == "Speaker"


def test_build_speaker_can_use_macos_say_explicitly(monkeypatch):
    def fake_run(command, *, capture_output, text, check):
        class Result:
            stdout = "Yuna            ko_KR    # 안녕하세요?\\n"

        return Result()

    monkeypatch.setattr("agent_voice.providers.platform.system", lambda: "Darwin")
    monkeypatch.setattr("agent_voice.providers.shutil.which", lambda command: "/bin/say")
    monkeypatch.setattr("agent_voice.providers.subprocess.run", fake_run)

    speaker, backend = _build_speaker(
        backend="macos-say",
        cache_dir=Path(".cache/test"),
        tts_voice="am_michael",
        supertonic_voice="M2",
        tts_lang="ko",
        tts_speed=0.94,
        output_device=None,
        macos_say_voice=None,
        macos_say_rate=None,
        echo_canceller=None,
        aec_sample_rate=16000,
    )

    assert backend == "macos-say"
    assert isinstance(speaker, MacOSSaySpeaker)
    assert speaker.voice == "Yuna"


def test_build_speaker_auto_keeps_kokoro_for_non_korean(monkeypatch):
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
        supertonic_voice="M2",
        tts_lang="en-us",
        tts_speed=0.94,
        output_device="Speaker",
        macos_say_voice=None,
        macos_say_rate=None,
        echo_canceller=None,
        aec_sample_rate=16000,
    )

    assert backend == "kokoro"
    assert speaker == "kokoro-speaker"
    assert calls[0]["voice"] == "am_michael"
    assert calls[0]["lang"] == "en-us"
    assert calls[0]["output_device"] == "Speaker"


def test_build_speaker_wraps_supertonic_player_when_echo_canceller_is_enabled(
    monkeypatch,
):
    calls = []

    def fake_from_cache(**kwargs):
        calls.append(kwargs)
        return "supertonic-speaker"

    monkeypatch.setattr(
        "agent_voice.providers.SupertonicSpeaker.from_cache",
        fake_from_cache,
    )

    echo_canceller = FakeEchoCanceller()
    speaker, backend = _build_speaker(
        backend="supertonic",
        cache_dir=Path(".cache/test"),
        tts_voice="am_michael",
        supertonic_voice="M2",
        tts_lang="ko",
        tts_speed=0.94,
        output_device="Speaker",
        macos_say_voice=None,
        macos_say_rate=None,
        echo_canceller=echo_canceller,
        aec_sample_rate=16000,
    )

    assert backend == "supertonic"
    assert speaker == "supertonic-speaker"
    assert isinstance(calls[0]["player"], AecAudioPlayer)
    assert calls[0]["player"].echo_canceller is echo_canceller


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


def test_microphone_source_transcribes_livekit_echo_cancelled_audio(monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np

    captured_audio = []

    class FakeInputStream:
        def __init__(self, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    class Segment:
        text = "clean speech"

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            return None

        def transcribe(self, audio, **kwargs):
            captured_audio.append(audio.copy())
            return [Segment()], None

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

    echo_canceller = FakeEchoCanceller()
    source = MicrophoneWhisperTranscriptSource(
        sample_rate=16000,
        chunk_ms=10,
        vad_threshold=0.01,
        silence_seconds=0.01,
        min_speech_seconds=0.001,
        use_smart_turn=False,
        echo_canceller=echo_canceller,
    )
    try:
        source._on_audio(np.full((160, 1), 0.5, dtype=np.float32), 160, None, None)
        source._on_audio(np.zeros((160, 1), dtype=np.float32), 160, None, None)

        transcript = None
        for _ in range(50):
            transcript = source.next_transcript()
            if transcript is not None:
                break
            time.sleep(0.01)
    finally:
        source.close()

    assert transcript is not None
    assert transcript.text == "clean speech"
    assert len(echo_canceller.capture_frames) >= 2
    assert captured_audio[0].max() == pytest.approx(0.25)


def test_microphone_source_can_use_injected_transcriber(monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np

    class FakeInputStream:
        def __init__(self, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream),
    )

    transcriber = FakeTranscriber("아까 말했던 수정 전략이 뭔지 설명해줘")
    source = MicrophoneWhisperTranscriptSource(
        sample_rate=16000,
        chunk_ms=10,
        vad_threshold=0.01,
        silence_seconds=0.01,
        min_speech_seconds=0.001,
        use_smart_turn=False,
        transcriber=transcriber,
    )
    try:
        source._on_audio(np.full((160, 1), 0.5, dtype=np.float32), 160, None, None)
        source._on_audio(np.zeros((160, 1), dtype=np.float32), 160, None, None)

        transcript = None
        for _ in range(50):
            transcript = source.next_transcript()
            if transcript is not None:
                break
            time.sleep(0.01)
    finally:
        source.close()

    assert transcript is not None
    assert transcript.text == "아까 말했던 수정 전략이 뭔지 설명해줘"
    assert transcriber.calls[0][1] == 16000


def test_debug_audio_recorder_saves_wav_and_manifest(tmp_path):
    import soundfile as sf

    import numpy as np

    recorder = DebugAudioRecorder(tmp_path)

    entry = recorder.record(
        np.array([0.0, 0.25, -0.25], dtype=np.float32),
        sample_rate=16000,
        transcript="좌우 반전되어 보이는 문제가 있습니다.",
        source="microphone",
        stage="stt_input",
    )

    assert entry["transcript"] == "좌우 반전되어 보이는 문제가 있습니다."
    assert entry["sample_rate"] == 16000
    assert entry["source"] == "microphone"
    assert entry["stage"] == "stt_input"
    assert (tmp_path / entry["audio"]).is_file()

    audio, sample_rate = sf.read(tmp_path / entry["audio"], dtype="float32")
    assert sample_rate == 16000
    assert audio.tolist() == pytest.approx([0.0, 0.25, -0.25], abs=1e-4)

    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [entry]


def test_microphone_source_records_debug_audio_with_transcript(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np

    class FakeInputStream:
        def __init__(self, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream),
    )

    source = MicrophoneWhisperTranscriptSource(
        sample_rate=16000,
        chunk_ms=10,
        vad_threshold=0.01,
        silence_seconds=0.01,
        min_speech_seconds=0.001,
        use_smart_turn=False,
        transcriber=FakeTranscriber("과우 반전되어 보이는 문제가 있습니다."),
        debug_audio_recorder=DebugAudioRecorder(tmp_path),
    )
    try:
        source._on_audio(np.full((160, 1), 0.5, dtype=np.float32), 160, None, None)
        source._on_audio(np.zeros((160, 1), dtype=np.float32), 160, None, None)

        for _ in range(50):
            if source.next_transcript() is not None:
                break
            time.sleep(0.01)
    finally:
        source.close()

    entries = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert entries[0]["transcript"] == "과우 반전되어 보이는 문제가 있습니다."
    assert entries[0]["stage"] == "stt_input"
    assert (tmp_path / entries[0]["audio"]).is_file()


def test_microphone_source_records_smart_turn_rejected_audio(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np

    class FakeInputStream:
        def __init__(self, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    class IncompleteSmartTurnSource(MicrophoneWhisperTranscriptSource):
        def _smart_turn_complete(self, audio):
            return False

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream),
    )

    source = IncompleteSmartTurnSource(
        sample_rate=16000,
        chunk_ms=10,
        vad_threshold=0.01,
        silence_seconds=0.01,
        min_speech_seconds=0.001,
        use_smart_turn=True,
        transcriber=FakeTranscriber("should not be used"),
        debug_audio_recorder=DebugAudioRecorder(tmp_path),
    )
    try:
        source._on_audio(np.full((160, 1), 0.5, dtype=np.float32), 160, None, None)
        source._on_audio(np.zeros((160, 1), dtype=np.float32), 160, None, None)
        time.sleep(0.05)
    finally:
        source.close()

    assert source.next_transcript() is None
    entries = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert entries[0]["transcript"] == ""
    assert entries[0]["stage"] == "smart_turn_incomplete"
    assert entries[0]["smart_turn_complete"] is False


def test_whisper_cpp_transcriber_invokes_cli_with_accuracy_defaults(tmp_path):
    import numpy as np
    from types import SimpleNamespace

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=" 아까 말했던 수정 전략이 뭔지 설명해줘. \n",
            stderr="",
        )

    transcriber = WhisperCppTranscriber(
        model_path=tmp_path / "ggml-large-v3-q5_0.bin",
        executable="whisper-cli",
        language="ko",
        runner=fake_run,
    )

    text = transcriber.transcribe(
        np.zeros(160, dtype=np.float32),
        sample_rate=16000,
    )

    command, kwargs = calls[0]
    assert text == "아까 말했던 수정 전략이 뭔지 설명해줘."
    assert command[:2] == ["whisper-cli", "-m"]
    assert str(tmp_path / "ggml-large-v3-q5_0.bin") in command
    assert "-nt" in command
    assert "-np" in command
    assert command[command.index("-bs") + 1] == "1"
    assert command[command.index("-bo") + 1] == "1"
    assert "-nf" in command
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


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


def test_managed_voice_loop_prints_agent_status_after_start():
    loop = FakeLoop()
    agent = FakeStatusAgent()
    output = StringIO()
    runner = ManagedVoiceLoop(
        loop=loop,
        agent=agent,
        status_lines=("static status",),
        output=output,
    )

    runner.run_forever()

    assert output.getvalue().splitlines() == [
        "static status",
        "agent status: ready",
    ]


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
