import pytest

from agent_voice.providers import KokoroSpeaker, ManagedVoiceLoop, _download_if_missing


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

    def fake_urlretrieve(url, target):
        target.write_bytes(b"x" * 2048)

    monkeypatch.setattr(
        "agent_voice.providers.urllib.request.urlretrieve",
        fake_urlretrieve,
    )

    _download_if_missing("https://example.invalid/model.onnx", path, min_bytes=1024)

    assert path.read_bytes() == b"x" * 2048


def test_download_if_missing_rejects_too_small_download(tmp_path, monkeypatch):
    path = tmp_path / "kokoro-v1.0.onnx"

    def fake_urlretrieve(url, target):
        target.write_bytes(b"<html>not a model</html>")

    monkeypatch.setattr(
        "agent_voice.providers.urllib.request.urlretrieve",
        fake_urlretrieve,
    )

    with pytest.raises(RuntimeError, match="Downloaded asset kokoro-v1.0.onnx"):
        _download_if_missing("https://example.invalid/model.onnx", path, min_bytes=1024)

    assert not path.exists()
    assert not (tmp_path / "kokoro-v1.0.onnx.download").exists()
