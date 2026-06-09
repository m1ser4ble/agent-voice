import json

from agent_voice.recording import RecordingConfig, record_wav


class FakeSoundDevice:
    def __init__(self):
        self.calls = []
        self.waits = 0

    def rec(self, frames, *, samplerate, channels, dtype, device):
        self.calls.append(
            {
                "frames": frames,
                "samplerate": samplerate,
                "channels": channels,
                "dtype": dtype,
                "device": device,
            }
        )
        return [[0.1], [-0.1]]

    def wait(self):
        self.waits += 1


class FakeSoundFile:
    def __init__(self):
        self.writes = []

    def write(self, path, audio, sample_rate, *, format, subtype):
        self.writes.append(
            {
                "path": path,
                "audio": audio,
                "sample_rate": sample_rate,
                "format": format,
                "subtype": subtype,
            }
        )


def test_record_wav_saves_audio_and_appends_manifest(tmp_path):
    sd = FakeSoundDevice()
    sf = FakeSoundFile()
    output = tmp_path / "fixtures" / "mine.wav"
    manifest = tmp_path / "fixtures" / "manifest.jsonl"

    result = record_wav(
        RecordingConfig(
            output=output,
            seconds=1.5,
            sample_rate=16_000,
            input_device="MacBook Pro 마이크",
            text="헬로 월드 출력하는 C 코드 작성해줘",
            manifest=manifest,
        ),
        sounddevice=sd,
        soundfile=sf,
    )

    assert result == output
    assert sd.calls == [
        {
            "frames": 24_000,
            "samplerate": 16_000,
            "channels": 1,
            "dtype": "float32",
            "device": "MacBook Pro 마이크",
        }
    ]
    assert sd.waits == 1
    assert sf.writes == [
        {
            "path": str(output),
            "audio": [[0.1], [-0.1]],
            "sample_rate": 16_000,
            "format": "WAV",
            "subtype": "PCM_16",
        }
    ]

    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert row == {
        "audio": "mine.wav",
        "text": "헬로 월드 출력하는 C 코드 작성해줘",
        "language": "ko",
        "sample_rate": 16_000,
        "source": "local-mic",
    }
