from __future__ import annotations

import asyncio
import importlib.util
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "provider-smoke"
WAV_PATH = CACHE / "kokoro-smoke.wav"

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {path.name}")
    urllib.request.urlretrieve(url, path)


async def main() -> None:
    text = "Authentication bug fixed. One test passed."
    model_path = CACHE / "kokoro-v1.0.onnx"
    voices_path = CACHE / "voices-v1.0.bin"

    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("voice-onnx smoke should not require torch")

    download_if_missing(KOKORO_MODEL_URL, model_path)
    download_if_missing(KOKORO_VOICES_URL, voices_path)

    print("loading kokoro-onnx")
    kokoro = Kokoro(str(model_path), str(voices_path))
    print("creating speech")
    audio, sample_rate = kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")
    sf.write(WAV_PATH, audio, sample_rate)
    print(f"kokoro wav={WAV_PATH} sample_rate={sample_rate} samples={len(audio)}")

    print("loading faster-whisper tiny.en")
    whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = whisper.transcribe(str(WAV_PATH), language="en", beam_size=1)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    print(f"whisper language={info.language} probability={info.language_probability:.3f}")
    print(f"whisper transcript={transcript!r}")
    expected_words = ("authentication", "bug", "fixed", "test", "passed")
    missing_words = [word for word in expected_words if word not in transcript.casefold()]
    if missing_words:
        raise RuntimeError(f"whisper transcript missing expected words: {missing_words}")

    print("loading smart-turn-v3")
    turn = LocalSmartTurnAnalyzerV3(
        sample_rate=sample_rate,
        params=SmartTurnParams(stop_secs=3.0, pre_speech_ms=0.0, max_duration_secs=8.0),
    )
    int16_audio = np.clip(audio, -1.0, 1.0)
    int16_audio = (int16_audio * 32767).astype(np.int16)
    turn.append_audio(int16_audio.tobytes(), is_speech=True)
    state, metrics = await turn.analyze_end_of_turn()
    print(f"smart_turn state={state}")
    if metrics:
        print(
            "smart_turn "
            f"complete={metrics.is_complete} "
            f"probability={metrics.probability:.3f} "
            f"latency_ms={metrics.e2e_processing_time_ms:.1f}"
        )
        if not metrics.is_complete:
            raise RuntimeError("smart-turn-v3 did not classify the generated speech as complete")
    else:
        raise RuntimeError("smart-turn-v3 did not return metrics")


if __name__ == "__main__":
    asyncio.run(main())
