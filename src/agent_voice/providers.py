from __future__ import annotations

import asyncio
import platform
import queue
import shutil
import subprocess
import sys
import threading
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from agent_voice.adapter import Agent, PexpectAgent
from agent_voice.loop import CollectOutput, Transcript, TranscriptInput, VoiceLoop
from agent_voice.loop import VoiceLoopEvent
from agent_voice.presenter import VoicePresenter


KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)
KOKORO_MODEL_MIN_BYTES = 50 * 1024 * 1024
KOKORO_VOICES_MIN_BYTES = 1 * 1024 * 1024


class AudioPlayer(Protocol):
    def play(self, audio: Any, sample_rate: int) -> None:
        """Play audio and block until playback finishes."""

    def stop(self) -> None:
        """Stop currently playing audio."""


class ChunkedAudioPlayer(AudioPlayer, Protocol):
    def play_chunks(
        self,
        audio: Any,
        sample_rate: int,
        *,
        chunk_size: int,
        before_play: Callable[[Any], None],
    ) -> None:
        """Play audio in chunks, invoking a hook immediately before each chunk."""


class EchoCanceller(Protocol):
    def analyze_render(self, frame: Any) -> None:
        """Feed played speaker audio as the AEC reverse stream."""

    def process_capture(self, frame: Any) -> Any:
        """Return microphone capture with render echo reduced."""


class Closeable(Protocol):
    def close(self) -> None:
        """Release owned resources."""


class DownloadReporter(Protocol):
    def invalid_cached_asset(self, path: Path, size: int, min_bytes: int) -> None:
        """Report that an existing cached file cannot be reused."""

    def download_start(self, path: Path, url: str) -> None:
        """Report that a download is starting."""

    def download_progress(
        self,
        path: Path,
        block_count: int,
        block_size: int,
        total_size: int,
    ) -> None:
        """Report download progress from urllib's reporthook."""

    def download_complete(self, path: Path, size: int) -> None:
        """Report that a download finished and passed validation."""


class NullDownloadReporter:
    def invalid_cached_asset(self, path: Path, size: int, min_bytes: int) -> None:
        return None

    def download_start(self, path: Path, url: str) -> None:
        return None

    def download_progress(
        self,
        path: Path,
        block_count: int,
        block_size: int,
        total_size: int,
    ) -> None:
        return None

    def download_complete(self, path: Path, size: int) -> None:
        return None


class TerminalVoiceObserver:
    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output

    def __call__(self, event: VoiceLoopEvent) -> None:
        output = self.output or sys.stdout

        if event.kind == "transcript":
            print(
                f"[{_format_transcript_source(event.source)}] {event.text}",
                file=output,
                flush=True,
            )
            return

        if event.kind == "agent_input":
            print(f"[agent input] {event.text}", file=output, flush=True)
            return

        if event.kind == "agent_output":
            text = event.text.rstrip()
            if text:
                print("[agent output]", file=output, flush=True)
                print(text, file=output, flush=True)
            return

        if event.kind == "speech_summary":
            print(f"[voice summary] {event.text}", file=output, flush=True)
            return

        if event.kind == "interrupt":
            print(f"[interrupt] {event.text}", file=output, flush=True)
            return

        if event.kind == "exit":
            print(f"[exit] {event.text}", file=output, flush=True)
            return

        if event.kind == "queued_transcript":
            print(f"[queued typed input] {event.text}", file=output, flush=True)
            return

        if event.kind == "ignored_transcript":
            print(
                f"[ignored while speaking] {event.text}",
                file=output,
                flush=True,
            )
            return

        if event.kind == "ignored_self_echo":
            print(
                f"[ignored self echo] {event.text}",
                file=output,
                flush=True,
            )


class StderrDownloadReporter:
    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output
        self._last_percent_by_path: dict[Path, int] = {}

    def invalid_cached_asset(self, path: Path, size: int, min_bytes: int) -> None:
        print(
            f"Cached asset {path.name} is too small "
            f"({size} bytes; expected at least {min_bytes}); re-downloading.",
            file=self._output,
        )

    def download_start(self, path: Path, url: str) -> None:
        print(f"Downloading {path.name} from {url}", file=self._output)

    def download_progress(
        self,
        path: Path,
        block_count: int,
        block_size: int,
        total_size: int,
    ) -> None:
        if total_size <= 0:
            return

        downloaded = min(block_count * block_size, total_size)
        percent = int(downloaded * 100 / total_size)
        last_percent = self._last_percent_by_path.get(path, -10)
        if percent < 100 and percent < last_percent + 10:
            return

        self._last_percent_by_path[path] = percent
        print(
            f"{path.name}: {percent}% ({_format_bytes(downloaded)} / "
            f"{_format_bytes(total_size)})",
            file=self._output,
        )

    def download_complete(self, path: Path, size: int) -> None:
        print(f"Downloaded {path.name} ({_format_bytes(size)}).", file=self._output)

    @property
    def _output(self) -> TextIO:
        return self.output or sys.stderr


@dataclass
class SoundDevicePlayer:
    output_device: int | str | None = None

    def play(self, audio: Any, sample_rate: int) -> None:
        import sounddevice as sd

        sd.play(audio, sample_rate, blocking=True, device=self.output_device)

    def play_chunks(
        self,
        audio: Any,
        sample_rate: int,
        *,
        chunk_size: int,
        before_play: Callable[[Any], None],
    ) -> None:
        import numpy as np
        import sounddevice as sd

        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            stream_audio = array.reshape(-1, 1)
        else:
            stream_audio = array

        channels = 1 if stream_audio.ndim == 1 else int(stream_audio.shape[1])
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=self.output_device,
        ) as stream:
            for start in range(0, len(stream_audio), chunk_size):
                chunk = stream_audio[start : start + chunk_size]
                if len(chunk) == 0:
                    continue
                before_play(chunk)
                stream.write(chunk)

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()


@dataclass
class LiveKitEchoCanceller:
    sample_rate: int = 16000
    frame_ms: int = 10
    stream_delay_ms: int = 0
    apm: Any | None = None
    frame_factory: Any | None = None

    def __post_init__(self) -> None:
        if self.apm is None or self.frame_factory is None:
            from livekit import rtc

            if self.apm is None:
                self.apm = rtc.AudioProcessingModule(
                    echo_cancellation=True,
                    noise_suppression=True,
                    high_pass_filter=True,
                    auto_gain_control=False,
                )
            if self.frame_factory is None:
                self.frame_factory = rtc.AudioFrame
        if self.stream_delay_ms:
            self.apm.set_stream_delay_ms(self.stream_delay_ms)

    def analyze_render(self, frame: Any) -> None:
        for audio_frame, _sample_count in self._livekit_frames(frame):
            self.apm.process_reverse_stream(audio_frame)

    def process_capture(self, frame: Any) -> Any:
        import numpy as np

        outputs = []
        for audio_frame, sample_count in self._livekit_frames(frame):
            self.apm.process_stream(audio_frame)
            processed = np.asarray(audio_frame.data, dtype=np.int16)
            outputs.append(processed[:sample_count].astype(np.float32) / 32767.0)
        if not outputs:
            return np.asarray(frame, dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32)

    def _livekit_frames(self, frame: Any) -> list[tuple[Any, int]]:
        import numpy as np

        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []

        frame_samples = max(1, int(self.sample_rate * self.frame_ms / 1000))
        livekit_frames = []
        for start in range(0, len(samples), frame_samples):
            chunk = samples[start : start + frame_samples]
            sample_count = int(chunk.size)
            if sample_count < frame_samples:
                chunk = np.pad(chunk, (0, frame_samples - sample_count))
            clipped = np.clip(chunk, -1.0, 1.0)
            int16_chunk = (clipped * 32767).astype(np.int16)
            audio_frame = self.frame_factory(
                bytearray(int16_chunk.tobytes()),
                self.sample_rate,
                1,
                frame_samples,
            )
            livekit_frames.append((audio_frame, sample_count))
        return livekit_frames


@dataclass
class AecAudioPlayer:
    player: AudioPlayer
    echo_canceller: EchoCanceller
    target_sample_rate: int
    frame_ms: int = 10

    def play(self, audio: Any, sample_rate: int) -> None:
        output_audio = _float32_audio(audio)
        chunk_size = max(1, int(sample_rate * self.frame_ms / 1000))
        play_chunks = getattr(self.player, "play_chunks", None)
        if play_chunks is None:
            self._play_chunks_fallback(output_audio, sample_rate, chunk_size)
            return

        play_chunks(
            output_audio,
            sample_rate,
            chunk_size=chunk_size,
            before_play=lambda chunk: self._feed_render_reference(chunk, sample_rate),
        )

    def stop(self) -> None:
        self.player.stop()

    def _play_chunks_fallback(
        self,
        audio: Any,
        sample_rate: int,
        chunk_size: int,
    ) -> None:
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            if len(chunk) == 0:
                continue
            self._feed_render_reference(chunk, sample_rate)
            self.player.play(chunk, sample_rate)

    def _feed_render_reference(self, chunk: Any, sample_rate: int) -> None:
        render = _float32_mono_audio(chunk)
        render = _resample_mono_audio(
            render,
            source_sample_rate=sample_rate,
            target_sample_rate=self.target_sample_rate,
        )
        for frame in _audio_frames(render, self.target_sample_rate, self.frame_ms):
            self.echo_canceller.analyze_render(frame)


@dataclass
class KokoroSpeaker:
    kokoro: Any
    player: AudioPlayer
    voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"

    @classmethod
    def from_cache(
        cls,
        *,
        cache_dir: Path,
        voice: str = "af_sarah",
        speed: float = 1.0,
        lang: str = "en-us",
        output_device: int | str | None = None,
        player: AudioPlayer | None = None,
    ) -> KokoroSpeaker:
        from kokoro_onnx import Kokoro

        model_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"
        reporter = StderrDownloadReporter()
        _download_if_missing(
            KOKORO_MODEL_URL,
            model_path,
            min_bytes=KOKORO_MODEL_MIN_BYTES,
            reporter=reporter,
        )
        _download_if_missing(
            KOKORO_VOICES_URL,
            voices_path,
            min_bytes=KOKORO_VOICES_MIN_BYTES,
            reporter=reporter,
        )
        return cls(
            kokoro=Kokoro(str(model_path), str(voices_path)),
            player=player or SoundDevicePlayer(output_device=output_device),
            voice=voice,
            speed=speed,
            lang=lang,
        )

    def say(self, text: str) -> None:
        audio, sample_rate = self.kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )
        self.player.play(audio, sample_rate)

    def stop(self) -> None:
        self.player.stop()


@dataclass
class SupertonicSpeaker:
    tts: Any
    player: AudioPlayer
    voice: str = "M2"
    speed: float = 1.0
    lang: str = "ko"
    sample_rate: int = 44100

    @classmethod
    def from_cache(
        cls,
        *,
        voice: str = "M2",
        speed: float = 1.0,
        lang: str = "ko",
        output_device: int | str | None = None,
        player: AudioPlayer | None = None,
    ) -> SupertonicSpeaker:
        from supertonic import TTS

        tts = TTS(auto_download=True)
        sample_rate = int(getattr(tts, "sample_rate", 44100) or 44100)
        return cls(
            tts=tts,
            player=player or SoundDevicePlayer(output_device=output_device),
            voice=voice,
            speed=speed,
            lang=lang,
            sample_rate=sample_rate,
        )

    def say(self, text: str) -> None:
        style = self.tts.get_voice_style(voice_name=self.voice)
        audio, _ = self.tts.synthesize(
            text,
            voice_style=style,
            speed=self.speed,
            lang=_supertonic_lang(self.lang),
        )
        self.player.play(_mono_audio(audio), self.sample_rate)

    def stop(self) -> None:
        self.player.stop()


@dataclass
class MacOSSaySpeaker:
    voice: str | None = None
    rate: int | None = None
    _process: subprocess.Popen[Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def say(self, text: str) -> None:
        self.stop()
        command = ["say"]
        if self.voice:
            command.extend(["-v", self.voice])
        if self.rate is not None:
            command.extend(["-r", str(self.rate)])
        command.append(text)

        self._process = subprocess.Popen(command)
        try:
            self._process.wait()
        finally:
            self._process = None

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=0.5)


@dataclass
class ManagedVoiceLoop:
    loop: VoiceLoop
    agent: Agent
    closeables: Sequence[Closeable] = ()
    status_lines: Sequence[str] = ()
    output: TextIO | None = None

    def run_forever(self) -> int:
        try:
            for line in self.status_lines:
                print(line, file=self.output or sys.stdout, flush=True)
            self.agent.start()
            return self.loop.run_forever()
        finally:
            self.agent.stop()
            for closeable in self.closeables:
                closeable.close()


class MicrophoneWhisperTranscriptSource:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        vad_threshold: float = 0.01,
        silence_seconds: float = 0.7,
        min_speech_seconds: float = 0.25,
        max_utterance_seconds: float = 12.0,
        whisper_model: str = "tiny",
        whisper_language: str | None = None,
        compute_type: str = "int8",
        use_smart_turn: bool = True,
        input_device: int | str | None = None,
        echo_canceller: EchoCanceller | None = None,
    ) -> None:
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel

        self._np = np
        self._sd = sd
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = max(1, int(sample_rate * chunk_ms / 1000))
        self.vad_threshold = vad_threshold
        self.silence_seconds = silence_seconds
        self.min_speech_seconds = min_speech_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.whisper_language = whisper_language
        self.use_smart_turn = use_smart_turn
        self.echo_canceller = echo_canceller
        self._transcripts: queue.Queue[str] = queue.Queue()
        self._audio_chunks: queue.Queue[Any] = queue.Queue()
        self._stop = threading.Event()
        self._model = WhisperModel(whisper_model, device="cpu", compute_type=compute_type)
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._on_audio,
            device=input_device,
        )
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._stream.start()
        self._worker.start()

    def next_transcript(self) -> Transcript | None:
        try:
            return Transcript(
                text=self._transcripts.get_nowait(),
                source="microphone",
            )
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        stop = getattr(self._stream, "stop", None)
        close = getattr(self._stream, "close", None)
        if stop is not None:
            stop()
        if close is not None:
            close()
        self._worker.join(timeout=1.0)

    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if self._stop.is_set():
            return
        self._audio_chunks.put(indata.copy())

    def _run(self) -> None:
        buffer: list[Any] = []
        silence_seconds = 0.0
        speech_seconds = 0.0

        while not self._stop.is_set():
            try:
                chunk = self._audio_chunks.get(timeout=0.1)
            except queue.Empty:
                continue

            mono = self._np.asarray(chunk, dtype=self._np.float32).reshape(-1)
            if self.echo_canceller is not None:
                mono = self._np.asarray(
                    self.echo_canceller.process_capture(mono),
                    dtype=self._np.float32,
                ).reshape(-1)
            chunk_seconds = len(mono) / self.sample_rate
            rms = float(self._np.sqrt(self._np.mean(mono * mono))) if len(mono) else 0.0
            is_speech = rms >= self.vad_threshold

            if is_speech:
                buffer.append(mono)
                speech_seconds += chunk_seconds
                silence_seconds = 0.0
            elif buffer:
                buffer.append(mono)
                silence_seconds += chunk_seconds

            if not buffer:
                continue

            if speech_seconds >= self.max_utterance_seconds:
                self._finalize(buffer)
                buffer = []
                silence_seconds = 0.0
                speech_seconds = 0.0
                continue

            if silence_seconds >= self.silence_seconds:
                if speech_seconds >= self.min_speech_seconds:
                    self._finalize(buffer)
                buffer = []
                silence_seconds = 0.0
                speech_seconds = 0.0

    def _finalize(self, chunks: Sequence[Any]) -> None:
        audio = self._np.concatenate(chunks).astype(self._np.float32)
        if self.use_smart_turn and not self._smart_turn_complete(audio):
            return

        segments, _ = self._model.transcribe(
            audio,
            language=self.whisper_language,
            beam_size=1,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        if transcript:
            self._transcripts.put(transcript)

    def _smart_turn_complete(self, audio: Any) -> bool:
        try:
            from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
                LocalSmartTurnAnalyzerV3,
            )

            analyzer = LocalSmartTurnAnalyzerV3(
                sample_rate=self.sample_rate,
                params=SmartTurnParams(
                    stop_secs=self.silence_seconds,
                    pre_speech_ms=0.0,
                    max_duration_secs=self.max_utterance_seconds,
                ),
            )
            clipped = self._np.clip(audio, -1.0, 1.0)
            int16_audio = (clipped * 32767).astype(self._np.int16)
            analyzer.append_audio(int16_audio.tobytes(), is_speech=True)
            _, metrics = asyncio.run(analyzer.analyze_end_of_turn())
            return metrics is None or metrics.is_complete
        except Exception:
            return True


class KeyboardTranscriptSource:
    def __init__(self, input_stream: TextIO | None = None) -> None:
        self._input_stream = input_stream or sys.stdin
        self._transcripts: queue.Queue[Transcript] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def next_transcript(self) -> Transcript | None:
        try:
            return self._transcripts.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=0.1)

    def _run(self) -> None:
        while not self._stop.is_set():
            line = self._input_stream.readline()
            if line == "":
                return
            text = line.strip()
            if text:
                self._transcripts.put(Transcript(text=text, source="keyboard"))


class MergedTranscriptSource:
    def __init__(self, sources: Sequence[Any]) -> None:
        self._sources = tuple(sources)

    def next_transcript(self) -> TranscriptInput | None:
        for source in self._sources:
            transcript = source.next_transcript()
            if transcript is not None:
                return transcript
        return None

    def close(self) -> None:
        for source in self._sources:
            close = getattr(source, "close", None)
            if close is not None:
                close()


def build_local_voice_loop(
    *,
    command: tuple[str, ...],
    language: str,
    collect_output: CollectOutput,
    cache_dir: Path,
    whisper_model: str = "tiny",
    whisper_language: str | None = None,
    tts_voice: str = "af_sarah",
    tts_lang: str = "en-us",
    tts_speed: float = 1.0,
    sample_rate: int = 16000,
    vad_threshold: float = 0.01,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    keyboard_input: bool = True,
    keyboard_input_stream: TextIO | None = None,
    terminal_output: TextIO | None = None,
    transparent_io: bool = True,
    tts_backend: str = "auto",
    supertonic_voice: str = "M2",
    macos_say_voice: str | None = None,
    macos_say_rate: int | None = None,
    agent: Agent | None = None,
    aec_enabled: bool = True,
    aec_delay_ms: int = 120,
) -> ManagedVoiceLoop:
    agent = agent or PexpectAgent(command=command)
    echo_canceller = (
        LiveKitEchoCanceller(
            sample_rate=sample_rate,
            stream_delay_ms=max(0, aec_delay_ms),
        )
        if aec_enabled
        else None
    )
    microphone_source = MicrophoneWhisperTranscriptSource(
        sample_rate=sample_rate,
        vad_threshold=vad_threshold,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        input_device=input_device,
        echo_canceller=echo_canceller,
    )
    sources: list[Any] = [microphone_source]
    keyboard_source = _build_keyboard_source(
        enabled=keyboard_input,
        input_stream=keyboard_input_stream,
    )
    if keyboard_source is not None:
        sources.insert(0, keyboard_source)
    transcript_source = MergedTranscriptSource(sources)
    try:
        speaker, resolved_tts_backend = _build_speaker(
            backend=tts_backend,
            cache_dir=cache_dir,
            tts_voice=tts_voice,
            supertonic_voice=supertonic_voice,
            tts_lang=tts_lang,
            tts_speed=tts_speed,
            output_device=output_device,
            macos_say_voice=macos_say_voice,
            macos_say_rate=macos_say_rate,
            echo_canceller=echo_canceller,
            aec_sample_rate=sample_rate,
        )
    except Exception:
        transcript_source.close()
        raise
    loop = VoiceLoop(
        transcript_source=transcript_source,
        agent=agent,
        presenter=VoicePresenter(language=language),
        speaker=speaker,
        collect_output=collect_output,
        observer=(
            TerminalVoiceObserver(terminal_output) if transparent_io else None
        ),
    )
    status_lines = _build_status_lines(
        command=command,
        keyboard_enabled=keyboard_source is not None,
        language=language,
        whisper_language=whisper_language,
        tts_backend=resolved_tts_backend,
        aec_enabled=aec_enabled,
    )
    return ManagedVoiceLoop(
        loop=loop,
        agent=agent,
        closeables=(transcript_source,),
        status_lines=status_lines if transparent_io else (),
        output=terminal_output,
    )


def _build_speaker(
    *,
    backend: str,
    cache_dir: Path,
    tts_voice: str,
    supertonic_voice: str,
    tts_lang: str,
    tts_speed: float,
    output_device: int | str | None,
    macos_say_voice: str | None,
    macos_say_rate: int | None,
    echo_canceller: EchoCanceller | None,
    aec_sample_rate: int,
) -> tuple[Any, str]:
    if backend not in {"auto", "kokoro", "macos-say", "supertonic"}:
        raise ValueError(
            "tts backend must be one of: auto, kokoro, macos-say, supertonic"
        )

    if backend == "auto" and _should_use_supertonic(tts_lang):
        player = _build_audio_player(
            output_device=output_device,
            echo_canceller=echo_canceller,
            aec_sample_rate=aec_sample_rate,
        )
        return (
            SupertonicSpeaker.from_cache(
                voice=supertonic_voice,
                speed=tts_speed,
                lang=tts_lang,
                output_device=output_device,
                player=player,
            ),
            "supertonic",
        )

    if backend == "auto" and _should_use_macos_say(tts_lang):
        return (
            MacOSSaySpeaker(
                voice=macos_say_voice or _default_macos_say_voice(tts_lang),
                rate=macos_say_rate,
            ),
            "macos-say",
        )

    if backend == "supertonic":
        player = _build_audio_player(
            output_device=output_device,
            echo_canceller=echo_canceller,
            aec_sample_rate=aec_sample_rate,
        )
        return (
            SupertonicSpeaker.from_cache(
                voice=supertonic_voice,
                speed=tts_speed,
                lang=tts_lang,
                output_device=output_device,
                player=player,
            ),
            "supertonic",
        )

    if backend == "macos-say":
        if not _macos_say_available():
            raise RuntimeError("macos-say backend requires the macOS `say` command.")
        return (
            MacOSSaySpeaker(
                voice=macos_say_voice or _default_macos_say_voice(tts_lang),
                rate=macos_say_rate,
            ),
            "macos-say",
        )

    player = _build_audio_player(
        output_device=output_device,
        echo_canceller=echo_canceller,
        aec_sample_rate=aec_sample_rate,
    )
    return (
        KokoroSpeaker.from_cache(
            cache_dir=cache_dir / "kokoro",
            voice=tts_voice,
            speed=tts_speed,
            lang=tts_lang,
            output_device=output_device,
            player=player,
        ),
        "kokoro",
    )


def _build_audio_player(
    *,
    output_device: int | str | None,
    echo_canceller: EchoCanceller | None,
    aec_sample_rate: int,
) -> AudioPlayer:
    player = SoundDevicePlayer(output_device=output_device)
    if echo_canceller is None:
        return player
    return AecAudioPlayer(
        player=player,
        echo_canceller=echo_canceller,
        target_sample_rate=aec_sample_rate,
    )


def _should_use_supertonic(tts_lang: str) -> bool:
    return tts_lang.casefold().startswith("ko")


def _should_use_macos_say(tts_lang: str) -> bool:
    return (
        tts_lang.casefold().startswith("ko")
        and platform.system() == "Darwin"
        and _macos_say_available()
    )


def _macos_say_available() -> bool:
    return shutil.which("say") is not None


def _default_macos_say_voice(tts_lang: str) -> str | None:
    if not tts_lang.casefold().startswith("ko"):
        return None

    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    for line in result.stdout.splitlines():
        if "ko_KR" in line or "Korean" in line:
            parts = line.split()
            if parts:
                return parts[0]
    return None


def _supertonic_lang(tts_lang: str) -> str:
    normalized = tts_lang.casefold()
    if normalized.startswith("ko"):
        return "ko"
    if normalized.startswith("en"):
        return "en"
    return normalized.split("-", maxsplit=1)[0]


def _mono_audio(audio: Any) -> Any:
    import numpy as np

    array = np.asarray(audio)
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    return array


def _float32_mono_audio(audio: Any) -> Any:
    import numpy as np

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        if 1 in array.shape:
            return array.reshape(-1)
        return np.mean(array, axis=1).astype(np.float32)
    return array.reshape(-1)


def _float32_audio(audio: Any) -> Any:
    import numpy as np

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _resample_mono_audio(
    audio: Any,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> Any:
    import numpy as np

    if source_sample_rate == target_sample_rate or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)

    duration = len(audio) / float(source_sample_rate)
    target_length = max(1, int(round(duration * target_sample_rate)))
    source_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _audio_frames(audio: Any, sample_rate: int, frame_ms: int) -> list[Any]:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    frames = []
    for start in range(0, len(samples), frame_samples):
        frame = samples[start : start + frame_samples]
        if len(frame):
            frames.append(frame.astype(np.float32))
    return frames


def _build_keyboard_source(
    *,
    enabled: bool,
    input_stream: TextIO | None = None,
) -> KeyboardTranscriptSource | None:
    if not enabled:
        return None

    stream = input_stream or sys.stdin
    if input_stream is None and not _is_interactive_stream(stream):
        return None
    return KeyboardTranscriptSource(input_stream=stream)


def _is_interactive_stream(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty is not None and isatty())


def _build_status_lines(
    *,
    command: tuple[str, ...],
    keyboard_enabled: bool,
    language: str,
    whisper_language: str | None,
    tts_backend: str,
    aec_enabled: bool,
) -> tuple[str, ...]:
    lines = [
        f"agent-voice session started: {' '.join(command)}",
        f"summary language: {language}; stt language: {whisper_language or 'auto'}",
        f"tts backend: {tts_backend}",
        f"aec: {'livekit' if aec_enabled else 'disabled'}",
    ]
    if keyboard_enabled:
        lines.append(
            "Speak commands, or type a line and press Enter. "
            "Say/type '종료' or 'exit' to quit."
        )
    else:
        lines.append("Speak commands. Say '종료' or 'exit' to quit.")
    return tuple(lines)


def _format_transcript_source(source: str) -> str:
    if source == "microphone":
        return "voice transcript"
    if source == "keyboard":
        return "typed input"
    if source == "text":
        return "text input"
    return "transcript"


def _download_if_missing(
    url: str,
    path: Path,
    *,
    min_bytes: int = 1,
    reporter: DownloadReporter | None = None,
) -> None:
    reporter = reporter or NullDownloadReporter()
    if _asset_has_expected_size(path, min_bytes):
        return

    if path.exists():
        size = path.stat().st_size
        reporter.invalid_cached_asset(path, size, min_bytes)
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.download")
    temp_path.unlink(missing_ok=True)
    try:
        reporter.download_start(path, url)
        urllib.request.urlretrieve(
            url,
            temp_path,
            reporthook=lambda block_count, block_size, total_size: (
                reporter.download_progress(path, block_count, block_size, total_size)
            ),
        )
        if not _asset_has_expected_size(temp_path, min_bytes):
            size = temp_path.stat().st_size if temp_path.exists() else 0
            raise RuntimeError(
                f"Downloaded asset {path.name} is too small "
                f"({size} bytes; expected at least {min_bytes}). "
                "The download may have been interrupted or rate-limited."
            )
        temp_path.replace(path)
        reporter.download_complete(path, path.stat().st_size)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _asset_has_expected_size(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _format_bytes(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
