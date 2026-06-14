from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import soundfile as sf
import torch
import torchaudio


def load_audio(
    path: Path,
    *,
    sample_rate: int,
    max_seconds: float | None = None,
) -> torch.Tensor:
    audio, input_sample_rate = _read_audio(path, sample_rate=sample_rate)
    waveform = torch.from_numpy(audio).mean(dim=1)
    if input_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=input_sample_rate,
            new_freq=sample_rate,
        )
    if max_seconds is not None:
        max_samples = int(sample_rate * max_seconds)
        waveform = waveform[:max_samples]
    return waveform.contiguous()


def _read_audio(path: Path, *, sample_rate: int):
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return _read_audio_with_ffmpeg(path, sample_rate=sample_rate)


def _read_audio_with_ffmpeg(path: Path, *, sample_rate: int):
    with tempfile.TemporaryDirectory(prefix="supertonic-audio-") as temp_dir:
        wav_path = Path(temp_dir) / "decoded.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(wav_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ffmpeg is required to decode audio unsupported by soundfile: {path}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            if len(stderr) > 2000:
                stderr = stderr[-2000:]
            raise RuntimeError(
                "ffmpeg failed to decode audio "
                f"path={path} returncode={exc.returncode} stderr={stderr!r}"
            ) from exc
        return sf.read(wav_path, dtype="float32", always_2d=True)


class LogMelExtractor:
    def __init__(
        self,
        *,
        sample_rate: int = 44_100,
        n_mels: int = 228,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 1:
            raise ValueError(f"waveform must be mono [samples], got {tuple(waveform.shape)}")
        mel = self.transform(waveform)
        return torch.log1p(mel).contiguous()
