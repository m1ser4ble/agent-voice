from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_audio(
    path: Path,
    *,
    sample_rate: int,
    max_seconds: float | None = None,
) -> torch.Tensor:
    audio, input_sample_rate = sf.read(path, dtype="float32", always_2d=True)
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
