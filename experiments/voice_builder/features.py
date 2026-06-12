from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class ReferenceFeatures:
    duration: float
    f0: float
    centroid: float
    rolloff: float
    rms: float


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio, int(sample_rate)


def extract_reference_features(path: Path) -> ReferenceFeatures:
    audio, sample_rate = load_mono_audio(path)
    if audio.size == 0:
        raise ValueError(f"audio file is empty: {path}")
    return ReferenceFeatures(
        duration=float(audio.size / sample_rate),
        f0=estimate_f0(audio, sample_rate),
        centroid=spectral_centroid(audio, sample_rate),
        rolloff=spectral_rolloff(audio, sample_rate),
        rms=float(np.sqrt(np.mean(audio * audio))),
    )


def estimate_f0(audio: np.ndarray, sample_rate: int) -> float:
    frame = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.02)
    min_lag = max(1, int(sample_rate / 350))
    max_lag = max(min_lag + 1, int(sample_rate / 70))
    values: list[float] = []
    for start in range(0, max(0, len(audio) - frame), hop):
        segment = audio[start : start + frame]
        if segment.size < frame or np.sqrt(np.mean(segment * segment)) < 0.02:
            continue
        segment = segment - float(np.mean(segment))
        corr = signal.correlate(segment, segment, mode="full")[segment.size - 1 :]
        if max_lag >= corr.size:
            continue
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        strength = float(corr[lag] / max(corr[0], 1e-9))
        if strength > 0.25:
            values.append(sample_rate / lag)
    return float(np.median(values)) if values else 120.0


def spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    freqs, magnitude = _magnitude_spectrum(audio, sample_rate)
    energy = np.maximum(magnitude.sum(axis=0), 1e-9)
    centroid = (magnitude * freqs[:, None]).sum(axis=0) / energy
    return float(np.mean(centroid))


def spectral_rolloff(audio: np.ndarray, sample_rate: int, *, ratio: float = 0.85) -> float:
    freqs, magnitude = _magnitude_spectrum(audio, sample_rate)
    cumulative = np.cumsum(magnitude, axis=0)
    thresholds = cumulative[-1, :] * ratio
    rolloff = [
        freqs[min(np.searchsorted(cumulative[:, index], thresholds[index]), len(freqs) - 1)]
        for index in range(cumulative.shape[1])
    ]
    return float(np.mean(rolloff))


def _magnitude_spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, spec = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=min(2048, max(256, len(audio))),
        noverlap=min(1024, max(0, len(audio) // 2 - 1)),
        boundary=None,
    )
    return freqs, np.abs(spec)
