from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.voice_builder.features import extract_reference_features, load_mono_audio
from experiments.voice_builder.latent_basis import (
    DP_DIMS,
    TTL_DIMS,
    StyleArrays,
    build_directional_style,
    direction_alpha,
)
from experiments.voice_builder.optimize import coefficient_grid


StyleJson = dict[str, Any]
PreviewFn = Callable[..., None]
OptimizerFn = Callable[..., dict[str, float]]

TTL_STD = 0.0625
DP_STD = 0.25
DEFAULT_TTL_PROJECTION_SCALE = 0.5
DEFAULT_DP_PROJECTION_SCALE = 0.5
DEFAULT_PREVIEW_SPEED = 0.7


@dataclass(frozen=True)
class BuildResult:
    style_path: Path
    preview_path: Path


def build_voice_style_payload(
    audio_path: Path,
    *,
    ttl_projection_scale: float = DEFAULT_TTL_PROJECTION_SCALE,
    dp_projection_scale: float = DEFAULT_DP_PROJECTION_SCALE,
    alpha: float | None = None,
    calibration_styles: dict[str, StyleArrays | tuple[np.ndarray, np.ndarray]] | None = None,
) -> StyleJson:
    audio, sample_rate = load_mono_audio(audio_path)
    if audio.size == 0:
        raise ValueError(f"audio file is empty: {audio_path}")

    features = extract_reference_features(audio_path)
    brightness_gain = _brightness_gain(features.centroid)
    projected_ttl = _audio_to_ttl(
        audio,
        sample_rate,
        brightness_gain=brightness_gain,
    ) * ttl_projection_scale
    calibration = _normalize_calibration_styles(calibration_styles or {})
    if calibration:
        alpha = direction_alpha(features) if alpha is None else alpha
        ttl, dp = build_directional_style(
            projected_ttl=projected_ttl,
            calibration_styles=calibration,
            alpha=alpha,
        )
        method = "spectral_projection_v4"
        calibration_mode = "f2_plus_f1_minus_m2_direction"
    else:
        alpha = 0.0 if alpha is None else alpha
        ttl = projected_ttl
        dp = _audio_to_dp(audio, sample_rate) * dp_projection_scale
        method = "direct_projection_v1"
        calibration_mode = "none"

    return {
        "style_ttl": {
            "dims": TTL_DIMS,
            "data": ttl.reshape(TTL_DIMS).astype(float).tolist(),
        },
        "style_dp": {
            "dims": DP_DIMS,
            "data": dp.reshape(DP_DIMS).astype(float).tolist(),
        },
        "metadata": {
            "generator": "agent-voice voice_builder.mp3_to_json",
            "source_audio": str(audio_path),
            "source_sample_rate": sample_rate,
            "duration_seconds": float(audio.size / sample_rate),
            "method": method,
            "ttl_projection_scale": ttl_projection_scale,
            "dp_projection_scale": dp_projection_scale,
            "calibration_mode": calibration_mode,
            "direction_alpha": alpha,
            "brightness_gain": brightness_gain,
            "reference_features": {
                "duration": features.duration,
                "f0": features.f0,
                "centroid": features.centroid,
                "rolloff": features.rolloff,
                "rms": features.rms,
            },
        },
    }


def build_voice_style_json(
    input_audio: Path,
    *,
    out_dir: Path,
    text: str,
    speed: float = DEFAULT_PREVIEW_SPEED,
    lang: str = "ko",
    alpha: float | None = None,
    ttl_projection_scale: float = DEFAULT_TTL_PROJECTION_SCALE,
    optimize: bool = False,
    optimizer_fn: OptimizerFn | None = None,
    synthesize_preview_fn: PreviewFn = None,
) -> BuildResult:
    if optimize:
        optimizer = optimizer_fn or optimize_coefficients
        result = optimizer(
            input_audio=input_audio,
            text=text,
            speed=speed,
            seed_alpha=alpha,
            seed_projection_scale=ttl_projection_scale,
        )
        alpha = float(result["alpha"])
        ttl_projection_scale = float(result["ttl_projection_scale"])

    payload = build_voice_style_payload(
        input_audio,
        ttl_projection_scale=ttl_projection_scale,
        alpha=alpha,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    style_path = out_dir / "voice-style.json"
    preview_path = out_dir / "preview.wav"
    style_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    preview_fn = synthesize_preview_fn or synthesize_preview
    preview_fn(payload, text=text, output_path=preview_path, speed=speed, lang=lang)
    return BuildResult(style_path=style_path, preview_path=preview_path)


def synthesize_preview(
    payload: StyleJson,
    *,
    text: str,
    output_path: Path,
    speed: float,
    lang: str,
) -> None:
    from supertonic import TTS
    from supertonic.pipeline import Style

    tts = TTS(auto_download=True)
    style = Style(
        np.asarray(payload["style_ttl"]["data"], dtype=np.float32),
        np.asarray(payload["style_dp"]["data"], dtype=np.float32),
    )
    audio, _ = tts.synthesize(
        text,
        voice_style=style,
        speed=speed,
        lang=lang,
        total_steps=12,
        silence_duration=0.25,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        np.asarray(audio, dtype=np.float32).squeeze(),
        int(getattr(tts, "sample_rate", 44100) or 44100),
        format="WAV",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Supertonic voice-style JSON directly from an MP3/WAV file.",
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument(
        "--text",
        default="안녕하세요 작업을 시작할까요 무엇을 명령하실겁니까 휴먼",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".cache/agent-voice/voice-builder/mp3-to-json"),
    )
    parser.add_argument("--speed", type=float, default=DEFAULT_PREVIEW_SPEED)
    parser.add_argument(
        "--lang",
        default="ko",
        help="Supertonic synthesis language for the preview, for example ko or en.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Search nearby latent coefficients before writing the final JSON.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    synthesize_preview_fn: PreviewFn | None = None,
    optimizer_fn: OptimizerFn | None = None,
) -> int:
    args = parse_args(argv)
    result = build_voice_style_json(
        args.input,
        out_dir=args.out_dir,
        text=args.text,
        speed=args.speed,
        lang=args.lang,
        optimize=args.optimize,
        optimizer_fn=optimizer_fn,
        synthesize_preview_fn=synthesize_preview_fn,
    )
    print(result.style_path)
    print(result.preview_path)
    return 0


def optimize_coefficients(
    *,
    input_audio: Path,
    text: str,
    speed: float,
    seed_alpha: float | None,
    seed_projection_scale: float,
) -> dict[str, float]:
    features = extract_reference_features(input_audio)
    alpha = direction_alpha(features) if seed_alpha is None else seed_alpha
    # Full black-box scoring is intentionally a later hook; this returns the
    # center candidate while exposing the coefficient grid shape.
    grid = coefficient_grid(
        seed_alpha=alpha,
        seed_projection_scale=seed_projection_scale,
    )
    center = min(
        grid,
        key=lambda item: abs(item.alpha - alpha)
        + abs(item.ttl_projection_scale - seed_projection_scale),
    )
    return {
        "alpha": center.alpha,
        "ttl_projection_scale": center.ttl_projection_scale,
        "score": 0.0,
    }


def _audio_to_ttl(
    audio: np.ndarray,
    sample_rate: int,
    *,
    brightness_gain: float,
) -> np.ndarray:
    frame = max(512, int(sample_rate * 0.04))
    hop = max(128, frame // 4)
    _, _, spec = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=frame,
        noverlap=max(0, frame - hop),
        boundary=None,
    )
    magnitude = np.log1p(np.abs(spec).T)
    if magnitude.size == 0:
        magnitude = np.zeros((1, 1), dtype=np.float32)
    magnitude = _apply_brightness_tilt(magnitude, brightness_gain)

    projected = signal.resample(magnitude, TTL_DIMS[1], axis=0)
    projected = signal.resample(projected, TTL_DIMS[2], axis=1)
    return _normalize_latent(projected, target_std=TTL_STD, clip=0.85)


def _audio_to_dp(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    segments = np.array_split(audio, DP_DIMS[1])
    rows = []
    for segment in segments:
        if segment.size == 0:
            segment = np.zeros(1, dtype=np.float32)
        rows.append(_segment_features(segment, sample_rate))
    features = np.asarray(rows, dtype=np.float32)
    projected = signal.resample(features, DP_DIMS[2], axis=1)
    return _normalize_latent(projected, target_std=DP_STD, clip=0.85)


def _segment_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio))))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(np.int8)))))
    peak = float(np.max(np.abs(audio)))
    freqs, _, spec = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=min(512, max(64, len(audio))),
        noverlap=min(256, max(0, len(audio) // 2 - 1)),
        boundary=None,
    )
    magnitude = np.abs(spec)
    energy = np.maximum(magnitude.sum(axis=0), 1e-9)
    centroid = float(np.mean((magnitude * freqs[:, None]).sum(axis=0) / energy))
    rolloff = 0.0
    if magnitude.size:
        cumulative = np.cumsum(magnitude, axis=0)
        thresholds = cumulative[-1, :] * 0.85
        rolloff_values = [
            freqs[min(np.searchsorted(cumulative[:, index], thresholds[index]), len(freqs) - 1)]
            for index in range(cumulative.shape[1])
        ]
        rolloff = float(np.mean(rolloff_values))
    p10, p50, p90 = np.percentile(audio, [10, 50, 90])
    return np.asarray(
        [
            rms,
            zcr,
            peak,
            centroid / max(sample_rate / 2, 1),
            rolloff / max(sample_rate / 2, 1),
            float(np.mean(audio)),
            float(np.std(audio)),
            float(p10),
            float(p50),
            float(p90),
            float(len(audio) / sample_rate),
            rms / max(peak, 1e-6),
            float(np.mean(np.abs(audio))),
            float(np.max(audio)),
            float(np.min(audio)),
            1.0,
        ],
        dtype=np.float32,
    )


def _brightness_gain(centroid: float) -> float:
    return 1.0 + min(1.0, max(0.0, (centroid - 3000.0) / 2500.0))


def _apply_brightness_tilt(magnitude: np.ndarray, gain: float) -> np.ndarray:
    if magnitude.shape[1] <= 1 or gain <= 1.0:
        return magnitude
    tilt = np.linspace(1.0, gain, magnitude.shape[1], dtype=np.float32)
    return magnitude * tilt[None, :]


def _normalize_calibration_styles(
    styles: dict[str, StyleArrays | tuple[np.ndarray, np.ndarray]],
) -> dict[str, StyleArrays]:
    normalized: dict[str, StyleArrays] = {}
    for label, style in styles.items():
        if isinstance(style, StyleArrays):
            normalized[label] = style
        else:
            ttl, dp = style
            normalized[label] = StyleArrays(
                ttl=np.asarray(ttl, dtype=np.float32),
                dp=np.asarray(dp, dtype=np.float32),
            )
    return normalized


def _normalize_latent(
    values: np.ndarray,
    *,
    target_std: float,
    clip: float,
) -> np.ndarray:
    latent = np.asarray(values, dtype=np.float32)
    latent = latent - float(np.mean(latent))
    std = float(np.std(latent))
    if std > 1e-9:
        latent = latent / std
    latent = latent * target_std
    return np.clip(latent, -clip, clip).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
