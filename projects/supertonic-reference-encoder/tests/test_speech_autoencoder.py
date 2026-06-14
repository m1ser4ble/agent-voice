import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from supertonic_reference_encoder.speech_autoencoder import (
    AutoencoderTrainConfig,
    evaluate_autoencoder_audio,
    load_autoencoder_checkpoint,
    MultiResolutionMelLoss,
    SpeechAutoencoder,
    WaveformDataset,
    collate_waveforms,
    train_autoencoder,
    train_autoencoder_one_step,
)


def _write_wav(path: Path, frequency: float = 220.0) -> None:
    sample_rate = 16_000
    t = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_waveform_dataset_reads_audio_only_manifest(tmp_path):
    audio = tmp_path / "voice.wav"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    manifest.write_text(json.dumps({"audio": str(audio)}) + "\n", encoding="utf-8")

    dataset = WaveformDataset(manifest, sample_rate=16_000)
    item = dataset[0]

    assert item.waveform.ndim == 1
    assert item.waveform.numel() == 4_000
    assert item.metadata == {}


def test_speech_autoencoder_reconstructs_waveform_from_24_dimensional_latent(tmp_path):
    model = SpeechAutoencoder(sample_rate=16_000)
    waveform = torch.randn(2, 4_096)

    output = model(waveform)

    assert output.latent.shape[0] == 2
    assert output.latent.shape[1] == 24
    assert output.waveform.shape == waveform.shape


def test_multiresolution_mel_loss_is_zero_for_identical_audio():
    loss_fn = MultiResolutionMelLoss(sample_rate=16_000)
    waveform = torch.randn(2, 4_096)

    loss = loss_fn(waveform, waveform)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_train_autoencoder_one_step_runs_on_waveform_batch(tmp_path):
    audio = tmp_path / "voice.wav"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    manifest.write_text(json.dumps({"audio": str(audio)}) + "\n", encoding="utf-8")
    dataset = WaveformDataset(manifest, sample_rate=16_000)
    batch = collate_waveforms([dataset[0]])

    metrics = train_autoencoder_one_step(batch, device=torch.device("cpu"))

    assert metrics["loss"] > 0.0
    assert metrics["mel_loss"] > 0.0


def test_load_autoencoder_checkpoint_restores_model_weights(tmp_path):
    source = SpeechAutoencoder(sample_rate=16_000)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    target = SpeechAutoencoder(sample_rate=16_000)

    load_autoencoder_checkpoint(target, checkpoint)

    source_first = next(source.parameters()).detach()
    target_first = next(target.parameters()).detach()
    assert torch.allclose(source_first, target_first)


def test_evaluate_autoencoder_audio_returns_validation_metrics(tmp_path):
    audio = tmp_path / "validation.wav"
    _write_wav(audio)
    model = SpeechAutoencoder(sample_rate=16_000)
    loss_fn = MultiResolutionMelLoss(sample_rate=16_000)

    metrics = evaluate_autoencoder_audio(
        model,
        audio,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
        sample_rate=16_000,
        max_seconds=0.25,
    )

    assert metrics["validation_mel_loss"] >= 0.0
    assert metrics["validation_waveform_l1"] >= 0.0


def test_train_autoencoder_logs_validation_audio_metrics(tmp_path):
    audio = tmp_path / "train.wav"
    validation_audio = tmp_path / "validation.wav"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    _write_wav(validation_audio, frequency=440.0)
    manifest.write_text(json.dumps({"audio": str(audio)}) + "\n", encoding="utf-8")

    metrics = train_autoencoder(
        AutoencoderTrainConfig(
            manifest=manifest,
            output_dir=tmp_path / "runs",
            epochs=1,
            batch_size=1,
            sample_rate=16_000,
            max_seconds=0.25,
            validation_audio=validation_audio,
            device="cpu",
        )
    )

    assert metrics["validation_mel_loss"] >= 0.0
    logged = json.loads((tmp_path / "runs" / "metrics.jsonl").read_text().splitlines()[-1])
    assert "validation_mel_loss" in logged
