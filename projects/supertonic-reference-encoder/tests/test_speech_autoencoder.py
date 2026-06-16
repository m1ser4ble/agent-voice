import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from supertonic_reference_encoder.speech_autoencoder import (
    AutoencoderTrainConfig,
    evaluate_autoencoder_audio,
    load_autoencoder_checkpoint,
    MultiResolutionMelLoss,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    PaperAutoencoderAdversarialLoss,
    SpeechAutoencoder,
    WaveformDataset,
    collate_waveforms,
    train_autoencoder,
    train_autoencoder_one_step,
    _assert_finite_losses,
    _linear_log_spectrogram,
    _loss_precision_waveforms,
    _sanitize_waveform_for_discriminator,
    _resolve_mixed_precision,
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


def test_latent_decoder_uses_paper_dilation_pattern_and_projection_head():
    model = SpeechAutoencoder(sample_rate=16_000)

    assert model.decoder.input[0].padding == (0,)
    assert [block.depthwise.dilation[0] for block in model.decoder.blocks] == [
        1,
        2,
        4,
        1,
        2,
        4,
        1,
        1,
        1,
        1,
    ]
    assert model.decoder.post_norm.num_features == 512
    assert model.decoder.projection.kernel_size == (3,)
    assert model.decoder.projection.in_channels == 512
    assert model.decoder.projection.out_channels == 2048
    assert model.decoder.frame_projection.in_features == 2048
    assert model.decoder.frame_projection.out_features == 512
    assert model.decoder.blocks[0].norm.eps == 1e-6
    assert torch.allclose(model.decoder.blocks[0].gamma, torch.full((512,), 0.1))
    assert torch.allclose(
        model.decoder.blocks[0].depthwise.bias,
        torch.zeros_like(model.decoder.blocks[0].depthwise.bias),
    )


def test_resolution_discriminator_uses_clipped_log_linear_spectrogram():
    spectrogram = torch.tensor(
        [[[0.0 + 0.0j, 1.0 + 0.0j, 3.0 + 4.0j]]],
        dtype=torch.complex64,
    )

    logged = _linear_log_spectrogram(spectrogram)

    expected = torch.log(torch.clamp(torch.abs(spectrogram), min=1e-7))
    assert torch.allclose(logged, expected)


def test_multiresolution_mel_loss_is_zero_for_identical_audio():
    loss_fn = MultiResolutionMelLoss(sample_rate=16_000)
    waveform = torch.randn(2, 4_096)

    loss = loss_fn(waveform, waveform)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_paper_discriminators_return_scores_and_layer_features():
    waveform = torch.randn(2, 4096)
    mpd = MultiPeriodDiscriminator()
    mrd = MultiResolutionDiscriminator(sample_rate=16_000)

    mpd_outputs = mpd(waveform)
    mrd_outputs = mrd(waveform)

    assert [output.name for output in mpd_outputs] == [
        "mpd_2",
        "mpd_3",
        "mpd_5",
        "mpd_7",
        "mpd_11",
    ]
    assert [output.name for output in mrd_outputs] == [
        "mrd_512",
        "mrd_1024",
        "mrd_2048",
    ]
    assert all(output.score.ndim == 4 for output in mpd_outputs + mrd_outputs)
    assert all(len(output.features) == 6 for output in mpd_outputs + mrd_outputs)


def test_paper_adversarial_loss_updates_generator_and_discriminator():
    waveform = torch.randn(2, 4096)
    model = SpeechAutoencoder(sample_rate=16_000)
    discriminators = PaperAutoencoderAdversarialLoss(sample_rate=16_000)
    generator_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.AdamW(discriminators.parameters(), lr=1e-4)
    generator_before = _flatten_parameters(model)
    discriminator_before = _flatten_parameters(discriminators)

    output = model(waveform)
    metrics = discriminators.train_step(
        real=waveform,
        generated=output.waveform,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        reconstruction_loss_fn=MultiResolutionMelLoss(sample_rate=16_000),
    )

    assert metrics["loss"] > 0.0
    assert metrics["mel_loss"] > 0.0
    assert metrics["generator_adversarial_loss"] > 0.0
    assert metrics["feature_matching_loss"] >= 0.0
    assert metrics["discriminator_loss"] > 0.0
    assert not torch.allclose(generator_before, _flatten_parameters(model))
    assert not torch.allclose(discriminator_before, _flatten_parameters(discriminators))


def _flatten_parameters(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten().cpu() for parameter in module.parameters()])


class _TinyAutoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor):
        return type("Output", (), {"waveform": waveform * self.scale})()


class _RecordingAdversarialLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_mixed_precision_enabled = None
        self.seen_gradient_scaler = None

    def train_step(self, **kwargs):
        self.seen_mixed_precision_enabled = kwargs["mixed_precision_enabled"]
        self.seen_gradient_scaler = kwargs["gradient_scaler"]
        return {
            "loss": 1.0,
            "mel_loss": 1.0,
            "generator_adversarial_loss": 1.0,
            "feature_matching_loss": 0.0,
            "discriminator_loss": 1.0,
        }


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


def test_mixed_precision_defaults_to_cuda_only():
    config = AutoencoderTrainConfig(
        manifest=Path("manifest.jsonl"),
        output_dir=Path("runs"),
    )

    assert config.mixed_precision
    assert _resolve_mixed_precision(torch.device("cuda"), requested=True)
    assert not _resolve_mixed_precision(torch.device("cpu"), requested=True)
    assert not _resolve_mixed_precision(torch.device("cuda"), requested=False)


def test_train_autoencoder_one_step_passes_amp_state_to_adversarial_step(tmp_path):
    audio = tmp_path / "voice.wav"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    manifest.write_text(json.dumps({"audio": str(audio)}) + "\n", encoding="utf-8")
    dataset = WaveformDataset(manifest, sample_rate=16_000)
    batch = collate_waveforms([dataset[0]])
    model = _TinyAutoencoder()
    adversarial_loss = _RecordingAdversarialLoss()

    metrics = train_autoencoder_one_step(
        batch,
        device=torch.device("cpu"),
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        loss_fn=MultiResolutionMelLoss(sample_rate=16_000),
        adversarial_loss=adversarial_loss,
        discriminator_optimizer=torch.optim.AdamW(adversarial_loss.parameters(), lr=1e-4),
        mixed_precision_enabled=True,
        gradient_scaler=object(),
    )

    assert metrics["loss"] == 1.0
    assert adversarial_loss.seen_mixed_precision_enabled is False
    assert adversarial_loss.seen_gradient_scaler is None


def test_amp_generated_audio_is_promoted_to_float32_for_gan_losses():
    real = torch.randn(1, 128, dtype=torch.float32)
    generated = torch.randn(1, 128, dtype=torch.float16, requires_grad=True)

    real_loss, generated_loss = _loss_precision_waveforms(real, generated)
    generated_loss.sum().backward()

    assert real_loss.dtype == torch.float32
    assert generated_loss.dtype == torch.float32
    assert generated.grad is not None


def test_non_finite_losses_fail_with_component_name():
    with pytest.raises(RuntimeError, match="mel_loss"):
        _assert_finite_losses(
            {
                "loss": torch.tensor(1.0),
                "mel_loss": torch.tensor(float("nan")),
            }
        )


def test_discriminator_waveforms_are_finite_and_bounded():
    waveform = torch.tensor(
        [[float("nan"), float("inf"), float("-inf"), -3.0, 0.5, 4.0]],
        requires_grad=True,
    )

    sanitized = _sanitize_waveform_for_discriminator(waveform)
    sanitized.sum().backward()

    assert torch.isfinite(sanitized).all()
    assert sanitized.min() >= -1.0
    assert sanitized.max() <= 1.0
    assert waveform.grad is not None


def test_load_autoencoder_checkpoint_restores_model_weights(tmp_path):
    source = SpeechAutoencoder(sample_rate=16_000)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    target = SpeechAutoencoder(sample_rate=16_000)

    load_autoencoder_checkpoint(target, checkpoint)

    source_first = next(source.parameters()).detach()
    target_first = next(target.parameters()).detach()
    assert torch.allclose(source_first, target_first)


def test_load_autoencoder_checkpoint_tolerates_pre_layer_scale_checkpoints(tmp_path):
    source = SpeechAutoencoder(sample_rate=16_000)
    state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.endswith(".gamma")
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": state}, checkpoint)
    target = SpeechAutoencoder(sample_rate=16_000)

    load_autoencoder_checkpoint(target, checkpoint)


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


def test_train_autoencoder_logs_step_progress(tmp_path, monkeypatch, capsys):
    audio_a = tmp_path / "train-a.wav"
    audio_b = tmp_path / "train-b.wav"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio_a)
    _write_wav(audio_b, frequency=440.0)
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"audio": str(audio_a)}),
                json.dumps({"audio": str(audio_b)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_train_step(*args, **kwargs):
        return {
            "loss": 1.0,
            "mel_loss": 0.5,
            "generator_adversarial_loss": 0.25,
            "feature_matching_loss": 0.125,
            "discriminator_loss": 0.75,
        }

    monkeypatch.setattr(
        "supertonic_reference_encoder.speech_autoencoder.train_autoencoder_one_step",
        fake_train_step,
    )

    train_autoencoder(
        AutoencoderTrainConfig(
            manifest=manifest,
            output_dir=tmp_path / "runs",
            epochs=1,
            batch_size=1,
            sample_rate=16_000,
            max_seconds=0.25,
            device="cpu",
            log_every_steps=1,
        )
    )

    lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"event": "autoencoder_step"' in line
    ]
    assert len(lines) == 2
    assert lines[0]["epoch"] == 1
    assert lines[0]["step"] == 1
    assert lines[0]["total_steps"] == 2
    assert lines[0]["loss"] == 1.0
    assert "step_seconds" in lines[0]


def test_train_autoencoder_normalizes_non_wav_validation_audio(tmp_path, monkeypatch):
    audio = tmp_path / "train.wav"
    validation_audio = tmp_path / "validation.mp3"
    normalized_audio = tmp_path / "normalized.wav"
    manifest = tmp_path / "manifest.jsonl"
    _write_wav(audio)
    validation_audio.write_bytes(b"fake mp3")
    manifest.write_text(json.dumps({"audio": str(audio)}) + "\n", encoding="utf-8")

    def fake_normalize(source, *, output_dir, sample_rate):
        assert source == validation_audio
        _write_wav(normalized_audio, frequency=440.0)
        return normalized_audio

    monkeypatch.setattr(
        "supertonic_reference_encoder.speech_autoencoder.normalize_target_audio",
        fake_normalize,
    )

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
