from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from supertonic_reference_encoder.audio import LogMelExtractor, load_audio
from supertonic_reference_encoder.model import (
    ConvNeXtBlock1d,
    LATENT_DIM,
    MEL_BANDS,
    MelLatentEncoder,
    _init_vocos_conv_or_linear,
)
from supertonic_reference_encoder.target_audio_dataset import normalize_target_audio
from supertonic_reference_encoder.train import _resolve_device


@dataclass(frozen=True)
class WaveformItem:
    waveform: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WaveformBatch:
    waveform: torch.Tensor
    lengths: torch.Tensor
    metadata: list[dict[str, Any]]


@dataclass(frozen=True)
class SpeechAutoencoderOutput:
    latent: torch.Tensor
    waveform: torch.Tensor


class WaveformDataset(Dataset[WaveformItem]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        sample_rate: int = 44_100,
        max_seconds: float | None = 12.0,
    ) -> None:
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.records = _read_audio_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> WaveformItem:
        record = self.records[index]
        waveform = load_audio(
            _resolve(self.root, record["audio"]),
            sample_rate=self.sample_rate,
            max_seconds=self.max_seconds,
        )
        return WaveformItem(
            waveform=waveform,
            metadata={k: v for k, v in record.items() if k != "audio"},
        )


def collate_waveforms(items: list[WaveformItem]) -> WaveformBatch:
    if not items:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.as_tensor([item.waveform.numel() for item in items], dtype=torch.long)
    max_length = int(lengths.max().item())
    waveform = torch.zeros(len(items), max_length, dtype=torch.float32)
    for index, item in enumerate(items):
        waveform[index, : item.waveform.numel()] = item.waveform
    return WaveformBatch(
        waveform=waveform,
        lengths=lengths,
        metadata=[item.metadata for item in items],
    )


class CausalConvNeXtBlock1d(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        intermediate_dim: int,
        kernel_size: int,
        dilation: int,
        layer_scale_init_value: float | None,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_in = nn.Linear(dim, intermediate_dim)
        self.pointwise_out = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )
        self.apply(_init_vocos_conv_or_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pointwise_in(x)
        x = F.gelu(x)
        x = self.pointwise_out(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return x + residual


class LatentDecoder(nn.Module):
    """Causal latent decoder following the paper's speech-autoencoder decoder."""

    DILATIONS = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1)

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        frame_size: int = 512,
    ) -> None:
        super().__init__()
        self.frame_size = frame_size
        self.input = nn.Sequential(
            nn.Conv1d(LATENT_DIM, hidden_dim, kernel_size=7),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                CausalConvNeXtBlock1d(
                    dim=hidden_dim,
                    intermediate_dim=2048,
                    kernel_size=7,
                    dilation=dilation,
                    layer_scale_init_value=1 / len(self.DILATIONS),
                )
                for dilation in self.DILATIONS
            ]
        )
        self.post_norm = nn.BatchNorm1d(hidden_dim)
        self.projection = nn.Conv1d(hidden_dim, 2048, kernel_size=3)
        self.frame_projection = nn.Linear(2048, frame_size)

    def forward(self, latent: torch.Tensor, *, target_samples: int | None = None) -> torch.Tensor:
        x = self.blocks(self.input(F.pad(latent, (6, 0))))
        x = self.post_norm(x)
        x = F.pad(x, (2, 0))
        x = self.projection(x)
        frames = self.frame_projection(x.transpose(1, 2))
        waveform = frames.reshape(latent.shape[0], -1)
        if target_samples is not None:
            if waveform.shape[1] < target_samples:
                waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
            waveform = waveform[:, :target_samples]
        return waveform


class SpeechAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 44_100,
        frame_size: int = 512,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel = LogMelExtractor(
            sample_rate=sample_rate,
            n_mels=MEL_BANDS,
            hop_length=frame_size,
            win_length=2048,
        )
        self.encoder = MelLatentEncoder()
        self.decoder = LatentDecoder(frame_size=frame_size)

    def forward(self, waveform: torch.Tensor) -> SpeechAutoencoderOutput:
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be [batch, samples], got {tuple(waveform.shape)}")
        self.mel.transform.to(waveform.device)
        mel = torch.stack([self.mel(sample) for sample in waveform], dim=0).to(waveform.device)
        latent = self.encoder(mel)
        reconstructed = self.decoder(latent, target_samples=waveform.shape[1])
        return SpeechAutoencoderOutput(latent=latent, waveform=reconstructed)


class MultiResolutionMelLoss(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 44_100,
        resolutions: tuple[tuple[int, int], ...] = ((1024, 64), (2048, 128), (4096, 128)),
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=fft_size,
                    hop_length=fft_size // 4,
                    win_length=fft_size,
                    n_mels=n_mels,
                    power=1.0,
                )
                for fft_size, n_mels in resolutions
            ]
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for transform in self.transforms:
            pred_mel = torch.log1p(transform(prediction))
            target_mel = torch.log1p(transform(target))
            losses.append(F.l1_loss(pred_mel, target_mel))
        return torch.stack(losses).mean()


@dataclass(frozen=True)
class DiscriminatorOutput:
    name: str
    score: torch.Tensor
    features: list[torch.Tensor]


class MultiPeriodDiscriminator(nn.Module):
    PERIODS = (2, 3, 5, 7, 11)

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [_PeriodDiscriminator(period=period) for period in self.PERIODS]
        )

    def forward(self, waveform: torch.Tensor) -> list[DiscriminatorOutput]:
        return [discriminator(waveform) for discriminator in self.discriminators]


class _PeriodDiscriminator(nn.Module):
    def __init__(self, *, period: int) -> None:
        super().__init__()
        self.period = period
        channels = (1, 16, 64, 256, 512, 512, 1)
        layers = []
        for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            is_last = index == len(channels) - 2
            kernel_size = (3, 1) if is_last else (5, 1)
            stride = (1, 1) if is_last else (3, 1)
            padding = (1, 0) if is_last else (2, 0)
            layers.append(
                nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be [batch, samples], got {tuple(waveform.shape)}")
        pad = (-waveform.shape[1]) % self.period
        if pad:
            waveform = F.pad(waveform, (0, pad), mode="reflect")
        x = waveform.view(waveform.shape[0], 1, waveform.shape[1] // self.period, self.period)
        features = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.leaky_relu(x, 0.1)
            features.append(x)
        return DiscriminatorOutput(name=f"mpd_{self.period}", score=x, features=features)


class MultiResolutionDiscriminator(nn.Module):
    FFT_SIZES = (512, 1024, 2048)

    def __init__(self, *, sample_rate: int = 44_100) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.discriminators = nn.ModuleList(
            [_ResolutionDiscriminator(fft_size=fft_size) for fft_size in self.FFT_SIZES]
        )

    def forward(self, waveform: torch.Tensor) -> list[DiscriminatorOutput]:
        return [discriminator(waveform) for discriminator in self.discriminators]


class _ResolutionDiscriminator(nn.Module):
    def __init__(self, *, fft_size: int) -> None:
        super().__init__()
        self.fft_size = fft_size
        channels = (1, 16, 16, 16, 16, 16, 1)
        strides = ((1, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1))
        kernels = ((5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (3, 3))
        self.layers = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    )
                )
                for in_channels, out_channels, kernel_size, stride in zip(
                    channels,
                    channels[1:],
                    kernels,
                    strides,
                )
            ]
        )

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        spectrogram = torch.stft(
            waveform,
            n_fft=self.fft_size,
            hop_length=self.fft_size // 4,
            win_length=self.fft_size,
            window=torch.hann_window(self.fft_size, device=waveform.device),
            return_complex=True,
        )
        x = _linear_log_spectrogram(spectrogram).unsqueeze(1)
        features = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.leaky_relu(x, 0.1)
            features.append(x)
        return DiscriminatorOutput(name=f"mrd_{self.fft_size}", score=x, features=features)


class PaperAutoencoderAdversarialLoss(nn.Module):
    """GAN objectives described for the SupertonicTTS speech autoencoder."""

    def __init__(
        self,
        *,
        sample_rate: int = 44_100,
        feature_matching_weight: float = 2.0,
        adversarial_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator(sample_rate=sample_rate)
        self.feature_matching_weight = feature_matching_weight
        self.adversarial_weight = adversarial_weight

    def forward(self, waveform: torch.Tensor) -> list[DiscriminatorOutput]:
        return [*self.mpd(waveform), *self.mrd(waveform)]

    def train_step(
        self,
        *,
        real: torch.Tensor,
        generated: torch.Tensor,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        reconstruction_loss_fn: MultiResolutionMelLoss,
        mixed_precision_enabled: bool = False,
        gradient_scaler: torch.amp.GradScaler | None = None,
    ) -> dict[str, float]:
        real_loss, generated_loss = _loss_precision_waveforms(real, generated)
        real_discriminator = _sanitize_waveform_for_discriminator(real_loss)
        generated_discriminator = _sanitize_waveform_for_discriminator(generated_loss)
        discriminator_optimizer.zero_grad(set_to_none=True)
        real_outputs = self(real_discriminator)
        fake_outputs = self(generated_discriminator.detach())
        discriminator_loss = _discriminator_loss(real_outputs, fake_outputs)
        _assert_finite_discriminator_outputs("discriminator_real", real_outputs)
        _assert_finite_discriminator_outputs("discriminator_fake", fake_outputs)
        _assert_finite_losses({"discriminator_loss": discriminator_loss})
        _backward_and_step(
            discriminator_loss,
            optimizer=discriminator_optimizer,
            gradient_scaler=gradient_scaler,
        )

        generator_optimizer.zero_grad(set_to_none=True)
        real_outputs_for_generator = self(real_discriminator)
        fake_outputs_for_generator = self(generated_discriminator)
        mel_loss = reconstruction_loss_fn(generated_loss, real_loss)
        generator_adversarial_loss = _generator_adversarial_loss(fake_outputs_for_generator)
        feature_matching_loss = _feature_matching_loss(
            real_outputs_for_generator,
            fake_outputs_for_generator,
        )
        loss = (
            mel_loss
            + self.adversarial_weight * generator_adversarial_loss
            + self.feature_matching_weight * feature_matching_loss
        )
        _assert_finite_losses(
            {
                "loss": loss,
                "mel_loss": mel_loss,
                "generator_adversarial_loss": generator_adversarial_loss,
                "feature_matching_loss": feature_matching_loss,
                "discriminator_loss": discriminator_loss,
            }
        )
        _backward_and_step(
            loss,
            optimizer=generator_optimizer,
            gradient_scaler=gradient_scaler,
        )
        if gradient_scaler is not None:
            gradient_scaler.update()
        return {
            "loss": float(loss.detach().cpu()),
            "mel_loss": float(mel_loss.detach().cpu()),
            "generator_adversarial_loss": float(generator_adversarial_loss.detach().cpu()),
            "feature_matching_loss": float(feature_matching_loss.detach().cpu()),
            "discriminator_loss": float(discriminator_loss.detach().cpu()),
        }


def _discriminator_loss(
    real_outputs: list[DiscriminatorOutput],
    fake_outputs: list[DiscriminatorOutput],
) -> torch.Tensor:
    losses = []
    for real, fake in zip(real_outputs, fake_outputs):
        losses.append(F.mse_loss(real.score, torch.ones_like(real.score)))
        losses.append(F.mse_loss(fake.score, torch.zeros_like(fake.score)))
    return torch.stack(losses).mean()


def _generator_adversarial_loss(outputs: list[DiscriminatorOutput]) -> torch.Tensor:
    return torch.stack(
        [F.mse_loss(output.score, torch.ones_like(output.score)) for output in outputs]
    ).mean()


def _linear_log_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(torch.abs(spectrogram), min=1e-7))


def _feature_matching_loss(
    real_outputs: list[DiscriminatorOutput],
    fake_outputs: list[DiscriminatorOutput],
) -> torch.Tensor:
    losses = []
    for real, fake in zip(real_outputs, fake_outputs):
        for real_feature, fake_feature in zip(real.features, fake.features):
            losses.append(F.l1_loss(fake_feature, real_feature.detach()))
    return torch.stack(losses).mean()


def _loss_precision_waveforms(
    real: torch.Tensor,
    generated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return real.float(), generated.float()


def _sanitize_waveform_for_discriminator(waveform: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def _assert_finite_losses(losses: dict[str, torch.Tensor]) -> None:
    non_finite = [
        name
        for name, loss in losses.items()
        if not torch.isfinite(loss.detach()).all().item()
    ]
    if non_finite:
        raise RuntimeError(f"non-finite autoencoder loss: {', '.join(non_finite)}")


def _assert_finite_discriminator_outputs(
    prefix: str,
    outputs: list[DiscriminatorOutput],
) -> None:
    non_finite = []
    for output in outputs:
        if not torch.isfinite(output.score.detach()).all().item():
            non_finite.append(f"{prefix}.{output.name}.score")
        for index, feature in enumerate(output.features):
            if not torch.isfinite(feature.detach()).all().item():
                non_finite.append(f"{prefix}.{output.name}.features[{index}]")
    if non_finite:
        raise RuntimeError(f"non-finite discriminator output: {', '.join(non_finite)}")


@dataclass(frozen=True)
class AutoencoderTrainConfig:
    manifest: Path
    output_dir: Path
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    sample_rate: int = 44_100
    max_seconds: float = 12.0
    num_workers: int = 0
    device: str = "auto"
    save_optimizer: bool = False
    resume: Path | None = None
    validation_audio: Path | None = None
    mixed_precision: bool = True
    log_every_steps: int = 0


def train_autoencoder_one_step(
    batch: WaveformBatch,
    *,
    device: torch.device,
    model: SpeechAutoencoder | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    loss_fn: MultiResolutionMelLoss | None = None,
    adversarial_loss: PaperAutoencoderAdversarialLoss | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    mixed_precision_enabled: bool = False,
    gradient_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    mixed_precision_enabled = _resolve_mixed_precision(
        device,
        requested=mixed_precision_enabled,
    )
    gradient_scaler = gradient_scaler if mixed_precision_enabled else None
    model = (model or SpeechAutoencoder(sample_rate=16_000)).to(device)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = (loss_fn or MultiResolutionMelLoss(sample_rate=16_000)).to(device)
    adversarial_loss = (adversarial_loss or PaperAutoencoderAdversarialLoss(sample_rate=16_000)).to(
        device
    )
    discriminator_optimizer = discriminator_optimizer or torch.optim.AdamW(
        adversarial_loss.parameters(),
        lr=1e-4,
    )
    model.train()
    adversarial_loss.train()

    waveform = batch.waveform.to(device)
    with _autocast(device, enabled=mixed_precision_enabled):
        output = model(waveform)
    return adversarial_loss.train_step(
        real=waveform,
        generated=output.waveform,
        generator_optimizer=optimizer,
        discriminator_optimizer=discriminator_optimizer,
        reconstruction_loss_fn=loss_fn,
        mixed_precision_enabled=mixed_precision_enabled,
        gradient_scaler=gradient_scaler,
    )


def evaluate_autoencoder_audio(
    model: SpeechAutoencoder,
    audio_path: Path,
    *,
    device: torch.device,
    loss_fn: MultiResolutionMelLoss,
    sample_rate: int,
    max_seconds: float | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    waveform = load_audio(
        audio_path,
        sample_rate=sample_rate,
        max_seconds=max_seconds,
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(waveform)
        mel_loss = loss_fn(output.waveform, waveform)
        waveform_l1 = F.l1_loss(output.waveform, waveform)
    if was_training:
        model.train()
    return {
        "validation_mel_loss": float(mel_loss.detach().cpu()),
        "validation_waveform_l1": float(waveform_l1.detach().cpu()),
    }


def train_autoencoder(config: AutoencoderTrainConfig) -> dict[str, float]:
    device = _resolve_device(config.device)
    dataset = WaveformDataset(
        config.manifest,
        sample_rate=config.sample_rate,
        max_seconds=config.max_seconds,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_waveforms,
    )
    model = SpeechAutoencoder(sample_rate=config.sample_rate).to(device)
    if config.resume is not None:
        load_autoencoder_checkpoint(model, config.resume)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = MultiResolutionMelLoss(sample_rate=config.sample_rate).to(device)
    adversarial_loss = PaperAutoencoderAdversarialLoss(sample_rate=config.sample_rate).to(device)
    discriminator_optimizer = torch.optim.AdamW(
        adversarial_loss.parameters(),
        lr=config.learning_rate,
    )
    mixed_precision_enabled = _resolve_mixed_precision(
        device,
        requested=config.mixed_precision,
    )
    gradient_scaler = (
        torch.amp.GradScaler(device.type, enabled=True)
        if mixed_precision_enabled
        else None
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config)
    validation_audio = (
        normalize_target_audio(
            config.validation_audio,
            output_dir=config.output_dir / "validation",
            sample_rate=config.sample_rate,
        )
        if config.validation_audio is not None
        else None
    )

    best_loss = float("inf")
    last_metrics: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        totals = {
            "loss": 0.0,
            "mel_loss": 0.0,
            "generator_adversarial_loss": 0.0,
            "feature_matching_loss": 0.0,
            "discriminator_loss": 0.0,
        }
        steps = 0
        total_steps = len(loader)
        for batch in loader:
            step_started_at = time.perf_counter()
            metrics = train_autoencoder_one_step(
                batch,
                device=device,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                adversarial_loss=adversarial_loss,
                discriminator_optimizer=discriminator_optimizer,
                mixed_precision_enabled=mixed_precision_enabled,
                gradient_scaler=gradient_scaler,
            )
            _sync_if_cuda(device)
            step_seconds = time.perf_counter() - step_started_at
            for key in totals:
                totals[key] += metrics[key]
            steps += 1
            if _should_log_step(steps, config.log_every_steps):
                print(
                    json.dumps(
                        _step_log_payload(
                            epoch=epoch,
                            step=steps,
                            total_steps=total_steps,
                            step_seconds=step_seconds,
                            batch=batch,
                            metrics=metrics,
                            device=device,
                            mixed_precision_enabled=mixed_precision_enabled,
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        last_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
        if validation_audio is not None:
            last_metrics.update(
                evaluate_autoencoder_audio(
                    model,
                    validation_audio,
                    device=device,
                    loss_fn=loss_fn,
                    sample_rate=config.sample_rate,
                    max_seconds=config.max_seconds,
                )
            )
        last_metrics["epoch"] = float(epoch)
        _append_metrics(config.output_dir / "metrics.jsonl", last_metrics)
        _save_checkpoint(
            config.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            discriminator=adversarial_loss,
            discriminator_optimizer=discriminator_optimizer,
            config=config,
            metrics=last_metrics,
            save_optimizer=config.save_optimizer,
        )
        if last_metrics["loss"] < best_loss:
            best_loss = last_metrics["loss"]
            _save_checkpoint(
                config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                discriminator=adversarial_loss,
                discriminator_optimizer=discriminator_optimizer,
                config=config,
                metrics=last_metrics,
                save_optimizer=config.save_optimizer,
            )
        print(json.dumps(last_metrics, ensure_ascii=False))
    return last_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-style speech autoencoder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-audio", type=Path)
    parser.add_argument("--log-every-steps", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA AMP fp16 training when the resolved device is CUDA.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_autoencoder(
        AutoencoderTrainConfig(
            manifest=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sample_rate=args.sample_rate,
            max_seconds=args.max_seconds,
            num_workers=args.num_workers,
            device=args.device,
            save_optimizer=args.save_optimizer,
            resume=args.resume,
            validation_audio=args.validation_audio,
            mixed_precision=args.mixed_precision,
            log_every_steps=args.log_every_steps,
        )
    )
    return 0


def _read_audio_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "audio" not in record:
            raise ValueError(f"{path}:{line_number} missing required key: audio")
        records.append(record)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def load_autoencoder_checkpoint(model: SpeechAutoencoder, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected_keys = list(incompatible.unexpected_keys)
    missing_keys = [
        key
        for key in incompatible.missing_keys
        if not key.endswith(".gamma")
    ]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint is incompatible: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _resolve_mixed_precision(device: torch.device, *, requested: bool) -> bool:
    return requested and device.type == "cuda"


def _autocast(device: torch.device, *, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device.type, dtype=torch.float16, enabled=True)


def _backward_and_step(
    loss: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler | None,
) -> None:
    if gradient_scaler is None:
        loss.backward()
        optimizer.step()
        return
    gradient_scaler.scale(loss).backward()
    gradient_scaler.step(optimizer)


def _should_log_step(step: int, log_every_steps: int) -> bool:
    return log_every_steps > 0 and step % log_every_steps == 0


def _step_log_payload(
    *,
    epoch: int,
    step: int,
    total_steps: int,
    step_seconds: float,
    batch: WaveformBatch,
    metrics: dict[str, float],
    device: torch.device,
    mixed_precision_enabled: bool,
) -> dict[str, float | int | str | bool]:
    payload: dict[str, float | int | str | bool] = {
        "event": "autoencoder_step",
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "step_seconds": step_seconds,
        "batch_size": int(batch.waveform.shape[0]),
        "batch_max_samples": int(batch.waveform.shape[1]),
        "device": device.type,
        "mixed_precision": mixed_precision_enabled,
    }
    payload.update({key: float(value) for key, value in metrics.items()})
    if device.type == "cuda":
        payload["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / 1024**3
        payload["cuda_reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3
        payload["cuda_max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    return payload


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_config(config: AutoencoderTrainConfig) -> None:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def _save_checkpoint(
    path: Path,
    *,
    model: SpeechAutoencoder,
    optimizer: torch.optim.Optimizer,
    discriminator: PaperAutoencoderAdversarialLoss | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    config: AutoencoderTrainConfig,
    metrics: dict[str, float],
    save_optimizer: bool,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "metrics": metrics,
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        if discriminator_optimizer is not None:
            payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    torch.save(payload, path)


if __name__ == "__main__":
    raise SystemExit(main())
