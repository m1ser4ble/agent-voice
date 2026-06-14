from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from supertonic_reference_encoder.audio import LogMelExtractor, load_audio
from supertonic_reference_encoder.model import ConvNeXtBlock1d, LATENT_DIM, MEL_BANDS, MelLatentEncoder
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
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Linear(dim, intermediate_dim)
        self.pointwise_out = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pointwise_in(x)
        x = F.gelu(x)
        x = self.pointwise_out(x)
        x = x.transpose(1, 2)
        return x + residual


class LatentDecoder(nn.Module):
    """Causal latent decoder following the paper's speech-autoencoder decoder."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        frame_size: int = 512,
    ) -> None:
        super().__init__()
        self.frame_size = frame_size
        self.input = nn.Sequential(
            nn.Conv1d(LATENT_DIM, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                CausalConvNeXtBlock1d(
                    dim=hidden_dim,
                    intermediate_dim=2048,
                    kernel_size=7,
                    dilation=2 ** (index % 4),
                )
                for index in range(10)
            ]
        )
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(hidden_dim, frame_size, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, *, target_samples: int | None = None) -> torch.Tensor:
        frames = self.output(self.blocks(self.input(latent)))
        waveform = frames.transpose(1, 2).reshape(latent.shape[0], -1)
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


def train_autoencoder_one_step(
    batch: WaveformBatch,
    *,
    device: torch.device,
    model: SpeechAutoencoder | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    loss_fn: MultiResolutionMelLoss | None = None,
) -> dict[str, float]:
    model = (model or SpeechAutoencoder(sample_rate=16_000)).to(device)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = (loss_fn or MultiResolutionMelLoss(sample_rate=16_000)).to(device)
    model.train()

    waveform = batch.waveform.to(device)
    optimizer.zero_grad(set_to_none=True)
    output = model(waveform)
    mel_loss = loss_fn(output.waveform, waveform)
    mel_loss.backward()
    optimizer.step()
    return {
        "loss": float(mel_loss.detach().cpu()),
        "mel_loss": float(mel_loss.detach().cpu()),
    }


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
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config)

    best_loss = float("inf")
    last_metrics: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        totals = {"loss": 0.0, "mel_loss": 0.0}
        steps = 0
        for batch in loader:
            metrics = train_autoencoder_one_step(
                batch,
                device=device,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
            )
            totals["loss"] += metrics["loss"]
            totals["mel_loss"] += metrics["mel_loss"]
            steps += 1
        last_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
        if config.validation_audio is not None:
            last_metrics.update(
                evaluate_autoencoder_audio(
                    model,
                    config.validation_audio,
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
    model.load_state_dict(checkpoint["model"])


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


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
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


if __name__ == "__main__":
    raise SystemExit(main())
