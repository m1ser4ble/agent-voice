from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from supertonic_reference_encoder.audio import LogMelExtractor, load_audio
from supertonic_reference_encoder.data import _resolve
from supertonic_reference_encoder.duration_predictor import DurationPredictor, duration_l1_loss
from supertonic_reference_encoder.model import AudioToStyleEncoder, StyleTensors
from supertonic_reference_encoder.styles import load_style_json
from supertonic_reference_encoder.text_to_latent import TextToLatentFlow, flow_matching_loss
from supertonic_reference_encoder.train import _resolve_device


VOCAB_SIZE = 256


@dataclass(frozen=True)
class StageItem:
    mel: torch.Tensor
    text_tokens: torch.Tensor
    target: StyleTensors
    duration: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StageBatch:
    mel: torch.Tensor
    lengths: torch.Tensor
    text_tokens: torch.Tensor
    target: StyleTensors
    duration: torch.Tensor
    metadata: list[dict[str, Any]]


class StageDataset(Dataset[StageItem]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        sample_rate: int = 44_100,
        n_mels: int = 228,
        max_seconds: float | None = 12.0,
        max_text_length: int = 256,
    ) -> None:
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.max_text_length = max_text_length
        self.mel_extractor = LogMelExtractor(sample_rate=sample_rate, n_mels=n_mels)
        self.records = _read_stage_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> StageItem:
        record = self.records[index]
        waveform = load_audio(
            _resolve(self.root, record["audio"]),
            sample_rate=self.sample_rate,
            max_seconds=self.max_seconds,
        )
        return StageItem(
            mel=self.mel_extractor(waveform),
            text_tokens=tokenize_text(str(record["text"]), max_length=self.max_text_length),
            target=load_style_json(_resolve(self.root, record["style_json"])),
            duration=torch.tensor(float(waveform.numel()) / self.sample_rate, dtype=torch.float32),
            metadata={k: v for k, v in record.items() if k not in {"audio", "style_json", "text"}},
        )


def tokenize_text(text: str, *, max_length: int) -> torch.Tensor:
    ids = [min(ord(char), VOCAB_SIZE - 1) for char in text[:max_length]]
    ids.extend([0] * (max_length - len(ids)))
    return torch.as_tensor(ids, dtype=torch.long)


def collate_stage_items(items: list[StageItem]) -> StageBatch:
    if not items:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.as_tensor([item.mel.shape[1] for item in items], dtype=torch.long)
    max_length = int(lengths.max().item())
    n_mels = items[0].mel.shape[0]
    mel = torch.zeros(len(items), n_mels, max_length, dtype=torch.float32)
    for index, item in enumerate(items):
        mel[index, :, : item.mel.shape[1]] = item.mel
    return StageBatch(
        mel=mel,
        lengths=lengths,
        text_tokens=torch.stack([item.text_tokens for item in items]),
        target=StyleTensors(
            style_ttl=torch.stack([item.target.style_ttl for item in items]),
            style_dp=torch.stack([item.target.style_dp for item in items]),
        ),
        duration=torch.stack([item.duration for item in items]),
        metadata=[item.metadata for item in items],
    )


def train_text_to_latent_one_step(
    batch: StageBatch,
    *,
    device: torch.device,
    model: TextToLatentFlow | None = None,
    latent_extractor: AudioToStyleEncoder | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model = (model or TextToLatentFlow(vocab_size=VOCAB_SIZE)).to(device)
    latent_extractor = (latent_extractor or AudioToStyleEncoder()).to(device)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    latent_extractor.eval()

    mel = batch.mel.to(device)
    with torch.no_grad():
        latent = latent_extractor.temporal_compressor(latent_extractor.latent_encoder(mel))
    loss, metrics = flow_matching_loss(
        model,
        text_tokens=batch.text_tokens.to(device),
        target_latent=latent,
        style_ttl=batch.target.style_ttl.to(device),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return metrics


def train_duration_one_step(
    batch: StageBatch,
    *,
    device: torch.device,
    model: DurationPredictor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model = (model or DurationPredictor(vocab_size=VOCAB_SIZE)).to(device)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    loss, metrics = duration_l1_loss(
        model,
        text_tokens=batch.text_tokens.to(device),
        style_dp=batch.target.style_dp.to(device),
        target_duration=batch.duration.to(device),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return metrics


@dataclass(frozen=True)
class StageTrainConfig:
    stage: Literal["text-to-latent", "duration"]
    manifest: Path
    output_dir: Path
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    sample_rate: int = 44_100
    n_mels: int = 228
    max_seconds: float = 12.0
    max_text_length: int = 256
    num_workers: int = 0
    device: str = "auto"


def train_stage(config: StageTrainConfig) -> dict[str, float]:
    device = _resolve_device(config.device)
    dataset = StageDataset(
        config.manifest,
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        max_seconds=config.max_seconds,
        max_text_length=config.max_text_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_stage_items,
    )
    if config.stage == "text-to-latent":
        model: torch.nn.Module = TextToLatentFlow(vocab_size=VOCAB_SIZE).to(device)
        latent_extractor = AudioToStyleEncoder().to(device)
    else:
        model = DurationPredictor(vocab_size=VOCAB_SIZE).to(device)
        latent_extractor = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config)

    last_metrics: dict[str, float] = {}
    best_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        totals: dict[str, float] = {}
        steps = 0
        for batch in loader:
            if config.stage == "text-to-latent":
                metrics = train_text_to_latent_one_step(
                    batch,
                    device=device,
                    model=model,  # type: ignore[arg-type]
                    latent_extractor=latent_extractor,
                    optimizer=optimizer,
                )
            else:
                metrics = train_duration_one_step(
                    batch,
                    device=device,
                    model=model,  # type: ignore[arg-type]
                    optimizer=optimizer,
                )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        last_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
        last_metrics["epoch"] = float(epoch)
        _append_metrics(config.output_dir / "metrics.jsonl", last_metrics)
        _save_model(config.output_dir / "latest.pt", model=model, config=config, metrics=last_metrics)
        if last_metrics["loss"] < best_loss:
            best_loss = last_metrics["loss"]
            _save_model(config.output_dir / "best.pt", model=model, config=config, metrics=last_metrics)
        print(json.dumps(last_metrics, ensure_ascii=False))
    return last_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-to-latent or duration stage.")
    parser.add_argument("--stage", choices=["text-to-latent", "duration"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--n-mels", type=int, default=228)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_stage(
        StageTrainConfig(
            stage=args.stage,
            manifest=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            max_seconds=args.max_seconds,
            max_text_length=args.max_text_length,
            num_workers=args.num_workers,
            device=args.device,
        )
    )
    return 0


def _read_stage_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = [key for key in ("audio", "style_json", "text") if key not in record]
        if missing:
            raise ValueError(f"{path}:{line_number} missing required keys: {missing}")
        records.append(record)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _write_config(config: StageTrainConfig) -> None:
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


def _save_model(
    path: Path,
    *,
    model: torch.nn.Module,
    config: StageTrainConfig,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "metrics": metrics,
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
