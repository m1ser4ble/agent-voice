from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from supertonic_reference_encoder.data import (
    ReferenceStyleBatch,
    ReferenceStyleDataset,
    collate_reference_styles,
)
from supertonic_reference_encoder.losses import style_reconstruction_loss
from supertonic_reference_encoder.model import AudioToStyleEncoder, StyleTensors


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path
    output_dir: Path
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    sample_rate: int = 44_100
    n_mels: int = 228
    max_seconds: float = 12.0
    ttl_weight: float = 1.0
    dp_weight: float = 1.0
    num_workers: int = 0
    device: str = "auto"


def train_one_step(
    batch: ReferenceStyleBatch,
    *,
    device: torch.device,
    model: AudioToStyleEncoder | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    ttl_weight: float = 1.0,
    dp_weight: float = 1.0,
) -> dict[str, float]:
    model = (model or AudioToStyleEncoder()).to(device)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    mel = batch.mel.to(device)
    target = StyleTensors(
        style_ttl=batch.target.style_ttl.to(device),
        style_dp=batch.target.style_dp.to(device),
    )

    optimizer.zero_grad(set_to_none=True)
    prediction = model(mel)
    loss, metrics = style_reconstruction_loss(
        prediction,
        target,
        ttl_weight=ttl_weight,
        dp_weight=dp_weight,
    )
    loss.backward()
    optimizer.step()
    return metrics


def train(config: TrainConfig) -> dict[str, float]:
    device = _resolve_device(config.device)
    dataset = ReferenceStyleDataset(
        config.manifest,
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        max_seconds=config.max_seconds,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_reference_styles,
    )
    model = AudioToStyleEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config)

    best_loss = float("inf")
    last_metrics: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        totals = {"loss": 0.0, "ttl_loss": 0.0, "dp_loss": 0.0}
        steps = 0
        for batch in loader:
            metrics = train_one_step(
                batch,
                device=device,
                model=model,
                optimizer=optimizer,
                ttl_weight=config.ttl_weight,
                dp_weight=config.dp_weight,
            )
            for key in totals:
                totals[key] += metrics[key]
            steps += 1
        last_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
        last_metrics["epoch"] = float(epoch)
        _append_metrics(config.output_dir / "metrics.jsonl", last_metrics)
        _save_checkpoint(
            config.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            metrics=last_metrics,
        )
        if last_metrics["loss"] < best_loss:
            best_loss = last_metrics["loss"]
            _save_checkpoint(
                config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                metrics=last_metrics,
            )
        print(json.dumps(last_metrics, ensure_ascii=False))
    return last_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Supertonic reference encoder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--n-mels", type=int, default=228)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--ttl-weight", type=float, default=1.0)
    parser.add_argument("--dp-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train(
        TrainConfig(
            manifest=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            max_seconds=args.max_seconds,
            ttl_weight=args.ttl_weight,
            dp_weight=args.dp_weight,
            num_workers=args.num_workers,
            device=args.device,
        )
    )
    return 0


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _write_config(config: TrainConfig) -> None:
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
    model: AudioToStyleEncoder,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
