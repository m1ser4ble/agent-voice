from __future__ import annotations

import torch
from torch.nn import functional as F

from supertonic_reference_encoder.model import StyleTensors


def style_reconstruction_loss(
    prediction: StyleTensors,
    target: StyleTensors,
    *,
    ttl_weight: float = 1.0,
    dp_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ttl_loss = F.mse_loss(prediction.style_ttl, target.style_ttl)
    dp_loss = F.mse_loss(prediction.style_dp, target.style_dp)
    loss = (ttl_weight * ttl_loss) + (dp_weight * dp_loss)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "ttl_loss": float(ttl_loss.detach().cpu()),
        "dp_loss": float(dp_loss.detach().cpu()),
    }
