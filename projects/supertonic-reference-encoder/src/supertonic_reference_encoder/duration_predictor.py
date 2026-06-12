from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from supertonic_reference_encoder.model import ConvNeXtBlock1d


class DurationPredictor(nn.Module):
    """Paper-aligned utterance-level duration predictor scaffold."""

    def __init__(
        self,
        *,
        vocab_size: int,
        text_dim: int = 64,
        reference_dim: int = 64,
        hidden_dim: int = 164,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_conv = nn.Sequential(
            *[
                ConvNeXtBlock1d(dim=text_dim, intermediate_dim=256, kernel_size=5)
                for _ in range(6)
            ]
        )
        self.utterance_token = nn.Parameter(torch.randn(1, 1, text_dim) * 0.02)
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=2,
                dim_feedforward=256,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )
        self.text_output = nn.Linear(text_dim, text_dim)
        self.reference_output = nn.Linear(8 * 16, reference_dim)
        self.duration_estimator = nn.Sequential(
            nn.Linear(text_dim + reference_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, *, text_tokens: torch.Tensor, style_dp: torch.Tensor) -> torch.Tensor:
        text = self.text_embedding(text_tokens).transpose(1, 2)
        text = self.text_conv(text).transpose(1, 2)
        utterance = self.utterance_token.expand(text.shape[0], -1, -1)
        text = torch.cat([utterance, text], dim=1)
        text = self.self_attention(text)
        text_embedding = self.text_output(text[:, 0, :])
        reference_embedding = self.reference_output(style_dp.reshape(style_dp.shape[0], -1))
        x = torch.cat([text_embedding, reference_embedding], dim=-1)
        return F.softplus(self.duration_estimator(x).squeeze(-1)) + 1e-3


def duration_l1_loss(
    model: DurationPredictor,
    *,
    text_tokens: torch.Tensor,
    style_dp: torch.Tensor,
    target_duration: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = model(text_tokens=text_tokens, style_dp=style_dp)
    loss = F.l1_loss(predicted, target_duration)
    return loss, {"loss": float(loss.detach().cpu()), "duration_l1": float(loss.detach().cpu())}
