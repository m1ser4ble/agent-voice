from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from supertonic_reference_encoder.model import COMPRESSED_LATENT_DIM, ConvNeXtBlock1d


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * -(math.log(10_000.0) / max(half - 1, 1))
        )
        angles = time[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class TextToLatentFlow(nn.Module):
    """Flow-matching velocity estimator for compressed speech latents.

    This follows the public paper structure at the scaffold level: text
    conditioning, reference-style conditioning, time conditioning, ConvNeXt
    blocks, and output in the 144-d compressed latent space.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int = 256,
        text_dim: int = 128,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_encoder = nn.Sequential(
            nn.Conv1d(text_dim, text_dim, kernel_size=5, padding=2),
            nn.GELU(),
            ConvNeXtBlock1d(
                dim=text_dim,
                intermediate_dim=512,
                kernel_size=5,
                layer_scale_init_value=1 / 2,
            ),
            ConvNeXtBlock1d(
                dim=text_dim,
                intermediate_dim=512,
                kernel_size=5,
                layer_scale_init_value=1 / 2,
            ),
        )
        self.style_key = nn.Linear(128, hidden_dim)
        self.style_value = nn.Linear(128, hidden_dim)
        self.latent_input = nn.Conv1d(COMPRESSED_LATENT_DIM, hidden_dim, kernel_size=1)
        self.text_project = nn.Linear(text_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_project = nn.Linear(time_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock1d(
                    dim=hidden_dim,
                    intermediate_dim=1024,
                    kernel_size=5,
                    layer_scale_init_value=1 / 4,
                )
                for _ in range(4)
            ]
        )
        self.output = nn.Conv1d(hidden_dim, COMPRESSED_LATENT_DIM, kernel_size=1)

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        noisy_latent: torch.Tensor,
        style_ttl: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        text = self.text_embedding(text_tokens).transpose(1, 2)
        text = self.text_encoder(text).transpose(1, 2)
        text_context = self.text_project(text.mean(dim=1))

        style_key, style_value = style_ttl.chunk(2, dim=-1)
        style_context = self.style_key(style_key).mean(dim=1) + self.style_value(style_value).mean(dim=1)
        time_context = self.time_project(self.time_embedding(time))

        x = self.latent_input(noisy_latent)
        context = (text_context + style_context + time_context).unsqueeze(-1)
        x = self.blocks(x + context)
        return self.output(x)


def flow_matching_loss(
    model: TextToLatentFlow,
    *,
    text_tokens: torch.Tensor,
    target_latent: torch.Tensor,
    style_ttl: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    noise = torch.randn_like(target_latent)
    time = torch.rand(target_latent.shape[0], device=target_latent.device)
    interpolated = (time[:, None, None] * target_latent) + ((1.0 - time[:, None, None]) * noise)
    target_velocity = target_latent - noise
    predicted_velocity = model(
        text_tokens=text_tokens,
        noisy_latent=interpolated,
        style_ttl=style_ttl,
        time=time,
    )
    loss = F.mse_loss(predicted_velocity, target_velocity)
    return loss, {"loss": float(loss.detach().cpu()), "flow_loss": float(loss.detach().cpu())}
