from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


MEL_BANDS = 228
LATENT_DIM = 24
COMPRESSION_FACTOR = 6
COMPRESSED_LATENT_DIM = LATENT_DIM * COMPRESSION_FACTOR
TTL_TOKENS = 50
TTL_STYLE_DIM = 256
TTL_ATTN_DIM = 128
DP_TOKENS = 8
DP_STYLE_DIM = 16


@dataclass(frozen=True)
class StyleTensors:
    style_ttl: torch.Tensor
    style_dp: torch.Tensor


class ConvNeXtBlock1d(nn.Module):
    """Small 1D ConvNeXt-style block matching the paper's reference encoders."""

    def __init__(
        self,
        *,
        dim: int,
        intermediate_dim: int,
        kernel_size: int,
        layer_scale_init_value: float | None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
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


class MelLatentEncoder(nn.Module):
    """Approximate paper A.1.1 latent encoder front-end.

    The public Supertonic package does not ship the private speech autoencoder
    encoder. This module provides a trainable mel-to-latent front-end so the
    reference heads can be trained end to end later.
    """

    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 512
        self.input = nn.Sequential(
            nn.Conv1d(MEL_BANDS, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
        )
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock1d(
                    dim=hidden_dim,
                    intermediate_dim=2048,
                    kernel_size=7,
                    layer_scale_init_value=1 / 10,
                )
                for _ in range(10)
            ]
        )
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, LATENT_DIM, kernel_size=1),
            _ChannelLayerNorm(LATENT_DIM),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"mel must be [batch, {MEL_BANDS}, time], got {mel.shape}")
        if mel.shape[1] != MEL_BANDS:
            raise ValueError(f"mel must have {MEL_BANDS} bands, got {mel.shape[1]}")
        x = self.input(mel)
        x = self.blocks(x)
        return self.output(x)


class TemporalCompressor(nn.Module):
    """Concatenate adjacent latent frames to match the paper's 144-d input."""

    def __init__(
        self,
        *,
        compression_factor: int = COMPRESSION_FACTOR,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.compression_factor = compression_factor
        self.latent_dim = latent_dim

    @property
    def output_dim(self) -> int:
        return self.latent_dim * self.compression_factor

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError(f"latents must be [batch, channels, time], got {latents.shape}")
        if latents.shape[1] != self.latent_dim:
            raise ValueError(f"latents must have {self.latent_dim} channels, got {latents.shape[1]}")

        batch, channels, time = latents.shape
        pad = (-time) % self.compression_factor
        if pad:
            latents = F.pad(latents, (0, pad))
            time = latents.shape[2]
        groups = time // self.compression_factor
        x = latents.view(batch, channels, groups, self.compression_factor)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.view(batch, self.output_dim, groups)


class TTLReferenceEncoder(nn.Module):
    """Reference encoder for text-to-latent style tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(COMPRESSED_LATENT_DIM, TTL_ATTN_DIM, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock1d(
                    dim=TTL_ATTN_DIM,
                    intermediate_dim=512,
                    kernel_size=5,
                    layer_scale_init_value=1 / 6,
                )
                for _ in range(6)
            ]
        )
        self.value_queries = nn.Parameter(torch.randn(TTL_TOKENS, TTL_ATTN_DIM) * 0.02)
        self.key_queries = nn.Parameter(torch.randn(TTL_TOKENS, TTL_ATTN_DIM) * 0.02)
        self.value_attn1 = nn.MultiheadAttention(
            embed_dim=TTL_ATTN_DIM,
            num_heads=4,
            batch_first=True,
        )
        self.value_attn2 = nn.MultiheadAttention(
            embed_dim=TTL_ATTN_DIM,
            num_heads=4,
            batch_first=True,
        )
        self.key_attn1 = nn.MultiheadAttention(
            embed_dim=TTL_ATTN_DIM,
            num_heads=4,
            batch_first=True,
        )
        self.key_attn2 = nn.MultiheadAttention(
            embed_dim=TTL_ATTN_DIM,
            num_heads=4,
            batch_first=True,
        )
        self.value_output = nn.Linear(TTL_ATTN_DIM, TTL_ATTN_DIM)
        self.key_output = nn.Linear(TTL_ATTN_DIM, TTL_ATTN_DIM)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.input(latents)
        x = self.blocks(x).transpose(1, 2)
        value_queries = self.value_queries.unsqueeze(0).expand(latents.shape[0], -1, -1)
        key_queries = self.key_queries.unsqueeze(0).expand(latents.shape[0], -1, -1)
        values, _ = self.value_attn1(value_queries, x, x, need_weights=False)
        values, _ = self.value_attn2(values, x, x, need_weights=False)
        keys, _ = self.key_attn1(key_queries, x, x, need_weights=False)
        keys, _ = self.key_attn2(keys, x, x, need_weights=False)
        return torch.cat([self.key_output(keys), self.value_output(values)], dim=-1)


class DPReferenceEncoder(nn.Module):
    """Reference encoder for duration-predictor style tokens."""

    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 64
        self.input = nn.Conv1d(COMPRESSED_LATENT_DIM, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock1d(
                    dim=hidden_dim,
                    intermediate_dim=256,
                    kernel_size=5,
                    layer_scale_init_value=1 / 4,
                )
                for _ in range(4)
            ]
        )
        self.queries = nn.Parameter(torch.randn(DP_TOKENS, DP_STYLE_DIM) * 0.02)
        self.key = nn.Linear(hidden_dim, DP_STYLE_DIM)
        self.value = nn.Linear(hidden_dim, DP_STYLE_DIM)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=DP_STYLE_DIM,
            num_heads=4,
            batch_first=True,
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=DP_STYLE_DIM,
            num_heads=4,
            batch_first=True,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # The paper describes an 8-query, 16-d attention projection and then a
        # stacked 64-d reference embedding. Public Supertonic JSON stores
        # style_dp as [8, 16], so this scaffold preserves the runtime format.
        x = self.input(latents)
        x = self.blocks(x).transpose(1, 2)
        key = self.key(x)
        value = self.value(x)
        queries = self.queries.unsqueeze(0).expand(latents.shape[0], -1, -1)
        tokens, _ = self.attn1(queries, key, value, need_weights=False)
        tokens, _ = self.attn2(tokens, key, value, need_weights=False)
        return tokens


class AudioToStyleEncoder(nn.Module):
    """Trainable reference-audio encoder that predicts Supertonic style tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_encoder = MelLatentEncoder()
        self.temporal_compressor = TemporalCompressor()
        self.ttl_reference_encoder = TTLReferenceEncoder()
        self.dp_reference_encoder = DPReferenceEncoder()

    def forward(self, mel: torch.Tensor) -> StyleTensors:
        latents = self.latent_encoder(mel)
        latents = self.temporal_compressor(latents)
        return StyleTensors(
            style_ttl=self.ttl_reference_encoder(latents),
            style_dp=self.dp_reference_encoder(latents),
        )


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _init_vocos_conv_or_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d | nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
