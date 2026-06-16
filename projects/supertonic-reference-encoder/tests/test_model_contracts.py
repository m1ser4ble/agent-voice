import torch

from supertonic_reference_encoder.model import (
    AudioToStyleEncoder,
    ConvNeXtBlock1d,
    MelLatentEncoder,
    StyleTensors,
    TTLReferenceEncoder,
    TemporalCompressor,
)


def test_audio_to_style_encoder_returns_supertonic_style_shapes():
    model = AudioToStyleEncoder()
    mel = torch.randn(2, 228, 320)

    style = model(mel)

    assert isinstance(style, StyleTensors)
    assert style.style_ttl.shape == (2, 50, 256)
    assert style.style_dp.shape == (2, 8, 16)
    assert style.style_ttl.requires_grad
    assert style.style_dp.requires_grad


def test_latent_encoder_matches_paper_24_dimensional_latent_space():
    encoder = MelLatentEncoder()
    mel = torch.randn(2, 228, 320)

    latent = encoder(mel)

    assert latent.shape == (2, 24, 320)


def test_temporal_compressor_concatenates_six_24_dimensional_frames():
    compressor = TemporalCompressor(compression_factor=6, latent_dim=24)
    latent = torch.randn(2, 24, 13)

    compressed = compressor(latent)

    assert compressed.shape == (2, 144, 3)
    assert torch.allclose(compressed[:, :24, 0], latent[:, :, 0])
    assert torch.allclose(compressed[:, 24:48, 0], latent[:, :, 1])
    assert torch.allclose(compressed[:, 120:144, 0], latent[:, :, 5])


def test_ttl_reference_encoder_emits_key_value_concat():
    encoder = TTLReferenceEncoder()
    compressed_latent = torch.randn(2, 144, 64)

    style_ttl = encoder(compressed_latent)

    assert style_ttl.shape == (2, 50, 256)
    key, value = style_ttl.chunk(2, dim=-1)
    assert key.shape == (2, 50, 128)
    assert value.shape == (2, 50, 128)


def test_encoder_accepts_variable_time_lengths():
    model = AudioToStyleEncoder()

    short = model(torch.randn(1, 228, 120))
    long = model(torch.randn(1, 228, 480))

    assert short.style_ttl.shape == long.style_ttl.shape == (1, 50, 256)
    assert short.style_dp.shape == long.style_dp.shape == (1, 8, 16)


def test_convnext_block_preserves_sequence_shape():
    block = ConvNeXtBlock1d(
        dim=64,
        intermediate_dim=256,
        kernel_size=5,
        layer_scale_init_value=0.25,
    )
    x = torch.randn(3, 64, 128)

    y = block(x)

    assert y.shape == x.shape


def test_convnext_block_uses_vocos_layer_norm_and_layer_scale():
    block = ConvNeXtBlock1d(
        dim=64,
        intermediate_dim=256,
        kernel_size=5,
        layer_scale_init_value=0.25,
    )

    assert block.norm.eps == 1e-6
    assert block.gamma is not None
    assert block.gamma.shape == (64,)
    assert torch.allclose(block.gamma, torch.full((64,), 0.25))


def test_reference_encoder_stacks_use_vocos_backbone_layer_scale_defaults():
    latent_encoder = MelLatentEncoder()
    ttl_encoder = TTLReferenceEncoder()

    assert torch.allclose(latent_encoder.blocks[0].gamma, torch.full((512,), 0.1))
    assert torch.allclose(ttl_encoder.blocks[0].gamma, torch.full((128,), 1 / 6))
