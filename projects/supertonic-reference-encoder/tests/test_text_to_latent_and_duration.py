import torch

from supertonic_reference_encoder.duration_predictor import DurationPredictor
from supertonic_reference_encoder.text_to_latent import TextToLatentFlow


def test_text_to_latent_flow_predicts_compressed_latent_velocity():
    model = TextToLatentFlow(vocab_size=128)
    text_tokens = torch.randint(0, 128, (2, 32))
    noisy_latent = torch.randn(2, 144, 20)
    style_ttl = torch.randn(2, 50, 256)
    time = torch.rand(2)

    velocity = model(
        text_tokens=text_tokens,
        noisy_latent=noisy_latent,
        style_ttl=style_ttl,
        time=time,
    )

    assert velocity.shape == noisy_latent.shape


def test_duration_predictor_returns_utterance_duration():
    model = DurationPredictor(vocab_size=128)
    text_tokens = torch.randint(0, 128, (2, 32))
    style_dp = torch.randn(2, 8, 16)

    duration = model(text_tokens=text_tokens, style_dp=style_dp)

    assert duration.shape == (2,)
    assert torch.all(duration > 0)
