from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_intelligible_prepare_script_uses_speech_preserving_defaults():
    script = PROJECT_ROOT / "scripts" / "prepare_target_autoencoder_intelligible_ft.sh"

    text = script.read_text(encoding="utf-8")

    assert 'OUTPUT_DIR="${OUTPUT_DIR:-data/autoencoder-jarvis-intelligible}"' in text
    assert 'TARGET_REPEAT="${TARGET_REPEAT:-20}"' in text
    assert "--augmentations" in text
    assert '"original"' in text
    assert '"bandpass"' in text
    assert '"compressed"' in text
    assert '"distorted"' not in text
    assert '"codec"' not in text


def test_intelligible_cuda_train_script_uses_conservative_defaults():
    script = PROJECT_ROOT / "scripts" / "train_autoencoder_intelligible_ft_cuda.sh"

    text = script.read_text(encoding="utf-8")

    assert 'MANIFEST="${MANIFEST:-data/autoencoder-jarvis-intelligible/manifest.jsonl}"' in text
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-runs/autoencoder-jarvis-intelligible-ft}"' in text
    assert 'EPOCHS="${EPOCHS:-80}"' in text
    assert 'LEARNING_RATE="${LEARNING_RATE:-0.00001}"' in text
    assert "scripts/prepare_target_autoencoder_intelligible_ft.sh" in text
