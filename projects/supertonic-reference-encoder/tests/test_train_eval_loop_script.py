from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_train_eval_loop_script_runs_train_then_cer_eval_until_plateau():
    script = PROJECT_ROOT / "scripts" / "train_eval_autoencoder_loop.sh"

    text = script.read_text(encoding="utf-8")

    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-runs/autoencoder-cer-loop-3000}"' in text
    assert 'MAX_ROUNDS="${MAX_ROUNDS:-3}"' in text
    assert 'EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-1000}"' in text
    assert 'EVAL_LIMIT="${EVAL_LIMIT:-1000}"' in text
    assert 'MIN_CER_IMPROVEMENT="${MIN_CER_IMPROVEMENT:-0.005}"' in text
    assert "scripts/train_autoencoder_cuda.sh" in text
    assert "scripts/eval_fleurs_autoencoder_cer.py" in text
    assert '--checkpoint "prev=' in text
    assert '--checkpoint "current=' in text
    assert "generated_mean_cer" in text
    assert "cer_improvement" in text
    assert "stopping: CER improvement below threshold" in text
