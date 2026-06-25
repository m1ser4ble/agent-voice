from supertonic_reference_encoder.evaluation import (
    EvaluationRow,
    character_error_rate,
    summarize_evaluation_rows,
)


def test_character_error_rate_counts_character_edits_after_normalization():
    assert character_error_rate("다리 밑 수직 간격", "다리 및 수직 간격") == 1 / 7


def test_character_error_rate_ignores_punctuation_spaces_and_case():
    assert character_error_rate("Hello, World!", "hello world") == 0.0


def test_summarize_evaluation_rows_reports_mean_and_median():
    rows = [
        EvaluationRow(index=0, audio="a.wav", text="a", transcript="a", cer=0.0),
        EvaluationRow(index=1, audio="b.wav", text="b", transcript="x", cer=0.5),
        EvaluationRow(index=2, audio="c.wav", text="c", transcript="y", cer=1.0),
    ]

    summary = summarize_evaluation_rows(rows)

    assert summary == {
        "count": 3,
        "mean_cer": 0.5,
        "median_cer": 0.5,
        "max_cer": 1.0,
    }
