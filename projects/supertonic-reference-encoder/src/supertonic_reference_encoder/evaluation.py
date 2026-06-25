from __future__ import annotations

import statistics
import string
from dataclasses import dataclass


_PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation + " \t\r\n")


@dataclass(frozen=True)
class EvaluationRow:
    index: int
    audio: str
    text: str
    transcript: str
    cer: float


def normalize_for_cer(text: str) -> str:
    return text.lower().translate(_PUNCTUATION_TRANSLATION)


def character_error_rate(reference: str, hypothesis: str) -> float:
    normalized_reference = normalize_for_cer(reference)
    normalized_hypothesis = normalize_for_cer(hypothesis)
    distance = _levenshtein_distance(normalized_reference, normalized_hypothesis)
    return distance / max(1, len(normalized_reference))


def summarize_evaluation_rows(rows: list[EvaluationRow]) -> dict[str, float | int]:
    if not rows:
        return {
            "count": 0,
            "mean_cer": 0.0,
            "median_cer": 0.0,
            "max_cer": 0.0,
        }
    values = [row.cer for row in rows]
    return {
        "count": len(rows),
        "mean_cer": float(statistics.fmean(values)),
        "median_cer": float(statistics.median(values)),
        "max_cer": float(max(values)),
    }


def _levenshtein_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            substitution_cost = int(reference_character != hypothesis_character)
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]
