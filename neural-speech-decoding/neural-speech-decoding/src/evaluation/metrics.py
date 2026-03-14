"""
metrics.py
----------
Evaluation metrics for the neural speech decoding pipeline.

Provides:
  - phoneme_error_rate (PER)   — pure Python, no external dependency
  - word_error_rate (WER)      — wrapper around jiwer
  - align_and_classify_errors  — detailed error breakdown via DP backtrack
"""

import re
from collections import Counter

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PER
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = np.arange(m + 1)
    for i in range(1, n + 1):
        prev = dp.copy()
        dp[0] = i
        for j in range(1, m + 1):
            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
    return int(dp[m])


def phoneme_error_rate(refs: list, hyps: list) -> float:
    """
    Compute corpus-level PER (%).

    Parameters
    ----------
    refs : list of list[int]   reference phoneme ID sequences
    hyps : list of list[int]   hypothesis phoneme ID sequences

    Returns
    -------
    float  PER in percent [0, 100]
    """
    total_err = total_len = 0
    for ref, hyp in zip(refs, hyps):
        total_err += _levenshtein(ref, hyp)
        total_len += max(len(ref), 1)
    return total_err / total_len * 100.0


def calculate_per(reference: list, hypothesis: list) -> float:
    """Per-sample PER as a fraction [0, 1]."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    return _levenshtein(reference, hypothesis) / len(reference)


# ─────────────────────────────────────────────────────────────────────────────
# WER
# ─────────────────────────────────────────────────────────────────────────────

def clean_text_for_wer(text: str) -> str:
    """Lowercase + strip punctuation for fair WER comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate as a fraction [0, 1].

    Uses the built-in DP implementation (no external dependency).
    """
    ref_words = clean_text_for_wer(reference).split()
    hyp_words = clean_text_for_wer(hypothesis).split()
    m, n = len(ref_words), len(hyp_words)
    if m == 0:
        return 1.0 if n > 0 else 0.0

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n] / m


# ─────────────────────────────────────────────────────────────────────────────
# Detailed error classification
# ─────────────────────────────────────────────────────────────────────────────

def align_and_classify_errors(
    ref_seq: list,
    hyp_seq: list,
    confusion_map: dict,
) -> Counter:
    """
    Align two phoneme sequences with DP, then backtrack to classify each
    error as targeted substitution, random substitution, deletion, or
    insertion.

    Parameters
    ----------
    ref_seq       : ground-truth phoneme ID sequence
    hyp_seq       : hypothesis phoneme ID sequence
    confusion_map : dict {phoneme_id: [likely_confusion_ids]}

    Returns
    -------
    Counter with keys: targeted_sub, random_sub, deletion, insertion
    """
    n, m = len(ref_seq), len(hyp_seq)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1): dp[i][0] = i
    for j in range(1, m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_seq[i - 1] == hyp_seq[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i, j = n, m
    errors = Counter(
        {'targeted_sub': 0, 'random_sub': 0, 'deletion': 0, 'insertion': 0}
    )

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_seq[i - 1] == hyp_seq[j - 1]:
            i -= 1; j -= 1
        elif (i > 0 and j > 0
              and dp[i][j] == dp[i - 1][j - 1] + 1):
            ref_ph = ref_seq[i - 1]
            hyp_ph = hyp_seq[j - 1]
            if ref_ph in confusion_map and hyp_ph in confusion_map[ref_ph]:
                errors['targeted_sub'] += 1
            else:
                errors['random_sub'] += 1
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            errors['deletion'] += 1; i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            errors['insertion'] += 1; j -= 1
        else:
            # Fallback (should not occur)
            i = max(i - 1, 0); j = max(j - 1, 0)

    return errors
