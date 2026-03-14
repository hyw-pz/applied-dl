"""
metrics.py
----------
Decoding utilities and error-rate metrics for neural speech decoding.
"""

import re

import numpy as np
import torch
from torch import Tensor

from src.vocabulary import LOGIT_TO_PHONEME


# ─────────────────────────────────────────────────────────────────────────────
# CTC decoding
# ─────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: Tensor, blank_id: int = 0) -> list:
    """
    Greedy CTC decode: argmax per frame, collapse repeats, remove blanks.

    Args:
        log_probs: (T, B, C) log-probability tensor from the acoustic model
        blank_id:  index of the CTC blank token (default 0)

    Returns:
        List of B decoded int sequences
    """
    pred_ids = torch.argmax(log_probs, dim=-1).transpose(0, 1)  # (B, T)
    decoded_preds = []
    for seq in pred_ids:
        decoded, prev_token = [], -1
        for token in seq.tolist():
            if token != prev_token and token != blank_id:
                decoded.append(token)
            prev_token = token
        decoded_preds.append(decoded)
    return decoded_preds


def decode_ids_to_string(seq_ids: list) -> str:
    """
    Converts a list of phoneme integer IDs to a space-separated ARPAbet string.
    Skips BLANK (0) and out-of-vocabulary indices.

    Example:
        [16, 11, 21, 25] → "HH EH L OW"
    """
    return " ".join(
        LOGIT_TO_PHONEME[p]
        for p in seq_ids
        if p != 0 and p < len(LOGIT_TO_PHONEME)
    )


def decode_text_labels(labels: list) -> list:
    """Decodes bytes labels to UTF-8 strings; passes str labels through."""
    return [t.decode('utf-8') if isinstance(t, bytes) else t for t in labels]


# ─────────────────────────────────────────────────────────────────────────────
# Error rate metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_per(reference: list, hypothesis: list) -> float:
    """
    Phoneme Error Rate via Levenshtein edit distance.

    Args:
        reference:  list of int phoneme IDs (ground truth)
        hypothesis: list of int phoneme IDs (prediction)

    Returns:
        PER in [0, ∞)  (can exceed 1 if many insertions)
    """
    m, n = len(reference), len(hypothesis)
    if m == 0:
        return 0.0 if n == 0 else 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n] / m


def phoneme_error_rate(refs: list, hyps: list) -> float:
    """
    Corpus-level PER (%).

    Args:
        refs:  list of reference int-ID sequences
        hyps:  list of hypothesis int-ID sequences

    Returns:
        PER as a percentage in [0, 100]
    """
    def levenshtein(a, b):
        n, m = len(a), len(b)
        dp = np.arange(m + 1)
        for i in range(1, n + 1):
            prev = dp.copy()
            dp[0] = i
            for j in range(1, m + 1):
                dp[j] = min(prev[j] + 1, dp[j-1] + 1,
                            prev[j-1] + (0 if a[i-1] == b[j-1] else 1))
        return dp[m]

    total_err, total_len = 0, 0
    for ref, hyp in zip(refs, hyps):
        total_err += levenshtein(ref, hyp)
        total_len += max(len(ref), 1)
    return total_err / total_len * 100.0


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate via Levenshtein edit distance on whitespace-tokenised words.
    Lowercases input and strips punctuation before comparison.

    Args:
        reference:  ground truth sentence string
        hypothesis: predicted sentence string

    Returns:
        WER in [0, ∞)
    """
    ref_words = re.sub(r'[^\w\s]', '', reference.lower()).split()
    hyp_words = re.sub(r'[^\w\s]', '', hypothesis.lower()).split()

    m, n = len(ref_words), len(hyp_words)
    if m == 0:
        return 0.0 if n == 0 else 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n] / m
