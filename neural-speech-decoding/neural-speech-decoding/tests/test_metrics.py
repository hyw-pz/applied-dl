"""
tests/test_metrics.py
Unit tests for PER, WER, and error classification.
"""

import pytest
from src.evaluation.metrics import (
    phoneme_error_rate,
    calculate_per,
    calculate_wer,
    clean_text_for_wer,
    align_and_classify_errors,
)


class TestPER:
    def test_perfect(self):
        assert phoneme_error_rate([[1, 2, 3]], [[1, 2, 3]]) == 0.0

    def test_full_error(self):
        per = phoneme_error_rate([[1, 2, 3]], [[4, 5, 6]])
        assert per == pytest.approx(100.0)

    def test_one_deletion(self):
        per = phoneme_error_rate([[1, 2, 3]], [[1, 3]])
        assert per == pytest.approx(100.0 / 3, rel=1e-3)

    def test_empty_hyp(self):
        per = phoneme_error_rate([[1, 2]], [[]])
        assert per == pytest.approx(100.0)

    def test_calculate_per_fraction(self):
        assert calculate_per([1, 2, 3], [1, 2, 3]) == 0.0
        assert calculate_per([], []) == 0.0
        assert calculate_per([], [1]) == 1.0


class TestWER:
    def test_perfect(self):
        assert calculate_wer("hello world", "hello world") == 0.0

    def test_one_substitution(self):
        wer = calculate_wer("hello world", "hello earth")
        assert wer == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert calculate_wer("Hello World", "hello world") == 0.0

    def test_punctuation_stripped(self):
        assert calculate_wer("hello, world!", "hello world") == 0.0

    def test_empty_reference(self):
        assert calculate_wer("", "") == 0.0
        assert calculate_wer("", "word") == 1.0

    def test_clean_text(self):
        assert clean_text_for_wer("Hello, World!") == "hello world"
        assert clean_text_for_wer("  spaces  ") == "spaces"


class TestErrorClassification:
    CONFUSION_MAP = {
        1: [2, 3],   # phoneme 1 can be confused with 2 or 3
        4: [5],
    }

    def test_no_errors(self):
        errors = align_and_classify_errors([1, 2, 3], [1, 2, 3], self.CONFUSION_MAP)
        assert sum(errors.values()) == 0

    def test_targeted_substitution(self):
        # phoneme 1 confused with 2 → targeted
        errors = align_and_classify_errors([1], [2], self.CONFUSION_MAP)
        assert errors['targeted_sub'] == 1
        assert errors['random_sub']   == 0

    def test_random_substitution(self):
        # phoneme 1 confused with 9 (not in map) → random
        errors = align_and_classify_errors([1], [9], self.CONFUSION_MAP)
        assert errors['random_sub'] == 1

    def test_deletion(self):
        errors = align_and_classify_errors([1, 2], [1], self.CONFUSION_MAP)
        assert errors['deletion'] == 1

    def test_insertion(self):
        errors = align_and_classify_errors([1], [1, 2], self.CONFUSION_MAP)
        assert errors['insertion'] == 1
