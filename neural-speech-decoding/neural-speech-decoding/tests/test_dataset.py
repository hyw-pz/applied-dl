"""
tests/test_dataset.py
Unit tests for Dataset, BucketBatchSampler, and collate functions.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import SpeechDataset, SpeechDatasetWithText
from src.data.dataloader import (
    BucketBatchSampler,
    collate_fn,
    collate_fn_with_text,
    compute_token_lengths,
    make_attention_mask,
)


def _make_data(n=6, max_t=60, channels=16):
    """Build a minimal fake data dict."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(20, max_t, size=n).tolist()
    return {
        'neural_features': [
            rng.random((l, channels)).astype(np.float32) for l in lengths
        ],
        'n_time_steps':  lengths,
        'seq_class_ids': [
            rng.integers(1, 40, size=rng.integers(3, 8)).astype(np.int64)
            for _ in range(n)
        ],
        'seq_len':       [rng.integers(3, 8) for _ in range(n)],
        'sentence_label': [b'hello world'] * n,
        'transcriptions': [None] * n,
        'session':        ['s1'] * n,
        'block_num':      list(range(n)),
        'trial_num':      list(range(n)),
    }


class TestSpeechDataset:
    def test_length(self):
        ds = SpeechDataset(_make_data(4), max_time_steps=50)
        assert len(ds) == 4

    def test_item_shapes(self):
        ds  = SpeechDataset(_make_data(4), max_time_steps=50)
        neural, phon, slen, n_steps = ds[0]
        assert neural.ndim == 3                  # (1, C, T_eff)
        assert neural.shape[0] == 1
        assert neural.shape[1] == 16
        assert phon.ndim == 1
        assert isinstance(slen, int)
        assert n_steps <= 50

    def test_max_time_steps_clipping(self):
        ds = SpeechDataset(_make_data(4), max_time_steps=10)
        for i in range(len(ds)):
            _, _, _, n_steps = ds[i]
            assert n_steps <= 10


class TestSpeechDatasetWithText:
    def test_text_decoding(self):
        ds = SpeechDatasetWithText(_make_data(3), max_time_steps=50)
        _, _, _, _, text = ds[0]
        assert isinstance(text, str)
        assert text == 'hello world'


class TestBucketBatchSampler:
    def test_total_samples(self):
        ds      = SpeechDataset(_make_data(6), max_time_steps=50)
        sampler = BucketBatchSampler(ds, batch_size=2)
        indices = [idx for batch in sampler for idx in batch]
        assert sorted(indices) == list(range(6))

    def test_drop_last(self):
        ds      = SpeechDataset(_make_data(5), max_time_steps=50)
        sampler = BucketBatchSampler(ds, batch_size=2, drop_last=True)
        assert len(sampler) == 2   # floor(5/2) = 2


class TestCollateFn:
    def test_batch_shape(self):
        ds    = SpeechDataset(_make_data(4), max_time_steps=50)
        items = [ds[i] for i in range(4)]
        neural_batch, phon_batch, seq_lens, time_steps = collate_fn(items)
        assert neural_batch.shape[0] == 4
        assert neural_batch.shape[1] == 1
        assert neural_batch.ndim == 4
        assert phon_batch.shape[0] == 4

    def test_with_text(self):
        ds    = SpeechDatasetWithText(_make_data(3), max_time_steps=50)
        items = [ds[i] for i in range(3)]
        neural_batch, phon_batch, seq_lens, time_steps, texts = collate_fn_with_text(items)
        assert len(texts) == 3
        assert all(isinstance(t, str) for t in texts)


class TestTokenLengths:
    def test_values(self):
        steps = torch.tensor([80, 60, 40])
        lens  = compute_token_lengths(steps, pool_kernel=15, pool_stride=8)
        expected = (steps - 15) // 8 + 1
        assert (lens == expected).all()

    def test_attention_mask_shape(self):
        il   = torch.tensor([8, 5, 3])
        mask = make_attention_mask(il, max_len=10)
        assert mask.shape == (3, 1, 1, 10)
        # First sample: all 8 positions valid
        assert mask[0, 0, 0, :8].all()
        assert not mask[0, 0, 0, 8]
