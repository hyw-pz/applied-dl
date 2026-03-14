"""
dataset.py
----------
Dataset classes, samplers, and collate functions for neural speech decoding.
All classes expect data dicts loaded from HDF5 session files.
"""

import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class SpeechDataset(Dataset):
    """
    Lazy-loading dataset returning neural features + phoneme labels.
    Returns 4-item tuples: (neural_tensor, phoneme_tensor, seq_len, n_steps)
    """

    def __init__(self, data_dict: dict, max_time_steps: int = 1500):
        self.data = data_dict
        self.length = len(data_dict['neural_features'])
        self.max_time_steps = max_time_steps

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n_time_steps = int(self.data['n_time_steps'][idx])
        seq_len = int(self.data['seq_len'][idx])
        effective_n_steps = min(n_time_steps, self.max_time_steps)

        neural = self.data['neural_features'][idx][:effective_n_steps]
        phonemes = self.data['seq_class_ids'][idx][:seq_len]

        neural_tensor = (torch.from_numpy(neural).float()
                         .transpose(0, 1).unsqueeze(0))        # (1, 512, T)
        phoneme_tensor = torch.from_numpy(phonemes.astype(np.int64))

        return neural_tensor, phoneme_tensor, seq_len, effective_n_steps


class SpeechDatasetWithText(Dataset):
    """
    Same as SpeechDataset but also returns the sentence-level text label.
    Returns 5-item tuples: (neural_tensor, phoneme_tensor, seq_len, n_steps, text)
    """

    def __init__(self, data_dict: dict, max_time_steps: int = 1500):
        self.data = data_dict
        self.length = len(data_dict['neural_features'])
        self.max_time_steps = max_time_steps

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n_time_steps = int(self.data['n_time_steps'][idx])
        seq_len = int(self.data['seq_len'][idx])
        effective_n_steps = min(n_time_steps, self.max_time_steps)

        neural = self.data['neural_features'][idx][:effective_n_steps]
        phonemes = self.data['seq_class_ids'][idx][:seq_len]

        text_label = self.data['sentence_label'][idx]
        if isinstance(text_label, bytes):
            text_label = text_label.decode('utf-8')

        neural_tensor = (torch.from_numpy(neural).float()
                         .transpose(0, 1).unsqueeze(0))
        phoneme_tensor = torch.from_numpy(phonemes.astype(np.int64))

        return neural_tensor, phoneme_tensor, seq_len, effective_n_steps, text_label


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

class BucketBatchSampler(Sampler):
    """
    Sorts trials by n_time_steps, groups into batches, then shuffles batch
    order. Minimises within-batch padding variance for more accurate CTC lengths.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sorted_indices = sorted(
            range(len(dataset)),
            key=lambda i: dataset.data['n_time_steps'][i]
        )

    def __iter__(self):
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        random.shuffle(batches)
        for batch in batches:
            if not self.drop_last or len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Collate functions
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Collate for SpeechDataset (no text).
    Pads neural sequences to the longest trial in the batch.

    Returns
    -------
    neural_batch  : (B, 1, 512, T_max)
    phoneme_batch : (B, S_max)  zero-padded
    seq_lens      : (B,)
    time_steps    : (B,)
    """
    neurals, phonemes_list, seq_lens, time_steps = zip(*batch)
    neurals_reshaped = [n.squeeze(0).transpose(0, 1) for n in neurals]
    neural_batch = (pad_sequence(neurals_reshaped, batch_first=True)
                    .transpose(1, 2).unsqueeze(1))
    phoneme_batch = pad_sequence(phonemes_list, batch_first=True, padding_value=0)
    return (neural_batch, phoneme_batch,
            torch.tensor(seq_lens, dtype=torch.long),
            torch.tensor(time_steps, dtype=torch.long))


def collate_fn_with_text(batch):
    """
    Collate for SpeechDatasetWithText (includes text labels).

    Returns
    -------
    neural_batch  : (B, 1, 512, T_max)
    phoneme_batch : (B, S_max)  zero-padded
    seq_lens      : (B,)
    time_steps    : (B,)
    texts         : tuple of str
    """
    neurals, phonemes_list, seq_lens, time_steps, texts = zip(*batch)
    neurals_reshaped = [n.squeeze(0).transpose(0, 1) for n in neurals]
    neural_batch = (pad_sequence(neurals_reshaped, batch_first=True)
                    .transpose(1, 2).unsqueeze(1))
    phoneme_batch = pad_sequence(phonemes_list, batch_first=True, padding_value=0)
    return (neural_batch, phoneme_batch,
            torch.tensor(seq_lens, dtype=torch.long),
            torch.tensor(time_steps, dtype=torch.long),
            texts)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute CTC input lengths after temporal pooling
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_lengths(input_time_steps: torch.Tensor,
                          pool_kernel: int,
                          pool_stride: int) -> torch.Tensor:
    """Returns valid CTC frame counts after AvgPool1d downsampling."""
    return ((input_time_steps - pool_kernel) // pool_stride + 1).long()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(data_dict: dict,
                     batch_size: int,
                     max_time_steps: int = 1500,
                     with_text: bool = True,
                     num_workers: int = 2,
                     pin_memory: bool = True) -> DataLoader:
    """
    Convenience factory that wires up dataset, sampler, and DataLoader.

    Args:
        data_dict:       dict loaded from HDF5 session files
        batch_size:      samples per batch
        max_time_steps:  clip neural sequences at this length
        with_text:       if True, uses SpeechDatasetWithText and collate_fn_with_text
        num_workers:     DataLoader worker processes
        pin_memory:      pin memory for faster GPU transfers

    Returns:
        DataLoader ready for iteration
    """
    if with_text:
        dataset = SpeechDatasetWithText(data_dict, max_time_steps)
        collate = collate_fn_with_text
    else:
        dataset = SpeechDataset(data_dict, max_time_steps)
        collate = collate_fn

    sampler = BucketBatchSampler(dataset, batch_size)
    return DataLoader(dataset, batch_sampler=sampler,
                      collate_fn=collate,
                      num_workers=num_workers,
                      pin_memory=pin_memory)
