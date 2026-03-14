"""
dataset.py
----------
PyTorch Dataset classes for the neural speech decoding pipeline.

Two variants:
  - SpeechDataset         : returns (neural, phonemes, seq_len, n_steps)
  - SpeechDatasetWithText : adds the sentence label as a 5th return value
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    """
    Lazy-loading dataset.  Neural features stay as raw (T_i, 512) arrays
    in memory; clipping to *max_time_steps* and tensor conversion happen
    at __getitem__ time.

    Returns
    -------
    neural_tensor   : FloatTensor  (1, 512, T_eff)
    phoneme_tensor  : LongTensor   (seq_len,)   — variable length
    seq_len         : int
    effective_n_steps : int
    """

    def __init__(self, data_dict: dict, max_time_steps: int = 1500):
        self.data           = data_dict
        self.length         = len(data_dict['neural_features'])
        self.max_time_steps = max_time_steps

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n_time_steps      = int(self.data['n_time_steps'][idx])
        seq_len           = int(self.data['seq_len'][idx])
        effective_n_steps = min(n_time_steps, self.max_time_steps)

        # (T_eff, 512) → (512, T_eff) → (1, 512, T_eff)
        neural   = self.data['neural_features'][idx][:effective_n_steps]
        phonemes = self.data['seq_class_ids'][idx][:seq_len]

        neural_tensor  = (torch.from_numpy(neural).float()
                          .transpose(0, 1).unsqueeze(0))
        phoneme_tensor = torch.from_numpy(phonemes.astype(np.int64))

        return neural_tensor, phoneme_tensor, seq_len, effective_n_steps


class SpeechDatasetWithText(Dataset):
    """
    Identical to :class:`SpeechDataset` but also returns the sentence label
    (decoded to a plain Python ``str``).

    Returns
    -------
    neural_tensor   : FloatTensor  (1, 512, T_eff)
    phoneme_tensor  : LongTensor   (seq_len,)
    seq_len         : int
    effective_n_steps : int
    text_label      : str
    """

    def __init__(self, data_dict: dict, max_time_steps: int = 1500):
        self.data           = data_dict
        self.length         = len(data_dict['neural_features'])
        self.max_time_steps = max_time_steps

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n_time_steps      = int(self.data['n_time_steps'][idx])
        seq_len           = int(self.data['seq_len'][idx])
        effective_n_steps = min(n_time_steps, self.max_time_steps)

        neural   = self.data['neural_features'][idx][:effective_n_steps]
        phonemes = self.data['seq_class_ids'][idx][:seq_len]

        text_label = self.data['sentence_label'][idx]
        if isinstance(text_label, bytes):
            text_label = text_label.decode('utf-8')

        neural_tensor  = (torch.from_numpy(neural).float()
                          .transpose(0, 1).unsqueeze(0))
        phoneme_tensor = torch.from_numpy(phonemes.astype(np.int64))

        return neural_tensor, phoneme_tensor, seq_len, effective_n_steps, text_label
