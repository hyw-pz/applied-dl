"""
dataloader.py
-------------
Custom sampler and collate functions for the neural speech decoding pipeline.

Key components:
  - BucketBatchSampler  : sort by trial length → less within-batch padding
  - collate_fn          : dynamic padding for (neural, phonemes) batches
  - collate_fn_with_text: same + passes sentence labels through
  - compute_token_lengths: CTC frame count after temporal pooling
"""

import random

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

class BucketBatchSampler(Sampler):
    """
    Sort trials by ``n_time_steps``, group into batches, then shuffle the
    batch order.  Minimises within-batch padding variance → more accurate
    CTC input lengths.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.sorted_indices = sorted(
            range(len(dataset)),
            key=lambda i: dataset.data['n_time_steps'][i],
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
    Dynamic padding — pads to the longest trial in *this* batch only.

    Input  : list of (neural (1,512,T_i), phonemes, seq_len, n_steps)
    Output :
        neural_batch  : (B, 1, 512, T_max)
        phoneme_batch : (B, S_max)  — 0-padded
        seq_lens      : (B,)
        time_steps    : (B,)
    """
    neurals, phonemes_list, seq_lens, time_steps = zip(*batch)

    # (1,512,T) → (T,512) for pad_sequence → pad → (B,T_max,512) → back
    neurals_reshaped = [n.squeeze(0).transpose(0, 1) for n in neurals]
    neural_batch = (pad_sequence(neurals_reshaped, batch_first=True)
                    .transpose(1, 2).unsqueeze(1))

    phoneme_batch = pad_sequence(phonemes_list, batch_first=True, padding_value=0)

    return (
        neural_batch,
        phoneme_batch,
        torch.tensor(seq_lens,   dtype=torch.long),
        torch.tensor(time_steps, dtype=torch.long),
    )


def collate_fn_with_text(batch):
    """
    Same as :func:`collate_fn` but also collects the sentence labels.

    Output adds:
        texts : tuple of str
    """
    neurals, phonemes_list, seq_lens, time_steps, texts = zip(*batch)

    neurals_reshaped = [n.squeeze(0).transpose(0, 1) for n in neurals]
    neural_batch = (pad_sequence(neurals_reshaped, batch_first=True)
                    .transpose(1, 2).unsqueeze(1))

    phoneme_batch = pad_sequence(phonemes_list, batch_first=True, padding_value=0)

    return (
        neural_batch,
        phoneme_batch,
        torch.tensor(seq_lens,   dtype=torch.long),
        torch.tensor(time_steps, dtype=torch.long),
        texts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CTC length helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_lengths(
    input_time_steps: Tensor,
    pool_kernel: int,
    pool_stride: int,
) -> Tensor:
    """
    Compute valid CTC frame counts after temporal AvgPool1d.

    Parameters
    ----------
    input_time_steps : (B,)  actual T values (before padding)
    pool_kernel      : AvgPool1d kernel size
    pool_stride      : AvgPool1d stride

    Returns
    -------
    LongTensor (B,)
    """
    return ((input_time_steps - pool_kernel) // pool_stride + 1).long()


# ─────────────────────────────────────────────────────────────────────────────
# Padding mask helper
# ─────────────────────────────────────────────────────────────────────────────

def make_attention_mask(input_lengths: Tensor, max_len: int) -> Tensor:
    """
    Build a boolean attention mask (B, 1, 1, P) suitable for
    ``F.scaled_dot_product_attention``.

    True  = valid token
    False = padding (will be masked out)
    """
    B = input_lengths.size(0)
    device = input_lengths.device
    seq_range = torch.arange(max_len, device=device).expand(B, max_len)
    bool_mask = seq_range < input_lengths.unsqueeze(1)   # (B, P)
    return bool_mask.unsqueeze(1).unsqueeze(2)            # (B, 1, 1, P)
