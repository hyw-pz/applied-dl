"""
am_uncertainty.py
-----------------
Acoustic model (AM) uncertainty estimation.

Two methods:
  1. Sequence score  : negative mean of per-timestep max log-prob
  2. MC-Dropout      : variance across N stochastic forward passes
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_am_sequence_score(
    log_probs: Tensor,
    input_lengths: Tensor,
) -> Tensor:
    """
    Compute per-sample AM uncertainty as the negative mean of the
    maximum log-prob at each valid CTC timestep.

    A high positive value → model is uncertain (low max log-prob).
    A value close to 0    → model is confident.

    Parameters
    ----------
    log_probs      : (T, B, C)  log-softmax probabilities
    input_lengths  : (B,)       valid frame counts per sample

    Returns
    -------
    Tensor (B,)  uncertainty scores (higher = more uncertain)
    """
    max_log_probs = log_probs.max(dim=-1)[0]   # (T, B)
    scores = []
    for i in range(log_probs.shape[1]):
        valid_len = int(input_lengths[i].item())
        score = max_log_probs[:valid_len, i].mean().item()
        scores.append(-score)
    return torch.tensor(scores)


def mc_dropout_uncertainty(
    model,
    neural_batch: Tensor,
    input_lengths: Tensor,
    n_samples: int = 20,
) -> tuple:
    """
    Monte-Carlo Dropout uncertainty estimation.

    Forces ``model.train()`` so Dropout layers remain active, then
    runs *n_samples* stochastic forward passes.

    Parameters
    ----------
    model          : acoustic model (DBConformerCTC or NeuralSpeechModel)
    neural_batch   : (B, 1, C, T)  input neural features
    input_lengths  : (B,)          valid CTC frame counts
    n_samples      : number of MC samples

    Returns
    -------
    mean_probs   : (T, B, C)  mean probability over MC samples
    uncertainty  : (B,)       mean predictive entropy per sample
    """
    model.train()   # keep dropout active
    probs_list = []

    with torch.no_grad():
        for _ in range(n_samples):
            log_probs = model(neural_batch).log_softmax(dim=-1)
            probs_list.append(log_probs.exp())   # (T, B, C)

    probs_stack = torch.stack(probs_list, dim=0)  # (N, T, B, C)
    mean_probs  = probs_stack.mean(dim=0)          # (T, B, C)

    # Predictive entropy H[y] = -sum_c p_c * log(p_c)
    entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=-1)  # (T, B)

    # Average over valid timesteps
    uncertainty = []
    for i in range(mean_probs.shape[1]):
        valid_len = int(input_lengths[i].item())
        uncertainty.append(entropy[:valid_len, i].mean().item())

    model.eval()
    return mean_probs, torch.tensor(uncertainty)
