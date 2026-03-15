"""
decode.py
---------
CTC decoding utilities: greedy decoding and (optionally) beam search.
"""

import torch
from torch import Tensor


def greedy_ctc_decode(log_probs: Tensor, blank_id: int = 0) -> list:
    """
    Greedy CTC decode.

    Parameters
    ----------
    log_probs : (T, B, C)  log-softmax probabilities
    blank_id  : index of the CTC blank token (default 0)

    Returns
    -------
    list of lists — one decoded integer sequence per batch item
    """
    tokens = log_probs.argmax(dim=-1).cpu().numpy()   # (T, B)
    decoded = []
    for b in range(tokens.shape[1]):
        seq, prev = [], -1
        for t in range(tokens.shape[0]):
            tok = int(tokens[t, b])
            if tok != blank_id and tok != prev:
                seq.append(tok)
            prev = tok
        decoded.append(seq)
    return decoded


def beam_ctc_decode(
    log_probs: Tensor,
    beam_width: int = 10,
    blank_id: int = 0,
) -> list:
    """
    Simple prefix-beam CTC decode (pure Python, no external library).

    For production use consider ``ctcdecode`` with a language model.

    Parameters
    ----------
    log_probs  : (T, B, C) log-softmax probabilities
    beam_width : number of beams to maintain per batch item
    blank_id   : CTC blank index

    Returns
    -------
    list of lists — best decoded sequence per batch item
    """
    import numpy as np
    from collections import defaultdict

    probs = log_probs.exp().cpu().numpy()   # (T, B, C)
    T, B, C = probs.shape
    results  = []

    for b in range(B):
        # beams: dict { sequence_tuple : (prob_blank, prob_non_blank) }
        beams = {(): (1.0, 0.0)}

        for t in range(T):
            new_beams = defaultdict(lambda: (0.0, 0.0))
            for seq, (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = probs[t, b, c]
                    if c == blank_id:
                        nb, nnb = new_beams[seq]
                        new_beams[seq] = (nb + (p_b + p_nb) * p, nnb)
                    else:
                        new_seq = seq + (c,)
                        if seq and seq[-1] == c:
                            nb, nnb = new_beams[new_seq]
                            new_beams[new_seq] = (nb, nnb + p_nb * p)
                            nb2, nnb2 = new_beams[seq]
                            new_beams[seq] = (nb2, nnb2 + p_b * p)
                        else:
                            nb, nnb = new_beams[new_seq]
                            new_beams[new_seq] = (nb, nnb + (p_b + p_nb) * p)

            # Keep top-k beams
            beams = dict(
                sorted(new_beams.items(),
                       key=lambda x: x[1][0] + x[1][1],
                       reverse=True)[:beam_width]
            )

        best_seq = max(beams, key=lambda s: sum(beams[s]))
        results.append(list(best_seq))

    return results
