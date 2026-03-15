"""
synthetic_data.py
-----------------
Phoneme confusion map and synthetic error injection for language model
training data augmentation.

The confusion map is derived from acoustic phonetics: phonemes that share
place/manner of articulation are grouped as likely confusable pairs.

Error distribution (from validation set analysis):
  Targeted substitution : 21.26 %
  Random substitution   : 45.08 %
  Deletion              : 21.34 %
  Insertion             : 12.32 %
"""

import random
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Phoneme confusion map
# ─────────────────────────────────────────────────────────────────────────────

PHONEME_CONFUSION_MAP = {
    # Plosives — voicing pairs
    27: [7, 31, 20],   # P → B, T, K
    7:  [27, 9, 15],   # B → P, D, G
    31: [9, 27, 20],   # T → D, P, K
    9:  [31, 7, 15],   # D → T, B, G
    20: [15, 31, 27],  # K → G, T, P
    15: [20, 9, 7],    # G → K, D, B
    # Fricatives — voicing / similar hiss
    14: [35, 32, 29],  # F  → V, TH, S
    35: [14, 10, 38],  # V  → F, DH, Z
    32: [10, 14, 29],  # TH → DH, F, S
    10: [32, 35, 38],  # DH → TH, V, Z
    29: [38, 30, 32],  # S  → Z, SH, TH
    38: [29, 39, 10],  # Z  → S, ZH, DH
    30: [39, 29, 8],   # SH → ZH, S, CH
    39: [30, 38, 19],  # ZH → SH, Z, JH
    16: [32, 14],      # HH → TH, F
    # Affricates
    8:  [19, 30, 31],  # CH → JH, SH, T
    19: [8, 39, 9],    # JH → CH, ZH, D
    # Nasals
    22: [23, 24],      # M  → N, NG
    23: [22, 24],      # N  → M, NG
    24: [23, 22],      # NG → N, M
    # Liquids / glides
    21: [28, 36, 37],  # L  → R, W, Y
    28: [21, 36],      # R  → L, W
    36: [21, 28, 34],  # W  → L, R, UW
    37: [21, 18],      # Y  → L, IY
    # Front vowels
    18: [17, 13],      # IY → IH, EY
    17: [18, 11],      # IH → IY, EH
    11: [17, 2],       # EH → IH, AE
    2:  [11, 1],       # AE → EH, AA
    # Back / central vowels
    1:  [4, 3],        # AA → AO, AH
    4:  [1, 33],       # AO → AA, UH
    33: [34, 4],       # UH → UW, AO
    34: [33, 25],      # UW → UH, OW
    3:  [11, 33, 12],  # AH → EH, UH, ER
    12: [3, 28],       # ER → AH, R
    # Diphthongs
    13: [11, 18, 6],   # EY → EH, IY, AY
    6:  [1, 18, 13],   # AY → AA, IY, EY
    26: [4, 18],       # OY → AO, IY
    5:  [1, 34, 25],   # AW → AA, UW, OW
    25: [4, 34, 5],    # OW → AO, UW, AW
}

# Valid phoneme IDs (1–39; 0 = BLANK, 40 = '|')
VOCAB_PHONEMES = list(range(1, 40))

# Error type distribution from validation set
_ERR_TYPES  = ['targeted_sub', 'random_sub', 'deletion', 'insertion']
_ERR_PROBS  = [0.2126, 0.4508, 0.2134, 0.1232]


# ─────────────────────────────────────────────────────────────────────────────
# Error injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_synthetic_errors(
    sequence: list,
    target_per: float = 0.15,
    confusion_map: dict = None,
) -> list:
    """
    Inject realistic errors into a ground-truth phoneme sequence.

    Parameters
    ----------
    sequence    : list of phoneme IDs (ints)
    target_per  : approximate phoneme error rate to inject [0, 1]
    confusion_map : phoneme → confusable phonemes mapping
                    (defaults to PHONEME_CONFUSION_MAP)

    Returns
    -------
    list of int  — corrupted phoneme sequence
    """
    if confusion_map is None:
        confusion_map = PHONEME_CONFUSION_MAP

    synthetic_seq = []

    for ph in sequence:
        if random.random() < target_per:
            err_type = np.random.choice(_ERR_TYPES, p=_ERR_PROBS)

            if err_type == 'targeted_sub':
                if ph in confusion_map:
                    synthetic_seq.append(random.choice(confusion_map[ph]))
                else:
                    choices = [p for p in VOCAB_PHONEMES if p != ph]
                    synthetic_seq.append(random.choice(choices))

            elif err_type == 'random_sub':
                choices = [p for p in VOCAB_PHONEMES if p != ph]
                synthetic_seq.append(random.choice(choices))

            elif err_type == 'deletion':
                continue   # drop this phoneme

            elif err_type == 'insertion':
                # Insert a random phoneme before the current one
                synthetic_seq.append(random.choice(VOCAB_PHONEMES))
                synthetic_seq.append(ph)
        else:
            synthetic_seq.append(ph)

    return synthetic_seq


# ─────────────────────────────────────────────────────────────────────────────
# Mixed dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def create_merged_lm_dataset(
    train_loader,
    acoustic_model,
    device,
    gt_ratio: float  = 0.20,
    syn_ratio: float = 0.60,
    pred_ratio: float = 0.20,
    target_per: float = 0.10,
):
    """
    Build a mixed phoneme→text training dataset:

      - GT ratio   : ground-truth phoneme sequences
      - Syn ratio  : synthetically corrupted GT sequences
      - Pred ratio : real acoustic model predictions

    Total: ``gt_ratio + syn_ratio + pred_ratio`` should equal 1.
    In practice, synth copies = round(syn_ratio / gt_ratio) per GT sample.

    Parameters
    ----------
    train_loader    : DataLoader yielding (neural, ids, lens, steps, texts)
    acoustic_model  : trained acoustic model in eval mode
    device          : torch.device
    gt_ratio, syn_ratio, pred_ratio : mixing proportions

    Returns
    -------
    merged_phonemes : list[list[int]]
    merged_texts    : list[str]
    """
    import torch
    from src.evaluation.decode import greedy_ctc_decode

    acoustic_model.eval()
    merged_phonemes, merged_texts = [], []

    n_synth = max(1, round(syn_ratio / max(gt_ratio, 1e-6)))

    print('Generating mixed dataset…')
    with torch.no_grad():
        for batch_feat, batch_ids, batch_lens, _, texts in train_loader:
            batch_feat = batch_feat.to(device)
            log_probs  = acoustic_model(batch_feat)
            predicted_seqs = greedy_ctc_decode(log_probs, blank_id=0)

            for b in range(len(batch_ids)):
                ref_seq = batch_ids[b][:int(batch_lens[b].item())].tolist()
                ref_seq = [x for x in ref_seq if x != 0]
                text_label = texts[b]

                # Ground truth
                merged_phonemes.append(ref_seq)
                merged_texts.append(text_label)

                # Synthetic
                for _ in range(n_synth):
                    merged_phonemes.append(
                        inject_synthetic_errors(ref_seq, target_per=target_per)
                    )
                    merged_texts.append(text_label)

                # Real predicted
                merged_phonemes.append(predicted_seqs[b])
                merged_texts.append(text_label)

    print(f'Total sequences in mixed dataset: {len(merged_phonemes)}')
    return merged_phonemes, merged_texts
