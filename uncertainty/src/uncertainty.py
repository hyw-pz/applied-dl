"""
uncertainty.py
--------------
Uncertainty estimation for the neural speech decoding pipeline.

Two uncertainty sources are measured:

  AM Uncertainty  — negated mean max log-prob over valid CTC frames.
                    Computed directly from acoustic model output, zero overhead.

  LM Uncertainty  — depends on the language model architecture:
    • BART (Seq2Seq): negated length-normalised beam search score
                      (outputs.sequences_scores from generate()).
    • Qwen (Causal):  negated mean per-token log-prob via
                      model.compute_transition_scores().

Both are defined such that higher values = less confident.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.dataset import compute_token_lengths
from src.metrics import calculate_per, calculate_wer, ctc_greedy_decode
from src.vocabulary import LOGIT_TO_PHONEME


# ─────────────────────────────────────────────────────────────────────────────
# AM Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def compute_am_sequence_score(log_probs: torch.Tensor,
                              input_lengths: torch.Tensor) -> torch.Tensor:
    """
    Acoustic model uncertainty: negated mean max log-prob over valid CTC frames.

    Formally:  U_AM = -(1/T) Σ_t max_c log p(c | t)

    Args:
        log_probs:     (T, B, C) log-probabilities from the acoustic model
        input_lengths: (B,)     valid frame counts after temporal pooling

    Returns:
        Tensor of shape (B,) — higher value means less confident
    """
    max_log_probs = log_probs.max(dim=-1)[0]  # (T, B)
    scores = []
    for i in range(log_probs.shape[1]):
        valid_len = input_lengths[i].item()
        score = max_log_probs[:valid_len, i].mean().item()
        scores.append(-score)
    return torch.tensor(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone AM uncertainty evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_am_uncertainty(acoustic_model, data_loader,
                            device, args) -> tuple:
    """
    Evaluates acoustic model uncertainty against phoneme error rate.

    Expects a DataLoader that yields 4-item batches (no text labels):
        (neural_batch, phoneme_batch, seq_lens, input_time_steps)

    Args:
        acoustic_model: DBConformerCTC in eval mode
        data_loader:    DataLoader using SpeechDataset (no text)
        device:         torch.device
        args:           namespace with pool_kernel and pool_stride

    Returns:
        (all_pers, all_am_uncertainties)
    """
    acoustic_model.eval()
    all_pers, all_am_uncertainties = [], []
    samples = []

    k_w, s_w = args.pool_kernel, args.pool_stride

    print("Running AM Uncertainty Evaluation...")
    with torch.no_grad():
        for neural_batch, phoneme_batch, seq_lens, input_time_steps in tqdm(data_loader):
            neural_batch = neural_batch.to(device)

            input_lengths = compute_token_lengths(
                input_time_steps, k_w, s_w
            ).to(device)

            B, T_p = neural_batch.size(0), neural_batch.size(-1)
            P = compute_token_lengths(torch.tensor([T_p]), k_w, s_w).item()
            seq_range = torch.arange(P, device=device).expand(B, P)
            attn_mask = (seq_range < input_lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)

            log_probs = acoustic_model(neural_batch, mask=attn_mask).log_softmax(dim=-1)
            am_uncertainty = compute_am_sequence_score(log_probs, input_lengths)
            predictions_ids = ctc_greedy_decode(log_probs)

            for i, phoneme_ids in enumerate(predictions_ids):
                true_len = seq_lens[i].item()
                true_seq = phoneme_batch[i][:true_len].tolist()
                per = calculate_per(true_seq, phoneme_ids)
                all_pers.append(per)
                all_am_uncertainties.append(am_uncertainty[i].item())
                if len(samples) < 15:
                    samples.append({
                        'am_unc': am_uncertainty[i].item(), 'per': per,
                        'pred': [LOGIT_TO_PHONEME[idx] for idx in phoneme_ids],
                        'true': [LOGIT_TO_PHONEME[idx] for idx in true_seq],
                    })

    corr = np.corrcoef(all_am_uncertainties, all_pers)[0, 1]
    print(f"\n--- AM Uncertainty Analysis ---")
    print(f"AM uncertainty vs PER correlation: {corr:.3f}")
    print(f"Mean PER: {np.mean(all_pers):.2%} | Median PER: {np.median(all_pers):.2%}")

    unc_array = np.array(all_am_uncertainties)
    q25, q50, q75 = np.percentile(unc_array, [25, 50, 75])
    print(f"\n--- AM Confidence Stratification (quartile-based) ---")
    for lo, hi, label in [(0, q25, 'HIGH'), (q25, q50, 'MEDIUM'),
                          (q50, q75, 'LOW'), (q75, 9999, 'VERY_LOW')]:
        group = [p for p, u in zip(all_pers, all_am_uncertainties) if lo <= u < hi]
        if group:
            print(f"  {label:10s}: {len(group):4d} samples ({len(group)/len(all_pers):.1%}), "
                  f"Mean PER={np.mean(group):.2%}, Median PER={np.median(group):.2%}")

    _plot_calibration_and_scatter(
        x_vals=all_am_uncertainties, y_vals=all_pers,
        x_label='AM Uncertainty', y_label='PER',
        title='AM Uncertainty Analysis', corr=corr,
        save_path='am_uncertainty.png'
    )
    return all_pers, all_am_uncertainties


# ─────────────────────────────────────────────────────────────────────────────
# BART pipeline uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline_uncertainty_bart(acoustic_model, lm_model, lm_tokenizer,
                                       data_loader, device, args) -> tuple:
    """
    Joint AM + BART uncertainty evaluation.

    LM uncertainty = -outputs.sequences_scores (negated normalised beam score).

    Args:
        acoustic_model: DBConformerCTC in eval mode
        lm_model:       BART AutoModelForSeq2SeqLM in eval mode
        lm_tokenizer:   corresponding tokenizer
        data_loader:    DataLoader using SpeechDatasetWithText
        device:         torch.device
        args:           namespace with pool_kernel and pool_stride

    Returns:
        (all_wers, all_pers, all_am_uncertainties, all_lm_uncertainties)
    """
    acoustic_model.eval()
    lm_model.eval()

    all_wers, all_pers = [], []
    all_lm_uncertainties, all_am_uncertainties = [], []
    samples = []

    k_w, s_w = args.pool_kernel, args.pool_stride

    print("Running BART Pipeline Uncertainty Evaluation...")
    with torch.no_grad():
        for neural_batch, phoneme_batch, seq_lens, input_time_steps, text_labels in tqdm(data_loader):
            neural_batch = neural_batch.to(device)
            input_lengths = compute_token_lengths(
                input_time_steps, pool_kernel=k_w, pool_stride=s_w
            ).to(device)

            B, T_p = neural_batch.size(0), neural_batch.size(-1)
            P = compute_token_lengths(torch.tensor([T_p]), k_w, s_w).item()
            seq_range = torch.arange(P, device=device).expand(B, P)
            attn_mask = (seq_range < input_lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)

            # Acoustic model
            log_probs = acoustic_model(neural_batch, mask=attn_mask).log_softmax(dim=-1)
            am_uncertainty = compute_am_sequence_score(log_probs, input_lengths)
            predictions_ids = ctc_greedy_decode(log_probs, blank_id=0)

            predicted_phoneme_strings, batch_pers = [], []
            for i, phoneme_ids in enumerate(predictions_ids):
                true_len = seq_lens[i].item()
                true_seq = [x for x in phoneme_batch[i][:true_len].tolist() if x != 0]
                batch_pers.append(calculate_per(true_seq, phoneme_ids))
                all_pers.append(batch_pers[-1])
                predicted_phoneme_strings.append(
                    " ".join(LOGIT_TO_PHONEME[idx] for idx in phoneme_ids
                             if idx < len(LOGIT_TO_PHONEME))
                )

            # BART inference
            enc = lm_tokenizer(predicted_phoneme_strings, return_tensors="pt",
                               padding=True, max_length=256, truncation=True).to(device)
            outputs = lm_model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_length=256, num_beams=4, length_penalty=0.6,
                output_scores=True, return_dict_in_generate=True
            )

            lm_uncertainty = -outputs.sequences_scores.cpu()
            predicted_texts = lm_tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )

            for i, pred_text in enumerate(predicted_texts):
                true_text = text_labels[i]
                if isinstance(true_text, bytes):
                    true_text = true_text.decode('utf-8')
                wer = calculate_wer(true_text, pred_text)
                all_wers.append(wer)
                all_lm_uncertainties.append(lm_uncertainty[i].item())
                all_am_uncertainties.append(am_uncertainty[i].item())
                if len(samples) < 15:
                    samples.append({'am_unc': am_uncertainty[i].item(),
                                    'lm_unc': lm_uncertainty[i].item(),
                                    'per': batch_pers[i], 'pred': pred_text,
                                    'true': true_text, 'wer': wer})

    _print_pipeline_results(samples, all_wers, all_pers,
                             all_am_uncertainties, all_lm_uncertainties,
                             model_name="BART",
                             lm_boundaries=[(0, 0.05, 'HIGH'),
                                            (0.05, 0.15, 'MEDIUM'),
                                            (0.15, 0.25, 'LOW'),
                                            (0.25, 9999, 'VERY_LOW')])
    _plot_pipeline(all_wers, all_pers, all_am_uncertainties, all_lm_uncertainties,
                   title="BART Pipeline Uncertainty Analysis",
                   save_path="bart_uncertainty.png")
    return all_wers, all_pers, all_am_uncertainties, all_lm_uncertainties


# ─────────────────────────────────────────────────────────────────────────────
# Qwen pipeline uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline_uncertainty_qwen(acoustic_model, qwen_model, qwen_tokenizer,
                                       data_loader, device, args) -> tuple:
    """
    Joint AM + Qwen uncertainty evaluation.

    LM uncertainty = negated mean per-token log-prob via compute_transition_scores.
    Formally: U_LM = -(1/|y|) Σ_i log p(y_i | y_{<i}, x)

    Args:
        acoustic_model:  DBConformerCTC in eval mode
        qwen_model:      Qwen2.5 PeftModel in eval mode
        qwen_tokenizer:  corresponding tokenizer
        data_loader:     DataLoader using SpeechDatasetWithText
        device:          torch.device
        args:            namespace with pool_kernel and pool_stride

    Returns:
        (all_wers, all_pers, all_am_uncertainties, all_lm_uncertainties)
    """
    acoustic_model.eval()
    qwen_model.eval()
    qwen_tokenizer.padding_side = "left"  # required for decoder-only batched generation

    all_wers, all_pers = [], []
    all_lm_uncertainties, all_am_uncertainties = [], []
    samples = []

    k_w, s_w = args.pool_kernel, args.pool_stride
    system_prompt = (
        "You are an expert speech decoding system. Translate the noisy ARPAbet phonemes "
        "into English text. CRITICAL CONSTRAINTS: You must ONLY output valid English "
        "dictionary words. Output ONLY the final text."
    )

    print("Running Qwen Pipeline Uncertainty Evaluation...")
    with torch.no_grad():
        for neural_batch, phoneme_batch, seq_lens, input_time_steps, text_labels in tqdm(data_loader):
            neural_batch = neural_batch.to(device)
            input_lengths = compute_token_lengths(
                input_time_steps, pool_kernel=k_w, pool_stride=s_w
            ).to(device)

            B, T_p = neural_batch.size(0), neural_batch.size(-1)
            P = compute_token_lengths(torch.tensor([T_p]), k_w, s_w).item()
            seq_range = torch.arange(P, device=device).expand(B, P)
            attn_mask = (seq_range < input_lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)

            # Acoustic model
            log_probs = acoustic_model(neural_batch, mask=attn_mask).log_softmax(dim=-1)
            am_uncertainty = compute_am_sequence_score(log_probs, input_lengths)
            predictions_ids = ctc_greedy_decode(log_probs, blank_id=0)

            predicted_phoneme_strings, batch_pers = [], []
            for i, phoneme_ids in enumerate(predictions_ids):
                true_len = seq_lens[i].item()
                true_seq = [x for x in phoneme_batch[i][:true_len].tolist() if x != 0]
                batch_pers.append(calculate_per(true_seq, phoneme_ids))
                all_pers.append(batch_pers[-1])
                predicted_phoneme_strings.append(
                    " ".join(LOGIT_TO_PHONEME[idx] for idx in phoneme_ids
                             if idx < len(LOGIT_TO_PHONEME))
                )

            # Qwen inference
            prompts = [
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nPhonemes: {ph}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                for ph in predicted_phoneme_strings
            ]
            inputs = qwen_tokenizer(
                prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=512
            ).to(qwen_model.device)

            outputs = qwen_model.generate(
                **inputs, max_new_tokens=64, num_beams=4,
                do_sample=False, output_scores=True,
                return_dict_in_generate=True, early_stopping=True,
                length_penalty=0.8, repetition_penalty=1.1
            )

            # Qwen LM uncertainty via per-token log-probs
            transition_scores = qwen_model.compute_transition_scores(
                outputs.sequences, outputs.scores,
                outputs.beam_indices, normalize_logits=True
            )
            lm_uncertainty = []
            for k in range(len(transition_scores)):
                valid = transition_scores[k][transition_scores[k] != 0]
                lm_uncertainty.append(-valid.mean().item() if len(valid) > 0 else 10.0)

            prompt_len = inputs.input_ids.shape[1]
            predicted_texts = qwen_tokenizer.batch_decode(
                outputs.sequences[:, prompt_len:], skip_special_tokens=True
            )

            for i, pred_text in enumerate(predicted_texts):
                true_text = text_labels[i]
                if isinstance(true_text, bytes):
                    true_text = true_text.decode('utf-8')
                wer = calculate_wer(true_text, pred_text)
                all_wers.append(wer)
                all_lm_uncertainties.append(lm_uncertainty[i])
                all_am_uncertainties.append(am_uncertainty[i].item())
                if len(samples) < 15:
                    samples.append({'am_unc': am_uncertainty[i].item(),
                                    'lm_unc': lm_uncertainty[i],
                                    'per': batch_pers[i], 'pred': pred_text,
                                    'true': true_text, 'wer': wer})

    q25, q50, q75 = np.percentile(all_lm_uncertainties, [25, 50, 75])
    _print_pipeline_results(samples, all_wers, all_pers,
                             all_am_uncertainties, all_lm_uncertainties,
                             model_name="Qwen",
                             lm_boundaries=[(0, q25, 'HIGH'), (q25, q50, 'MEDIUM'),
                                            (q50, q75, 'LOW'), (q75, 9999, 'VERY_LOW')])
    _plot_pipeline(all_wers, all_pers, all_am_uncertainties, all_lm_uncertainties,
                   title="Qwen2.5 Pipeline Uncertainty Analysis",
                   save_path="qwen_uncertainty.png")
    return all_wers, all_pers, all_am_uncertainties, all_lm_uncertainties


# ─────────────────────────────────────────────────────────────────────────────
# ECE
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(wers: list, uncertainties: list,
                n_bins: int = 10, correct_threshold: float = 0.3) -> float:
    """
    Expected Calibration Error.

    Converts uncertainty to confidence via confidence = 1 / (1 + uncertainty).
    Samples with WER < correct_threshold are treated as correctly decoded.

    Args:
        wers:               per-sample WER values
        uncertainties:      per-sample LM uncertainty values
        n_bins:             number of equally-spaced confidence bins
        correct_threshold:  WER threshold below which a sample counts as correct

    Returns:
        ECE scalar — lower is better, 0 = perfect calibration
    """
    confidences = [1 / (1 + u) for u in uncertainties]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [lo <= c < hi for c in confidences]
        if not any(mask):
            continue
        bin_confs = [c for c, m in zip(confidences, mask) if m]
        bin_wers  = [w for w, m in zip(wers, mask) if m]
        bin_acc   = sum(w < correct_threshold for w in bin_wers) / len(bin_wers)
        ece += (len(bin_wers) / len(wers)) * abs(np.mean(bin_confs) - bin_acc)

    return ece


def evaluate_ece(wers: list, uncertainties: list,
                 thresholds: tuple = (0.0, 0.2, 0.3, 0.5)) -> None:
    """Prints ECE for multiple correctness thresholds."""
    print("--- ECE across correctness thresholds ---")
    for thresh in thresholds:
        ece = compute_ece(wers, uncertainties, correct_threshold=thresh)
        print(f"  WER < {thresh:.1f} as correct: ECE = {ece:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Coverage vs WER trade-off
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_wer_tradeoff(all_wers: list, all_am_uncertainties: list,
                                all_lm_uncertainties: list,
                                save_path: str = "coverage_vs_wer.png") -> None:
    """
    Plots coverage vs mean WER when filtering by joint AM + LM percentile thresholds.
    Coverage = fraction of samples whose joint uncertainty falls below both thresholds.
    """
    am_thresholds = np.percentile(all_am_uncertainties, np.arange(5, 100, 5))
    lm_thresholds = np.percentile(all_lm_uncertainties, np.arange(5, 100, 5))

    coverages, mean_wers = [], []
    for am_t, lm_t in zip(am_thresholds, lm_thresholds):
        subset = [w for w, a, l in zip(all_wers, all_am_uncertainties, all_lm_uncertainties)
                  if a < am_t and l < lm_t]
        if subset:
            coverages.append(len(subset) / len(all_wers))
            mean_wers.append(np.mean(subset))

    plt.figure(figsize=(8, 5))
    plt.plot(coverages, mean_wers, 'o-', color='blue')
    plt.xlabel('Coverage (fraction of samples predicted)')
    plt.ylabel('Mean WER')
    plt.title('Coverage vs WER Trade-off')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_pipeline_results(samples, all_wers, all_pers,
                             all_am_uncertainties, all_lm_uncertainties,
                             model_name, lm_boundaries):
    print(f"\n{'AM_UNC':<10} | {'LM_UNC':<10} | {'PER':<10} | {'WER':<10} | "
          f"{'Predicted':<35} | {'True'}")
    print("-" * 115)
    for s in samples:
        print(f"{s['am_unc']:.4f}    | {s['lm_unc']:.4f}    | "
              f"{s['per']:.2%}    | {s['wer']:.2%}    | "
              f"{s['pred'][:33]:<35} | {s['true']}")

    print(f"\n--- {model_name} Uncertainty Correlations ---")
    print(f"  AM uncertainty vs PER: {np.corrcoef(all_am_uncertainties, all_pers)[0,1]:.3f}")
    print(f"  AM uncertainty vs WER: {np.corrcoef(all_am_uncertainties, all_wers)[0,1]:.3f}")
    print(f"  LM uncertainty vs WER: {np.corrcoef(all_lm_uncertainties, all_wers)[0,1]:.3f}")
    print(f"  LM uncertainty vs PER: {np.corrcoef(all_lm_uncertainties, all_pers)[0,1]:.3f}")
    print(f"  Mean WER:   {np.mean(all_wers):.2%}  |  Median WER: {np.median(all_wers):.2%}")
    print(f"  Mean PER:   {np.mean(all_pers):.2%}")

    print(f"\n--- LM Confidence Tiers ---")
    for lo, hi, label in lm_boundaries:
        group = [w for w, u in zip(all_wers, all_lm_uncertainties) if lo <= u < hi]
        if group:
            print(f"  {label:10s}: {len(group):4d} samples ({len(group)/len(all_wers):.1%}), "
                  f"Mean WER={np.mean(group):.2%}, Median WER={np.median(group):.2%}")


def _plot_calibration_and_scatter(x_vals, y_vals, x_label, y_label,
                                   title, corr, save_path, n_bins=10):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    sorted_idx  = np.argsort(x_vals)
    s_y = [y_vals[i] for i in sorted_idx]
    s_x = [x_vals[i] for i in sorted_idx]
    bin_size = len(s_y) // n_bins
    bin_y = [np.mean(s_y[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)]
    bin_x = [np.mean(s_x[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)]

    axes[0].plot(bin_x, bin_y, 'o-', color='red')
    axes[0].set_xlabel(x_label); axes[0].set_ylabel(f'Average {y_label}')
    axes[0].set_title('Calibration Curve'); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x_vals, y_vals, alpha=0.3, s=20, color='green')
    axes[1].set_xlabel(x_label); axes[1].set_ylabel(y_label)
    axes[1].set_title(f'{x_label} vs {y_label} (corr={corr:.3f})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def _plot_pipeline(all_wers, all_pers, all_am_uncertainties,
                   all_lm_uncertainties, title, save_path, n_bins=10):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    def _calibration(ax, uncs, errs, color, x_label, y_label, ax_title):
        sorted_idx = np.argsort(uncs)
        s_e = [errs[i] for i in sorted_idx]
        s_u = [uncs[i] for i in sorted_idx]
        bin_size = len(s_e) // n_bins
        bin_e = [np.mean(s_e[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)]
        bin_u = [np.mean(s_u[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)]
        ax.plot(bin_u, bin_e, 'o-', color=color)
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_title(ax_title); ax.grid(True, alpha=0.3)

    _calibration(axes[0, 0], all_lm_uncertainties, all_wers,
                 'blue', 'LM Uncertainty', 'Average WER', 'LM Calibration Curve')

    corr_lm = np.corrcoef(all_lm_uncertainties, all_wers)[0, 1]
    axes[0, 1].scatter(all_lm_uncertainties, all_wers, alpha=0.3, s=20, color='purple')
    axes[0, 1].set_xlabel('LM Uncertainty'); axes[0, 1].set_ylabel('WER')
    axes[0, 1].set_title(f'LM Uncertainty vs WER (corr={corr_lm:.3f})')
    axes[0, 1].grid(True, alpha=0.3)

    _calibration(axes[1, 0], all_am_uncertainties, all_wers,
                 'red', 'AM Uncertainty', 'Average WER', 'AM Calibration Curve')

    corr_am = np.corrcoef(all_am_uncertainties, all_pers)[0, 1]
    axes[1, 1].scatter(all_am_uncertainties, all_pers, alpha=0.3, s=20, color='green')
    axes[1, 1].set_xlabel('AM Uncertainty'); axes[1, 1].set_ylabel('PER')
    axes[1, 1].set_title(f'AM Uncertainty vs PER (corr={corr_am:.3f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")
