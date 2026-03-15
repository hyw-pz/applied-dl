"""
calibration.py
--------------
Calibration analysis for the neural speech decoding pipeline.

Provides:
  - compute_ece          : Expected Calibration Error
  - confidence_report    : human-readable confidence level for one prediction
  - coverage_wer_curve   : coverage vs. WER trade-off analysis
  - plot_uncertainty     : 4-panel calibration / scatter plots
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# ECE
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    wers: list,
    uncertainties: list,
    n_bins: int = 10,
    correct_threshold: float = 0.3,
) -> float:
    """
    Expected Calibration Error.

    Converts uncertainty to confidence via ``1 / (1 + u)``, then checks
    whether samples with high confidence actually have low WER.

    Parameters
    ----------
    wers            : per-sample WER values [0, 1]
    uncertainties   : per-sample uncertainty values (higher = less confident)
    n_bins          : number of equal-width confidence bins
    correct_threshold: WER below this value is treated as "correct"

    Returns
    -------
    float ECE ∈ [0, 1]  (lower is better)
    """
    confidences  = [1 / (1 + u) for u in uncertainties]
    bin_edges    = np.linspace(0, 1, n_bins + 1)
    ece          = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = [lo <= c < hi for c in confidences]
        if not any(mask):
            continue
        bin_confs  = [c for c, m in zip(confidences, mask) if m]
        bin_wers   = [w for w, m in zip(wers, mask) if m]
        bin_acc    = sum(w <= correct_threshold for w in bin_wers) / len(bin_wers)
        bin_conf   = float(np.mean(bin_confs))
        ece       += (len(bin_wers) / len(wers)) * abs(bin_acc - bin_conf)

    return ece


# ─────────────────────────────────────────────────────────────────────────────
# Confidence report
# ─────────────────────────────────────────────────────────────────────────────

def confidence_report(
    predicted_text: str,
    lm_uncertainty: float,
    threshold_high: float = 0.05,
    threshold_low: float  = 0.25,
) -> dict:
    """
    Assign a confidence level label based on LM uncertainty.

    Thresholds should be tuned to the validation set distribution.

    Returns
    -------
    dict with keys: text, confidence, action, expected_wer, raw_uncertainty
    """
    if lm_uncertainty < threshold_high:
        level = 'HIGH'
        action = 'Use directly'
        expected_wer = '< 10%'
    elif lm_uncertainty < threshold_low:
        level = 'MEDIUM'
        action = 'Usable, but consider review'
        expected_wer = '10% – 40%'
    else:
        level = 'LOW'
        action = 'Recommend retry / manual check'
        expected_wer = '> 60%'

    return {
        'text':            predicted_text,
        'confidence':      level,
        'action':          action,
        'expected_wer':    expected_wer,
        'raw_uncertainty': lm_uncertainty,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coverage–WER trade-off
# ─────────────────────────────────────────────────────────────────────────────

def coverage_wer_curve(
    wers: list,
    am_uncertainties: list,
    lm_uncertainties: list,
    n_steps: int = 19,
) -> tuple:
    """
    Sweep joint (AM, LM) uncertainty thresholds using percentiles and
    compute the coverage / mean-WER at each threshold.

    Returns
    -------
    coverages : list[float]
    mean_wers : list[float]
    """
    am_thresholds = np.percentile(am_uncertainties, np.arange(5, 100, 100 // n_steps))
    lm_thresholds = np.percentile(lm_uncertainties, np.arange(5, 100, 100 // n_steps))

    coverages, mean_wers = [], []
    for am_t, lm_t in zip(am_thresholds, lm_thresholds):
        subset = [
            w for w, a, l in zip(wers, am_uncertainties, lm_uncertainties)
            if a < am_t and l < lm_t
        ]
        if subset:
            coverages.append(len(subset) / len(wers))
            mean_wers.append(float(np.mean(subset)))

    return coverages, mean_wers


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty(
    wers: list,
    pers: list,
    am_uncertainties: list,
    lm_uncertainties: list,
    title: str = 'Pipeline Uncertainty Analysis',
) -> plt.Figure:
    """
    4-panel plot:
      (0,0) LM calibration curve     (0,1) LM uncertainty vs WER scatter
      (1,0) AM calibration curve     (1,1) AM uncertainty vs PER scatter
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    n_bins    = 10

    def _calibration_curve(uncs, errors, ax, xlabel, title, color):
        sorted_idx = np.argsort(uncs)
        s_e = [errors[i] for i in sorted_idx]
        s_u = [uncs[i]   for i in sorted_idx]
        bin_size = max(len(s_e) // n_bins, 1)
        be = [np.mean(s_e[i * bin_size:(i + 1) * bin_size]) for i in range(n_bins)]
        bu = [np.mean(s_u[i * bin_size:(i + 1) * bin_size]) for i in range(n_bins)]
        ax.plot(bu, be, 'o-', color=color)
        ax.set_xlabel(xlabel); ax.set_ylabel('Average Error Rate')
        ax.set_title(title); ax.grid(True, alpha=0.3)

    _calibration_curve(lm_uncertainties, wers, axes[0, 0],
                       'LM Uncertainty', 'LM Calibration Curve', 'blue')
    _calibration_curve(am_uncertainties, pers, axes[1, 0],
                       'AM Uncertainty', 'AM Calibration Curve', 'red')

    corr_lm = np.corrcoef(lm_uncertainties, wers)[0, 1]
    axes[0, 1].scatter(lm_uncertainties, wers, alpha=0.3, s=20, color='purple')
    axes[0, 1].set_xlabel('LM Uncertainty')
    axes[0, 1].set_ylabel('WER')
    axes[0, 1].set_title(f'LM Uncertainty vs WER  (r={corr_lm:.3f})')
    axes[0, 1].grid(True, alpha=0.3)

    corr_am = np.corrcoef(am_uncertainties, pers)[0, 1]
    axes[1, 1].scatter(am_uncertainties, pers, alpha=0.3, s=20, color='green')
    axes[1, 1].set_xlabel('AM Uncertainty')
    axes[1, 1].set_ylabel('PER')
    axes[1, 1].set_title(f'AM Uncertainty vs PER  (r={corr_am:.3f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
