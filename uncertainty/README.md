# Neural Speech Decoding — Uncertainty Estimation

Uncertainty estimation for a two-stage neural speech decoding pipeline:

```
ECoG neural signal
    │
    ▼
DBConformerCTC  (acoustic model, CTC)   →  phoneme sequence + AM uncertainty
    │
    ▼
BART-large  /  Qwen2.5-7B  (language model)  →  text + LM uncertainty
```

---

## Repository structure

```
neural_speech_decoding/
├── src/
│   ├── __init__.py       # public API — import everything from here
│   ├── vocabulary.py     # ARPAbet phoneme vocab and confusion map
│   ├── dataset.py        # Dataset classes, BucketBatchSampler, collate_fn
│   ├── model.py          # DBConformer architecture + load_acoustic_model()
│   ├── metrics.py        # CTC decoding, PER, WER helpers
│   ├── uncertainty.py    # AM + LM uncertainty evaluation, ECE, plots
│   └── lm_models.py      # load_bart() and load_qwen() factory functions
│
├── notebooks/
│   └── uncertainty_demo.ipynb   # end-to-end demo (configure paths, run cells)
│
└── README.md
```

---

## Quickstart

```python
# In a notebook or script, add the repo root to sys.path first:
import sys
sys.path.insert(0, '/path/to/neural_speech_decoding')

from src import (
    build_dataloader, load_acoustic_model,
    load_bart, load_qwen,
    evaluate_pipeline_uncertainty_bart,
    evaluate_pipeline_uncertainty_qwen,
    evaluate_ece, plot_coverage_wer_tradeoff,
)
```

Then open `notebooks/uncertainty_demo.ipynb`, set the paths in **Section 1**, and run all cells.

---

## Uncertainty methods

### Acoustic Model (AM) Uncertainty

Defined as the negated mean maximum log-probability across valid CTC output frames:

$$U_{AM} = -\frac{1}{T}\sum_{t=1}^{T} \max_c \log p(c \mid t)$$

Computed in `src/uncertainty.py → compute_am_sequence_score()`.  
No additional inference overhead — derived directly from the CTC output.

### BART LM Uncertainty

Negated length-normalised beam search score returned by `generate()`:

$$U_{LM}^{BART} = -\text{score}_{beam}$$

Computed via `outputs.sequences_scores` with `return_dict_in_generate=True`.

### Qwen LM Uncertainty

Negated mean per-token log-probability computed via `compute_transition_scores()`:

$$U_{LM}^{Qwen} = -\frac{1}{|y|}\sum_{i=1}^{|y|} \log p(y_i \mid y_{<i}, x)$$

This is the mean negative log-likelihood of the generated token sequence, providing token-level confidence information not available from beam scores alone.

---

## Dependencies

```
torch
transformers
peft
trl
einops
timm
evaluate
jiwer
rouge_score
bitsandbytes>=0.46.1
h5py
numpy
matplotlib
tqdm
```

Install via:
```bash
pip install torch transformers peft trl einops timm evaluate jiwer rouge_score "bitsandbytes>=0.46.1" h5py matplotlib tqdm
```
