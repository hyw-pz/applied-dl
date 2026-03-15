# Neural Speech Decoding: Brain-to-Text via Phoneme Prediction

A two-stage neural speech decoding pipeline that converts intracortical neural signals into text. The system combines an acoustic model (phoneme prediction) with a language model (phoneme-to-text), and includes uncertainty-aware inference to flag low-confidence predictions.

---

## Architecture Overview

```
Neural Signal (512 channels, T timesteps)
        │
        ▼
┌───────────────────────────┐
│     Acoustic Model        │  ← EEGConformer or DBConformer
│  (Neural → Phonemes, CTC) │
└───────────────────────────┘
        │  Phoneme sequence + AM uncertainty
        ▼
┌───────────────────────────┐
│     Language Model        │  ← BART-base / BART-large / Qwen2.5-7B
│  (Phonemes → English Text)│
└───────────────────────────┘
        │  Text + LM uncertainty
        ▼
   Final Prediction
   + Confidence Report
```

---

## Repository Structure

```
neural-speech-decoding/
├── src/
│   ├── data/
│   │   ├── dataset.py            # SpeechDataset, SpeechDatasetWithText
│   │   ├── dataloader.py         # BucketBatchSampler, collate_fn variants
│   │   └── preprocessing.py      # HDF5 loading, z-score normalisation
│   ├── models/
│   │   ├── eeg_conformer.py      # EEGConformer acoustic model (v1)
│   │   ├── db_conformer.py       # DBConformer acoustic model (v2)
│   │   └── heads.py              # CTCHead, PhonemeHead
│   ├── training/
│   │   ├── train_acoustic.py     # CTC training loop (AMP, OneCycleLR)
│   │   └── train_language.py     # Seq2Seq / SFT training for LM
│   ├── evaluation/
│   │   ├── metrics.py            # PER, WER, BLEU, ROUGE
│   │   ├── decode.py             # greedy_ctc_decode, beam_ctc_decode
│   │   └── evaluate_pipeline.py  # End-to-end AM+LM evaluation
│   ├── uncertainty/
│   │   ├── am_uncertainty.py     # AM sequence score + MC-Dropout
│   │   ├── lm_uncertainty.py     # LM beam score uncertainty
│   │   ├── calibration.py        # ECE, calibration curves
│   │   └── confidence_filter.py  # Threshold-based confidence report
│   └── language_model/
│       ├── synthetic_data.py     # Phoneme confusion map + error injection
│       ├── bart_trainer.py       # BART-base / BART-large training
│       └── qwen_trainer.py       # Qwen2.5 QLoRA fine-tuning
├── configs/
│   ├── eeg_conformer_config.yaml
│   ├── db_conformer_config.yaml
│   └── language_model_config.yaml
├── scripts/
│   ├── run_train_acoustic.py     # Entry point: train acoustic model
│   ├── run_train_lm.py           # Entry point: train language model
│   ├── run_evaluate.py           # Entry point: full pipeline evaluation
│   └── run_uncertainty.py        # Entry point: uncertainty analysis
├── notebooks/
│   ├── 01_EEGConformer_acoustic.ipynb
│   ├── 02_DBConformer_acoustic.ipynb
│   └── 03_LM_and_Uncertainty.ipynb
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_metrics.py
├── docs/
│   └── model_architecture.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## Models

### Acoustic Models

| Model | Description | Key Features |
|---|---|---|
| **EEGConformer** | CNN + Transformer, single-branch | 2D temporal conv, sinusoidal pos-encoding |
| **DBConformer v2** | Dual-branch (temporal + spatial) | FlashAttention, AMP, OneCycleLR |

Both models output CTC logits over a 41-class phoneme vocabulary (ARPAbet + BLANK + word-boundary `|`).

### Language Models

All subsequent results were generated using a beam size of 4 during decoding.

| Model | Training Data| Average WER | Median WER |
|---|---|---|---|
| Qwen2.5-7B (QLoRA) | 20% GT + 60% synth + 20% predicted | 0.224 | 0.12 |
| GPT-4o-mini (few-shot) | Few shots prompting | 0.38 | 0.29 |
| BART-base | 50% GT + 50% predicted | 0.248 | 0.20 |
| BART-base + synthetic | 25% GT + 25% synth + 50% predicted | 0.273 | 0.20 |
| BART-large | 50% GT + 50% predicted | 0.249 | 0.20 |
| BART-large + synthetic | 20% GT + 60% synth + 20% predicted | 0.275 | 0.20 |
| Byt5-small | 50% GT + 50% predicted  | 0.407 | 0.40 |

---

## Uncertainty Estimation

Two uncertainty signals are computed at inference time:

- **AM Uncertainty**: negative mean of max log-prob per CTC timestep (lower = more confident)
- **LM Uncertainty**: negative beam-search sequence score (lower = more confident)

Confidence levels are assigned based on LM uncertainty thresholds:

| Level | LM Uncertainty | Typical WER |
|---|---|---|
| HIGH | < 0.05 | < 10% |
| MEDIUM | 0.05 – 0.15 | 10–40% |
| LOW | 0.15 – 0.25 | 40–60% |
| VERY LOW | > 0.25 | > 60% |

---

## Phoneme Vocabulary (ARPAbet)

41 classes: `BLANK` + 39 phonemes + `|` (word boundary)

```python
LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B',  'CH', 'D',  'DH',
    'EH', 'ER', 'EY', 'F',  'G',  'HH', 'IH', 'IY', 'JH', 'K',
    'L',  'M',  'N',  'NG', 'OW', 'OY', 'P',  'R',  'S',  'SH',
    'T',  'TH', 'UH', 'UW', 'V',  'W',  'Y',  'Z',  'ZH', '|',
]
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the acoustic model (DBConformer)
```bash
python scripts/run_train_acoustic.py \
    --config configs/db_conformer_config.yaml \
    --drive_dir /path/to/hdf5_data \
    --local_dir /path/to/local_cache
```

### 3. Train the language model (BART-large)
```bash
python scripts/run_train_lm.py \
    --model bart-large \
    --acoustic_ckpt /path/to/best.ckpt \
    --synthetic_ratio 0.6
```

### 4. Run end-to-end evaluation
```bash
python scripts/run_evaluate.py \
    --acoustic_ckpt /path/to/acoustic.ckpt \
    --lm_path /path/to/lm
```

### 5. Run uncertainty analysis
```bash
python scripts/run_uncertainty.py \
    --acoustic_ckpt /path/to/acoustic.ckpt \
    --lm_path /path/to/lm
```

---

## Data Format

The pipeline expects intracortical neural data in `.hdf5` format, with the following structure per trial:

```
trial_key/
  input_features    (T, 512)   float32
  seq_class_ids     (max_seq,) int64
  transcription     bytes
  attrs:
    n_time_steps    int
    seq_len         int
    sentence_label  str
    session         str
    block_num       int
    trial_num       int
```

Data is loaded via `src/data/preprocessing.py`.

---

## Requirements

See `requirements.txt`. Key dependencies:

- PyTorch ≥ 2.0
- HuggingFace Transformers ≥ 4.40
- PEFT (for Qwen QLoRA)
- TRL (SFTTrainer)
- einops, timm, h5py, evaluate, jiwer

---

## Citation

If you use this code, please cite the original data source and acknowledge the two-stage neural speech decoding approach.

---

## Notes on Checkpoints

Trained model weights are **not included** in this repository due to size. To reproduce results:
1. Train the acoustic model using the provided scripts
2. Train the language model using the acoustic model's predictions
3. Checkpoint paths are configured via YAML files in `configs/`
