# Neural Speech Decoding: Brain-to-Text via Phoneme Prediction

A two-stage neural speech decoding pipeline that converts intracortical neural signals into text. The system combines an acoustic model (phoneme prediction) with a language model (phoneme-to-text), and includes uncertainty-aware inference to flag low-confidence predictions. The final results for the validation set could be found in Qwen_Synthetic_Data_Results.csv.

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
results
├── BART_large_synthetic_results.csv
├── Qwen_synthetic_results.csv
```

---

## Models

### Acoustic Models

| Model | Description | Key Features |
|---|---|---|
| **EEGConformer** | CNN + Transformer, single-branch | 2D temporal conv, sinusoidal pos-encoding |
| **DBConformer** | Dual-branch (temporal + spatial) | FlashAttention, AMP, OneCycleLR |

Both models output CTC logits over a 41-class phoneme vocabulary (ARPAbet + BLANK + word-boundary `|`).

Below are the results for the First stage. 
| Model | Embedding Size| Parameter Count | Average PER | Median PER |
|---|---|---|---|---|
|EEG Conformer| 128 | 1.3 Million | 0.267 | 0.25 |
|DBConformer| 128 | 2.6 Million |0.193  |0.175  |
|DBConformer| 256 | 9.8 Million | 0.179 |0.154  |
|DBConformer| 512 | 38.5 Million | 0.124 |0.090  |

### Language Models

All subsequent results were generated using a BEAM size of 4 during decoding.

| Model | Training Data| Average WER | Median WER | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (QLoRA) | 50% GT + 50% predicted | 0.243 | 0.17 | 0.595 | 0.783 | 0.658 | 0.7829 |
| Qwen2.5-7B (QLoRA) | 20% GT + 60% synth + 20% predicted | 0.224 | 0.12 | 0.645 | 0.804 | 0.693 | 0.804 |
| BART-base | 50% GT + 50% predicted | 0.248 | 0.20 | 0.591 | 0.760 | 0.629 | 0.7594 |
| BART-base + synthetic | 25% GT + 25% synth + 50% predicted | 0.273 | 0.20 | 0.582 | 0.744 | 0.622 | 0.744 |
| BART-large | 50% GT + 50% predicted | 0.249 | 0.20 | 0.588 | 0.758 | 0.626 | 0.757 |
| BART-large + synthetic | 20% GT + 60% synth + 20% predicted | 0.275 | 0.20 | 0.571 | 0.742 | 0.613 | 0.742 |
| Byt5-small | 50% GT + 50% predicted  | 0.407 | 0.40 | 0.37 | 0.655 | 0.482 | 0.654 |

---
## Training Time & Hardware

| Model | Variant | Hardware | Time | Epochs |
|---|---|---|---|---|
| EEGConformer | emb=128 |L4 |10h |120 |
| DBConformer | emb=128 |T4 |0.5h |100 |
| DBConformer | emb=256 |L4 |0.5h |100 |
| DBConformer | emb=512 |L4 |~1h |100 |
| BART-base | 50% GT + 50% predicted |L4 |0.5h |50 |
| BART-large | 50% GT + 50% predicted |L4 |1h |50 |
| Qwen2.5-7B (QLoRA) | 20% GT + 60% synth + 20% predicted |L4 |7h |3|

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
## Pretrained Checkpoints & Data

### Model Weights

All pretrained checkpoints are available on Google Drive. Download and place them in the corresponding paths before running any notebook.

| Model | Download |
|---|---|
| EEGConformer (Phase 2 best) | [Google Drive](YOUR_LINK_HERE) |
| DBConformer v2 (Phase 3 best, emb=512) | [Google Drive](https://drive.google.com/file/d/1HDd2hpQWyXWH61_SrKY3DA5t30iMq1FZ/view?usp=drive_link) |
| BART-large + synthetic data | [Google Drive](https://drive.google.com/drive/folders/1lUDsz2-Bvsg623hJzEg7X9eizkTYKgLi?usp=drive_link) |
| Qwen2.5-7B LoRA adapter | [Google Drive](https://drive.google.com/drive/folders/1iQFunPJHNK-2_kzuwg4_z3-u7PaslCn9?usp=drive_link) |

### Data

The preprocessed data index files are not included in this repository due to file size (~2 GB). Download them and place both files in the repo root before running any notebook.

| File | Download |
|---|---|
| `train_index_merged.pkl` | [Google Drive](https://drive.google.com/file/d/10mMEx-yn7JWrduQ-7DJEcGzqCZPwCEy3/view?usp=drive_link) |
| `val_index_merged.pkl` | [Google Drive](https://drive.google.com/file/d/1jumosK6v5lngly8MzqbpkDEjyaSdxMRE/view?usp=drive_link) |

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
