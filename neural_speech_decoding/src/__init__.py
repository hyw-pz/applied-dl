"""
src/__init__.py
---------------
Public API for the neural_speech_decoding package.
Import from here to keep notebook imports clean.

Example (in a notebook):
    from src import build_dataloader, build_model, load_acoustic_model
    from src import evaluate_pipeline_uncertainty_bart, compute_ece
"""

from src.vocabulary import (
    LOGIT_TO_PHONEME,
    NUM_PHONEMES,
    PHONEME_TO_LOGIT,
    VOCAB_PHONEMES,
    PHONEME_CONFUSION_MAP,
)

from src.dataset import (
    SpeechDataset,
    SpeechDatasetWithText,
    BucketBatchSampler,
    collate_fn,
    collate_fn_with_text,
    compute_token_lengths,
    build_dataloader,
)

from src.model import (
    DBConformer,
    DBConformerCTC,
    build_model,
    load_acoustic_model,
)

from src.metrics import (
    ctc_greedy_decode,
    decode_ids_to_string,
    decode_text_labels,
    calculate_per,
    phoneme_error_rate,
    calculate_wer,
)

from src.uncertainty import (
    compute_am_sequence_score,
    evaluate_am_uncertainty,
    evaluate_pipeline_uncertainty_bart,
    evaluate_pipeline_uncertainty_qwen,
    compute_ece,
    evaluate_ece,
    plot_coverage_wer_tradeoff,
)

from src.lm_models import (
    load_bart,
    load_qwen,
)

__all__ = [
    # vocabulary
    "LOGIT_TO_PHONEME", "NUM_PHONEMES", "PHONEME_TO_LOGIT",
    "VOCAB_PHONEMES", "PHONEME_CONFUSION_MAP",
    # dataset
    "SpeechDataset", "SpeechDatasetWithText", "BucketBatchSampler",
    "collate_fn", "collate_fn_with_text", "compute_token_lengths",
    "build_dataloader",
    # model
    "DBConformer", "DBConformerCTC", "build_model", "load_acoustic_model",
    # metrics
    "ctc_greedy_decode", "decode_ids_to_string", "decode_text_labels",
    "calculate_per", "phoneme_error_rate", "calculate_wer",
    # uncertainty
    "compute_am_sequence_score",
    "evaluate_am_uncertainty",
    "evaluate_pipeline_uncertainty_bart",
    "evaluate_pipeline_uncertainty_qwen",
    "compute_ece", "evaluate_ece",
    "plot_coverage_wer_tradeoff",
    # lm models
    "load_bart", "load_qwen",
]
