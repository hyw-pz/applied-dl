"""
lm_uncertainty.py
-----------------
Language model (LM) uncertainty from beam-search sequence scores.

Works for both BART (seq2seq) and Qwen2.5 (causal LM with LoRA).
"""

import torch
from torch import Tensor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def get_bart_uncertainty(
    lm_model,
    lm_tokenizer,
    phoneme_strings: list,
    device,
    num_beams: int = 4,
    max_length: int = 256,
    length_penalty: float = 0.6,
) -> tuple:
    """
    Run BART beam-search and return predicted texts + uncertainty scores.

    Uncertainty = -sequence_score (beam normalised log-prob).
    Lower uncertainty → model is more confident.

    Parameters
    ----------
    phoneme_strings : list of ARPAbet phoneme strings (one per batch item)

    Returns
    -------
    predicted_texts : list[str]
    lm_uncertainty  : Tensor (B,)
    """
    encodings = lm_tokenizer(
        phoneme_strings,
        return_tensors='pt',
        padding=True,
        max_length=256,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = lm_model.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    lm_uncertainty  = -outputs.sequences_scores.cpu()
    predicted_texts = lm_tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )
    return predicted_texts, lm_uncertainty


def get_qwen_uncertainty(
    qwen_model,
    qwen_tokenizer,
    phoneme_strings: list,
    device,
    num_beams: int = 4,
    max_new_tokens: int = 64,
    system_prompt: str = (
        "You are an expert speech decoding system. "
        "Translate the noisy ARPAbet phonemes into English text. "
        "CRITICAL CONSTRAINTS: You must ONLY output valid English dictionary words. "
        "Output ONLY the final text."
    ),
) -> tuple:
    """
    Run Qwen2.5 beam-search and return predicted texts + uncertainty scores.

    Uses ``compute_transition_scores`` to get per-token log-probs,
    then averages over generated tokens (excluding padding).

    Parameters
    ----------
    phoneme_strings : list of ARPAbet phoneme strings

    Returns
    -------
    predicted_texts : list[str]
    lm_uncertainty  : list[float]
    """
    prompts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nPhonemes: {ph}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for ph in phoneme_strings
    ]

    inputs = qwen_tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(qwen_model.device)

    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            length_penalty=0.8,
            repetition_penalty=1.1,
        )

    transition_scores = qwen_model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        outputs.beam_indices if num_beams > 1 else None,
        normalize_logits=True,
    )

    lm_uncertainty = []
    for k in range(len(transition_scores)):
        valid = transition_scores[k][transition_scores[k] != 0]
        lm_uncertainty.append(-valid.mean().item() if len(valid) > 0 else 10.0)

    prompt_len      = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, prompt_len:]
    predicted_texts  = qwen_tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )
    return predicted_texts, lm_uncertainty
