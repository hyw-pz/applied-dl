"""
evaluate_pipeline.py
--------------------
End-to-end evaluation of the AM + LM pipeline with full metrics:
  WER, PER, BLEU, ROUGE-1/2/L, and optional per-sample CSV export.
"""

import random
import re

import evaluate as hf_evaluate
import jiwer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from src.data.dataloader import compute_token_lengths, make_attention_mask
from src.evaluation.decode import greedy_ctc_decode
from src.evaluation.metrics import calculate_per, clean_text_for_wer
from src.data.preprocessing import LOGIT_TO_PHONEME


# ─────────────────────────────────────────────────────────────────────────────
# BART / seq2seq pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bart_model(
    model_path: str,
    dataset,
    device,
    num_examples: int = 15,
    beam_size:    int = 4,
    batch_size:   int = 120,
    output_path:  str = None,
):
    """
    Load a BART (or ByT5) checkpoint and evaluate on *dataset*.

    Parameters
    ----------
    dataset     : HuggingFace Dataset with columns
                  'input_text', 'target_text', 'real_phonemes'
    output_path : if given, save per-sample results to this CSV path

    Returns
    -------
    dict with keys: corpus_wer, corpus_per, bleu, rouge1, rouge2, rougeL
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f'\n{"="*60}\nEvaluating: {model_path}\n{"="*60}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    bleu_metric  = hf_evaluate.load('bleu')
    rouge_metric = hf_evaluate.load('rouge')

    all_clean_preds, all_clean_refs = [], []
    all_in_phonemes, all_real_phonemes = [], []
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc='Generating'):
        batch = dataset[i:i + batch_size]
        inputs = tokenizer(
            batch['input_text'],
            return_tensors='pt', padding=True,
            truncation=True, max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=beam_size,
                max_length=128,
                early_stopping=True,
            )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for in_ph, real_ph, ref, pred in zip(
            batch['input_text'], batch['real_phonemes'],
            batch['target_text'], batch_preds,
        ):
            clean_ref  = clean_text_for_wer(ref)
            clean_pred = clean_text_for_wer(pred.strip())
            try:
                wer = jiwer.wer(clean_ref, clean_pred) if clean_ref else 1.0
                per = jiwer.wer(real_ph, in_ph) if real_ph.strip() else 1.0
            except ValueError:
                wer = per = 1.0

            all_clean_preds.append(clean_pred)
            all_clean_refs.append(clean_ref)
            if real_ph.strip():
                all_real_phonemes.append(real_ph)
                all_in_phonemes.append(in_ph)
            results.append({'Input': in_ph, 'Real_Phonemes': real_ph,
                            'Reference': ref, 'Prediction': pred,
                            'WER': wer, 'PER': per})

    # Corpus metrics
    bleu_refs   = [[r] for r in all_clean_refs]
    final_bleu  = bleu_metric.compute(predictions=all_clean_preds, references=bleu_refs)
    final_rouge = rouge_metric.compute(predictions=all_clean_preds, references=all_clean_refs)
    corpus_wer  = jiwer.wer(all_clean_refs, all_clean_preds)
    corpus_per  = jiwer.wer(all_real_phonemes, all_in_phonemes) if all_real_phonemes else float('nan')

    # Print examples
    print(f'\n--- {num_examples} Random Examples ---')
    for idx, res in enumerate(random.sample(results, min(num_examples, len(results)))):
        print(f"\nExample {idx+1}:")
        print(f"  Input Phonemes : {res['Input'][:60]}")
        print(f"  Reference      : {res['Reference']}")
        print(f"  Prediction     : {res['Prediction']}")
        print(f"  WER: {res['WER']:.4f} | PER: {res['PER']:.4f}")

    print(f'\n{"="*40}\nFINAL CORPUS METRICS\n{"="*40}')
    print(f'--> Corpus WER  : {corpus_wer:.4f}')
    print(f'--> Corpus PER  : {corpus_per:.4f}')
    print(f'--> BLEU        : {final_bleu["bleu"]:.4f}')
    print(f'--> ROUGE-1     : {final_rouge["rouge1"]:.4f}')
    print(f'--> ROUGE-2     : {final_rouge["rouge2"]:.4f}')
    print(f'--> ROUGE-L     : {final_rouge["rougeL"]:.4f}')

    if output_path:
        import pandas as pd
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f'\nResults saved to {output_path}')

    return dict(
        corpus_wer=corpus_wer,
        corpus_per=corpus_per,
        bleu=final_bleu['bleu'],
        rouge1=final_rouge['rouge1'],
        rouge2=final_rouge['rouge2'],
        rougeL=final_rouge['rougeL'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Qwen pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_qwen_model(
    model,
    tokenizer,
    dataset,
    device=None,
    num_examples: int = 15,
    beam_size:    int = 4,
    batch_size:   int = 8,
    output_path:  str = None,
):
    """
    Evaluate a loaded Qwen2.5 LoRA model on *dataset*.

    Returns
    -------
    dict with keys: avg_wer, avg_per, bleu, rougeL
    """
    bleu_metric  = hf_evaluate.load('bleu')
    rouge_metric = hf_evaluate.load('rouge')

    model.eval()
    tokenizer.padding_side = 'left'

    system_prompt = (
        "You are an expert speech decoding system. "
        "Translate the noisy ARPAbet phonemes into English text. "
        "CRITICAL CONSTRAINTS: You must ONLY output valid English dictionary words. "
        "Output ONLY the final text."
    )

    all_clean_preds, all_clean_refs = [], []
    w_errors, p_errors = [], []
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc='Generating (Qwen)'):
        batch = dataset[i:i + batch_size]
        prompts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nPhonemes: {ph}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for ph in batch['input_text']
        ]
        inputs = tokenizer(
            prompts, return_tensors='pt',
            padding=True, truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=beam_size,
                do_sample=False,
                early_stopping=True,
                length_penalty=0.8,
                repetition_penalty=1.1,
            )

        prompt_len = inputs.input_ids.shape[1]
        batch_preds = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )

        for in_ph, real_ph, ref, pred in zip(
            batch['input_text'], batch['real_phonemes'],
            batch['target_text'], batch_preds,
        ):
            clean_ref  = clean_text_for_wer(ref)
            clean_pred = clean_text_for_wer(pred.strip())
            try:
                wer = jiwer.wer(clean_ref, clean_pred) if clean_ref else 1.0
                per = jiwer.wer(real_ph, in_ph) if real_ph.strip() else 1.0
            except ValueError:
                wer = per = 1.0
            w_errors.append(wer)
            p_errors.append(per)
            all_clean_preds.append(clean_pred)
            all_clean_refs.append(clean_ref)
            results.append({'Input': in_ph, 'Real_Phonemes': real_ph,
                            'Reference': ref, 'Prediction': pred,
                            'WER': wer, 'PER': per})

    bleu_refs   = [[r] for r in all_clean_refs]
    final_bleu  = bleu_metric.compute(predictions=all_clean_preds, references=bleu_refs)
    final_rouge = rouge_metric.compute(predictions=all_clean_preds, references=all_clean_refs)

    print(f'\n--- {num_examples} Random Examples ---')
    for idx, res in enumerate(random.sample(results, min(num_examples, len(results)))):
        print(f"\nExample {idx+1}:")
        print(f"  Input     : {res['Input'][:60]}")
        print(f"  Reference : {res['Reference']}")
        print(f"  Prediction: {res['Prediction']}")
        print(f"  WER: {res['WER']:.4f} | PER: {res['PER']:.4f}")

    print(f'\n{"="*40}\nFINAL CORPUS METRICS\n{"="*40}')
    print(f'--> Avg WER : {np.mean(w_errors):.4f}')
    print(f'--> Avg PER : {np.mean(p_errors):.4f}')
    print(f'--> BLEU    : {final_bleu["bleu"]:.4f}')
    print(f'--> ROUGE-L : {final_rouge["rougeL"]:.4f}')

    if output_path:
        import pandas as pd
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f'\nResults saved to {output_path}')

    return dict(
        avg_wer=float(np.mean(w_errors)),
        avg_per=float(np.mean(p_errors)),
        bleu=final_bleu['bleu'],
        rougeL=final_rouge['rougeL'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty-aware pipeline evaluation (BART or Qwen)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline_uncertainty(
    acoustic_model,
    lm_model,
    lm_tokenizer,
    data_loader,
    device,
    args,
    lm_type: str = 'bart',
):
    """
    Run the full AM → LM pipeline on *data_loader* and collect
    WER, PER, AM uncertainty, and LM uncertainty per sample.

    Parameters
    ----------
    lm_type : 'bart' or 'qwen'

    Returns
    -------
    all_wers, all_pers, all_am_uncertainties, all_lm_uncertainties
    """
    from src.uncertainty.am_uncertainty import compute_am_sequence_score
    from src.uncertainty.lm_uncertainty import get_bart_uncertainty, get_qwen_uncertainty
    from src.evaluation.metrics import calculate_per, calculate_wer

    acoustic_model.eval()
    lm_model.eval()
    if lm_type == 'qwen':
        lm_tokenizer.padding_side = 'left'

    all_wers, all_pers = [], []
    all_am_uncs, all_lm_uncs = [], []
    samples = []

    k_w, s_w = args.pool_kernel, args.pool_stride

    print(f'Running {lm_type.upper()} uncertainty pipeline…')
    with torch.no_grad():
        for neural_batch, phoneme_batch, seq_lens, input_time_steps, text_labels in tqdm(data_loader):
            neural_batch = neural_batch.to(device)
            input_lengths = compute_token_lengths(
                input_time_steps, k_w, s_w
            ).to(device)

            T_p = neural_batch.size(-1)
            P   = compute_token_lengths(torch.tensor([T_p]), k_w, s_w).item()
            attn_mask = make_attention_mask(input_lengths, P)

            log_probs  = acoustic_model(neural_batch, mask=attn_mask).log_softmax(dim=-1)
            am_unc     = compute_am_sequence_score(log_probs, input_lengths)
            preds_ids  = greedy_ctc_decode(log_probs, blank_id=0)

            phoneme_strings, batch_pers = [], []
            for i, pid in enumerate(preds_ids):
                tl = seq_lens[i].item()
                ts = [x for x in phoneme_batch[i][:tl].tolist() if x != 0]
                batch_pers.append(calculate_per(ts, pid))
                all_pers.append(batch_pers[-1])
                phoneme_strings.append(
                    ' '.join(LOGIT_TO_PHONEME[p] for p in pid if p < len(LOGIT_TO_PHONEME))
                )

            if lm_type == 'bart':
                pred_texts, lm_unc = get_bart_uncertainty(
                    lm_model, lm_tokenizer, phoneme_strings, device
                )
                lm_unc = lm_unc.tolist()
            else:
                pred_texts, lm_unc = get_qwen_uncertainty(
                    lm_model, lm_tokenizer, phoneme_strings, device
                )

            for i, pred_text in enumerate(pred_texts):
                true_text = text_labels[i]
                if isinstance(true_text, bytes):
                    true_text = true_text.decode('utf-8')
                wer = calculate_wer(true_text, pred_text)
                all_wers.append(wer)
                all_am_uncs.append(am_unc[i].item())
                all_lm_uncs.append(lm_unc[i])

                if len(samples) < 15:
                    samples.append(dict(
                        am_unc=am_unc[i].item(), lm_unc=lm_unc[i],
                        per=batch_pers[i], pred=pred_text,
                        true=true_text, wer=wer,
                    ))

    # Print table
    print(f"\n{'AM_UNC':<10} | {'LM_UNC':<10} | {'PER':<8} | {'WER':<8} | {'Predicted':<35} | True")
    print('-' * 100)
    for s in samples:
        print(f"{s['am_unc']:.4f}    | {s['lm_unc']:.4f}    | "
              f"{s['per']:.2%}  | {s['wer']:.2%}  | "
              f"{s['pred'][:33]:<35} | {s['true']}")

    print(f'\n--- Uncertainty Correlations ---')
    print(f'AM ↔ PER : {np.corrcoef(all_am_uncs, all_pers)[0,1]:.3f}')
    print(f'AM ↔ WER : {np.corrcoef(all_am_uncs, all_wers)[0,1]:.3f}')
    print(f'LM ↔ WER : {np.corrcoef(all_lm_uncs, all_wers)[0,1]:.3f}')
    print(f'LM ↔ PER : {np.corrcoef(all_lm_uncs, all_pers)[0,1]:.3f}')
    print(f'\nMean WER  : {np.mean(all_wers):.2%}')
    print(f'Median WER: {np.median(all_wers):.2%}')
    print(f'Mean PER  : {np.mean(all_pers):.2%}')

    return all_wers, all_pers, all_am_uncs, all_lm_uncs
