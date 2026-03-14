"""
bart_trainer.py
---------------
BART-base and BART-large training for phoneme→text sequence-to-sequence.

Three training variants:
  1. bart_base_train          : BART-base, no synthetic data
  2. bart_base_synthetic_train: BART-base, with synthetic augmentation
  3. bart_large_train         : BART-large, with label smoothing + warmup
"""

import gc

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_preprocess_fn(tokenizer, max_input_length=128, max_target_length=128):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['input_text'],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples['target_text'],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    return preprocess_function


def _compute_metrics_fn(tokenizer):
    wer_metric = evaluate.load('wer')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {'wer': wer_metric.compute(predictions=decoded_preds, references=decoded_labels)}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# BART-base
# ─────────────────────────────────────────────────────────────────────────────

def bart_base_train(
    train_dataset_hf: Dataset,
    val_dataset_hf:   Dataset,
    save_dir:         str,
    cfg:              dict,
):
    """
    Fine-tune BART-base for phoneme→text.

    Parameters
    ----------
    train_dataset_hf / val_dataset_hf : HuggingFace Dataset with columns
        'input_text' (ARPAbet string) and 'target_text' (English sentence)
    save_dir : output directory on Drive
    cfg      : dict from language_model_config.yaml (bart_base section)
    """
    gc.collect()
    torch.cuda.empty_cache()

    model_checkpoint = cfg.get('model_checkpoint', 'facebook/bart-base')
    tokenizer  = AutoTokenizer.from_pretrained(model_checkpoint)
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    preprocess_fn = _build_preprocess_fn(
        tokenizer,
        max_input_length=cfg.get('max_input_length', 128),
        max_target_length=cfg.get('max_target_length', 128),
    )
    tokenized_train = train_dataset_hf.map(preprocess_fn, batched=True)
    tokenized_val   = val_dataset_hf.map(preprocess_fn, batched=True)
    data_collator   = DataCollatorForSeq2Seq(tokenizer, model=bart_model)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./bart-phoneme-to-text',
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        learning_rate=cfg.get('learning_rate', 1e-5),
        per_device_train_batch_size=cfg.get('train_batch_size', 80),
        per_device_eval_batch_size=cfg.get('eval_batch_size', 80),
        weight_decay=cfg.get('weight_decay', 0.01),
        save_total_limit=3,
        num_train_epochs=cfg.get('num_epochs', 50),
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=bart_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.get('early_stopping', 5)
        )],
    )

    print('Starting BART-base training…')
    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'Saved to {save_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# BART-large
# ─────────────────────────────────────────────────────────────────────────────

def bart_large_train(
    train_dataset_hf: Dataset,
    val_dataset_hf:   Dataset,
    save_dir:         str,
    cfg:              dict,
):
    """
    Fine-tune BART-large with label smoothing, warmup, and gradient clipping.

    Parameters
    ----------
    Same as :func:`bart_base_train` but uses the bart_large cfg section.
    """
    gc.collect()
    torch.cuda.empty_cache()

    model_checkpoint = cfg.get('model_checkpoint', 'facebook/bart-large')
    tokenizer  = AutoTokenizer.from_pretrained(model_checkpoint)
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # BART-large specific fixes
    bart_model.tie_weights()
    bart_model.config.forced_bos_token_id   = None
    bart_model.config.decoder_start_token_id = tokenizer.eos_token_id

    preprocess_fn = _build_preprocess_fn(
        tokenizer,
        max_input_length=cfg.get('max_input_length', 128),
        max_target_length=cfg.get('max_target_length', 128),
    )
    tokenized_train = train_dataset_hf.map(preprocess_fn, batched=True)
    tokenized_val   = val_dataset_hf.map(preprocess_fn, batched=True)
    data_collator   = DataCollatorForSeq2Seq(tokenizer, model=bart_model)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./bart-phoneme-to-text',
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        learning_rate=cfg.get('learning_rate', 2e-7),
        warmup_ratio=cfg.get('warmup_ratio', 0.1),
        max_grad_norm=cfg.get('max_grad_norm', 1.0),
        per_device_train_batch_size=cfg.get('train_batch_size', 60),
        per_device_eval_batch_size=cfg.get('eval_batch_size', 60),
        label_smoothing_factor=cfg.get('label_smoothing_factor', 0.1),
        weight_decay=cfg.get('weight_decay', 0.05),
        save_total_limit=3,
        num_train_epochs=cfg.get('num_epochs', 50),
        predict_with_generate=True,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=bart_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.get('early_stopping', 5)
        )],
    )

    print('Starting BART-large training…')
    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'Saved to {save_dir}')
