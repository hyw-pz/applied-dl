"""
qwen_trainer.py
---------------
QLoRA fine-tuning of Qwen2.5-7B-Instruct for phoneme→text.

Uses 4-bit NF4 quantisation (bitsandbytes) + LoRA adapters (peft)
trained with SFTTrainer (trl).
"""

import gc
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatter
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert speech decoding system. "
    "Translate the noisy ARPAbet phonemes into English text. "
    "CRITICAL CONSTRAINTS: You must ONLY output valid English dictionary words. "
    "Output ONLY the final text."
)


def format_chatml(example: dict) -> dict:
    """Format one sample into Qwen2.5 ChatML structure."""
    full_prompt = (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPhonemes: {example['input_text']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['target_text']}<|im_end|>"
    )
    return {'text': full_prompt}


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────────────────

def qwen_lora_train(
    train_dataset_hf: Dataset,
    val_dataset_hf:   Dataset,
    save_dir:         str,
    cfg:              dict,
):
    """
    QLoRA fine-tune Qwen2.5-7B-Instruct.

    Parameters
    ----------
    train_dataset_hf / val_dataset_hf : HuggingFace Dataset with columns
        'input_text' and 'target_text'
    save_dir : where to save the final LoRA adapter
    cfg      : dict from language_model_config.yaml (qwen section)
    """
    gc.collect()
    torch.cuda.empty_cache()

    base_model_id = cfg.get('base_model', 'Qwen/Qwen2.5-7B-Instruct')

    # 4-bit quantisation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=cfg.get('double_quant', True),
        bnb_4bit_quant_type=cfg.get('quant_type', 'nf4'),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f'Loading base model: {base_model_id}')
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_config = LoraConfig(
        r=cfg.get('lora_r', 16),
        lora_alpha=cfg.get('lora_alpha', 32),
        target_modules=cfg.get('lora_target_modules', [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ]),
        lora_dropout=cfg.get('lora_dropout', 0.1),
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Format datasets
    train_sft = train_dataset_hf.map(format_chatml)
    val_sft   = val_dataset_hf.map(format_chatml)

    training_args = SFTConfig(
        output_dir='./qwen2.5-phoneme-to-text-lora',
        dataset_text_field='text',
        max_length=cfg.get('max_seq_length', 256),
        per_device_train_batch_size=cfg.get('train_batch_size', 8),
        gradient_accumulation_steps=cfg.get('grad_accum_steps', 2),
        per_device_eval_batch_size=cfg.get('eval_batch_size', 8),
        learning_rate=cfg.get('learning_rate', 1e-4),
        weight_decay=cfg.get('weight_decay', 0.08),
        num_train_epochs=cfg.get('num_epochs', 3),
        eval_strategy='epoch',
        save_strategy='epoch',
        bf16=True,
        fp16=False,
        max_grad_norm=cfg.get('max_grad_norm', 0.3),
        warmup_ratio=cfg.get('warmup_ratio', 0.1),
        lr_scheduler_type=cfg.get('lr_scheduler', 'cosine'),
        optim='paged_adamw_8bit',
        report_to='none',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        save_total_limit=2,
        logging_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_sft,
        eval_dataset=val_sft,
        processing_class=tokenizer,
        args=training_args,
    )

    print('Starting Qwen2.5 QLoRA fine-tuning…')
    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'LoRA adapter saved to {save_dir}')
