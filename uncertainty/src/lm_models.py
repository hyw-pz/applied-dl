"""
lm_models.py
------------
Loading utilities for the language model components:
  - BART-large (Seq2Seq, fine-tuned on phoneme→text)
  - Qwen2.5-7B-Instruct (Causal LM, QLoRA fine-tuned)
"""

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel


def load_bart(model_path: str, device: torch.device):
    """
    Loads a fine-tuned BART model and tokenizer from a local path or HF Hub.

    Args:
        model_path: path to saved model directory (contains config.json etc.)
        device:     torch.device to move the model to

    Returns:
        (lm_model, lm_tokenizer)  — model is in eval mode
    """
    lm_tokenizer = AutoTokenizer.from_pretrained(model_path)
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    lm_model.eval()
    print(f"BART loaded from: {model_path}")
    return lm_model, lm_tokenizer


def load_qwen(base_model_id: str,
              lora_path: str,
              use_quantization: bool = True):
    """
    Loads Qwen2.5-7B-Instruct base model and merges a QLoRA adapter.

    Args:
        base_model_id:    HuggingFace model ID, e.g. "Qwen/Qwen2.5-7B-Instruct"
        lora_path:        local path to the saved LoRA adapter directory
                          (must contain adapter_config.json + adapter_model.safetensors)
        use_quantization: if True, loads in 4-bit NF4 (requires ~5 GB VRAM on top
                          of other models); if False, loads in bfloat16 (~14 GB VRAM,
                          faster inference, recommended when VRAM > 22 GB)

    Returns:
        (qwen_model, tokenizer)  — model is in eval mode
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    qwen_model = PeftModel.from_pretrained(base_model, lora_path)
    qwen_model.eval()
    print(f"Qwen loaded — base: {base_model_id}, adapter: {lora_path}")
    return qwen_model, tokenizer
