"""
run_evaluate.py
---------------
Entry point: evaluate the full AM + LM pipeline.

Usage:
    # Evaluate all BART variants
    python scripts/run_evaluate.py \
        --acoustic_config configs/db_conformer_config.yaml \
        --acoustic_ckpt /path/to/best.ckpt \
        --lm_paths \
            /path/to/BART_base \
            /path/to/BART_large \
        --val_pkl /content/val_index_merged.pkl \
        --model_type bart

    # Evaluate Qwen
    python scripts/run_evaluate.py \
        --acoustic_config configs/db_conformer_config.yaml \
        --acoustic_ckpt /path/to/best.ckpt \
        --lm_paths /path/to/Qwen_LoRA \
        --val_pkl /content/val_index_merged.pkl \
        --model_type qwen \
        --qwen_base Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
import pickle

import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_val_hf(val_index, acoustic_model, args, device):
    """Run acoustic inference on val set and return a HuggingFace Dataset."""
    from torch.utils.data import DataLoader
    from datasets import Dataset

    from src.data.dataset import SpeechDatasetWithText
    from src.data.dataloader import (BucketBatchSampler, collate_fn_with_text,
                                     compute_token_lengths, make_attention_mask)
    from src.evaluation.decode import greedy_ctc_decode
    from src.data.preprocessing import LOGIT_TO_PHONEME

    val_ds      = SpeechDatasetWithText(val_index, args.max_time_steps)
    val_sampler = BucketBatchSampler(val_ds, batch_size=32)
    val_loader  = DataLoader(val_ds, batch_sampler=val_sampler,
                             collate_fn=collate_fn_with_text,
                             num_workers=2, pin_memory=True)

    pred_phonemes, target_texts, real_phonemes = [], [], []
    acoustic_model.eval()

    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(val_loader, desc='AM inference'):
            feat, ids, lens, steps, texts = batch
            feat = feat.to(device)
            il   = compute_token_lengths(steps, args.pool_kernel, args.pool_stride).to(device)
            T_p  = feat.size(-1)
            P    = compute_token_lengths(torch.tensor([T_p]),
                                         args.pool_kernel, args.pool_stride).item()
            mask = make_attention_mask(il, P)
            lp   = acoustic_model(feat, mask=mask).log_softmax(dim=-1)
            preds = greedy_ctc_decode(lp)

            for b, pid in enumerate(preds):
                pred_phonemes.append(
                    ' '.join(LOGIT_TO_PHONEME[p] for p in pid if p < len(LOGIT_TO_PHONEME))
                )
                ref_seq = [x for x in ids[b][:lens[b]].tolist() if x != 0]
                real_phonemes.append(
                    ' '.join(LOGIT_TO_PHONEME[p] for p in ref_seq if p < len(LOGIT_TO_PHONEME))
                )
            target_texts.extend(
                t.decode('utf-8') if isinstance(t, bytes) else str(t)
                for t in texts
            )

    return Dataset.from_dict({
        'input_text':   pred_phonemes,
        'target_text':  target_texts,
        'real_phonemes': real_phonemes,
    })


def main():
    parser = argparse.ArgumentParser(description='Evaluate AM + LM pipeline')
    parser.add_argument('--acoustic_config', required=True)
    parser.add_argument('--acoustic_ckpt',   required=True)
    parser.add_argument('--lm_paths', nargs='+', required=True,
                        help='One or more LM checkpoint directories')
    parser.add_argument('--val_pkl', default=None)
    parser.add_argument('--model_type', default='bart',
                        choices=['bart', 'qwen'])
    parser.add_argument('--qwen_base', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--beam_size',  type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=120)
    cli = parser.parse_args()

    ac_cfg = load_yaml(cli.acoustic_config)
    import argparse as _ap
    args = _ap.Namespace(**ac_cfg)
    args.data_name      = ac_cfg.get('data_name', 'NeuralPhoneme')
    args.class_num      = ac_cfg.get('class_num', 41)
    args.max_time_steps = ac_cfg.get('time_sample_num', 1500)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load acoustic model ────────────────────────────────────────────────
    from src.models.db_conformer import build_dbconformer_ctc
    acoustic_model = build_dbconformer_ctc(args).to(device)
    acoustic_model.load_state_dict(
        torch.load(cli.acoustic_ckpt, map_location=device)
    )
    acoustic_model.eval()
    print(f'Loaded acoustic model from {cli.acoustic_ckpt}')

    # ── Load val data ──────────────────────────────────────────────────────
    if cli.val_pkl and os.path.exists(cli.val_pkl):
        with open(cli.val_pkl, 'rb') as f:
            val_index = pickle.load(f)
    else:
        from src.data.preprocessing import load_all_files
        val_index = load_all_files(args.drive_dir, args.local_dir, 'val')

    val_hf = build_val_hf(val_index, acoustic_model, args, device)
    print(f'Val dataset: {len(val_hf)} samples')

    # ── Evaluate each LM ──────────────────────────────────────────────────
    if cli.model_type == 'bart':
        from src.evaluation.evaluate_pipeline import evaluate_bart_model
        for lm_path in cli.lm_paths:
            evaluate_bart_model(
                lm_path, val_hf, device,
                beam_size=cli.beam_size,
                batch_size=cli.batch_size,
            )
    else:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from src.evaluation.evaluate_pipeline import evaluate_qwen_model

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer  = AutoTokenizer.from_pretrained(cli.qwen_base)
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        base_model = AutoModelForCausalLM.from_pretrained(
            cli.qwen_base, quantization_config=bnb_config, device_map='auto'
        )
        for lm_path in cli.lm_paths:
            model = PeftModel.from_pretrained(base_model, lm_path)
            model.eval()
            evaluate_qwen_model(
                model, tokenizer, val_hf,
                beam_size=cli.beam_size,
                batch_size=cli.batch_size,
            )


if __name__ == '__main__':
    main()
