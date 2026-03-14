"""
run_train_lm.py
---------------
Entry point: build mixed training data and fine-tune a language model
(BART-base, BART-large, or Qwen2.5) for phoneme→text.

Usage:
    python scripts/run_train_lm.py \
        --lm_config configs/language_model_config.yaml \
        --acoustic_config configs/db_conformer_config.yaml \
        --model bart-large \
        --val_pkl /content/val_index_merged.pkl \
        --train_pkl /content/train_index_merged.pkl
"""

import argparse
import os
import pickle

import torch
import yaml
from datasets import Dataset


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def decode_phonemes(seq_ids, logit_to_phoneme):
    return ' '.join(
        logit_to_phoneme[p]
        for p in seq_ids
        if p != 0 and p < len(logit_to_phoneme)
    )


def decode_text(label):
    if isinstance(label, bytes):
        return label.decode('utf-8')
    return str(label)


def main():
    parser = argparse.ArgumentParser(description='Train language model')
    parser.add_argument('--lm_config',       required=True)
    parser.add_argument('--acoustic_config', required=True)
    parser.add_argument('--model', default='bart-large',
                        choices=['bart-base', 'bart-large', 'qwen'])
    parser.add_argument('--train_pkl', default=None)
    parser.add_argument('--val_pkl',   default=None)
    cli = parser.parse_args()

    lm_cfg  = load_yaml(cli.lm_config)
    ac_cfg  = load_yaml(cli.acoustic_config)

    # Merge acoustic config into Namespace for model building
    import argparse as _ap
    args = _ap.Namespace(**ac_cfg)
    args.data_name      = ac_cfg.get('data_name', 'NeuralPhoneme')
    args.class_num      = ac_cfg.get('class_num', 41)
    args.max_time_steps = ac_cfg.get('time_sample_num', 1500)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load acoustic model ────────────────────────────────────────────────
    from src.models.db_conformer import build_dbconformer_ctc
    acoustic_model = build_dbconformer_ctc(args).to(device)
    ckpt = lm_cfg.get('acoustic_ckpt')
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        acoustic_model.load_state_dict(state)
        print(f'Loaded acoustic checkpoint: {ckpt}')
    else:
        print('Warning: no acoustic checkpoint found — using random weights')
    acoustic_model.eval()

    # ── Load data ──────────────────────────────────────────────────────────
    if cli.train_pkl and os.path.exists(cli.train_pkl):
        with open(cli.train_pkl, 'rb') as f:
            train_index = pickle.load(f)
        with open(cli.val_pkl, 'rb') as f:
            val_index = pickle.load(f)
    else:
        from src.data.preprocessing import load_all_files
        train_index = load_all_files(
            args.drive_dir, args.local_dir, 'train', getattr(args, 'max_files', None)
        )
        val_index = load_all_files(
            args.drive_dir, args.local_dir, 'val', getattr(args, 'max_files', None)
        )

    from src.data.dataset import SpeechDatasetWithText
    from src.data.dataloader import BucketBatchSampler, collate_fn_with_text
    from torch.utils.data import DataLoader

    train_ds      = SpeechDatasetWithText(train_index, args.max_time_steps)
    val_ds        = SpeechDatasetWithText(val_index,   args.max_time_steps)
    train_sampler = BucketBatchSampler(train_ds, batch_size=args.batch_size)
    val_sampler   = BucketBatchSampler(val_ds,   batch_size=args.batch_size)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler,
                               collate_fn=collate_fn_with_text,
                               num_workers=2, pin_memory=True, prefetch_factor=2)
    val_loader    = DataLoader(val_ds, batch_sampler=val_sampler,
                               collate_fn=collate_fn_with_text,
                               num_workers=2, pin_memory=True, prefetch_factor=2)

    # ── Build mixed training dataset ───────────────────────────────────────
    from src.language_model.synthetic_data import create_merged_lm_dataset
    from src.data.preprocessing import LOGIT_TO_PHONEME
    from src.evaluation.decode import greedy_ctc_decode
    from src.data.dataloader import compute_token_lengths, make_attention_mask

    mixed_phonemes, mixed_texts = create_merged_lm_dataset(
        train_loader, acoustic_model, device,
        gt_ratio   = lm_cfg.get('gt_ratio',   0.20),
        syn_ratio  = lm_cfg.get('syn_ratio',   0.60),
        pred_ratio = lm_cfg.get('pred_ratio',  0.20),
        target_per = lm_cfg.get('target_per',  0.10),
    )

    train_input_texts  = [decode_phonemes(s, LOGIT_TO_PHONEME) for s in mixed_phonemes]
    train_target_texts = [decode_text(t) for t in mixed_texts]
    train_hf = Dataset.from_dict({
        'input_text':  train_input_texts,
        'target_text': train_target_texts,
    })

    # Val: real acoustic predictions only
    val_preds, val_labels = [], []
    acoustic_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            feat, _, _, steps, texts = batch
            feat = feat.to(device)
            il   = compute_token_lengths(steps, args.pool_kernel, args.pool_stride).to(device)
            T_p  = feat.size(-1)
            P    = compute_token_lengths(torch.tensor([T_p]), args.pool_kernel, args.pool_stride).item()
            mask = make_attention_mask(il, P)
            lp   = acoustic_model(feat, mask=mask).log_softmax(dim=-1)
            for pred in greedy_ctc_decode(lp):
                val_preds.append(decode_phonemes(pred, LOGIT_TO_PHONEME))
            val_labels.extend([decode_text(t) for t in texts])

    val_hf = Dataset.from_dict({
        'input_text':  val_preds,
        'target_text': val_labels,
    })
    print(f'Train: {len(train_hf)} samples  |  Val: {len(val_hf)} samples')

    # ── Train LM ───────────────────────────────────────────────────────────
    if cli.model == 'bart-base':
        from src.language_model.bart_trainer import bart_base_train
        bart_base_train(train_hf, val_hf,
                        save_dir=lm_cfg['bart_base']['save_dir'],
                        cfg=lm_cfg['bart_base'])
    elif cli.model == 'bart-large':
        from src.language_model.bart_trainer import bart_large_train
        bart_large_train(train_hf, val_hf,
                         save_dir=lm_cfg['bart_large']['save_dir'],
                         cfg=lm_cfg['bart_large'])
    elif cli.model == 'qwen':
        from src.language_model.qwen_trainer import qwen_lora_train
        qwen_lora_train(train_hf, val_hf,
                        save_dir=lm_cfg['qwen']['save_dir'],
                        cfg=lm_cfg['qwen'])


if __name__ == '__main__':
    main()
