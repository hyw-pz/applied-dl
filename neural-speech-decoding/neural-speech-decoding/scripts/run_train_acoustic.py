"""
run_train_acoustic.py
---------------------
Entry point: train DBConformer or EEGConformer acoustic model.

Usage (Google Colab):
    python scripts/run_train_acoustic.py \
        --config configs/db_conformer_config.yaml \
        --model dbconformer \
        --phase 1

Usage for Phase 3 fine-tuning:
    python scripts/run_train_acoustic.py \
        --config configs/db_conformer_config.yaml \
        --model dbconformer \
        --phase 3 \
        --ckpt /path/to/phase1_best.ckpt
"""

import argparse
import os
import pickle

import torch
import yaml

from torch.utils.data import DataLoader


def load_config(path: str) -> argparse.Namespace:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # flatten nested keys into a Namespace
    ns = argparse.Namespace(**cfg)
    # ensure required fields exist with defaults
    ns.data_name      = getattr(ns, 'data_name',      'NeuralPhoneme')
    ns.class_num      = getattr(ns, 'class_num',      41)
    ns.max_time_steps = getattr(ns, 'time_sample_num', 1500)
    return ns


def main():
    parser = argparse.ArgumentParser(description='Train acoustic model')
    parser.add_argument('--config', required=True,
                        help='Path to YAML config (e.g. configs/db_conformer_config.yaml)')
    parser.add_argument('--model', default='dbconformer',
                        choices=['dbconformer', 'eegconformer'],
                        help='Which acoustic model to train')
    parser.add_argument('--phase', type=int, default=1,
                        choices=[1, 3],
                        help='Training phase: 1=OneCycleLR, 3=ReduceLROnPlateau fine-tune')
    parser.add_argument('--ckpt', default=None,
                        help='Checkpoint to resume from (required for phase 3)')
    parser.add_argument('--save_dir', default='./runs',
                        help='Root directory for checkpoints')
    parser.add_argument('--train_pkl', default=None,
                        help='Path to pre-cached train_index.pkl (optional)')
    parser.add_argument('--val_pkl',   default=None,
                        help='Path to pre-cached val_index.pkl (optional)')
    cli = parser.parse_args()

    args = load_config(cli.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load data ──────────────────────────────────────────────────────────
    if cli.train_pkl and os.path.exists(cli.train_pkl):
        with open(cli.train_pkl, 'rb') as f:
            train_index = pickle.load(f)
        with open(cli.val_pkl, 'rb') as f:
            val_index = pickle.load(f)
        print(f'Loaded data from pkl: {len(train_index["neural_features"])} train, '
              f'{len(val_index["neural_features"])} val trials')
    else:
        from src.data.preprocessing import load_all_files
        train_index = load_all_files(args.drive_dir, args.local_dir, 'train', args.max_files)
        val_index   = load_all_files(args.drive_dir, args.local_dir, 'val',   args.max_files)

    from src.data.dataset import SpeechDataset
    from src.data.dataloader import BucketBatchSampler, collate_fn

    train_ds      = SpeechDataset(train_index, max_time_steps=args.max_time_steps)
    val_ds        = SpeechDataset(val_index,   max_time_steps=args.max_time_steps)
    train_sampler = BucketBatchSampler(train_ds, batch_size=args.batch_size)
    val_sampler   = BucketBatchSampler(val_ds,   batch_size=args.batch_size)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler,
                               collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_ds,   batch_sampler=val_sampler,
                               collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # ── Build model ────────────────────────────────────────────────────────
    if cli.model == 'dbconformer':
        from src.models.db_conformer import build_dbconformer_ctc
        model = build_dbconformer_ctc(args).to(device)
    else:
        from src.models.eeg_conformer import NeuralSpeechModel
        cfg_dict = vars(args)
        model    = NeuralSpeechModel(cfg_dict).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params:,}')

    if cli.ckpt:
        state = torch.load(cli.ckpt, map_location=device)
        model.load_state_dict(state)
        print(f'Resumed from {cli.ckpt}')

    # ── Train ──────────────────────────────────────────────────────────────
    from src.training.train_acoustic import train_model, finetune_phase3

    if cli.phase == 1:
        best_per = train_model(
            model, train_loader, val_loader, args, device,
            save_dir=cli.save_dir, phase='phase1',
        )
    else:
        if cli.ckpt is None:
            raise ValueError('--ckpt must be provided for phase 3 fine-tuning')
        best_per = finetune_phase3(
            model, train_loader, val_loader, args, device,
            save_dir=cli.save_dir,
            extra_epochs=getattr(args, 'phase3_epochs', 50),
            starting_lr=getattr(args, 'phase3_lr', 1e-4),
        )

    print(f'\nFinal best Val PER: {best_per:.2f}%')


if __name__ == '__main__':
    main()
