"""
train_acoustic.py
-----------------
CTC training loop for both EEGConformer and DBConformerCTC.

Features:
  - AMP (torch.autocast + GradScaler)
  - OneCycleLR scheduler
  - AdamW with weight decay
  - OOM guard
  - Attention mask construction for variable-length batches
  - Phoneme Error Rate (PER) validation
"""

import gc
import os

import numpy as np
import torch
import torch.nn as nn

from src.data.dataloader import compute_token_lengths, make_attention_mask
from src.evaluation.metrics import phoneme_error_rate
from src.evaluation.decode import greedy_ctc_decode


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    ctc_loss_fn,
    device,
    pool_kernel: int,
    pool_stride: int,
) -> float:
    """Run one training epoch. Returns mean batch loss."""
    model.train()
    total_loss, num_batches = 0.0, 0

    for batch_feat, batch_ids, batch_lens, input_time_steps in train_loader:
        try:
            batch_feat = batch_feat.to(device)
            batch_ids  = batch_ids.to(device)
            batch_lens = batch_lens.to(device)

            # Valid CTC frame counts
            input_lengths = compute_token_lengths(
                input_time_steps, pool_kernel, pool_stride
            ).to(device)

            # Padding mask for attention
            T_p = batch_feat.size(-1)
            P   = compute_token_lengths(
                torch.tensor([T_p]), pool_kernel, pool_stride
            ).item()
            attn_mask = make_attention_mask(input_lengths, P)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                log_probs = model(batch_feat, mask=attn_mask)
                log_probs = log_probs.log_softmax(dim=-1)

            B           = batch_feat.size(0)
            targets_cat = torch.cat([
                batch_ids[i, :int(batch_lens[i].item())]
                for i in range(B)
            ])
            loss = ctc_loss_fn(log_probs, targets_cat, input_lengths, batch_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss  += loss.item()
            num_batches += 1

        except torch.cuda.OutOfMemoryError:
            print('\n  [OOM] Skipping batch…')
            torch.cuda.empty_cache()
            optimizer.zero_grad()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    pool_kernel: int,
    pool_stride: int,
) -> float:
    """Run validation; returns PER (%)."""
    model.eval()
    all_refs, all_hyps = [], []

    for batch_feat, batch_ids, batch_lens, input_time_steps in val_loader:
        batch_feat = batch_feat.to(device)
        input_lengths = compute_token_lengths(
            input_time_steps, pool_kernel, pool_stride
        ).to(device)

        T_p = batch_feat.size(-1)
        P   = compute_token_lengths(
            torch.tensor([T_p]), pool_kernel, pool_stride
        ).item()
        attn_mask = make_attention_mask(input_lengths, P)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            log_probs = model(batch_feat, mask=attn_mask).log_softmax(dim=-1)

        hyps = greedy_ctc_decode(log_probs)
        for i in range(len(hyps)):
            slen = int(batch_lens[i].item())
            all_refs.append(batch_ids[i, :slen].tolist())
            all_hyps.append(hyps[i])

    return phoneme_error_rate(all_refs, all_hyps)


def train_model(
    model,
    train_loader,
    val_loader,
    args,
    device,
    save_dir: str = './runs',
    phase: str = 'phase1',
) -> float:
    """
    Full training loop with checkpoint saving.

    Parameters
    ----------
    model       : DBConformerCTC or NeuralSpeechModel
    train_loader / val_loader : DataLoaders
    args        : argparse.Namespace with lr, max_epoch, eval_interval,
                  pool_kernel, pool_stride, data_name, session
    device      : torch.device
    save_dir    : root directory for checkpoints
    phase       : string tag included in the checkpoint filename

    Returns
    -------
    float : best validation PER achieved
    """
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer   = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.max_epoch,
    )
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_per  = 100.0
    ckpt_dir  = os.path.join(save_dir, args.data_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.max_epoch + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            ctc_loss_fn, device, args.pool_kernel, args.pool_stride,
        )

        if epoch % args.eval_interval == 0 or epoch == args.max_epoch:
            per = validate(
                model, val_loader, device,
                args.pool_kernel, args.pool_stride,
            )
            print(f'Epoch {epoch:03d}/{args.max_epoch} | '
                  f'Loss: {avg_loss:.4f} | Val PER: {per:.2f}%')

            if per < best_per:
                best_per = per
                ckpt_name = f'DBConformerCTC_{args.session}_{phase}_best.ckpt'
                torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))
                print(f'  → Saved best checkpoint (PER={best_per:.2f}%)')

    print(f'Training complete. Best Val PER: {best_per:.2f}%')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_per


def finetune_phase3(
    model,
    train_loader,
    val_loader,
    args,
    device,
    save_dir: str = './runs',
    extra_epochs: int = 50,
    starting_lr: float = 1e-4,
) -> float:
    """
    Phase 3: stable fine-tuning with ReduceLROnPlateau (no warm-up oscillation).

    Parameters match :func:`train_model`; ``extra_epochs`` and
    ``starting_lr`` override ``args.max_epoch`` and ``args.lr``.
    """
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer   = torch.optim.AdamW(
        model.parameters(), lr=starting_lr, weight_decay=1e-4
    )
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_per = 100.0
    ckpt_dir = os.path.join(save_dir, args.data_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, extra_epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer,
            # ReduceLROnPlateau is stepped per-epoch not per-step; pass a
            # dummy lambda scheduler so the signature stays the same
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            scaler, ctc_loss_fn, device,
            args.pool_kernel, args.pool_stride,
        )

        per = validate(
            model, val_loader, device,
            args.pool_kernel, args.pool_stride,
        )
        scheduler.step(per)

        print(f'Phase3 Epoch {epoch:03d}/{extra_epochs} | '
              f'Loss: {avg_loss:.4f} | Val PER: {per:.2f}%')

        if per < best_per:
            best_per = per
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir,
                             f'DBConformerCTC_{args.session}_phase3_best.ckpt'),
            )

    return best_per
