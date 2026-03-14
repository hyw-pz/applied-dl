"""
run_uncertainty.py
------------------
Entry point: run uncertainty analysis on the full AM + LM pipeline.

Outputs:
  - Calibration curves (AM and LM)
  - Coverage–WER trade-off plot
  - ECE score
  - Stratified confidence report

Usage:
    python scripts/run_uncertainty.py \
        --acoustic_config configs/db_conformer_config.yaml \
        --acoustic_ckpt /path/to/best.ckpt \
        --lm_path /path/to/BART_large \
        --val_pkl /content/val_index_merged.pkl \
        --lm_type bart
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Uncertainty analysis')
    parser.add_argument('--acoustic_config', required=True)
    parser.add_argument('--acoustic_ckpt',   required=True)
    parser.add_argument('--lm_path',         required=True)
    parser.add_argument('--val_pkl',         default=None)
    parser.add_argument('--lm_type', default='bart', choices=['bart', 'qwen'])
    parser.add_argument('--qwen_base', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', default='./uncertainty_results')
    cli = parser.parse_args()

    os.makedirs(cli.output_dir, exist_ok=True)

    ac_cfg = load_yaml(cli.acoustic_config)
    import argparse as _ap
    args = _ap.Namespace(**ac_cfg)
    args.data_name      = ac_cfg.get('data_name', 'NeuralPhoneme')
    args.class_num      = ac_cfg.get('class_num', 41)
    args.max_time_steps = ac_cfg.get('time_sample_num', 1500)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Acoustic model ─────────────────────────────────────────────────────
    from src.models.db_conformer import build_dbconformer_ctc
    acoustic_model = build_dbconformer_ctc(args).to(device)
    acoustic_model.load_state_dict(
        torch.load(cli.acoustic_ckpt, map_location=device)
    )
    acoustic_model.eval()

    # ── Val data ───────────────────────────────────────────────────────────
    if cli.val_pkl and os.path.exists(cli.val_pkl):
        with open(cli.val_pkl, 'rb') as f:
            val_index = pickle.load(f)
    else:
        from src.data.preprocessing import load_all_files
        val_index = load_all_files(args.drive_dir, args.local_dir, 'val')

    from src.data.dataset import SpeechDatasetWithText
    from src.data.dataloader import BucketBatchSampler, collate_fn_with_text
    from torch.utils.data import DataLoader

    val_ds      = SpeechDatasetWithText(val_index, args.max_time_steps)
    val_sampler = BucketBatchSampler(val_ds, batch_size=cli.batch_size)
    val_loader  = DataLoader(val_ds, batch_sampler=val_sampler,
                             collate_fn=collate_fn_with_text,
                             num_workers=2, pin_memory=True)

    # ── Language model ─────────────────────────────────────────────────────
    if cli.lm_type == 'bart':
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        lm_tokenizer = AutoTokenizer.from_pretrained(cli.lm_path)
        lm_model     = AutoModelForSeq2SeqLM.from_pretrained(cli.lm_path).to(device)
    else:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
        )
        lm_tokenizer = AutoTokenizer.from_pretrained(cli.qwen_base)
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            cli.qwen_base, quantization_config=bnb_config, device_map='auto'
        )
        lm_model = PeftModel.from_pretrained(base, cli.lm_path)
    lm_model.eval()

    # ── Run pipeline ───────────────────────────────────────────────────────
    from src.evaluation.evaluate_pipeline import evaluate_pipeline_uncertainty
    wers, pers, am_uncs, lm_uncs = evaluate_pipeline_uncertainty(
        acoustic_model, lm_model, lm_tokenizer,
        val_loader, device, args, lm_type=cli.lm_type,
    )

    # ── Calibration metrics ────────────────────────────────────────────────
    from src.uncertainty.calibration import (
        compute_ece, coverage_wer_curve, plot_uncertainty
    )

    ece = compute_ece(wers, lm_uncs)
    print(f'\nECE: {ece:.4f}')

    coverages, mean_wers = coverage_wer_curve(wers, am_uncs, lm_uncs)

    # ── Plots ──────────────────────────────────────────────────────────────
    fig_unc = plot_uncertainty(
        wers, pers, am_uncs, lm_uncs,
        title=f'{cli.lm_type.upper()} Pipeline Uncertainty Analysis',
    )
    fig_unc.savefig(os.path.join(cli.output_dir, 'uncertainty_analysis.png'),
                   dpi=150, bbox_inches='tight')

    plt.figure(figsize=(8, 5))
    plt.plot(coverages, mean_wers, 'o-', color='blue')
    plt.xlabel('Coverage (fraction of samples predicted)')
    plt.ylabel('Mean WER')
    plt.title('Coverage vs WER Trade-off')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cli.output_dir, 'coverage_wer.png'),
                dpi=150, bbox_inches='tight')

    print(f'\nPlots saved to {cli.output_dir}/')

    # ── Confidence stratification ──────────────────────────────────────────
    from src.uncertainty.calibration import confidence_report
    print('\n--- LM Confidence Stratification ---')
    thresholds = [
        (0.00, 0.05,  'HIGH     '),
        (0.05, 0.15,  'MEDIUM   '),
        (0.15, 0.25,  'LOW      '),
        (0.25, 1e9,   'VERY LOW '),
    ]
    for lo, hi, label in thresholds:
        grp = [w for w, u in zip(wers, lm_uncs) if lo <= u < hi]
        if grp:
            print(f'{label}: {len(grp):4d} samples ({len(grp)/len(wers):.1%}),  '
                  f'mean WER={np.mean(grp):.2%},  median WER={np.median(grp):.2%}')


if __name__ == '__main__':
    main()
