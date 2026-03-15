"""
tests/test_models.py
Smoke tests for EEGConformer and DBConformer forward passes.
"""

import argparse

import pytest
import torch


# ── EEGConformer ──────────────────────────────────────────────────────────────

class TestEEGConformer:
    @pytest.fixture
    def config(self):
        return {
            'emb_size':          64,
            'num_electrodes':    16,
            'temporal_kernel':   (1, 5),
            'pool_kernel':       (1, 5),
            'pool_stride':       (1, 4),
            'transformer_depth': 2,
            'num_heads':         4,
            'ffn_hidden':        64,
            'num_classes':       41,
            'dropout':           0.1,
        }

    def test_forward_shape(self, config):
        from src.models.eeg_conformer import NeuralSpeechModel
        model = NeuralSpeechModel(config)
        model.eval()
        x = torch.randn(2, 1, config['num_electrodes'], 100)
        with torch.no_grad():
            out = model(x)
        # (T', B, C)
        assert out.ndim == 3
        assert out.shape[1] == 2
        assert out.shape[2] == 41

    def test_gradient_flow(self, config):
        from src.models.eeg_conformer import NeuralSpeechModel
        model = NeuralSpeechModel(config)
        x   = torch.randn(2, 1, config['num_electrodes'], 60)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                pytest.fail(f'No gradient for {name}')


# ── DBConformer ───────────────────────────────────────────────────────────────

class TestDBConformer:
    @pytest.fixture
    def args(self):
        return argparse.Namespace(
            data_name             = 'test',
            chn                   = 16,
            time_sample_num       = 80,
            emb_size              = 32,
            spa_dim               = 4,
            transformer_depth_tem = 2,
            transformer_depth_chn = 2,
            temporal_kernel       = 5,
            pool_kernel           = 5,
            pool_stride           = 4,
            gate_flag             = False,
            posemb_flag           = True,
            chn_atten_flag        = True,
            branch                = 'all',
            class_num             = 41,
            ffn_hidden            = 64,
            dropoutRate           = 0.1,
        )

    def test_backbone_output_shape(self, args):
        from src.models.db_conformer import DBConformer
        model = DBConformer(args, emb_size=32, tem_depth=2, chn_depth=2,
                             chn=16, n_classes=41)
        model.eval()
        x = torch.randn(2, 1, 16, 80)
        with torch.no_grad():
            t_out, s_out = model(x)
        assert t_out.ndim == 3   # (B, P, D)
        assert t_out.shape[0] == 2
        assert t_out.shape[2] == 32

    def test_ctc_model_output_shape(self, args):
        from src.models.db_conformer import build_dbconformer_ctc
        model = build_dbconformer_ctc(args)
        model.eval()
        x = torch.randn(2, 1, 16, 80)
        with torch.no_grad():
            out = model(x)
        assert out.ndim == 3          # (T', B, C)
        assert out.shape[1] == 2
        assert out.shape[2] == 41

    def test_attention_mask(self, args):
        from src.models.db_conformer import build_dbconformer_ctc
        from src.data.dataloader import compute_token_lengths, make_attention_mask
        model = build_dbconformer_ctc(args)
        model.eval()
        x    = torch.randn(2, 1, 16, 80)
        lens = torch.tensor([80, 60])
        il   = compute_token_lengths(lens, args.pool_kernel, args.pool_stride)
        T_p  = x.size(-1)
        P    = compute_token_lengths(torch.tensor([T_p]), args.pool_kernel, args.pool_stride).item()
        mask = make_attention_mask(il, P)
        with torch.no_grad():
            out = model(x, mask=mask)
        assert out.shape[1] == 2


# ── CTC decode ────────────────────────────────────────────────────────────────

class TestGreedyDecode:
    def test_basic(self):
        from src.evaluation.decode import greedy_ctc_decode
        # T=4, B=1, C=5 — argmax at positions [0, 1, 1, 2] → collapse → [1, 2]
        lp = torch.zeros(4, 1, 5)
        lp[0, 0, 0] = 10   # blank
        lp[1, 0, 1] = 10
        lp[2, 0, 1] = 10   # repeated — collapsed
        lp[3, 0, 2] = 10
        decoded = greedy_ctc_decode(lp)
        assert decoded[0] == [1, 2]

    def test_all_blank(self):
        from src.evaluation.decode import greedy_ctc_decode
        lp = torch.zeros(3, 1, 5)
        lp[:, 0, 0] = 10
        assert greedy_ctc_decode(lp)[0] == []
