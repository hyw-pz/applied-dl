"""
eeg_conformer.py
----------------
EEGConformer: CNN + Transformer neural speech decoder (v1).

Architecture:
  PatchEmbedding (2D temporal Conv) → PositionalEncoding
  → TransformerEncoder → PhonemeHead (CTC)

This is the first acoustic model in the pipeline, operating directly on
(B, 1, num_electrodes, T) neural inputs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# CNN front-end
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    2D depthwise-separable CNN that extracts spatio-temporal patch embeddings.

    Input  : (B, 1, C, T)   where C = num_electrodes
    Output : (B, P, emb_size)
    """

    def __init__(
        self,
        emb_size: int = 128,
        num_electrodes: int = 512,
        temporal_kernel=(1, 10),
        pool_kernel=(1, 15),
        pool_stride=(1, 8),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.emb_size = emb_size

        self.conv = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, emb_size, kernel_size=temporal_kernel,
                      padding=(0, temporal_kernel[1] // 2), bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            # Spatial (depthwise) convolution
            nn.Conv2d(emb_size, emb_size, kernel_size=(num_electrodes, 1),
                      groups=emb_size, bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Dropout(dropout),
        )

        # After spatial pooling the electrode dim collapses to 1
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=2),   # (B, emb_size, P)
        )

    def forward(self, x: Tensor) -> Tensor:  # x: (B, 1, C, T)
        x = self.conv(x)          # (B, emb_size, 1, P)
        x = x.squeeze(2)          # (B, emb_size, P)
        return x.permute(0, 2, 1) # (B, P, emb_size)


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.queries    = nn.Linear(emb_size, emb_size)
        self.keys       = nn.Linear(emb_size, emb_size)
        self.values     = nn.Linear(emb_size, emb_size)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        Q = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.keys(x),    'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.values(x),  'b n (h d) -> b h n d', h=self.num_heads)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 8,
        drop_p: float = 0.3,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.3,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn  = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.drop1 = nn.Dropout(drop_p)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff    = FeedForwardBlock(emb_size, forward_expansion, forward_drop_p)
        self.drop2 = nn.Dropout(drop_p)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x   = self.norm1(x)
        x   = self.attn(x)
        x   = res + self.drop1(x)
        res = x
        x   = self.norm2(x)
        x   = self.ff(x)
        x   = res + self.drop2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, emb_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(emb_size) for _ in range(depth)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# CTC head
# ─────────────────────────────────────────────────────────────────────────────

class PhonemeHead(nn.Module):
    """
    FFN CTC head.

    Input  : (B, P, emb_size)
    Output : (P, B, num_classes)  — ready for nn.CTCLoss
    """

    def __init__(self, emb_size: int, num_classes: int,
                 ffn_hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.ffn(x)              # (B, P, C)
        return logits.permute(1, 0, 2)   # (P, B, C)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSpeechModel(nn.Module):
    """
    EEGConformer: PatchEmbedding → PositionalEncoding → TransformerEncoder
    → PhonemeHead (CTC).

    Parameters
    ----------
    config : dict with keys:
        emb_size, num_electrodes, temporal_kernel, pool_kernel, pool_stride,
        transformer_depth, num_heads, dropout, num_classes
    """

    def __init__(self, config: dict):
        super().__init__()
        emb_size      = config['emb_size']
        num_electrodes = config.get('num_electrodes', 512)
        num_classes   = config.get('num_classes', 41)
        depth         = config.get('transformer_depth', 6)
        dropout       = config.get('dropout', 0.3)

        self.patch_embedding = PatchEmbedding(
            emb_size=emb_size,
            num_electrodes=num_electrodes,
            temporal_kernel=config.get('temporal_kernel', (1, 10)),
            pool_kernel=config.get('pool_kernel', (1, 15)),
            pool_stride=config.get('pool_stride', (1, 8)),
            dropout=dropout,
        )
        self.pos_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.transformer  = TransformerEncoder(depth, emb_size)
        self.head         = PhonemeHead(emb_size, num_classes,
                                        ffn_hidden=config.get('ffn_hidden', 256),
                                        dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, 1, C, T)
        Returns (T', B, num_classes) CTC logits
        """
        x = self.patch_embedding(x)   # (B, P, D)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.head(x)           # (P, B, C)
