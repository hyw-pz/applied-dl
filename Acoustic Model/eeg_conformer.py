# -*- coding: utf-8 -*-
"""

EEG Conformer Architecture (CNN + Transformer)
@author: Zhe Wang, YiWei Hou

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=128, num_electrodes=512,
                 temporal_kernel=(1, 10), pool_kernel=(1, 15), pool_stride=(1, 8)):
        super().__init__()

        # same-padding
        pad_w = temporal_kernel[1] // 2

        # Temporal Conv (same padding), conv over time dimension (1,5) = receptive field of 5 bins x 20 ms = 100 ms
        self.temporal_conv = nn.Conv2d(
            in_channels=1, out_channels=emb_size,
            kernel_size=temporal_kernel, stride=(1, 1), padding=(0, pad_w)
        )

        # Spatial dimension is collapsed from 512 -> 1
        self.spatial_conv = nn.Conv2d(emb_size, emb_size, kernel_size=(num_electrodes, 1),
                                      stride=(1, 1), groups=emb_size)

        # Batch Normalisation + Activation
        self.bn  = nn.BatchNorm2d(emb_size)
        self.act = nn.ELU()

        # Average Pooling  (temporal downsampling)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.dropout = nn.Dropout(0.5)

        # Projection feature channels (128) to target embedding dimension for transformer encoder
        # Shape  (Batch, emb_size, Height (1), num_tokens) -> (Batch, num_tokens, emb_size)
        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=(1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, emb_size)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))

        pe = torch.zeros(1, max_len, emb_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # register_buffer ensures it's saved with the model but not updated by the optimizer
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (Batch, num_tokens, emb_size)
        # Add the positional encoding up to the current sequence length
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert emb_size % num_heads == 0, f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})"

        self.emb_size  = emb_size
        self.num_heads = num_heads
        self.head_dim  = emb_size // num_heads
        self.dropout_p = dropout

        self.queries    = nn.Linear(emb_size, emb_size)
        self.keys       = nn.Linear(emb_size, emb_size)
        self.values     = nn.Linear(emb_size, emb_size)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        Q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)

        # FlashAttention
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        residual = x
        x = self.fn(x, **kwargs)
        return x + residual


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),  # expand
            nn.GELU(),                                  # smooth activation
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),  # project back
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 8,
        drop_p: float = 0.3,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.3
    ):
        super().__init__()
        # Sub-layer 1: self-attention
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.drop1 = nn.Dropout(drop_p)

        # Sub-layer 2: feed-forward
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.drop2 = nn.Dropout(drop_p)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Self-attention with residual connection
        res = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = res + self.drop1(x)

        # Feed-forward with residual connection
        res = x
        x = self.norm2(x)
        x = self.ff(x)
        x = res + self.drop2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, emb_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class NeuralEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            emb_size=config['emb_size'],
            num_electrodes=config['num_electrodes'],
            temporal_kernel=config['temporal_kernel'],
            pool_kernel=config['pool_kernel'],
            pool_stride=config['pool_stride']
        )
        self.pos_encoder = PositionalEncoding(
            emb_size=config['emb_size'],
            dropout=config['drop_p']
        )
        self.transformer = TransformerEncoder(depth=config['depth'], emb_size=config['emb_size'])
        self.norm = nn.LayerNorm(config['emb_size'])

    def forward(self, x: Tensor, input_lengths: Tensor = None) -> Tensor:
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)

        # --- ATTENTION MASK CREATION ---
        mask = None
        if input_lengths is not None:
            B, T, _ = x.shape
            seq_range = torch.arange(T, device=x.device).expand(B, T)
            bool_mask = seq_range < input_lengths.unsqueeze(1)
            mask = bool_mask.unsqueeze(1).unsqueeze(2)
        # -------------------------------

        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        return x


class PhonemeHead(nn.Module):
    """
    FFN head for per-timestep phoneme prediction.
    Input : (B, num_tokens, emb_size)
    Output: (B, num_tokens, num_phonemes + 1)
    """
    def __init__(
        self,
        emb_size    : int,
        num_phonemes: int,
        ffn_hidden  : int   = 256,
        drop_p      : float = 0.3,
    ):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(ffn_hidden, num_phonemes + 1),  # +1 for blank token
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


# ===============================================================
# Main Model Wrapper
# ===============================================================
class EEGConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = NeuralEncoder(config)
        self.head = PhonemeHead(
            emb_size=config['emb_size'],
            num_phonemes=config['num_phonemes'],
            ffn_hidden=config['ffn_hidden'],
            drop_p=config['drop_p'],
        )

    def forward(self, x: Tensor, input_lengths: Tensor = None):
        x = self.encoder(x, input_lengths)
        x = self.head(x)
        x = F.log_softmax(x, dim=-1)

        num_tokens = x.shape[1]
        x = x.permute(1, 0, 2)
        return x, num_tokens