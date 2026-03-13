# -*- coding: utf-8 -*-
"""
DBConformer Architecture (Dual-Branch Convolutional Transformer)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


# ════════════════════════════════════════════════════════════════════════════
# 1. Convolutional & Stem Blocks
# ════════════════════════════════════════════════════════════════════════════

class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        super().__init__()
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def forward(self, x):
        return F.gelu(sum(x))


class Stem(nn.Module):
    """
    Two-stage pooling scheme (small conv + AvgPool), giving ~T/8 output tokens.
    """
    def __init__(self, data_name, in_planes, out_planes=64,
                 kernel_size=15, pool_kernel=15, pool_stride=8, radix=2):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.radix = radix
        self.data_name = data_name

        self.sconv = Conv(
            nn.Conv1d(in_planes, self.mid_planes, 1, bias=False, groups=radix),
            bn=nn.BatchNorm1d(self.mid_planes))

        self.tconv = nn.ModuleList()
        ks = kernel_size
        for _ in range(radix):
            self.tconv.append(Conv(
                nn.Conv1d(out_planes, out_planes, ks, 1,
                          groups=out_planes, padding=ks // 2, bias=False),
                bn=nn.BatchNorm1d(out_planes)))
            ks = max(ks // 2, 3)

        self.interFre = InterFre()
        self.downSampling = nn.AvgPool1d(pool_kernel, pool_stride)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(xi) for xi, m in zip(out, self.tconv)]
        out = self.interFre(out)
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class PatchEmbeddingTemporal(nn.Module):
    def __init__(self, data_name, in_planes, out_planes,
                 kernel_size, radix, pool_kernel, pool_stride,
                 time_points, num_classes):
        super().__init__()
        self.stem = Stem(
            data_name=data_name,
            in_planes=in_planes * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            radix=radix,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):          # x: (B, C, T)
        out = self.stem(x)         # -> (B, D, P)
        return out.permute(0, 2, 1)  # -> (B, P, D)


class PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spa_dim, emb_size),
        )

    def forward(self, x):          # x: (B, C, T)
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        x = self.encoder(x)        # -> (B*C, emb_size)
        return x.view(B, C, -1)    # -> (B, C, emb_size)


# ════════════════════════════════════════════════════════════════════════════
# 2. Attention & Transformer Blocks
# ════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert emb_size % num_heads == 0, f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})"
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        Q = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.keys(x),    'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.values(x),  'b n (h d) -> b h n d', h=self.num_heads)

        # Uses PyTorch's native scaled dot product attention (FlashAttention if available)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.3):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 8,
                 drop_p: float = 0.3, forward_expansion: int = 4,
                 forward_drop_p: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.drop1 = nn.Dropout(drop_p)

        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = FeedForwardBlock(emb_size, forward_expansion, forward_drop_p)
        self.drop2 = nn.Dropout(drop_p)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        res = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = res + self.drop1(x)

        res = x
        x = self.norm2(x)
        x = self.ff(x)
        x = res + self.drop2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, emb_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(emb_size) for _ in range(depth)]
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


# ════════════════════════════════════════════════════════════════════════════
# 3. CTC Head
# ════════════════════════════════════════════════════════════════════════════

class CTCHead(nn.Module):
    """
    Input  : (B, T', emb_size)
    Output : (T', B, NUM_PHONEMES)  for nn.CTCLoss
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
        logits = self.ffn(x)              # (B, T', C)
        return logits.permute(1, 0, 2)    # (T', B, C)


# ════════════════════════════════════════════════════════════════════════════
# 4. Main Model Wrappers
# ════════════════════════════════════════════════════════════════════════════

class DBConformer(nn.Module):
    """
    Dual-Branch Convolutional Transformer.
    """
    def __init__(self, args, emb_size=128, tem_depth=6, chn_depth=6,
                 chn=512, n_classes=41):
        super().__init__()

        self.embedding = PatchEmbeddingTemporal(
            data_name=args.data_name,
            in_planes=args.chn,
            out_planes=emb_size,
            kernel_size=args.temporal_kernel,
            radix=1,
            pool_kernel=args.pool_kernel,
            pool_stride=args.pool_stride,
            time_points=args.time_sample_num,
            num_classes=args.class_num,
        )
        self.channel_embedding = PatchEmbeddingSpatial(
            spa_dim=args.spa_dim, emb_size=emb_size
        )

        self.C = args.chn
        self.D = emb_size
        self.gate_flag = args.gate_flag
        self.posemb_flag = args.posemb_flag
        self.branch = args.branch
        self.chn_atten_flag = args.chn_atten_flag

        # Learnable positional embeddings (size determined at first forward)
        if args.posemb_flag:
            P_init = max(1, args.time_sample_num // args.pool_stride)
            self.pos_embedding_temporal = nn.Parameter(
                torch.randn(1, P_init, emb_size)
            )
            self.pos_embedding_spatial = nn.Parameter(
                torch.randn(1, self.C, emb_size)
            )

        self.temporal_transformer = TransformerEncoder(tem_depth, emb_size)
        self.spatial_transformer = TransformerEncoder(chn_depth, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: (B, 1, C, T)
        x = x.squeeze(1)                                  # (B, C, T)

        x_embed = self.embedding(x)                       # (B, P, D)
        x_embed_spatial = self.channel_embedding(x)       # (B, C, D)

        if self.posemb_flag:
            pos_tem = F.interpolate(
                self.pos_embedding_temporal.permute(0, 2, 1),
                size=x_embed.shape[1], mode='linear', align_corners=False
            ).permute(0, 2, 1)
            x_embed = x_embed + pos_tem

            pos_spa = F.interpolate(
                self.pos_embedding_spatial.permute(0, 2, 1),
                size=x_embed_spatial.shape[1], mode='linear', align_corners=False
            ).permute(0, 2, 1)
            x_embed_spatial = x_embed_spatial + pos_spa

        # Thread the padding mask through temporal branch only;
        # spatial branch operates over electrodes (always fully valid).
        x_temporal = self.temporal_transformer(x_embed, mask=mask)   # (B, P, D)
        x_spatial = self.spatial_transformer(x_embed_spatial)        # (B, C, D)

        return x_temporal, None


class DBConformerCTC(nn.Module):
    """DBConformer backbone + CTC head."""
    def __init__(self, backbone: DBConformer, emb_size: int,
                 num_classes: int, ffn_hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.ctc_head = CTCHead(emb_size, num_classes, ffn_hidden, dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        features, _ = self.backbone(x, mask=mask)   # (B, T', D)
        return self.ctc_head(features)              # (T', B, C)


def backbone_net_dbconformer(args) -> DBConformer:
    return DBConformer(
        args,
        emb_size=args.emb_size,
        tem_depth=args.transformer_depth_tem,
        chn_depth=args.transformer_depth_chn,
        chn=args.chn,
        n_classes=args.class_num,
    )