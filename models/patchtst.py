"""PatchTST adapted for multi-label ECG classification.

Architecture (channel-independent):
  Input  : (B, 12, 5000)
  Patches: (B*12, N_patches, patch_size)  via unfold
  Embed  : (B*12, N_patches, d_model)  linear projection + learnable pos-enc
  Encoder: L × TransformerEncoderLayer (pre-norm)
  Pool   : mean over patches → (B, 12, d_model)
  Head   : mean over leads → Dropout → Linear(d_model, n_classes)

Reference: Nie et al. 2023 "A Time Series is Worth 64 Words"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    """Extract non-overlapping patches and project to d_model.

    Input : (B_leads, L)          — one lead at a time, batch-flattened
    Output: (B_leads, N_patches, d_model)
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(patch_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B_leads, L)
        patches = x.unfold(-1, self.patch_size, self.stride)  # (B_leads, N, patch_size)
        return self.dropout(self.proj(patches))               # (B_leads, N, d_model)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding added to patch tokens."""

    def __init__(self, n_patches: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe


class PatchTST(nn.Module):
    """PatchTST for multi-label ECG classification.

    Input : (B, n_leads, L)   where L = 5000 at 500 Hz
    Output: (B, n_classes)    raw logits

    The model is channel-independent: each lead is processed by the same
    shared transformer, then lead representations are mean-pooled before
    the classification head.

    Args:
        n_classes:   Number of output classes (binary multi-label).
        n_leads:     Number of ECG leads (12 for standard 12-lead).
        signal_length: Number of time steps per lead (5000 at 500 Hz).
        patch_size:  Samples per patch (50 ≈ 100 ms at 500 Hz).
        stride:      Patch stride; equal to patch_size gives no overlap.
        d_model:     Transformer embedding dimension.
        n_heads:     Number of attention heads (d_model must be divisible).
        n_layers:    Number of transformer encoder layers.
        ff_dim:      Feed-forward hidden dimension (typically 2-4 × d_model).
        attn_dropout: Dropout inside attention weights.
        dropout:     Dropout on embeddings and after pooling.
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_leads: int = 12,
        signal_length: int = 5000,
        patch_size: int = 50,
        stride: int = 50,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 1024,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_leads = n_leads
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model

        # Number of patches per lead
        self.n_patches = math.floor((signal_length - patch_size) / stride) + 1

        # Patch embedding (shared across leads)
        self.patch_embed = PatchEmbedding(patch_size, stride, d_model, dropout)
        self.pos_enc = LearnablePositionalEncoding(self.n_patches, d_model)

        # Transformer encoder (pre-norm via norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN is more stable
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_leads, L)
        B, C, L = x.shape

        # Flatten leads into batch dim so the same transformer processes every lead
        x = x.reshape(B * C, L)                   # (B*C, L)
        x = self.patch_embed(x)                   # (B*C, N, d_model)
        x = self.pos_enc(x)                        # (B*C, N, d_model)

        x = self.encoder(x)                        # (B*C, N, d_model)

        x = x.mean(dim=1)                          # (B*C, d_model) — pool patches
        x = x.reshape(B, C, self.d_model)          # (B, C, d_model)
        x = x.mean(dim=1)                          # (B, d_model)   — pool leads

        x = self.head_dropout(x)
        return self.fc(x)                          # (B, n_classes) logits

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for sr, L in [(500, 5000), (100, 1000)]:
        patch_size = 50 if sr == 500 else 10
        model = PatchTST(
            n_classes=5,
            signal_length=L,
            patch_size=patch_size,
            stride=patch_size,
        )
        x = torch.randn(2, 12, L)  # batch=2 to stay within CPU memory limits
        logits = model(x)
        assert logits.shape == (2, 5), f"Bad shape: {logits.shape}"
        print(
            f"[{sr} Hz] input={x.shape}  output={logits.shape}  "
            f"n_patches={model.n_patches}  params={model.n_parameters:,}"
        )
    print("patchtst.py smoke test passed.")
