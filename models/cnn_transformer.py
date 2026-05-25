from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

"""CNN al inicio: extrae características locales del ECG (picos QRS, forma de onda, morfología beat-a-beat).
Es rápida y eficiente en señales 1D.Transformer después: recibe esas características como "tokens" y
aplica self-attention para capturar dependencias a largo plazo (ej. cómo se relaciona el segmento ST de un latido con el siguiente)."""
"""
CNN-Transformer Hybrid for multi-label ECG classification (PTB-XL+).

Architecture:
  1. CNN Encoder  — stack of 1-D depthwise-separable residual blocks that
                    compress (12, 5000) → (d_model, T) feature maps.
  2. Positional Encoding — learnable 1-D positional embeddings added to
                    each of the T temporal tokens.
  3. Transformer Encoder — N layers of multi-head self-attention + FFN,
                    capturing long-range dependencies across the sequence.
  4. Classification Head — global average-pool over tokens → Dropout →
                    Linear(d_model, n_classes).  Returns logits (no sigmoid).

Input shape : (B, 12, L)   where L = 5000 at 500 Hz, or 1000 at 100 Hz
Output shape: (B, n_classes)                         logits, no sigmoid

~6.2 M parameters with default settings.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth: drop entire residual branch with probability p."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep) / keep
        return x * mask


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D signals."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        scale = self.fc(x).unsqueeze(-1)          # (B, C, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# CNN Encoder
# ─────────────────────────────────────────────────────────────────────────────

class DSConvBlock(nn.Module):
    """
    Depthwise-separable residual block (1-D).

    Conv(k=7, depthwise) → BN → ReLU → Conv(k=1, pointwise) → BN → SE → DropPath
    + skip (proj if channels differ) → ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # depthwise
            nn.Conv1d(in_ch, in_ch, kernel_size=7, stride=stride,
                      padding=3, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            # pointwise
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se = SEBlock1D(out_ch)
        self.drop_path = DropPath(drop_path_rate)

        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop_path(self.se(self.conv(x))) + self.skip(x))


class CNNEncoder(nn.Module):
    """
    Four-stage CNN that maps (B, 12, L) → (B, d_model, T).

    Stage  | out_ch | stride | purpose
    -------|--------|--------|----------------------------
    Stem   |  64    |   2    | initial downsampling
    Stage1 |  64    |   2    | local morphology
    Stage2 | 128    |   2    | beat-level patterns
    Stage3 | 256    |   2    | rhythm features
    Stage4 | d_model|   2    | high-level representation
    → total stride = 2^5 = 32  ⟹  5000 / 32 = 156 tokens at 500 Hz
                                   1000 / 32 =  31 tokens at 100 Hz
    """

    def __init__(self, d_model: int = 256, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        dpr = [drop_path_rate * i / 7 for i in range(8)]  # 8 blocks total

        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            DSConvBlock(64,  64,  stride=2, drop_path_rate=dpr[0]),
            DSConvBlock(64,  64,  stride=1, drop_path_rate=dpr[1]),
        )
        self.stage2 = nn.Sequential(
            DSConvBlock(64,  128, stride=2, drop_path_rate=dpr[2]),
            DSConvBlock(128, 128, stride=1, drop_path_rate=dpr[3]),
        )
        self.stage3 = nn.Sequential(
            DSConvBlock(128, 256, stride=2, drop_path_rate=dpr[4]),
            DSConvBlock(256, 256, stride=1, drop_path_rate=dpr[5]),
        )
        self.stage4 = nn.Sequential(
            DSConvBlock(256, d_model, stride=2, drop_path_rate=dpr[6]),
            DSConvBlock(d_model, d_model, stride=1, drop_path_rate=dpr[7]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x   # (B, d_model, T)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings (preferred over sinusoidal for short seqs)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions)  # broadcast over B


# ─────────────────────────────────────────────────────────────────────────────
# CNN-Transformer Model
# ─────────────────────────────────────────────────────────────────────────────

class CNNTransformer(nn.Module):
    """
    CNN-Transformer hybrid for multi-label 12-lead ECG classification.

    Args:
        n_classes       : number of output classes (default 5 for PTB-XL+)
        d_model         : width of Transformer and CNN output (default 256)
        n_heads         : number of attention heads (default 8)
        n_layers        : number of Transformer encoder layers (default 4)
        ffn_dim         : FFN hidden size inside Transformer (default 1024)
        dropout         : dropout rate inside Transformer and head (default 0.1)
        drop_path_rate  : stochastic depth in CNN encoder (default 0.1)
        max_seq_len     : max token sequence length for positional encoding
        sampling_rate   : informational only (stored as attribute)
    """

    def __init__(
        self,
        n_classes: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        max_seq_len: int = 512,
        sampling_rate: int = 500,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate

        # ── CNN Encoder ───────────────────────────────────────────────────
        self.cnn = CNNEncoder(d_model=d_model, drop_path_rate=drop_path_rate)

        # ── Positional Encoding ───────────────────────────────────────────
        self.pos_enc = LearnablePositionalEncoding(d_model, max_len=max_seq_len)

        # ── Transformer Encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,      # (B, T, d_model) convention
            norm_first=True,       # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # ── Classification Head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._init_weights()

    # ─── Weight init ──────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─── Forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 12, L) — normalised ECG signal

        Returns:
            logits : (B, n_classes) — raw logits, no sigmoid
        """
        # 1. CNN feature extraction: (B, 12, L) → (B, d_model, T)
        x = self.cnn(x)

        # 2. Reshape for Transformer: (B, d_model, T) → (B, T, d_model)
        x = x.permute(0, 2, 1)

        # 3. Add positional encoding
        x = self.pos_enc(x)

        # 4. Transformer self-attention across T tokens
        x = self.transformer(x)   # (B, T, d_model)

        # 5. Global average pooling over time
        x = x.mean(dim=1)         # (B, d_model)

        # 6. Classify
        return self.head(x)       # (B, n_classes)

    # ─── Utility ──────────────────────────────────────────────────────────
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  CNN-Transformer smoke test")
    print("=" * 60)

    for sr, L in [(500, 5000), (100, 1000)]:
        model = CNNTransformer(n_classes=5, d_model=256, n_heads=8,
                               n_layers=4, sampling_rate=sr)
        model.eval()
        dummy = torch.randn(4, 12, L)
        with torch.no_grad():
            out = model(dummy)
        T = model.cnn(dummy).shape[-1]
        print(f"\n  sampling_rate={sr} Hz  |  input=({4}, 12, {L})")
        print(f"  CNN output tokens : {T}")
        print(f"  Model output      : {out.shape}  (expected (4, 5))")
        print(f"  Parameters        : {model.n_parameters:,}")
        assert out.shape == (4, 5), "Output shape mismatch!"

    print("\n  All checks passed.\n")
    sys.exit(0)