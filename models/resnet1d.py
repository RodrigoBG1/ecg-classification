"""ResNet-1D with SE attention and stochastic depth for multi-label ECG classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DropPath(nn.Module):
    """Stochastic depth: randomly drop entire residual branches during training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(survival)
        return x * noise / survival


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D signals."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        s = self.gap(x).squeeze(-1)       # (B, C)
        s = self.fc(s).unsqueeze(-1)       # (B, C, 1)
        return x * s


class ResidualBlock1D(nn.Module):
    """
    Residual block with SE attention and optional stochastic depth.

    Conv(k=7) → BN → ReLU → Conv(k=7) → BN → SE → DropPath → + skip → ReLU

    Kernel size 7 gives a ~3.5× larger receptive field than k=3 per block,
    matching the low-frequency nature of ECG waveforms.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        drop_path: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, se_reduction)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.downsample: nn.Sequential | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


def _make_stage(
    in_ch: int,
    out_ch: int,
    n_blocks: int,
    stride: int,
    drop_path_rates: list[float],
) -> nn.Sequential:
    blocks: list[nn.Module] = [
        ResidualBlock1D(in_ch, out_ch, stride=stride, drop_path=drop_path_rates[0])
    ]
    for dp in drop_path_rates[1:]:
        blocks.append(ResidualBlock1D(out_ch, out_ch, stride=1, drop_path=dp))
    return nn.Sequential(*blocks)


class ResNet1D(nn.Module):
    """
    ResNet-1D backbone with SE attention and stochastic depth.

    Input:  (B, 12, L)  where L=5000 at 500 Hz or L=1000 at 100 Hz.
    Output: (B, n_classes) — raw logits (apply sigmoid for probabilities).

    Architecture (500 Hz, L=5000):
        Stem  : Conv(12→64, k=15, s=2) → BN → ReLU → MaxPool(k=4, s=4) → 625
        Stage1: 2×Block(64→64,   s=1)  → 625
        Stage2: 2×Block(64→128,  s=2)  → 313
        Stage3: 2×Block(128→256, s=2)  → 157
        Stage4: 2×Block(256→512, s=2)  → 79
        Head  : GAP → Dropout(0.3) → Linear(512, n_classes)

    At 100 Hz (L=1000) the MaxPool is replaced with Identity, giving ~4× fewer
    temporal positions at the head — still sufficient for GAP.
    """

    _CHANNELS  = [64, 128, 256, 512]
    _N_BLOCKS  = [2, 2, 2, 2]

    def __init__(
        self,
        n_classes: int = 5,
        drop_path_rate: float = 0.2,
        sampling_rate: int = 500,
    ) -> None:
        super().__init__()

        total_blocks = sum(self._N_BLOCKS)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        idx = 0

        # Stem: at 500 Hz we need extra downsampling so the temporal dimension
        # entering stage1 is comparable to the 100 Hz case.
        if sampling_rate == 500:
            # 5000 → stride=2 → 2500 → maxpool(k=4,s=4) → 625
            stem_pool: nn.Module = nn.MaxPool1d(kernel_size=4, stride=4)
        else:
            # 1000 → stride=2 → 500 (no additional pool needed)
            stem_pool = nn.Identity()

        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            stem_pool,
        )

        ch = self._CHANNELS
        nb = self._N_BLOCKS

        self.stage1 = _make_stage(ch[0], ch[0], nb[0], stride=1, drop_path_rates=dp_rates[idx:idx+nb[0]]); idx += nb[0]
        self.stage2 = _make_stage(ch[0], ch[1], nb[1], stride=2, drop_path_rates=dp_rates[idx:idx+nb[1]]); idx += nb[1]
        self.stage3 = _make_stage(ch[1], ch[2], nb[2], stride=2, drop_path_rates=dp_rates[idx:idx+nb[2]]); idx += nb[2]
        self.stage4 = _make_stage(ch[2], ch[3], nb[3], stride=2, drop_path_rates=dp_rates[idx:idx+nb[3]])

        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc      = nn.Linear(ch[3], n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for sr, L in [(500, 5000), (100, 1000)]:
        model = ResNet1D(n_classes=5, sampling_rate=sr)
        x = torch.randn(4, 12, L)
        logits = model(x)
        assert logits.shape == (4, 5), f"Bad shape: {logits.shape}"
        print(f"[{sr} Hz] input={x.shape}  output={logits.shape}  params={model.n_parameters:,}")
    print("resnet1d.py smoke test passed.")
