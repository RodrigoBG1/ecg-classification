"""ResNet-1D for multi-label ECG classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock1D(nn.Module):
    """
    Basic residual block for 1-D signals.

    Applies: Conv → BN → ReLU → Conv → BN, then adds a skip connection.
    When in_channels != out_channels or stride > 1 a 1×1 projection
    is used to match dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
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

        # Projection shortcut when shape changes
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet1D(nn.Module):
    """
    ResNet-1D backbone for multi-label ECG classification.

    Input:  (B, 12, 1000) — 12-lead ECG at 100 Hz
    Output: (B, n_classes) — raw logits (apply sigmoid for probabilities)

    Architecture:
        Stem  : Conv1d(12, 64, k=15, s=2) → BN → ReLU
        Block1: 64  → 64   (stride 1)
        Block2: 64  → 128  (stride 2)
        Block3: 128 → 256  (stride 2)
        Block4: 256 → 512  (stride 2)
        Head  : GlobalAvgPool → Linear(512, n_classes)
    """

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        self.layer1 = ResidualBlock1D(64, 64,   stride=1)
        self.layer2 = ResidualBlock1D(64, 128,  stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)
        self.layer4 = ResidualBlock1D(256, 512, stride=2)

        # Classifier head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, n_classes)

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
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 12, 1000)
        Returns:
            logits: (B, n_classes)
        """
        x = self.stem(x)       # (B, 64, 500)
        x = self.layer1(x)     # (B, 64, 500)
        x = self.layer2(x)     # (B, 128, 250)
        x = self.layer3(x)     # (B, 256, 125)
        x = self.layer4(x)     # (B, 512, 63)
        x = self.gap(x)        # (B, 512, 1)
        x = x.squeeze(-1)      # (B, 512)
        return self.fc(x)      # (B, n_classes)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ResNet1D(n_classes=5)
    print(f"Parameters: {model.n_parameters:,}")

    x = torch.randn(4, 12, 1000)
    logits = model(x)
    assert logits.shape == (4, 5), f"Unexpected output shape: {logits.shape}"
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")
    print("resnet1d.py smoke test passed.")
