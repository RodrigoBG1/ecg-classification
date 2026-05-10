"""Normalization and augmentation transforms for ECG signals."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def fit_normalizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-lead mean and std from the training set.

    Args:
        X_train: array of shape (N, 12, 1000).

    Returns:
        mean: shape (12, 1)
        std:  shape (12, 1)
    """
    mean = X_train.mean(axis=(0, 2), keepdims=False)[:, np.newaxis]  # (12, 1)
    std = X_train.std(axis=(0, 2), keepdims=False)[:, np.newaxis]    # (12, 1)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def normalize(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization using pre-computed stats."""
    return (X - mean) / std


def save_norm_stats(mean: np.ndarray, std: np.ndarray, path: Path) -> None:
    """Save normalizer statistics to a .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), {"mean": mean, "std": std})
    print(f"Saved normalizer stats to {path}")


def load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalizer statistics from a .npy file."""
    data = np.load(str(path), allow_pickle=True).item()
    return data["mean"], data["std"]


# ---------------------------------------------------------------------------
# Augmentation transforms  (torchvision-style callable classes)
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Add per-sample Gaussian noise to an ECG tensor."""

    def __init__(self, sigma: float = 0.02) -> None:
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (12, 1000).
        Returns:
            Noisy tensor of the same shape.
        """
        noise = torch.randn_like(x) * self.sigma
        return x + noise

    def __repr__(self) -> str:
        return f"GaussianNoise(sigma={self.sigma})"


class RandomAmplitudeScale:
    """Multiply the entire signal by a random scalar in [low, high]."""

    def __init__(self, low: float = 0.8, high: float = 1.2) -> None:
        self.low = low
        self.high = high

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (12, 1000).
        Returns:
            Scaled tensor of the same shape.
        """
        scale = random.uniform(self.low, self.high)
        return x * scale

    def __repr__(self) -> str:
        return f"RandomAmplitudeScale(low={self.low}, high={self.high})"


class RandomCrop:
    """
    Randomly crop a contiguous segment of length `crop_size` then
    zero-pad back to `output_size`.

    This simulates shorter recordings without changing the tensor shape
    expected by downstream convolutions.
    """

    def __init__(self, crop_size: int = 900, output_size: int = 1000) -> None:
        if crop_size > output_size:
            raise ValueError("crop_size must be <= output_size")
        self.crop_size = crop_size
        self.output_size = output_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (12, L) where L >= crop_size.
        Returns:
            Tensor of shape (12, output_size) with the cropped segment
            left-aligned and zero-padded on the right.
        """
        L = x.shape[-1]
        max_start = max(0, L - self.crop_size)
        start = random.randint(0, max_start)
        cropped = x[..., start : start + self.crop_size]

        out = torch.zeros(*x.shape[:-1], self.output_size, dtype=x.dtype)
        out[..., : self.crop_size] = cropped
        return out

    def __repr__(self) -> str:
        return f"RandomCrop(crop_size={self.crop_size}, output_size={self.output_size})"


class Compose:
    """Chain multiple transforms sequentially."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        lines = ["Compose("]
        for t in self.transforms:
            lines.append(f"  {t},")
        lines.append(")")
        return "\n".join(lines)


def default_train_transforms() -> Compose:
    """Return the standard augmentation pipeline used during training."""
    return Compose([
        GaussianNoise(sigma=0.02),
        RandomAmplitudeScale(low=0.8, high=1.2),
        RandomCrop(crop_size=900, output_size=1000),
    ])


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(42)
    X_fake = rng.standard_normal((10, 12, 1000)).astype(np.float32)

    mean, std = fit_normalizer(X_fake)
    print(f"mean shape: {mean.shape}, std shape: {std.shape}")

    X_normed = normalize(X_fake, mean, std)
    print(f"Normalized mean≈0: {np.abs(X_normed.mean(axis=(0,2))).max():.4f}")

    x = torch.from_numpy(X_fake[0])
    pipeline = default_train_transforms()
    out = pipeline(x)
    assert out.shape == (12, 1000), f"Unexpected shape: {out.shape}"
    print(f"Transform output shape: {out.shape}")
    print("preprocessing.py smoke test passed.")
