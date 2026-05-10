"""SCP code → 5-class binary label vector for PTB-XL+."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}


def _parse_scp_codes(raw: str) -> dict[str, float]:
    """Convert the CSV string-dict representation to a Python dict."""
    return ast.literal_eval(raw)


def load_scp_statements(scp_path: Path) -> dict[str, str]:
    """Return mapping from SCP code → diagnostic_class string."""
    df = pd.read_csv(scp_path, index_col=0)
    return {
        code: str(row["diagnostic_class"]).strip()
        for code, row in df.iterrows()
        if pd.notna(row.get("diagnostic_class", float("nan")))
    }


def build_label_vector(scp_codes_raw: str, code_to_class: dict[str, str]) -> torch.Tensor:
    """
    Parse one record's scp_codes string and return a float32 binary vector
    of shape (5,) corresponding to [NORM, MI, STTC, CD, HYP].

    A class is positive if *any* SCP code with confidence > 0 maps to it.
    """
    label = torch.zeros(len(CLASSES), dtype=torch.float32)
    try:
        codes: dict[str, float] = _parse_scp_codes(scp_codes_raw)
    except (ValueError, SyntaxError):
        return label

    for code, confidence in codes.items():
        if confidence <= 0:
            continue
        diag_class = code_to_class.get(code)
        if diag_class in CLASS_INDEX:
            label[CLASS_INDEX[diag_class]] = 1.0

    return label


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """
    Compute per-class pos_weight = (N - n_pos) / n_pos for BCEWithLogitsLoss.

    Args:
        labels: float array of shape (N, 5) with binary values.

    Returns:
        Tensor of shape (5,) suitable for BCEWithLogitsLoss(pos_weight=...).
    """
    n_pos = labels.sum(axis=0).clip(min=1)
    n_neg = len(labels) - n_pos
    weights = n_neg / n_pos
    return torch.tensor(weights, dtype=torch.float32)


def print_class_distribution(labels: np.ndarray) -> None:
    """Print per-class positive counts and frequencies."""
    n = len(labels)
    print(f"\nClass distribution (N={n}):")
    print(f"  {'Class':<8} {'Positives':>10} {'Frequency':>10}  pos_weight")
    pos_weight = compute_pos_weight(labels)
    for i, cls in enumerate(CLASSES):
        n_pos = int(labels[:, i].sum())
        freq = n_pos / n
        print(f"  {cls:<8} {n_pos:>10} {freq:>10.3f}  {pos_weight[i]:.2f}")
    print()


if __name__ == "__main__":
    # Smoke test with synthetic data
    fake_scp = "{'NDT': 100.0, 'NST_': 0.0, 'NORM': 100.0}"
    fake_code_to_class = {"NDT": "STTC", "NORM": "NORM", "NST_": "NORM"}
    vec = build_label_vector(fake_scp, fake_code_to_class)
    print(f"Label vector: {vec}")
    assert vec[CLASS_INDEX["NORM"]] == 1.0
    assert vec[CLASS_INDEX["STTC"]] == 1.0
    assert vec[CLASS_INDEX["MI"]] == 0.0

    fake_labels = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 1, 0, 0, 0]], dtype=float)
    print_class_distribution(fake_labels)
    pw = compute_pos_weight(fake_labels)
    print(f"pos_weight: {pw}")
    print("labels.py smoke test passed.")
