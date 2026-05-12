"""Evaluation metrics for multi-label ECG classification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def compute_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred_logits: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute classification metrics for multi-label ECG classification.

    Args:
        y_true:         Binary ground-truth array of shape (N, 5).
        y_pred_logits:  Raw model logits of shape (N, 5).
        threshold:      Decision threshold applied after sigmoid.

    Returns:
        Dictionary with keys:
            auc_roc_macro, accuracy_exact, hamming_accuracy,
            precision_macro, recall_macro, f1_macro,
            and per-class auc_<CLASS>, f1_<CLASS>, precision_<CLASS>, recall_<CLASS>.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_logits, torch.Tensor):
        y_pred_logits = y_pred_logits.cpu().numpy()

    y_true = y_true.astype(int)
    y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))   # sigmoid
    y_pred = (y_prob >= threshold).astype(int)

    shared_kwargs = dict(zero_division=0)

    # --- AUC-ROC (primary metric in ECG literature; threshold-independent) ---
    try:
        auc_macro = float(roc_auc_score(y_true, y_prob, average="macro"))
        auc_per   = roc_auc_score(y_true, y_prob, average=None)
    except ValueError:
        auc_macro = 0.0
        auc_per   = np.zeros(len(CLASSES))

    metrics: dict[str, float] = {
        "auc_roc_macro":   auc_macro,
        "accuracy_exact":  float(accuracy_score(y_true, y_pred)),
        "hamming_accuracy": float(1.0 - hamming_loss(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **shared_kwargs)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", **shared_kwargs)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", **shared_kwargs)),
    }

    # Per-class metrics
    f1_per   = f1_score(y_true, y_pred, average=None, **shared_kwargs)
    prec_per = precision_score(y_true, y_pred, average=None, **shared_kwargs)
    rec_per  = recall_score(y_true, y_pred, average=None, **shared_kwargs)

    for i, cls in enumerate(CLASSES):
        metrics[f"auc_{cls}"]       = float(auc_per[i])
        metrics[f"f1_{cls}"]        = float(f1_per[i])
        metrics[f"precision_{cls}"] = float(prec_per[i])
        metrics[f"recall_{cls}"]    = float(rec_per[i])

    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Return a compact single-line string of the key aggregate metrics."""
    return (
        f"AUC={metrics['auc_roc_macro']:.4f}  "
        f"F1={metrics['f1_macro']:.4f}  "
        f"hamming={metrics['hamming_accuracy']:.4f}  "
        f"prec={metrics['precision_macro']:.4f}  "
        f"rec={metrics['recall_macro']:.4f}"
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(128, 5)).astype(float)
    y_logits = rng.standard_normal((128, 5))
    m = compute_metrics(y_true, y_logits)
    print(format_metrics(m))
    for cls in CLASSES:
        print(f"  {cls}: AUC={m[f'auc_{cls}']:.3f}  F1={m[f'f1_{cls}']:.3f}  "
              f"P={m[f'precision_{cls}']:.3f}  R={m[f'recall_{cls}']:.3f}")
    print("metrics.py smoke test passed.")
