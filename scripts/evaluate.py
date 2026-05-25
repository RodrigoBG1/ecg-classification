"""
Evaluate a trained model on the PTB-XL+ test set.

Usage (PowerShell):
    python scripts/evaluate.py `
        --data_path "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" `
        --checkpoint checkpoints/cnn_transformer/best_model.pt `
        --norm_stats config/norm_stats.npy `
        --sampling_rate 500 `
        --model cnn_transformer

    python scripts/evaluate.py `
        --data_path "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" `
        --checkpoint checkpoints/resnet1d/best_model.pt `
        --norm_stats config/norm_stats.npy `
        --sampling_rate 500 `
        --model resnet1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from data.dataset import PTBXLDataset
from data.preprocessing import load_norm_stats
from utils.logger import get_logger

log = get_logger(__name__, log_file=Path("logs/evaluate.log"))

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


def compute_test_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute metrics directly here, avoiding the accuracy_score multilabel bug."""
    probs  = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    metrics = {}

    # Macro AUC-ROC
    metrics["auc_roc_macro"] = float(roc_auc_score(y_true, probs, average="macro"))

    # Per-class AUC
    for i, name in enumerate(CLASS_NAMES):
        try:
            metrics[f"auc_roc_{name}"] = float(roc_auc_score(y_true[:, i], probs[:, i]))
        except Exception:
            metrics[f"auc_roc_{name}"] = float("nan")

    # F1, precision, recall (macro)
    metrics["f1_macro"]        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"]    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Hamming accuracy  (fraction of correct individual labels)
    metrics["hamming_accuracy"] = float((y_pred == y_true).mean())

    return metrics


def load_model(model_name: str, checkpoint_path: Path, config: dict, sr: int):
    model_cfg = config.get("model", {})

    if model_name == "resnet1d":
        from models.resnet1d import ResNet1D
        model = ResNet1D(
            n_classes=model_cfg.get("n_classes", 5),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.2),
            sampling_rate=sr,
        )
    elif model_name == "cnn_transformer":
        from models.cnn_transformer import CNNTransformer
        cfg = {
            "n_classes": 5, "d_model": 256, "n_heads": 8,
            "n_layers": 4, "ffn_dim": 1024, "dropout": 0.1,
            "drop_path_rate": 0.1,
        }
        cfg.update(config.get("cnn_transformer", {}))
        model = CNNTransformer(
            n_classes=cfg["n_classes"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            ffn_dim=cfg["ffn_dim"], dropout=cfg["dropout"],
            drop_path_rate=cfg["drop_path_rate"], sampling_rate=sr,
        )
    else:
        log.error(f"Unknown model: {model_name}. Choose 'resnet1d' or 'cnn_transformer'.")
        sys.exit(1)

    state = torch.load(checkpoint_path, map_location="cpu")
    # support both raw state_dict and wrapped checkpoint
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits = torch.cat(all_logits)   # (N, 5)
    labels = torch.cat(all_labels)   # (N, 5)
    return compute_test_metrics(logits, labels)


def main(args: argparse.Namespace) -> None:
    Path("logs").mkdir(exist_ok=True)

    with open(args.config) as f:
        import yaml
        config = yaml.safe_load(f)

    sr = args.sampling_rate or config["data"]["sampling_rate"]
    norm_stats = load_norm_stats(Path(args.norm_stats))

    # ── Test dataset ──────────────────────────────────────────────────────
    test_ds = PTBXLDataset(
        Path(args.data_path), split="test",
        sampling_rate=sr,
        transform=None,
        norm_stats=norm_stats,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    log.info(f"Test set: {len(test_ds)} samples  |  model: {args.model}  |  sr={sr} Hz")

    # ── Load model ────────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model, checkpoint_path, config, sr)
    model  = model.to(device)
    log.info(f"Loaded checkpoint from {checkpoint_path}  ->  device={device}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TEST SET RESULTS  —  {args.model.upper()}")
    print("=" * 60)
    print(f"  Macro AUC-ROC : {metrics['auc_roc_macro']:.4f}")
    print(f"  Macro F1      : {metrics['f1_macro']:.4f}")
    print(f"  Hamming Acc   : {metrics.get('hamming_accuracy', float('nan')):.4f}")
    print(f"  Precision     : {metrics.get('precision_macro', float('nan')):.4f}")
    print(f"  Recall        : {metrics.get('recall_macro', float('nan')):.4f}")
    print("-" * 60)
    print("  Per-class AUC:")
    for i, name in enumerate(CLASS_NAMES):
        key = f"auc_roc_{name}"
        if key in metrics:
            print(f"    {name:6s}  {metrics[key]:.4f}")
    print("=" * 60 + "\n")

    log.info(
        f"Test results — AUC={metrics['auc_roc_macro']:.4f}  "
        f"F1={metrics['f1_macro']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained ECG model on the test set")
    p.add_argument("--data_path",      type=str, required=True)
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--norm_stats",     type=str, default="config/norm_stats.npy")
    p.add_argument("--config",         type=str, default="config/config.yaml")
    p.add_argument("--model",          type=str, required=True,
                   choices=["resnet1d", "cnn_transformer"],
                   help="Which model architecture to load")
    p.add_argument("--sampling_rate",  type=int, default=None, choices=[100, 500])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())