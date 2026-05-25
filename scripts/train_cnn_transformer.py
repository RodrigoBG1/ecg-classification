"""
Entry point for CNN-Transformer training on PTB-XL+.

Usage (PowerShell):
    python scripts/train_cnn_transformer.py `
        --data_path "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" `
        --norm_stats config/norm_stats.npy `
        --pos_weight data/processed/pos_weight.npy `
        --sampling_rate 500

Run scripts/preprocess.py first to generate norm_stats.npy and pos_weight.npy.
These artifacts are shared with the ResNet-1D run — no need to regenerate them.
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

from data.dataset import PTBXLDataset
from data.preprocessing import default_train_transforms, load_norm_stats
from models.cnn_transformer import CNNTransformer
from training.trainer import Trainer
from utils.logger import get_logger

log = get_logger(__name__, log_file=Path("logs/train_cnn_transformer.log"))


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_cfg(config: dict) -> dict:
    """
    Read CNN-Transformer hyperparameters from config.yaml.
    Falls back to sensible defaults so the run works even if the key is absent.
    """
    defaults = {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "ffn_dim": 1024,
        "dropout": 0.1,
        "drop_path_rate": 0.1,
        "n_classes": 5,
    }
    return {**defaults, **config.get("cnn_transformer", {})}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    Path("logs").mkdir(exist_ok=True)

    config    = load_config(Path(args.config))
    data_cfg  = config["data"]
    train_cfg = config["training"]
    model_cfg = get_model_cfg(config)

    sr         = args.sampling_rate or data_cfg["sampling_rate"]
    sig_length = 5000 if sr == 500 else 1000

    # ── Normalizer stats (shared with ResNet run) ──────────────────────────
    norm_stats_path = Path(args.norm_stats)
    if not norm_stats_path.exists():
        log.error(
            f"Normalizer stats not found at {norm_stats_path}. "
            f"Run: python scripts/preprocess.py --data_path <data_path> --sampling_rate {sr}"
        )
        sys.exit(1)
    norm_stats = load_norm_stats(norm_stats_path)
    log.info(f"Loaded normalizer stats from {norm_stats_path}")

    # ── pos_weight (shared with ResNet run) ───────────────────────────────
    pos_weight: torch.Tensor | None = None
    if args.pos_weight and Path(args.pos_weight).exists():
        pw_array   = np.load(args.pos_weight)
        pos_weight = torch.from_numpy(pw_array).float()
        log.info(f"Loaded pos_weight: {pos_weight.tolist()}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    data_path        = Path(args.data_path)
    train_transforms = default_train_transforms(signal_length=sig_length)

    train_ds = PTBXLDataset(
        data_path, split="train",
        sampling_rate=sr,
        transform=train_transforms,
        norm_stats=norm_stats,
    )
    val_ds = PTBXLDataset(
        data_path, split="val",
        sampling_rate=sr,
        transform=None,
        norm_stats=norm_stats,
    )
    log.info(
        f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples  "
        f"sampling_rate={sr} Hz  signal_length={sig_length}"
    )

    num_workers  = train_cfg["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = CNNTransformer(
        n_classes      = model_cfg["n_classes"],
        d_model        = model_cfg["d_model"],
        n_heads        = model_cfg["n_heads"],
        n_layers       = model_cfg["n_layers"],
        ffn_dim        = model_cfg["ffn_dim"],
        dropout        = model_cfg["dropout"],
        drop_path_rate = model_cfg["drop_path_rate"],
        sampling_rate  = sr,
    )
    log.info(f"CNN-Transformer parameters: {model.n_parameters:,}")

    # ── Trainer (same as ResNet — reuse existing infrastructure) ──────────
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        pos_weight     = pos_weight,
        checkpoint_dir = Path(args.checkpoint_dir),
        learning_rate  = train_cfg["learning_rate"],
        weight_decay   = train_cfg["weight_decay"],
        mixup_alpha    = train_cfg.get("mixup_alpha", 0.4),
        grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0),
        label_smoothing= train_cfg.get("label_smoothing", 0.1),
        warmup_epochs  = train_cfg.get("warmup_epochs", 5),
    )

    n_epochs = args.epochs or train_cfg["epochs"]
    log.info(f"Starting training for {n_epochs} epochs …")
    history = trainer.fit(n_epochs=n_epochs)

    best = max(history, key=lambda r: r["auc_roc_macro"])
    log.info(
        f"\nTraining complete. "
        f"Best val AUC={best['auc_roc_macro']:.4f}  "
        f"F1={best['f1_macro']:.4f}  "
        f"at epoch {int(best['epoch'])}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN-Transformer on PTB-XL+")
    p.add_argument("--data_path",      type=str, default="data/raw/")
    p.add_argument("--norm_stats",     type=str, default="config/norm_stats.npy")
    p.add_argument("--pos_weight",     type=str, default="data/processed/pos_weight.npy")
    p.add_argument("--config",         type=str, default="config/config.yaml")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/cnn_transformer")
    p.add_argument(
        "--sampling_rate", type=int, default=None, choices=[100, 500],
        help="Override sampling rate from config (100 or 500 Hz)"
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs from config"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())