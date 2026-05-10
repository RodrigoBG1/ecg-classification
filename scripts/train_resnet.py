"""
Entry point for ResNet-1D training on PTB-XL+.

Usage:
    python scripts/train_resnet.py \\
        --data_path data/raw/ \\
        --norm_stats config/norm_stats.npy \\
        --pos_weight data/processed/pos_weight.npy

Run scripts/preprocess.py first to generate norm_stats.npy and pos_weight.npy.
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
from models.resnet1d import ResNet1D
from training.trainer import Trainer
from utils.logger import get_logger

log = get_logger(__name__, log_file=Path("logs/train_resnet.log"))


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    Path("logs").mkdir(exist_ok=True)

    config = load_config(Path(args.config))
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    # ── Normalizer stats ───────────────────────────────────────────────────
    norm_stats_path = Path(args.norm_stats)
    if not norm_stats_path.exists():
        log.error(
            f"Normalizer stats not found at {norm_stats_path}. "
            "Run: python scripts/preprocess.py --data_path <data_path>"
        )
        sys.exit(1)
    norm_stats = load_norm_stats(norm_stats_path)
    log.info(f"Loaded normalizer stats from {norm_stats_path}")

    # ── pos_weight ─────────────────────────────────────────────────────────
    pos_weight: torch.Tensor | None = None
    if args.pos_weight and Path(args.pos_weight).exists():
        pw_array = np.load(args.pos_weight)
        pos_weight = torch.from_numpy(pw_array).float()
        log.info(f"Loaded pos_weight: {pos_weight.tolist()}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    data_path = Path(args.data_path)
    train_transforms = default_train_transforms()

    train_ds = PTBXLDataset(
        data_path, split="train",
        transform=train_transforms,
        norm_stats=norm_stats,
    )
    val_ds = PTBXLDataset(
        data_path, split="val",
        transform=None,
        norm_stats=norm_stats,
    )
    log.info(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ResNet1D(n_classes=model_cfg["n_classes"])
    log.info(f"ResNet-1D parameters: {model.n_parameters:,}")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pos_weight=pos_weight,
        checkpoint_dir=Path(args.checkpoint_dir),
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    log.info(f"Starting training for {args.epochs or train_cfg['epochs']} epochs …")
    history = trainer.fit(n_epochs=args.epochs or train_cfg["epochs"])

    # Summary
    best = max(history, key=lambda r: r["f1_macro"])
    log.info(
        f"\nTraining complete. Best val macro F1: {best['f1_macro']:.4f} "
        f"at epoch {int(best['epoch'])}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet-1D on PTB-XL+")
    p.add_argument("--data_path",      type=str, default="data/raw/")
    p.add_argument("--norm_stats",     type=str, default="config/norm_stats.npy")
    p.add_argument("--pos_weight",     type=str, default="data/processed/pos_weight.npy")
    p.add_argument("--config",         type=str, default="config/config.yaml")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/resnet1d")
    p.add_argument("--epochs",         type=int, default=None,
                   help="Override epochs from config")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
