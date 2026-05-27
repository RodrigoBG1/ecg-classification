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
from models.patchtst_opt import PatchTST
from training.trainer import Trainer
from utils.logger import get_logger

log = get_logger(__name__, log_file=Path("logs/train_patchtst.log"))


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    Path("logs").mkdir(exist_ok=True)

    config    = load_config(Path(args.config))
    data_cfg  = config["data"]
    train_cfg = config["training"]
    model_cfg = config.get("patchtst", config["model"])

    sr         = args.sampling_rate or data_cfg["sampling_rate"]
    sig_length = 5000 if sr == 500 else 1000

    # ── Patch size: 50 samples = 100 ms at 500 Hz; 10 samples at 100 Hz ──
    patch_size = args.patch_size or (50 if sr == 500 else 10)

    # ── Normalizer stats ──────────────────────────────────────────────────
    norm_stats_path = Path(args.norm_stats)
    if not norm_stats_path.exists():
        log.error(
            f"Normalizer stats not found at {norm_stats_path}. "
            f"Run: python scripts/preprocess.py --data_path <data_path> --sampling_rate {sr}"
        )
        sys.exit(1)
    norm_stats = load_norm_stats(norm_stats_path)
    log.info(f"Loaded normalizer stats from {norm_stats_path}")

    # ── pos_weight ────────────────────────────────────────────────────────
    pos_weight: torch.Tensor | None = None
    if args.pos_weight and Path(args.pos_weight).exists():
        pw_array = np.load(args.pos_weight)
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
    # PatchTST transformers can be memory-heavy; halve batch if not overridden
    batch_size   = args.batch_size or train_cfg["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = PatchTST(
        n_classes     = model_cfg.get("n_classes",   5),
        n_leads       = data_cfg.get("n_leads",     12),
        signal_length = sig_length,
        patch_size    = patch_size,
        stride        = patch_size,          # non-overlapping patches
        d_model       = model_cfg.get("d_model",   256),
        n_heads       = model_cfg.get("n_heads",     8),
        n_layers      = model_cfg.get("n_layers",    6),
        ff_dim        = model_cfg.get("ff_dim",   1024),
        attn_dropout  = model_cfg.get("attn_dropout", 0.1),
        dropout       = model_cfg.get("dropout",    0.1),
    )
    log.info(
        f"PatchTST: patch_size={patch_size}  n_patches={model.n_patches}  "
        f"d_model={model.d_model}  parameters={model.n_parameters:,}"
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    # Transformers benefit from a lower LR and a longer warmup than CNNs.
    lr           = args.lr           or train_cfg.get("patchtst_lr",       3e-4)
    warmup       = args.warmup       or train_cfg.get("patchtst_warmup",     10)
    weight_decay = args.weight_decay or train_cfg.get("weight_decay",      1e-4)

    trainer = Trainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        pos_weight      = pos_weight,
        checkpoint_dir  = Path(args.checkpoint_dir),
        learning_rate   = lr,
        weight_decay    = weight_decay,
        mixup_alpha     = train_cfg.get("mixup_alpha", 0.4),
        grad_clip_norm  = train_cfg.get("grad_clip_norm", 1.0),
        label_smoothing = train_cfg.get("label_smoothing", 0.1),
        warmup_epochs   = warmup,
    )

    n_epochs = args.epochs or train_cfg["epochs"]
    log.info(f"Starting training for {n_epochs} epochs  lr={lr:.1e}  warmup={warmup}")
    history = trainer.fit(n_epochs=n_epochs)

    best = max(history, key=lambda r: r["auc_roc_macro"])
    log.info(
        f"\nTraining complete. Best val AUC={best['auc_roc_macro']:.4f}  "
        f"F1={best['f1_macro']:.4f}  at epoch {int(best['epoch'])}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PatchTST on PTB-XL+")
    p.add_argument("--data_path",      type=str,   default="data/raw/")
    p.add_argument("--norm_stats",     type=str,   default="config/norm_stats.npy")
    p.add_argument("--pos_weight",     type=str,   default="data/processed/pos_weight.npy")
    p.add_argument("--config",         type=str,   default="config/config.yaml")
    p.add_argument("--checkpoint_dir", type=str,   default="checkpoints/patchtst")
    p.add_argument("--sampling_rate",  type=int,   default=None, choices=[100, 500])
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch_size",     type=int,   default=None,
                   help="Override batch_size from config (reduce if OOM)")
    p.add_argument("--patch_size",     type=int,   default=None,
                   help="Patch size in samples (default: 50 at 500 Hz)")
    p.add_argument("--lr",             type=float, default=None,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--warmup",         type=int,   default=None,
                   help="Warmup epochs (default: 10)")
    p.add_argument("--weight_decay",   type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
