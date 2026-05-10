"""
Run the full PTB-XL+ preprocessing pipeline.

Usage:
    python scripts/preprocess.py \\
        --data_path data/raw/ \\
        --output_path data/processed/

What this does:
  1. Loads ptbxl_database.csv and partitions into train/val/test splits.
  2. Reads all training signals via PTBXLDataset to compute per-lead statistics.
  3. Fits a z-score normalizer on the training set only.
  4. Saves the normalizer statistics to config/norm_stats.npy.
  5. Computes and prints the class distribution + pos_weight for BCEWithLogitsLoss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PTBXLDataset
from data.labels import compute_pos_weight, print_class_distribution
from data.preprocessing import fit_normalizer, save_norm_stats
from utils.logger import get_logger

log = get_logger(__name__, log_file=Path("logs/preprocess.log"))


def collect_signals(dataset: PTBXLDataset, max_samples: int | None = None) -> np.ndarray:
    """
    Iterate the dataset and stack raw signals into a numpy array.

    Args:
        dataset:     PTBXLDataset instance (transform should be None).
        max_samples: If set, stop early after this many samples (useful for testing).

    Returns:
        Array of shape (N, 12, signal_length) in float32.
    """
    signals = []
    n = min(len(dataset), max_samples) if max_samples else len(dataset)

    for i in tqdm(range(n), desc="Loading signals"):
        x, _ = dataset[i]
        signals.append(x.numpy())

    return np.stack(signals)


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    norm_stats_path = Path(args.norm_stats_path)

    # Validate data is available
    required = ["ptbxl_database.csv", "scp_statements.csv", "records100"]
    for name in required:
        if not (data_path / name).exists():
            log.error(
                f"Missing: {data_path / name}. "
                "Run scripts/download_data.py for instructions."
            )
            sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Step 1: Build train dataset (no transforms, no norm) ──────────────
    log.info("Loading training split metadata …")
    train_ds = PTBXLDataset(data_path, split="train", transform=None, norm_stats=None)
    val_ds   = PTBXLDataset(data_path, split="val",   transform=None, norm_stats=None)
    test_ds  = PTBXLDataset(data_path, split="test",  transform=None, norm_stats=None)
    log.info(f"Split sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # ── Step 2: Compute normalizer on training signals ─────────────────────
    log.info("Collecting training signals to fit normalizer …")
    X_train = collect_signals(train_ds, max_samples=args.max_samples)
    log.info(f"Training signal array shape: {X_train.shape}")

    mean, std = fit_normalizer(X_train)
    log.info(f"Per-lead mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    log.info(f"Per-lead std  range: [{std.min():.4f},  {std.max():.4f}]")

    # ── Step 3: Save normalizer stats ──────────────────────────────────────
    save_norm_stats(mean, std, norm_stats_path)

    # ── Step 4: Class distribution + pos_weight ────────────────────────────
    log.info("Computing class distribution …")
    all_labels = train_ds.get_all_labels()
    print_class_distribution(all_labels)

    pos_weight = compute_pos_weight(all_labels)
    log.info(f"pos_weight (for BCEWithLogitsLoss): {pos_weight.tolist()}")

    # Save pos_weight alongside norm stats for use in train scripts
    pw_path = output_path / "pos_weight.npy"
    np.save(str(pw_path), pos_weight.numpy())
    log.info(f"Saved pos_weight to {pw_path}")

    log.info("Preprocessing complete.")
    print(
        f"\nDone! Artifacts written to:\n"
        f"  Normalizer stats : {norm_stats_path}\n"
        f"  pos_weight       : {pw_path}\n"
        f"\nNext step:\n"
        f"  python scripts/train_resnet.py "
        f"--data_path {data_path} --norm_stats {norm_stats_path}\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTB-XL+ preprocessing pipeline")
    p.add_argument("--data_path",   type=str, default="data/raw/",
                   help="Root directory of the PTB-XL+ dataset")
    p.add_argument("--output_path", type=str, default="data/processed/",
                   help="Directory for processed artifacts (pos_weight etc.)")
    p.add_argument("--norm_stats_path", type=str, default="config/norm_stats.npy",
                   help="Where to save the normalizer statistics")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit the number of training samples read (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
