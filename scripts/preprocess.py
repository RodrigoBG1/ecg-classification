"""
Run the full PTB-XL+ preprocessing pipeline.

Usage:
    python scripts/preprocess.py \\
        --data_path data/raw/ \\
        --output_path data/processed/ \\
        --sampling_rate 500

What this does:
  1. Loads ptbxl_database.csv and partitions into train/val/test splits.
  2. Computes per-lead z-score statistics on the training set using a
     streaming algorithm (constant memory — no loading the full 4 GB array).
  3. Saves the normalizer statistics to config/norm_stats.npy.
  4. Computes and saves the class distribution + pos_weight for BCEWithLogitsLoss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PTBXLDataset
from data.labels import compute_pos_weight, print_class_distribution
from data.preprocessing import save_norm_stats
from utils.logger import get_logger

log = get_logger(__name__, log_file=Path("logs/preprocess.log"))


def fit_normalizer_streaming(
    dataset: PTBXLDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-lead mean and std over the dataset without loading all signals
    into memory (O(1) memory, O(N) passes).

    At 500 Hz the full training set is ~4 GB; this approach keeps memory
    usage flat regardless of sampling rate.

    Returns:
        mean: shape (12, 1), float32
        std:  shape (12, 1), float32
    """
    n_leads = 12
    running_sum = np.zeros(n_leads, dtype=np.float64)
    running_sq  = np.zeros(n_leads, dtype=np.float64)
    total_pts   = 0

    for i in tqdm(range(len(dataset)), desc="Computing normalizer stats"):
        x, _ = dataset[i]
        x_np = x.numpy().astype(np.float64)   # (12, L)
        running_sum += x_np.sum(axis=1)
        running_sq  += (x_np ** 2).sum(axis=1)
        total_pts   += x_np.shape[1]

    mean     = running_sum / total_pts
    variance = running_sq / total_pts - mean ** 2
    std      = np.sqrt(np.clip(variance, 1e-8, None))

    mean = mean[:, np.newaxis].astype(np.float32)   # (12, 1)
    std  = std[:, np.newaxis].astype(np.float32)    # (12, 1)
    return mean, std


def main(args: argparse.Namespace) -> None:
    data_path      = Path(args.data_path)
    output_path    = Path(args.output_path)
    norm_stats_path = Path(args.norm_stats_path)
    sr             = args.sampling_rate

    # Validate required files/folders exist
    required = ["ptbxl_database.csv", "scp_statements.csv", f"records{sr}"]
    for name in required:
        if not (data_path / name).exists():
            log.error(
                f"Missing: {data_path / name}. "
                "Run scripts/download_data.py for instructions."
            )
            sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Step 1: Build datasets ─────────────────────────────────────────
    log.info(f"Loading split metadata (sampling_rate={sr}) …")
    train_ds = PTBXLDataset(data_path, split="train", sampling_rate=sr, transform=None, norm_stats=None)
    val_ds   = PTBXLDataset(data_path, split="val",   sampling_rate=sr, transform=None, norm_stats=None)
    test_ds  = PTBXLDataset(data_path, split="test",  sampling_rate=sr, transform=None, norm_stats=None)
    log.info(f"Split sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # ── Step 2: Streaming normalizer on training signals ──────────────
    log.info("Fitting per-lead z-score normalizer (streaming, constant memory) …")
    mean, std = fit_normalizer_streaming(train_ds)
    log.info(f"Per-lead mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    log.info(f"Per-lead std  range: [{std.min():.4f},  {std.max():.4f}]")

    # ── Step 3: Save normalizer stats ──────────────────────────────────
    save_norm_stats(mean, std, norm_stats_path)

    # ── Step 4: Class distribution + pos_weight ────────────────────────
    log.info("Computing class distribution …")
    all_labels = train_ds.get_all_labels()
    print_class_distribution(all_labels)

    pos_weight = compute_pos_weight(all_labels)
    log.info(f"pos_weight (for BCEWithLogitsLoss): {pos_weight.tolist()}")

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
        f"--data_path {data_path} --norm_stats {norm_stats_path} "
        f"--sampling_rate {sr}\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTB-XL+ preprocessing pipeline")
    p.add_argument("--data_path",        type=str, default="data/raw/",
                   help="Root directory of the PTB-XL+ dataset")
    p.add_argument("--output_path",      type=str, default="data/processed/",
                   help="Directory for processed artifacts (pos_weight etc.)")
    p.add_argument("--norm_stats_path",  type=str, default="config/norm_stats.npy",
                   help="Where to save the normalizer statistics")
    p.add_argument("--sampling_rate",    type=int, default=500, choices=[100, 500],
                   help="ECG sampling rate: 100 or 500 Hz (default: 500)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
