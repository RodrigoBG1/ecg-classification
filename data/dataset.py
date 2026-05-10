"""PTB-XL+ Dataset — torch.utils.data.Dataset implementation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import wfdb

from data.labels import build_label_vector, load_scp_statements


SPLIT_FOLDS: dict[str, list[int]] = {
    "train": list(range(1, 9)),  # folds 1-8
    "val": [9],
    "test": [10],
}


class PTBXLDataset(Dataset):
    """
    Loads ECG signals and multi-label targets from the PTB-XL+ dataset.

    Args:
        data_path:    Root directory containing ptbxl_database.csv,
                      scp_statements.csv, and records100/.
        split:        One of 'train', 'val', 'test'.
        sampling_rate: 100 (default) or 500.  Determines which subfolder
                       is read (records100 or records500).
        transform:    Optional callable applied to the (12, L) float32 tensor.
                      Applied only when split == 'train'.
        norm_stats:   Optional (mean, std) arrays of shape (12, 1) for
                      z-score normalization. If None, signals are returned as-is.
    """

    def __init__(
        self,
        data_path: Path | str,
        split: str = "train",
        sampling_rate: int = 100,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        norm_stats: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        if split not in SPLIT_FOLDS:
            raise ValueError(f"split must be one of {list(SPLIT_FOLDS)}, got '{split}'")

        self.data_path = Path(data_path)
        self.split = split
        self.sampling_rate = sampling_rate
        self.transform = transform if split == "train" else None
        self.norm_stats = norm_stats

        self._load_metadata()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        db_csv = self.data_path / "ptbxl_database.csv"
        scp_csv = self.data_path / "scp_statements.csv"

        df = pd.read_csv(db_csv, index_col="ecg_id")
        folds = SPLIT_FOLDS[self.split]
        self.df = df[df["strat_fold"].isin(folds)].reset_index()

        self.code_to_class = load_scp_statements(scp_csv)

        # Choose the correct filename column based on sampling rate
        self._filename_col = (
            "filename_lr" if self.sampling_rate == 100 else "filename_hr"
        )

    def _read_signal(self, record_path: Path) -> Optional[np.ndarray]:
        """Return signal array of shape (12, signal_length) or None on error."""
        try:
            record_str = str(record_path.with_suffix(""))
            signal, _ = wfdb.rdsamp(record_str)
            # signal shape from wfdb: (signal_length, n_leads) → transpose
            return signal.T.astype(np.float32)  # (12, 1000)
        except Exception as exc:
            warnings.warn(f"Could not read {record_path}: {exc}", stacklevel=2)
            return None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Build absolute record path from the relative path stored in CSV
        rel_path = row[self._filename_col]
        record_path = self.data_path / rel_path

        signal = self._read_signal(record_path)

        if signal is None:
            # Return zeros so the DataLoader doesn't crash on bad files
            signal_len = 1000 if self.sampling_rate == 100 else 5000
            signal = np.zeros((12, signal_len), dtype=np.float32)

        # Normalization (per-lead z-score)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            signal = (signal - mean) / std

        x = torch.from_numpy(signal)  # (12, 1000)

        if self.transform is not None:
            x = self.transform(x)

        # Build label vector
        y = build_label_vector(row["scp_codes"], self.code_to_class)

        return x, y

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_all_labels(self) -> np.ndarray:
        """Return all label vectors as a float32 array of shape (N, 5).

        Useful for computing class statistics without iterating __getitem__.
        """
        from data.labels import build_label_vector as _blv

        labels = []
        for _, row in self.df.iterrows():
            labels.append(_blv(row["scp_codes"], self.code_to_class).numpy())
        return np.stack(labels)


if __name__ == "__main__":
    import sys

    data_path = Path("data/raw/")
    if not (data_path / "ptbxl_database.csv").exists():
        print(
            "PTB-XL+ data not found at data/raw/. "
            "Run scripts/download_data.py for instructions."
        )
        sys.exit(0)

    ds = PTBXLDataset(data_path, split="train")
    print(f"Train samples: {len(ds)}")
    x, y = ds[0]
    print(f"Signal shape: {x.shape}  Label: {y}")
    print("dataset.py smoke test passed.")
