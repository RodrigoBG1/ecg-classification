"""Training loop for ECG classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import compute_metrics, format_metrics
from utils.logger import get_logger

log = get_logger(__name__)


class Trainer:
    """
    Manages the full training/validation lifecycle for a PyTorch model.

    Args:
        model:        The neural network (outputs logits, no sigmoid).
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        pos_weight:   Per-class weight tensor for BCEWithLogitsLoss (shape: (n_classes,)).
        checkpoint_dir: Directory where the best model checkpoint is saved.
        device:       'cuda', 'mps', or 'cpu'.  Auto-detected if None.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pos_weight: Optional[torch.Tensor] = None,
        checkpoint_dir: Path | str = Path("checkpoints"),
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        pw = pos_weight.to(self.device) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5, verbose=True
        )

        self.best_val_f1: float = -1.0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """Run one training epoch and return the mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for x, y in tqdm(self.train_loader, desc="Train", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_batches

    def validate(self) -> tuple[float, dict[str, float]]:
        """
        Run validation and return (val_loss, metrics_dict).

        Collects all logits and labels in CPU memory before computing metrics
        to avoid multiple forward passes.
        """
        self.model.eval()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Val  ", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        val_loss = total_loss / len(self.val_loader)
        y_pred = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_labels).numpy()
        metrics = compute_metrics(y_true, y_pred)
        return val_loss, metrics

    def fit(self, n_epochs: int) -> list[dict[str, float]]:
        """
        Train for `n_epochs` epochs, validate after each, and save the
        best checkpoint (by val macro F1).

        Returns:
            Training history — one dict per epoch with train_loss, val_loss,
            and all metric keys.
        """
        log.info(f"Training on {self.device} for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            val_f1 = metrics["f1_macro"]
            self.scheduler.step(val_f1)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }
            self.history.append(record)

            log.info(
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                + format_metrics(metrics)
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self._save_checkpoint(epoch, val_f1)

        return self.history

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_f1: float) -> None:
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_f1": val_f1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        log.info(f"  ✓ Saved best checkpoint (F1={val_f1:.4f}) → {ckpt_path}")

    def load_best_checkpoint(self) -> None:
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded checkpoint from epoch {ckpt['epoch']} (F1={ckpt['val_f1']:.4f})")


if __name__ == "__main__":
    # Smoke test with synthetic data (no real dataset needed)
    from torch.utils.data import TensorDataset

    n, n_classes = 64, 5
    X = torch.randn(n, 12, 1000)
    Y = (torch.rand(n, n_classes) > 0.7).float()
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=16)

    from models.resnet1d import ResNet1D

    model = ResNet1D(n_classes=n_classes)
    trainer = Trainer(model, loader, loader, checkpoint_dir=Path("/tmp/ckpt_test"))
    history = trainer.fit(n_epochs=2)
    print(f"History entries: {len(history)}")
    print("trainer.py smoke test passed.")
