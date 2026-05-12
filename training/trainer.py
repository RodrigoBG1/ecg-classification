"""Training loop for ECG classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import compute_metrics, format_metrics
from utils.logger import get_logger

log = get_logger(__name__)


class Trainer:
    """
    Manages the full training/validation lifecycle for a PyTorch model.

    Args:
        model:          The neural network (outputs logits, no sigmoid).
        train_loader:   DataLoader for the training split.
        val_loader:     DataLoader for the validation split.
        pos_weight:     Per-class weight tensor for BCEWithLogitsLoss (shape: (n_classes,)).
        checkpoint_dir: Directory where the best model checkpoint is saved.
        device:         'cuda', 'mps', or 'cpu'. Auto-detected if None.
        learning_rate:  Initial LR for AdamW.
        weight_decay:   L2 coefficient for AdamW.
        early_stopping_patience: Epochs without AUC improvement before stopping.
        mixup_alpha:    Beta distribution alpha for Mixup. 0 disables Mixup.
        grad_clip_norm: Max gradient norm for clipping. 0 disables clipping.
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
        early_stopping_patience: int = 15,
        mixup_alpha: float = 0.4,
        grad_clip_norm: float = 1.0,
        label_smoothing: float = 0.1,
        warmup_epochs: int = 5,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.mixup_alpha = mixup_alpha
        self.grad_clip_norm = grad_clip_norm
        self.label_smoothing = label_smoothing
        self.warmup_epochs = warmup_epochs

        pw = pos_weight.to(self.device) if pos_weight is not None else None
        self._pos_weight = pw
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler: SequentialLR | None = None
        self.early_stopping_patience = early_stopping_patience

        # Mixed precision: enabled only on CUDA
        self.use_amp: bool = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.best_val_auc: float = -1.0
        self._epochs_without_improvement: int = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _smooth_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """BCEWithLogitsLoss with label smoothing.

        Smooths 1 → (1 - ε/2) and 0 → ε/2, which prevents the model from
        becoming overconfident on hard labels and improves calibration.
        """
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self._pos_weight)

    # ------------------------------------------------------------------
    # Mixup helpers
    # ------------------------------------------------------------------

    def _mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[idx]
        return mixed_x, y, y[idx], lam

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """Run one training epoch and return the mean loss."""
        self.model.train()
        total_loss = 0.0

        for x, y in tqdm(self.train_loader, desc="Train", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.mixup_alpha > 0:
                x, y_a, y_b, lam = self._mixup(x, y)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x)
                if self.mixup_alpha > 0:
                    loss = lam * self._smooth_loss(logits, y_a) + (1.0 - lam) * self._smooth_loss(logits, y_b)
                else:
                    loss = self._smooth_loss(logits, y)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> tuple[float, dict[str, float]]:
        """
        Run validation and return (val_loss, metrics_dict).

        Collects all logits and labels in CPU memory before computing metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Val  ", leave=False):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(x)
                    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self._pos_weight)

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
        Train for up to `n_epochs` epochs with early stopping.

        Monitors macro AUC-ROC for early stopping and checkpointing.

        Returns:
            Training history — one dict per epoch.
        """
        log.info(
            f"Training on {self.device} for up to {n_epochs} epochs "
            f"(patience={self.early_stopping_patience}, amp={self.use_amp}, "
            f"mixup={self.mixup_alpha > 0}, warmup={self.warmup_epochs} epochs, "
            f"label_smoothing={self.label_smoothing})"
        )

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(n_epochs - self.warmup_epochs, 1),
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()
            self.scheduler.step()

            val_auc = metrics["auc_roc_macro"]
            lr = self.optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                **metrics,
            }
            self.history.append(record)

            log.info(
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={lr:.2e}  " + format_metrics(metrics)
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self._epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_auc, metrics["f1_macro"])
            else:
                self._epochs_without_improvement += 1
                if self._epochs_without_improvement >= self.early_stopping_patience:
                    log.info(
                        f"Early stopping: no AUC improvement for "
                        f"{self.early_stopping_patience} epochs. "
                        f"Best AUC={self.best_val_auc:.4f}"
                    )
                    break

        return self.history

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_auc: float, val_f1: float) -> None:
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        log.info(f"  ✓ Saved best checkpoint (AUC={val_auc:.4f}, F1={val_f1:.4f}) → {ckpt_path}")

    def load_best_checkpoint(self) -> None:
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        log.info(
            f"Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(AUC={ckpt['val_auc']:.4f}, F1={ckpt['val_f1']:.4f})"
        )


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    from models.resnet1d import ResNet1D

    n, n_classes = 64, 5
    X = torch.randn(n, 12, 5000)
    Y = (torch.rand(n, n_classes) > 0.7).float()
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=16)

    model = ResNet1D(n_classes=n_classes, sampling_rate=500)
    trainer = Trainer(model, loader, loader, checkpoint_dir=Path("/tmp/ckpt_test"))
    history = trainer.fit(n_epochs=2)
    print(f"History entries: {len(history)}")
    print("trainer.py smoke test passed.")
