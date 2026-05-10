# ECG Classification — CLAUDE.md

## Project overview
Multi-label ECG classification on PTB-XL+ (21,799 12-lead recordings, 100 Hz).
Phase 1 complete: project scaffold + ResNet-1D + data pipeline.
Planned phases: PatchTST, CNN-Transformer hybrid, ablation study.

## Architecture decisions
- Labels: 5-class binary vector [NORM, MI, STTC, CD, HYP] via SCP code → diagnostic_class mapping.
- Loss: BCEWithLogitsLoss with per-class pos_weight (handle class imbalance).
- Split: strat_fold column — folds 1-8 train, 9 val, 10 test. Do NOT shuffle or re-split.
- Normalization: per-lead z-score fitted on training set only; stats saved to config/norm_stats.npy.

## Key files
- `data/dataset.py`      — PTBXLDataset; reads wfdb records, applies transforms.
- `data/labels.py`       — SCP code parsing → binary label vector.
- `data/preprocessing.py`— Normalizer + augmentation classes.
- `models/resnet1d.py`   — ResNet-1D backbone (logits out, no sigmoid).
- `training/trainer.py`  — Trainer with BCEWithLogitsLoss, Adam, ReduceLROnPlateau.
- `training/metrics.py`  — compute_metrics() → dict with macro F1 and per-class breakdown.
- `scripts/preprocess.py`— Full pipeline: fit normalizer, save stats, print pos_weight.
- `scripts/train_resnet.py` — Main training entry point.

## Running order (once data is available)
```
python scripts/download_data.py --data_path data/raw/
python scripts/preprocess.py   --data_path data/raw/ --output_path data/processed/
python scripts/train_resnet.py --data_path data/raw/
```

## Conventions
- All functions have type hints (Python 3.10+).
- pathlib.Path everywhere — no hardcoded strings.
- Every module has an `if __name__ == "__main__"` smoke test.
- Model forward() returns logits; sigmoid/threshold applied only at eval time.
- Augmentations applied only to training split; PTBXLDataset handles this internally.
