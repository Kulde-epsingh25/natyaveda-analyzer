"""
NatyaVeda — Evaluation Script
Comprehensive test-set evaluation with per-class metrics, confusion matrix,
and temporal ensemble predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]


def evaluate(
    checkpoint_path: Path,
    test_data_dir: Path,
    report_dir: Path,
    device: str = "cuda",
    tta: bool = True,
    batch_size: int = 64,
) -> dict:
    """
    Run full evaluation pipeline.

    Returns metrics dict with accuracy, F1, per-class report, and confusion matrix.
    """
    import yaml
    from src.models.danceformer import DanceFormer
    from src.training.trainer import DanceDataset

    report_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    config = ckpt.get("config", {})
    model = DanceFormer.from_config(config)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    # Dataset
    test_ds = DanceDataset(
        data_dir=test_data_dir,
        split="test",
        clip_length=config.get("dataset", {}).get("clip_length_frames", 64),
        clip_stride=16,
        augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Test clips: {len(test_ds)}")

    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            kpts   = batch["keypoints"].to(device)
            vel    = batch["velocities"].to(device)
            acc    = batch["accelerations"].to(device)
            labels = batch["label"]

            logits = model(kpts, vel, acc)["dance_logits"]
            probs  = F.softmax(logits, dim=-1)

            if tta:
                # TTA: horizontal flip
                kpts_flip = _flip_keypoints(kpts)
                logits_flip = model(kpts_flip, vel, acc)["dance_logits"]
                probs = (probs + F.softmax(logits_flip, dim=-1)) / 2.0

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)   # [N, 8]
    all_labels = np.concatenate(all_labels, axis=0)   # [N]
    all_preds  = all_probs.argmax(axis=1)

    # Compute metrics
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )

    acc = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cls_report = classification_report(
        all_labels, all_preds, target_names=DANCE_CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "per_class": cls_report,
        "confusion_matrix": cm.tolist(),
        "num_test_clips": len(all_labels),
        "checkpoint": str(checkpoint_path),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("  NatyaVeda Dance Classifier — Test Results")
    print("=" * 70)
    print(f"  Accuracy       : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 (weighted)  : {f1_w:.4f}")
    print(f"  F1 (macro)     : {f1_m:.4f}")
    print(f"  Test clips     : {len(all_labels)}")
    print()
    print(classification_report(all_labels, all_preds, target_names=DANCE_CLASSES, zero_division=0))

    # Confusion matrix
    print("  Confusion Matrix:")
    _print_confusion_matrix(cm, DANCE_CLASSES)

    # Save JSON report
    report_path = report_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    # Save confusion matrix plot
    _save_confusion_matrix_plot(cm, DANCE_CLASSES, report_dir / "confusion_matrix.png")

    return metrics


def _flip_keypoints(kpts: torch.Tensor) -> torch.Tensor:
    """Horizontal flip augmentation for TTA."""
    B, T, D = kpts.shape
    kpts_np = kpts.cpu().numpy().reshape(B, T, 133, 3).copy()
    kpts_np[:, :, :, 0] = -kpts_np[:, :, :, 0]

    # Swap left/right body pairs
    pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    for l, r in pairs:
        kpts_np[:, :, [l, r]] = kpts_np[:, :, [r, l]]
    # Swap hands
    left = kpts_np[:, :, 91:112].copy()
    right = kpts_np[:, :, 112:133].copy()
    kpts_np[:, :, 91:112] = right
    kpts_np[:, :, 112:133] = left

    return torch.from_numpy(kpts_np.reshape(B, T, D)).to(kpts.device)


def _print_confusion_matrix(cm: np.ndarray, classes: list[str]) -> None:
    """Print confusion matrix as ASCII table."""
    abbrev = [c[:4] for c in classes]
    header = "     " + "  ".join(f"{a:>4}" for a in abbrev)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{abbrev[i]:>4} " + "  ".join(f"{v:>4}" for v in row)
        print(row_str)


def _save_confusion_matrix_plot(
    cm: np.ndarray, classes: list[str], save_path: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(
            cm_norm,
            annot=True, fmt=".2f",
            xticklabels=[c[:5] for c in classes],
            yticklabels=[c[:5] for c in classes],
            cmap="Blues", ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("NatyaVeda — Dance Classification Confusion Matrix")
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix plot saved: {save_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping plot")


# ─────────────────────────────────────────────────────────────────────────────
# Per-video evaluation (temporal ensemble)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_video(
    model: torch.nn.Module,
    npz_path: Path,
    device: str = "cuda",
    clip_length: int = 64,
    clip_stride: int = 16,
) -> dict:
    """
    Evaluate a single video using sliding-window temporal ensemble.
    All clip-level predictions are averaged for final video-level prediction.
    """
    data = np.load(str(npz_path))
    kpts = data["keypoints"]                       # [T, 133, 3]
    T = kpts.shape[0]
    data.close()

    all_probs = []
    starts = list(range(0, T - clip_length + 1, clip_stride))
    if not starts:
        starts = [0]

    model.eval()
    with torch.no_grad():
        for start in starts:
            end = min(start + clip_length, T)
            clip = kpts[start:end].reshape(1, end - start, -1).astype(np.float32)
            if clip.shape[1] < clip_length:
                pad = np.zeros((1, clip_length - clip.shape[1], clip.shape[2]), dtype=np.float32)
                clip = np.concatenate([clip, pad], axis=1)
            clip_t = torch.from_numpy(clip).to(device)
            logits = model(clip_t)["dance_logits"]
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            all_probs.append(probs)

    ensemble_probs = np.stack(all_probs).mean(axis=0)   # [8]
    pred_idx = int(ensemble_probs.argmax())

    return {
        "prediction": DANCE_CLASSES[pred_idx],
        "confidence": float(ensemble_probs[pred_idx]),
        "probabilities": {d: float(p) for d, p in zip(DANCE_CLASSES, ensemble_probs)},
        "num_clips": len(all_probs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="NatyaVeda Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test-data", required=True, help="Path to test data dir")
    parser.add_argument("--report-dir", default="reports", help="Output directory for reports")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        test_data_dir=Path(args.test_data),
        report_dir=Path(args.report_dir),
        device=args.device,
        tta=not args.no_tta,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
