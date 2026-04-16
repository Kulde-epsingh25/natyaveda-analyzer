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

PROTOTYPE_BLEND = 0.45
PROTOTYPE_SCALE = 3.0


def _build_class_prototypes(
    model: torch.nn.Module,
    data_dir: Path,
    clip_length: int,
    device: str,
    save_path: Path,
) -> torch.Tensor:
    from src.training.trainer import DanceDataset

    dataset = DanceDataset(
        data_dir=data_dir,
        split="train",
        clip_length=clip_length,
        clip_stride=16,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}

    with torch.no_grad():
        for batch in loader:
            kpts = batch["keypoints"].to(device)
            vel = batch["velocities"].to(device)
            acc = batch["accelerations"].to(device)
            labels = batch["label"].to(device)
            features = model(kpts, vel, acc)["features"]
            for feature, label in zip(features, labels):
                key = int(label.item())
                if key not in sums:
                    sums[key] = feature.detach().clone()
                    counts[key] = 1
                else:
                    sums[key] += feature.detach()
                    counts[key] += 1

    prototypes = []
    for class_idx in range(len(DANCE_CLASSES)):
        if class_idx in sums:
            proto = sums[class_idx] / max(counts[class_idx], 1)
        else:
            proto = torch.zeros_like(next(iter(sums.values())))
        prototypes.append(proto)

    prototype_tensor = torch.stack(prototypes, dim=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        prototypes=prototype_tensor.cpu().numpy().astype(np.float32),
        counts=np.array([counts.get(i, 0) for i in range(len(DANCE_CLASSES))], dtype=np.int32),
    )
    logger.info("Prototype cache saved: %s", save_path)
    return prototype_tensor


def _blend_with_prototypes(logits: torch.Tensor, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    feature_norm = F.normalize(features, dim=-1)
    proto_norm = F.normalize(prototypes.to(features.device, dtype=features.dtype), dim=-1)
    proto_scores = feature_norm @ proto_norm.T
    return logits + PROTOTYPE_BLEND * (proto_scores * PROTOTYPE_SCALE)


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
    from src.models.danceformer import DanceFormer, MODEL_REGISTRY
    from src.training.trainer import DanceDataset

    report_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    config = ckpt.get("config", {})
    state_dict = ckpt["state_dict"]

    # Prefer checkpoint config, but fall back to auto-detecting a compatible model variant.
    model = DanceFormer.from_config(config)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.warning("Checkpoint/config mismatch in evaluate(): %s", e)
        loaded = False
        for name in ("danceformer-small", "danceformer-base", "danceformer-large"):
            candidate = MODEL_REGISTRY[name]()
            try:
                candidate.load_state_dict(state_dict)
                model = candidate
                loaded = True
                logger.info("Auto-selected model variant from checkpoint: %s", name)
                break
            except RuntimeError:
                continue
        if not loaded:
            raise

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

    train_dir = test_data_dir / "train"
    prototype_cache = report_dir / "class_prototypes.npz"
    if prototype_cache.exists():
        cached = np.load(str(prototype_cache))
        prototypes = torch.from_numpy(cached["prototypes"]).to(device)
        cached.close()
        logger.info("Loaded prototype cache: %s", prototype_cache)
    else:
        prototypes = _build_class_prototypes(
            model,
            train_dir,
            config.get("dataset", {}).get("clip_length_frames", 64),
            device,
            prototype_cache,
        )

    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            kpts   = batch["keypoints"].to(device)
            vel    = batch["velocities"].to(device)
            acc    = batch["accelerations"].to(device)
            labels = batch["label"]

            outputs = model(kpts, vel, acc)
            logits = _blend_with_prototypes(outputs["dance_logits"], outputs["features"], prototypes)
            probs  = F.softmax(logits, dim=-1)

            if tta:
                probs_ensemble = [probs]

                kpts_flip = _flip_keypoints(kpts)
                vel_flip = _flip_motion(vel)
                acc_flip = _flip_motion(acc)
                out_flip = model(kpts_flip, vel_flip, acc_flip)
                probs_ensemble.append(
                    F.softmax(_blend_with_prototypes(out_flip["dance_logits"], out_flip["features"], prototypes), dim=-1)
                )

                kpts_rev = _reverse_time(kpts)
                vel_rev = _reverse_time(vel, negate=True)
                acc_rev = _reverse_time(acc)
                out_rev = model(kpts_rev, vel_rev, acc_rev)
                probs_ensemble.append(
                    F.softmax(_blend_with_prototypes(out_rev["dance_logits"], out_rev["features"], prototypes), dim=-1)
                )

                out_flip_rev = model(
                    _reverse_time(kpts_flip),
                    _reverse_time(vel_flip, negate=True),
                    _reverse_time(acc_flip),
                )
                probs_ensemble.append(
                    F.softmax(
                        _blend_with_prototypes(out_flip_rev["dance_logits"], out_flip_rev["features"], prototypes),
                        dim=-1,
                    )
                )

                probs = torch.stack(probs_ensemble, dim=0).mean(dim=0)

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


def _flip_motion(motion: torch.Tensor, preserve_sign: bool = True) -> torch.Tensor:
    """Mirror motion tensors to match horizontal flip augmentation."""
    B, T, D = motion.shape
    motion_np = motion.cpu().numpy().reshape(B, T, 133, 3).copy()
    if preserve_sign:
        motion_np[:, :, :, 0] = -motion_np[:, :, :, 0]

    pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    for l, r in pairs:
        motion_np[:, :, [l, r]] = motion_np[:, :, [r, l]]
    left = motion_np[:, :, 91:112].copy()
    right = motion_np[:, :, 112:133].copy()
    motion_np[:, :, 91:112] = right
    motion_np[:, :, 112:133] = left

    return torch.from_numpy(motion_np.reshape(B, T, D)).to(motion.device)


def _reverse_time(x: torch.Tensor, negate: bool = False) -> torch.Tensor:
    """Reverse the temporal axis; optionally negate motion direction."""
    out = x.flip(1)
    return -out if negate else out


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
