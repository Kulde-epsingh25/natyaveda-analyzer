"""
NatyaVeda — Build Dataset Splits
Creates stratified train / val / test splits from processed .npz feature files.
Supports single-split (80/10/10) and k-fold cross-validation modes.

Usage:
  python scripts/build_splits.py --input data/processed --output data/splits
  python scripts/build_splits.py --input data/processed --output data/splits --folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_files(data_dir: Path) -> dict[str, list[Path]]:
    """Collect all .npz files grouped by dance form."""
    files_by_class: dict[str, list[Path]] = defaultdict(list)
    for dance in DANCE_CLASSES:
        dance_dir = data_dir / dance
        if dance_dir.exists():
            npz_files = sorted(dance_dir.glob("*.npz"))
            # Exclude intermediate _pose.npz / _hands.npz / _videomae.npz files
            npz_files = [f for f in npz_files
                         if not any(f.name.endswith(s) for s in ("_pose.npz", "_hands.npz", "_videomae.npz"))]
            files_by_class[dance] = npz_files
            logger.info(f"  {dance:20s} : {len(npz_files):4d} files")
    return files_by_class


def _stratified_split(
    files_by_class: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Stratified split: each dance form is split independently, then combined.
    Returns (train_files, val_files, test_files).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)
    train_all, val_all, test_all = [], [], []

    for dance, files in files_by_class.items():
        files = files.copy()
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_test  = n - n_train - n_val

        if n_test < 1 and n >= 3:
            n_val  -= 1
            n_test  = 1
        if n < 3:
            # Very small class — put everything in train
            n_train, n_val, n_test = n, 0, 0

        train_all.extend(files[:n_train])
        val_all.extend(files[n_train:n_train + n_val])
        test_all.extend(files[n_train + n_val:])

    return train_all, val_all, test_all


def _kfold_splits(
    files_by_class: dict[str, list[Path]],
    k: int,
    seed: int,
) -> list[tuple[list[Path], list[Path]]]:
    """
    Generate k stratified folds.
    Returns list of (train_files, val_files) per fold.
    """
    rng = random.Random(seed)
    # Build per-class folds
    class_folds: dict[str, list[list[Path]]] = {}
    for dance, files in files_by_class.items():
        files = files.copy()
        rng.shuffle(files)
        folds = [[] for _ in range(k)]
        for i, f in enumerate(files):
            folds[i % k].append(f)
        class_folds[dance] = folds

    result = []
    for fold_idx in range(k):
        val_files   = []
        train_files = []
        for dance in files_by_class:
            folds = class_folds[dance]
            val_files.extend(folds[fold_idx])
            for j, fold in enumerate(folds):
                if j != fold_idx:
                    train_files.extend(fold)
        result.append((train_files, val_files))

    return result


def _copy_files(files: list[Path], dest_dir: Path, dance_class: str) -> None:
    """Copy files to destination directory under the correct dance subfolder."""
    (dest_dir / dance_class).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / dance_class / f.name)


def _count_by_class(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for f in files:
        # Parent dir name is the dance class
        counts[f.parent.name] += 1
    return counts


def _print_distribution(split_name: str, files: list[Path]) -> None:
    counts = _count_by_class(files)
    total  = sum(counts.values())
    print(f"\n  {split_name} ({total} clips):")
    for dance in DANCE_CLASSES:
        n = counts.get(dance, 0)
        bar = "█" * min(30, n // 2)
        print(f"    {dance:20s} : {n:4d}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NatyaVeda — Build Train/Val/Test Splits")
    parser.add_argument("--input",   default="data/processed", help="Processed .npz directory")
    parser.add_argument("--output",  default="data/splits",    help="Output splits directory")
    parser.add_argument("--train",   type=float, default=0.80, help="Train ratio (default: 0.80)")
    parser.add_argument("--val",     type=float, default=0.10, help="Val ratio   (default: 0.10)")
    parser.add_argument("--test",    type=float, default=0.10, help="Test ratio  (default: 0.10)")
    parser.add_argument("--folds",   type=int,   default=None, help="K-fold CV (overrides train/val/test ratios)")
    parser.add_argument("--seed",    type=int,   default=42,   help="Random seed")
    parser.add_argument("--symlink", action="store_true",      help="Create symlinks instead of copying files")
    args = parser.parse_args()

    data_dir   = Path(args.input)
    output_dir = Path(args.output)

    print(f"\n{'='*60}")
    print(f"  NatyaVeda — Building Dataset Splits")
    print(f"{'='*60}")
    print(f"  Input  : {data_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Seed   : {args.seed}")

    # Collect all processed files
    print(f"\n  Scanning {data_dir} ...")
    files_by_class = _collect_files(data_dir)

    total = sum(len(v) for v in files_by_class.values())
    if total == 0:
        logger.error(f"No .npz files found in {data_dir}. Run extract_features.py first.")
        return

    print(f"\n  Total files: {total}")

    if args.folds:
        # ── K-Fold Cross Validation mode
        print(f"\n  Mode: {args.folds}-Fold Cross Validation")
        folds = _kfold_splits(files_by_class, k=args.folds, seed=args.seed)

        fold_info = []
        for fold_idx, (train_files, val_files) in enumerate(folds):
            fold_dir = output_dir / f"fold_{fold_idx+1}"
            (fold_dir / "train").mkdir(parents=True, exist_ok=True)
            (fold_dir / "val").mkdir(parents=True, exist_ok=True)

            for f in train_files:
                _copy_files([f], fold_dir / "train", f.parent.name)
            for f in val_files:
                _copy_files([f], fold_dir / "val", f.parent.name)

            fold_info.append({
                "fold": fold_idx + 1,
                "train": len(train_files),
                "val":   len(val_files),
            })
            print(f"    Fold {fold_idx+1}: train={len(train_files)}  val={len(val_files)}")

        # Save metadata
        meta = {
            "mode": "kfold",
            "k": args.folds,
            "total_files": total,
            "seed": args.seed,
            "folds": fold_info,
            "classes": DANCE_CLASSES,
        }

    else:
        # ── Single train/val/test split mode
        print(f"\n  Mode: Single split  {args.train:.0%} / {args.val:.0%} / {args.test:.0%}")

        train_files, val_files, test_files = _stratified_split(
            files_by_class,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )

        # Create output dirs and copy files
        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                dance = f.parent.name
                dest  = split_dir / dance
                dest.mkdir(parents=True, exist_ok=True)
                if args.symlink:
                    link = dest / f.name
                    if not link.exists():
                        link.symlink_to(f.resolve())
                else:
                    shutil.copy2(f, dest / f.name)

        # Print distribution tables
        _print_distribution("TRAIN", train_files)
        _print_distribution("VAL",   val_files)
        _print_distribution("TEST",  test_files)

        # Save metadata
        meta = {
            "mode": "single_split",
            "train_ratio": args.train,
            "val_ratio":   args.val,
            "test_ratio":  args.test,
            "total_files": total,
            "train_count": len(train_files),
            "val_count":   len(val_files),
            "test_count":  len(test_files),
            "seed":        args.seed,
            "classes":     DANCE_CLASSES,
            "train_by_class": dict(_count_by_class(train_files)),
            "val_by_class":   dict(_count_by_class(val_files)),
            "test_by_class":  dict(_count_by_class(test_files)),
        }

    # Write split_info.json
    meta_path = output_dir / "split_info.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ Splits saved to: {output_dir}")
    print(f"  📄 Metadata: {meta_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
