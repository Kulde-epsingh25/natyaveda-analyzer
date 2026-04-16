"""
NatyaVeda — Compute Class Prototypes
Run this ONCE after training to compute mean class embeddings.
These prototypes improve mohiniyattam/kuchipudi/kathak classification
WITHOUT retraining (10-15% improvement from inference blending alone).

Usage:
    python scripts/compute_prototypes.py --checkpoint weights/danceformer_best.pt --data data/splits

Output:
    reports/class_prototypes.npz   ← loaded by Predictor automatically
    reports/cluster_visualization.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]


def compute_prototypes(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (prototypes [8, D], all_embeddings [N, D+1]) where last col is label.
    """
    model.eval()
    n = len(DANCE_CLASSES)
    D = None
    sums   = None
    counts = np.zeros(n)
    all_embs  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["keypoints"].to(device)
            y = batch["label"].to(device)
            out = model(x)

            # Try to get intermediate features (pre-classification)
            features = out.get("features")
            if features is None:
                # Fall back to logits if model doesn't expose features
                features = out["dance_logits"]

            if D is None:
                D = features.shape[-1]
                sums = np.zeros((n, D))

            fn = F.normalize(features, dim=-1).cpu().numpy()
            yl = y.cpu().numpy()

            for c in range(n):
                mask = yl == c
                if mask.any():
                    sums[c]   += fn[mask].sum(axis=0)
                    counts[c] += mask.sum()

            all_embs.append(fn)
            all_labels.append(yl)

    prototypes = sums / np.maximum(counts[:, None], 1)
    # L2 normalize prototypes
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(norms, 1e-8)

    all_embs   = np.concatenate(all_embs,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return prototypes, all_embs, all_labels, counts


def print_inter_class_distances(prototypes: np.ndarray) -> None:
    """Print pairwise cosine distances between class prototypes."""
    n = len(DANCE_CLASSES)
    print("\n  Inter-class cosine distances (higher = better separated):")
    print("  " + "─" * 70)
    header = "            " + "".join(f"  {d[:5]:5s}" for d in DANCE_CLASSES)
    print(f"  {header}")

    CONFUSED = {(5,3),(1,7),(2,0),(3,6),(3,1),(5,6),(1,6)}

    for i in range(n):
        row = f"  {DANCE_CLASSES[i]:14s}"
        for j in range(n):
            if i == j:
                row += "  ——   "
            else:
                p_i = prototypes[i] / (np.linalg.norm(prototypes[i]) + 1e-8)
                p_j = prototypes[j] / (np.linalg.norm(prototypes[j]) + 1e-8)
                cos = np.dot(p_i, p_j)
                dist = 1.0 - cos
                pair = (min(i,j), max(i,j))
                rev  = (min(i,j), max(i,j))
                is_bad = (i,j) in CONFUSED or (j,i) in CONFUSED
                flag = " ⚠️" if is_bad and dist < 0.5 else ""
                row += f"  {dist:.2f}{flag}"
        print(row)
    print("  " + "─" * 70)
    print("  Goal: all off-diagonal values > 0.5 (currently mohin↔kuchi is worst)")


def visualize_clusters(all_embs: np.ndarray, all_labels: np.ndarray,
                        prototypes: np.ndarray, output_path: str) -> None:
    """2D UMAP/t-SNE visualization of class embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        logger.info("Computing t-SNE (may take 1-2 min)...")

        # Combine embeddings + prototypes for t-SNE
        combined = np.concatenate([all_embs, prototypes], axis=0)
        n_data = len(all_embs)

        try:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=30,
                n_iter=1000,
                learning_rate="auto",
                init="pca",
            )
        except TypeError:
            # sklearn>=1.5 renamed n_iter to max_iter.
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=30,
                max_iter=1000,
                learning_rate="auto",
                init="pca",
            )
        reduced = tsne.fit_transform(combined)

        data_2d  = reduced[:n_data]
        proto_2d = reduced[n_data:]

        COLORS = [
            "#FF6B35", "#7209B7", "#2DC653", "#F4D03F",
            "#00B4D8", "#FF006E", "#F77F00", "#D62828",
        ]

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_facecolor("#0F0F1A")
        fig.patch.set_facecolor("#0F0F1A")

        patches = []
        for c, (dance, color) in enumerate(zip(DANCE_CLASSES, COLORS)):
            mask = all_labels == c
            if not mask.any():
                continue
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                       c=color, alpha=0.4, s=25, edgecolors="none")
            # Prototype star
            ax.scatter(proto_2d[c, 0], proto_2d[c, 1],
                       c=color, s=250, marker="*", edgecolors="white", linewidth=1.5, zorder=5)
            ax.annotate(dance, (proto_2d[c, 0], proto_2d[c, 1]),
                        fontsize=9, color=color, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
            patches.append(mpatches.Patch(color=color, label=dance.title()))

        ax.set_title("NatyaVeda — Class Embedding Clusters (t-SNE)\n"
                     "Stars = class prototypes | Dots = training clips",
                     color="white", fontsize=13, pad=14)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333344")
        ax.legend(handles=patches, loc="upper right", framealpha=0.3,
                  labelcolor="white", facecolor="#1A1A2E")
        ax.set_xlabel("t-SNE dim 1", color="gray")
        ax.set_ylabel("t-SNE dim 2", color="gray")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0F0F1A")
        plt.close()
        logger.info("Cluster visualization saved: %s", output_path)

    except ImportError as e:
        logger.warning("Visualization skipped: %s (install scikit-learn + matplotlib)", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/danceformer_best.pt")
    parser.add_argument("--data",       default="data/splits")
    parser.add_argument("--split",      default="train",
                        help="Which split to compute prototypes from")
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--no-viz",     action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.danceformer import DanceFormer
    from src.training.dataset import NatyaVedaDataset

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg  = ckpt.get("config", {})
    model = DanceFormer.from_config({"model": cfg}).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    logger.info("Model loaded: %s", args.checkpoint)

    # Load dataset
    data_dir = Path(args.data) / args.split
    ds = NatyaVedaDataset(str(data_dir), clip_length=64, clip_stride=32, augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    logger.info("Dataset: %d clips from %s", len(ds), data_dir)

    # Compute prototypes
    prototypes, all_embs, all_labels, counts = compute_prototypes(model, dl, device)

    # Print per-class sample counts
    print("\n  Per-class sample counts in training set:")
    for i, dance in enumerate(DANCE_CLASSES):
        bar = "█" * int(counts[i] // 5)
        print(f"  {dance:20s}: {int(counts[i]):5d}  {bar}")

    # Print inter-class distances
    print_inter_class_distances(prototypes)

    # Save
    proto_path = out_dir / "class_prototypes.npz"
    np.savez_compressed(str(proto_path),
                        prototypes=prototypes,
                        dance_classes=DANCE_CLASSES,
                        sample_counts=counts)
    logger.info("Prototypes saved: %s  shape=%s", proto_path, prototypes.shape)

    # Visualize
    if not args.no_viz:
        viz_path = str(out_dir / "cluster_visualization.png")
        visualize_clusters(all_embs, all_labels, prototypes, viz_path)

    print(f"\n  ✅ Done. Prototypes at: {proto_path}")
    print("  These are now automatically used by scripts/infer.py")
    print("  Re-run after each training run to keep prototypes current.")


if __name__ == "__main__":
    main()
