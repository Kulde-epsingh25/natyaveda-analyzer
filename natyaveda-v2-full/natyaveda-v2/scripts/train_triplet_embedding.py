"""
Train a compact triplet-loss embedding model for dance clustering.

This script learns a robust low-dimensional latent space from pre-extracted
pose clips in data/splits/{train,val}/<dance_class>/*.npz.

Example:
    python scripts/train_triplet_embedding.py --data data/splits --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Compute batch-hard triplet loss with fallback for small/degenerate batches."""
    if embeddings.size(0) < 3:
        return embeddings.sum() * 0.0

    dists = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

    pos_mask = same & ~eye
    neg_mask = ~same

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not torch.any(valid):
        return embeddings.sum() * 0.0

    hardest_pos = dists.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    hardest_neg = dists.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    losses = F.relu(hardest_pos - hardest_neg + margin)
    return losses[valid].mean()


def _knn1_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """1-NN accuracy inside a split as a quick embedding quality proxy."""
    if embeddings.size(0) < 2:
        return 0.0
    dists = torch.cdist(embeddings, embeddings, p=2)
    dists.fill_diagonal_(float("inf"))
    nn_idx = torch.argmin(dists, dim=1)
    pred = labels[nn_idx]
    return float((pred == labels).float().mean().item())


class TemporalTripletEncoder(nn.Module):
    """Lightweight temporal encoder for pose-clip embeddings."""

    def __init__(self, input_dim: int = 399, hidden_dim: int = 256, emb_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.pre = nn.LayerNorm(input_dim)
        self.frame_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 399]
        h = self.pre(x)
        h = self.frame_mlp(h)
        h = h.mean(dim=1)  # temporal pooling
        z = self.head(h)
        return F.normalize(z, dim=-1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, margin: float) -> dict[str, float]:
    model.eval()
    all_z = []
    all_y = []
    losses = []
    for batch in loader:
        x = batch["keypoints"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        z = model(x)
        loss = _batch_hard_triplet_loss(z, y, margin=margin)
        losses.append(float(loss.item()))
        all_z.append(z)
        all_y.append(y)

    if not all_z:
        return {"triplet_loss": 0.0, "knn1_acc": 0.0}

    emb = torch.cat(all_z, dim=0)
    lab = torch.cat(all_y, dim=0)
    return {
        "triplet_loss": float(np.mean(losses)) if losses else 0.0,
        "knn1_acc": _knn1_accuracy(emb, lab),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train triplet embedding model for dance clustering")
    parser.add_argument("--data", default="data/splits", help="Root split directory containing train/val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--clip-stride", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="weights/triplet_embedding_best.pt")
    parser.add_argument("--metrics-json", default="reports/triplet_embedding_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    data_root = (project_root / args.data).resolve()
    out_path = (project_root / args.output).resolve()
    metrics_path = (project_root / args.metrics_json).resolve()

    import sys
    sys.path.insert(0, str(project_root))
    from src.training.dataset import DANCE_CLASSES, NatyaVedaDataset

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    train_ds = NatyaVedaDataset(train_dir, clip_length=args.clip_length, clip_stride=args.clip_stride, augment=True)
    val_ds = NatyaVedaDataset(val_dir, clip_length=args.clip_length, clip_stride=args.clip_stride, augment=False)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training clips found in {train_dir}")

    device = torch.device(args.device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = TemporalTripletEncoder(
        input_dim=399,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    logger.info("=" * 60)
    logger.info("Triplet Embedding Training")
    logger.info("Data: %s", data_root)
    logger.info("Device: %s", device)
    logger.info("Train clips: %d | Val clips: %d", len(train_ds), len(val_ds))
    logger.info("Embedding dim: %d | Margin: %.3f", args.emb_dim, args.margin)
    logger.info("=" * 60)

    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []

        for batch in train_loader:
            x = batch["keypoints"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                z = model(x)
                loss = _batch_hard_triplet_loss(z, y, margin=args.margin)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.item()))

        train_loss = float(np.mean(running)) if running else 0.0
        val_stats = _evaluate(model, val_loader, device=device, margin=args.margin) if len(val_ds) > 0 else {
            "triplet_loss": 0.0,
            "knn1_acc": 0.0,
        }

        row = {
            "epoch": float(epoch),
            "train_triplet_loss": train_loss,
            "val_triplet_loss": float(val_stats["triplet_loss"]),
            "val_knn1_acc": float(val_stats["knn1_acc"]),
        }
        history.append(row)

        logger.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_knn1=%.3f",
            epoch,
            train_loss,
            row["val_triplet_loss"],
            row["val_knn1_acc"],
        )

        score = row["val_triplet_loss"] if len(val_ds) > 0 else train_loss
        if score < best_val:
            best_val = score
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "state_dict": model.state_dict(),
                "dance_classes": DANCE_CLASSES,
                "embedding_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "clip_length": args.clip_length,
                "margin": args.margin,
                "best_score": best_val,
            }
            torch.save(ckpt, out_path)
            logger.info("Saved best checkpoint: %s", out_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_loss": best_val}, f, indent=2)

    logger.info("Training finished. Metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
