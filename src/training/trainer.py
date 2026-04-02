"""
NatyaVeda — Training Dataset & Trainer
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]
DANCE_TO_IDX = {d: i for i, d in enumerate(DANCE_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DanceDataset(Dataset):
    """
    Loads preprocessed .npz pose feature files.

    Expected .npz structure:
        keypoints      : [T_total, 133, 3]
        velocities     : [T_total, 133, 2]  (optional)
        accelerations  : [T_total, 133, 2]  (optional)
        label          : scalar int (dance class index)

    Provides clips of fixed length via sliding window.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        clip_length: int = 64,
        clip_stride: int = 32,
        augment: bool = True,
        max_seq_len: int = 128,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.augment = augment and (split == "train")
        self.max_seq_len = max_seq_len

        self.clips: list[tuple[Path, int, int, int]] = []  # (path, start, end, label)
        self._build_clip_index()

    def _build_clip_index(self) -> None:
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            # Flat layout: look for per-dance subdirs
            split_dir = self.data_dir

        for dance_idx, dance in enumerate(DANCE_CLASSES):
            dance_dir = split_dir / dance
            if not dance_dir.exists():
                continue
            for npz_path in sorted(dance_dir.glob("*.npz")):
                data = np.load(str(npz_path))
                T = data["keypoints"].shape[0]
                data.close()
                # Sliding window clips
                start = 0
                while start + self.clip_length <= T:
                    self.clips.append((npz_path, start, start + self.clip_length, dance_idx))
                    start += self.clip_stride
                # Keep last clip if it has enough frames
                if T > self.clip_length and start < T:
                    self.clips.append((npz_path, T - self.clip_length, T, dance_idx))

        logger.info(f"DanceDataset [{self.split}]: {len(self.clips)} clips from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        path, start, end, label = self.clips[idx]
        data = np.load(str(path))

        kpts = data["keypoints"][start:end]     # [T, 133, 3]
        vel  = data.get("velocities", None)
        acc  = data.get("accelerations", None)

        if vel is not None:
            vel  = data["velocities"][start:end, :, :2]   # [T, 133, 2]
        if acc is not None:
            acc  = data["accelerations"][start:end, :, :2]

        data.close()

        # Reshape keypoints to flat vector
        kpts_flat = kpts.reshape(len(kpts), -1).astype(np.float32)   # [T, 399]

        vel_flat = vel.reshape(len(vel), -1).astype(np.float32) if vel is not None else np.zeros_like(kpts_flat)
        acc_flat = acc.reshape(len(acc), -1).astype(np.float32) if acc is not None else np.zeros_like(kpts_flat)

        # Normalization: hip-center, scale by torso height
        kpts_flat = self._normalize(kpts_flat)

        # Augmentation
        if self.augment:
            kpts_flat, vel_flat, acc_flat = self._augment(kpts_flat, vel_flat, acc_flat)

        return {
            "keypoints": torch.from_numpy(kpts_flat),    # [T, 399]
            "velocities": torch.from_numpy(vel_flat),    # [T, 399]
            "accelerations": torch.from_numpy(acc_flat), # [T, 399]
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _normalize(self, kpts_flat: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints to be hip-centered and scale-invariant.
        Assumes flat layout: [T, 399] with coords in [0,1] normalized screen space.
        """
        # Hip midpoint: indices 11 and 12 (left/right hip) in COCO layout
        # In flat 399 vector: keypoint k starts at k*3
        T = kpts_flat.shape[0]
        kpts = kpts_flat.reshape(T, 133, 3)

        left_hip  = kpts[:, 11, :2]   # [T, 2]
        right_hip = kpts[:, 12, :2]   # [T, 2]
        hip_center = (left_hip + right_hip) / 2.0  # [T, 2]

        left_shoulder  = kpts[:, 5, :2]
        right_shoulder = kpts[:, 6, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        torso_height = np.linalg.norm(shoulder_center - hip_center, axis=-1, keepdims=True)  # [T, 1]
        torso_height = np.clip(torso_height, 0.01, None)

        # Center and scale all xy coordinates
        kpts[:, :, 0] = (kpts[:, :, 0] - hip_center[:, 0:1]) / torso_height
        kpts[:, :, 1] = (kpts[:, :, 1] - hip_center[:, 1:2]) / torso_height

        return kpts.reshape(T, -1)

    def _augment(
        self, kpts: np.ndarray, vel: np.ndarray, acc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply pose-space augmentations."""
        # Random horizontal flip (mirror)
        if random.random() < 0.5:
            kpts, vel, acc = self._flip(kpts), self._flip(vel), self._flip(acc)

        # Add small Gaussian noise to keypoints
        noise_std = random.uniform(0.0, 0.015)
        kpts = kpts + np.random.randn(*kpts.shape).astype(np.float32) * noise_std

        # Random temporal jitter (drop 1-3 frames from start)
        jitter = random.randint(0, 3)
        if jitter > 0 and len(kpts) > jitter:
            kpts = kpts[jitter:]
            vel  = vel[jitter:]
            acc  = acc[jitter:]
            # Pad to original length
            pad = kpts[:jitter]
            kpts = np.concatenate([kpts, pad], axis=0)
            vel  = np.concatenate([vel, vel[:jitter]], axis=0)
            acc  = np.concatenate([acc, acc[:jitter]], axis=0)

        # Random scale ±10%
        scale = random.uniform(0.9, 1.1)
        kpts = kpts * scale

        return kpts, vel, acc

    def _flip(self, flat: np.ndarray) -> np.ndarray:
        """Mirror left-right by negating x coordinates and swapping paired keypoints."""
        T = flat.shape[0]
        kpts = flat.reshape(T, 133, 3).copy()

        # Negate x coordinates
        kpts[:, :, 0] = -kpts[:, :, 0]

        # Swap left/right pairs (body)
        swap_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),  # body
            (17, 20), (18, 21), (19, 22),  # feet (approximate)
        ]
        for l, r in swap_pairs:
            if max(l, r) < 133:
                kpts[:, [l, r]] = kpts[:, [r, l]]

        # Swap left hand (91-111) and right hand (112-132)
        left  = kpts[:, 91:112].copy()
        right = kpts[:, 112:133].copy()
        kpts[:, 91:112]  = right
        kpts[:, 112:133] = left

        return kpts.reshape(T, -1)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced sampling."""
        counts = [0] * len(DANCE_CLASSES)
        for _, _, _, label in self.clips:
            counts[label] += 1
        counts = [max(c, 1) for c in counts]
        weights = [1.0 / c for c in counts]
        return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced multi-class classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        ce_loss = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Training loop for DanceFormer with EMA, LR scheduling, and W&B logging."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: DanceDataset,
        val_dataset: DanceDataset,
        config: dict,
        device: str = "cuda",
        output_dir: Path | str = "weights",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.grad_clip = train_cfg.get("gradient", {}).get("clip_norm", 1.0)
        self.accum_steps = train_cfg.get("gradient", {}).get("accumulation_steps", 1)
        self.dance_loss_w = config.get("model", {}).get("dance_loss_weight", 1.0)
        self.mudra_loss_w = config.get("model", {}).get("mudra_loss_weight", 0.4)

        # DataLoaders with balanced sampling
        class_weights = train_dataset.get_class_weights()
        sample_weights = [class_weights[lbl] for _, _, _, lbl in train_dataset.clips]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
        )

        # Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # Scheduler
        sch_cfg = train_cfg.get("scheduler", {})
        warmup_epochs = sch_cfg.get("warmup_epochs", 10)
        warmup_steps = warmup_epochs * len(self.train_loader)
        total_steps = self.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=opt_cfg.get("lr", 1e-4),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )

        # Loss functions
        loss_cfg = train_cfg.get("loss", {})
        self.dance_loss_fn = FocalLoss(
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", 0.25),
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
        )
        self.mudra_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # EMA
        ema_cfg = train_cfg.get("ema", {})
        self.ema_enabled = ema_cfg.get("enabled", True)
        self.ema_decay = ema_cfg.get("decay", 0.999)
        if self.ema_enabled:
            from copy import deepcopy
            self.ema_model = deepcopy(model)
            self.ema_model.eval()

        # Logging
        self.use_wandb = train_cfg.get("logging", {}).get("use_wandb", False)
        self.best_val_f1 = 0.0
        self.global_step = 0

        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=train_cfg.get("logging", {}).get("project", "natyaveda"),
                    config=config,
                )
            except ImportError:
                self.use_wandb = False

    def train(self) -> None:
        logger.info(f"Starting training: {self.epochs} epochs, device={self.device}")
        logger.info(f"  Train clips: {len(self.train_loader.dataset)}")
        logger.info(f"  Val   clips: {len(self.val_loader.dataset)}")

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
                f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} f1={val_metrics['f1']:.3f}"
            )

            if self.use_wandb:
                import wandb
                wandb.log({**{f"train/{k}": v for k, v in train_metrics.items()},
                           **{f"val/{k}": v for k, v in val_metrics.items()},
                           "epoch": epoch})

            # Save best
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                ckpt_path = self.output_dir / "danceformer_best.pt"
                torch.save({
                    "epoch": epoch,
                    "state_dict": (self.ema_model if self.ema_enabled else self.model).state_dict(),
                    "val_f1": self.best_val_f1,
                    "config": self.config,
                }, str(ckpt_path))
                logger.info(f"  ✅ Best model saved (val F1={self.best_val_f1:.4f})")

            # Periodic checkpoint
            ckpt_cfg = self.config.get("training", {}).get("checkpointing", {})
            if epoch % ckpt_cfg.get("save_every_n_epochs", 5) == 0:
                torch.save({
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }, str(self.output_dir / f"danceformer_epoch{epoch:03d}.pt"))

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(self.train_loader):
            kpts = batch["keypoints"].to(self.device)
            vel  = batch["velocities"].to(self.device)
            acc  = batch["accelerations"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(self.device == "cuda")):
                out = self.model(kpts, vel, acc)
                dance_loss = self.dance_loss_fn(out["dance_logits"], labels)
                loss = self.dance_loss_w * dance_loss
                loss = loss / self.accum_steps

            loss.backward()

            if (step + 1) % self.accum_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.ema_enabled:
                    self._update_ema()

            total_loss += float(dance_loss)
            preds = out["dance_logits"].argmax(-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            self.global_step += 1

        return {"loss": total_loss / len(self.train_loader), "acc": correct / max(total, 1)}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        eval_model = self.ema_model if self.ema_enabled else self.model
        eval_model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            kpts   = batch["keypoints"].to(self.device)
            vel    = batch["velocities"].to(self.device)
            acc    = batch["accelerations"].to(self.device)
            labels = batch["label"].to(self.device)

            out = eval_model(kpts, vel, acc)
            loss = self.dance_loss_fn(out["dance_logits"], labels)
            total_loss += float(loss)
            all_preds.extend(out["dance_logits"].argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return {"loss": total_loss / len(self.val_loader), "acc": acc, "f1": f1}

    def _update_ema(self) -> None:
        """Exponential Moving Average update of model weights."""
        decay = self.ema_decay
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
