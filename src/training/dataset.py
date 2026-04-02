"""
NatyaVeda — PyTorch Dataset
Clean standalone dataset class for training and evaluation.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]
DANCE_TO_IDX = {d: i for i, d in enumerate(DANCE_CLASSES)}


class NatyaVedaDataset(Dataset):
    """
    Dataset that loads pre-extracted .npz pose feature files.

    Args:
        data_dir:     Root of data directory (e.g. data/splits/train)
        clip_length:  Frames per clip (default 64)
        clip_stride:  Sliding-window stride (default 32)
        augment:      Whether to apply pose augmentations
    """

    def __init__(
        self,
        data_dir: str | Path,
        clip_length: int = 64,
        clip_stride: int = 32,
        augment: bool = False,
    ) -> None:
        self.data_dir    = Path(data_dir)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.augment     = augment

        # (npz_path, start_frame, label_idx)
        self.clips: list[tuple[Path, int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        for label_idx, dance in enumerate(DANCE_CLASSES):
            dance_dir = self.data_dir / dance
            if not dance_dir.exists():
                continue
            for npz in sorted(dance_dir.glob("*.npz")):
                # Skip intermediate files
                if any(npz.name.endswith(s) for s in ("_pose.npz","_hands.npz","_videomae.npz")):
                    continue
                d = np.load(str(npz))
                T = int(d["keypoints"].shape[0])
                d.close()
                start = 0
                while start + self.clip_length <= T:
                    self.clips.append((npz, start, label_idx))
                    start += self.clip_stride
                # Last partial clip
                if T > self.clip_length and start < T:
                    self.clips.append((npz, T - self.clip_length, label_idx))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        npz_path, start, label = self.clips[idx]
        end = start + self.clip_length

        d    = np.load(str(npz_path))
        kpts = d["keypoints"][start:end].astype(np.float32)        # [T, 133, 3]
        vel  = d["velocities"][start:end, :, :2].astype(np.float32) if "velocities" in d \
               else np.zeros((self.clip_length, 133, 2), dtype=np.float32)
        acc  = d["accelerations"][start:end, :, :2].astype(np.float32) if "accelerations" in d \
               else np.zeros((self.clip_length, 133, 2), dtype=np.float32)
        d.close()

        # Flatten to [T, D]
        kpts_flat = kpts.reshape(self.clip_length, -1)  # [T, 399]
        vel_flat  = np.pad(vel.reshape(self.clip_length, -1),
                           ((0,0),(0,399-266)), mode="constant") if vel.shape[-1]*133 != 399 \
                    else vel.reshape(self.clip_length, -1)
        acc_flat  = np.pad(acc.reshape(self.clip_length, -1),
                           ((0,0),(0,399-266)), mode="constant") if acc.shape[-1]*133 != 399 \
                    else acc.reshape(self.clip_length, -1)

        # Normalize
        kpts_flat = self._normalize(kpts_flat)

        if self.augment:
            kpts_flat = self._augment(kpts_flat)

        return {
            "keypoints":     torch.from_numpy(kpts_flat),
            "velocities":    torch.from_numpy(vel_flat),
            "accelerations": torch.from_numpy(acc_flat),
            "label":         torch.tensor(label, dtype=torch.long),
            "dance_form":    DANCE_CLASSES[label],
        }

    def _normalize(self, kpts_flat: np.ndarray) -> np.ndarray:
        T = kpts_flat.shape[0]
        k = kpts_flat.reshape(T, 133, 3)
        hip  = ((k[:, 11, :2] + k[:, 12, :2]) / 2.0)
        sh   = ((k[:, 5,  :2] + k[:, 6,  :2]) / 2.0)
        scale = np.linalg.norm(sh - hip, axis=-1, keepdims=True).clip(0.01)
        k[:, :, 0] = (k[:, :, 0] - hip[:, 0:1]) / scale
        k[:, :, 1] = (k[:, :, 1] - hip[:, 1:2]) / scale
        return k.reshape(T, -1)

    def _augment(self, kpts_flat: np.ndarray) -> np.ndarray:
        # Small Gaussian noise
        kpts_flat = kpts_flat + np.random.randn(*kpts_flat.shape).astype(np.float32) * 0.01
        # Random scale
        kpts_flat = kpts_flat * random.uniform(0.9, 1.1)
        return kpts_flat

    def get_class_weights(self) -> torch.Tensor:
        counts = [0] * len(DANCE_CLASSES)
        for _, _, lbl in self.clips:
            counts[lbl] += 1
        weights = [1.0 / max(c, 1) for c in counts]
        return torch.tensor(weights, dtype=torch.float32)

    def class_distribution(self) -> dict[str, int]:
        counts = {d: 0 for d in DANCE_CLASSES}
        for _, _, lbl in self.clips:
            counts[DANCE_CLASSES[lbl]] += 1
        return counts
