"""NatyaVeda — Pose-Space Augmentations"""
from __future__ import annotations
import numpy as np
import random


class PoseAugmentor:
    """Collection of augmentations applied in keypoint space (not pixel space)."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_std: float = 0.01,
        scale_range: tuple = (0.9, 1.1),
        rotation_deg: float = 15.0,
        temporal_jitter: int = 3,
        drop_joint_prob: float = 0.05,
    ):
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotation_deg = rotation_deg
        self.temporal_jitter = temporal_jitter
        self.drop_joint_prob = drop_joint_prob

    def __call__(self, kpts: np.ndarray) -> np.ndarray:
        """Apply random augmentations. kpts: [T, 133, 3]"""
        if random.random() < self.flip_prob:
            kpts = self._horizontal_flip(kpts)
        kpts = self._add_noise(kpts)
        kpts = self._random_scale(kpts)
        kpts = self._random_rotation(kpts)
        kpts = self._drop_joints(kpts)
        return kpts

    def _horizontal_flip(self, kpts: np.ndarray) -> np.ndarray:
        out = kpts.copy()
        out[:, :, 0] = -out[:, :, 0]
        pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
        for l, r in pairs:
            out[:, [l, r]] = out[:, [r, l]]
        lh = out[:, 91:112].copy()
        rh = out[:, 112:133].copy()
        out[:, 91:112]  = rh
        out[:, 112:133] = lh
        return out

    def _add_noise(self, kpts: np.ndarray) -> np.ndarray:
        std = random.uniform(0, self.noise_std)
        noise = np.random.randn(*kpts.shape).astype(np.float32) * std
        noise[:, :, 2] = 0  # don't noise the confidence
        return kpts + noise

    def _random_scale(self, kpts: np.ndarray) -> np.ndarray:
        s = random.uniform(*self.scale_range)
        out = kpts.copy()
        out[:, :, :2] = kpts[:, :, :2] * s
        return out

    def _random_rotation(self, kpts: np.ndarray) -> np.ndarray:
        angle_rad = np.deg2rad(random.uniform(-self.rotation_deg, self.rotation_deg))
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        out = kpts.copy()
        x = kpts[:, :, 0].copy()
        y = kpts[:, :, 1].copy()
        out[:, :, 0] = cos_a * x - sin_a * y
        out[:, :, 1] = sin_a * x + cos_a * y
        return out

    def _drop_joints(self, kpts: np.ndarray) -> np.ndarray:
        """Randomly zero-out confidence of some joints to simulate occlusion."""
        out = kpts.copy()
        mask = np.random.rand(kpts.shape[1]) < self.drop_joint_prob
        out[:, mask, 2] = 0.0
        return out
