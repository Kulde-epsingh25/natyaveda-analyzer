"""NatyaVeda — Keypoint Utilities"""
from __future__ import annotations
import numpy as np


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Hip-center, torso-scale normalization.
    kpts: [T, 133, 3] or [133, 3]
    Returns same shape with x,y normalized; conf unchanged.
    """
    single = kpts.ndim == 2
    if single:
        kpts = kpts[None]  # [1, 133, 3]

    out = kpts.copy()
    left_hip  = kpts[:, 11, :2]
    right_hip = kpts[:, 12, :2]
    hip       = (left_hip + right_hip) / 2.0
    l_sh      = kpts[:, 5, :2]
    r_sh      = kpts[:, 6, :2]
    sh        = (l_sh + r_sh) / 2.0
    scale     = np.linalg.norm(sh - hip, axis=-1, keepdims=True).clip(0.01, None)

    out[:, :, 0] = (kpts[:, :, 0] - hip[:, 0:1]) / scale
    out[:, :, 1] = (kpts[:, :, 1] - hip[:, 1:2]) / scale

    return out[0] if single else out


def compute_joint_angles(kpts: np.ndarray) -> np.ndarray:
    """
    Compute angles at elbow, knee, shoulder for each frame.
    kpts: [T, 133, 3]
    Returns: [T, 6] — left/right elbow, knee, shoulder angles
    """
    def angle(a, b, c):
        ba = a - b; bc = c - b
        cos = np.sum(ba * bc, axis=-1) / (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
        )
        return np.arccos(np.clip(cos, -1, 1))

    T = kpts.shape[0]
    angles = np.zeros((T, 6), dtype=np.float32)

    # Left elbow: shoulder(5) → elbow(7) → wrist(9)
    angles[:, 0] = angle(kpts[:, 5, :2], kpts[:, 7, :2], kpts[:, 9, :2])
    # Right elbow: shoulder(6) → elbow(8) → wrist(10)
    angles[:, 1] = angle(kpts[:, 6, :2], kpts[:, 8, :2], kpts[:, 10, :2])
    # Left knee: hip(11) → knee(13) → ankle(15)
    angles[:, 2] = angle(kpts[:, 11, :2], kpts[:, 13, :2], kpts[:, 15, :2])
    # Right knee: hip(12) → knee(14) → ankle(16)
    angles[:, 3] = angle(kpts[:, 12, :2], kpts[:, 14, :2], kpts[:, 16, :2])
    # Left shoulder: hip(11) → shoulder(5) → elbow(7)
    angles[:, 4] = angle(kpts[:, 11, :2], kpts[:, 5, :2], kpts[:, 7, :2])
    # Right shoulder: hip(12) → shoulder(6) → elbow(8)
    angles[:, 5] = angle(kpts[:, 12, :2], kpts[:, 6, :2], kpts[:, 8, :2])

    return angles


def keypoints_to_dict(kpts: np.ndarray) -> dict:
    """Convert [133, 3] keypoints to named dict for debugging."""
    names = (
        ["nose","left_eye","right_eye","left_ear","right_ear",
         "left_shoulder","right_shoulder","left_elbow","right_elbow",
         "left_wrist","right_wrist","left_hip","right_hip",
         "left_knee","right_knee","left_ankle","right_ankle"]  # 0-16
        + [f"foot_{i}" for i in range(6)]                       # 17-22
        + [f"face_{i}" for i in range(68)]                      # 23-90
        + [f"lhand_{i}" for i in range(21)]                     # 91-111
        + [f"rhand_{i}" for i in range(21)]                     # 112-132
    )
    return {name: kpts[i].tolist() for i, name in enumerate(names)}
