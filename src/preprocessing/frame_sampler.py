"""NatyaVeda — Adaptive Frame Sampler"""
from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path


class FrameSampler:
    """Sample frames from a video at a target fps, with optional uniform or adaptive strategy."""

    def __init__(self, target_fps: float = 25.0, strategy: str = "uniform"):
        self.target_fps = target_fps
        self.strategy   = strategy  # "uniform" | "adaptive"

    def sample(self, video_path: Path, max_frames: int | None = None) -> list[tuple[int, np.ndarray]]:
        """
        Returns list of (frame_idx, frame_bgr) tuples sampled at target_fps.
        """
        cap = cv2.VideoCapture(str(video_path))
        orig_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(orig_fps / self.target_fps))

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_skip == 0:
                frames.append((idx, frame))
                if max_frames and len(frames) >= max_frames:
                    break
            idx += 1

        cap.release()
        return frames

    def sample_uniform(self, video_path: Path, n_frames: int) -> list[tuple[int, np.ndarray]]:
        """Sample exactly n_frames evenly spaced across the video."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((int(idx), frame))
        cap.release()
        return frames
