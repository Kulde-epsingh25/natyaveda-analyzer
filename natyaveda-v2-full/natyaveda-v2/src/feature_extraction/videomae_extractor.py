"""
NatyaVeda — VideoMAE-v2 Feature Extractor
Extracts holistic video-level tokens from VideoMAE-v2 (MCG-NJU/videomae-large).
These tokens are fused with pose features via cross-attention in DanceFormer.

Reference:
  VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking
  https://huggingface.co/MCG-NJU/videomae-large
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

VIDEOMAE_MODEL_ID = "MCG-NJU/videomae-large"
VIDEOMAE_FEATURE_DIM = 1024
CLIP_FRAMES = 16          # VideoMAE expects exactly 16 frames per clip
TARGET_SIZE = (224, 224)  # VideoMAE input resolution


class VideoMAEExtractor:
    """
    Extracts spatiotemporal tokens from VideoMAE-v2 Large.

    Usage
    -----
    >>> extractor = VideoMAEExtractor(device="cuda")
    >>> tokens = extractor.extract_video("dance.mp4")
    >>> print(tokens.shape)  # [N_clips, 1024]
    """

    def __init__(
        self,
        model_id: str = VIDEOMAE_MODEL_ID,
        device: str = "cuda",
        batch_clips: int = 4,
    ) -> None:
        self.device = device
        self.batch_clips = batch_clips
        self._processor = None
        self._model = None
        self.available = False
        self._load_model(model_id)

    def _load_model(self, model_id: str) -> None:
        try:
            from transformers import VideoMAEModel, VideoMAEImageProcessor
            self._processor = VideoMAEImageProcessor.from_pretrained(model_id)
            self._model = VideoMAEModel.from_pretrained(model_id)
            self._model = self._model.to(self.device).eval()
            self.available = True
            logger.info(f"VideoMAE-v2 loaded: {model_id}")
        except ImportError:
            logger.warning("transformers not installed — VideoMAE extractor disabled.")
        except Exception as e:
            logger.warning(f"VideoMAE load failed: {e}")

    def extract_video(
        self,
        video_path: Path | str,
        stride_frames: int = 8,
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Extract VideoMAE tokens for a full video using sliding 16-frame clips.

        Parameters
        ----------
        video_path : Path or str
        stride_frames : int
            Stride between consecutive clips. Default 8 (50% overlap).
        normalize : bool
            L2-normalize output features.

        Returns
        -------
        np.ndarray [N_clips, 1024] or None if extraction fails.
        """
        if not self.available:
            return None

        video_path = Path(video_path)
        frames = self._load_video_frames(video_path)
        if len(frames) < CLIP_FRAMES:
            logger.warning(f"Too few frames ({len(frames)}) in {video_path.name}")
            return None

        # Build clips
        clips = []
        starts = range(0, len(frames) - CLIP_FRAMES + 1, stride_frames)
        for start in starts:
            clip = frames[start : start + CLIP_FRAMES]
            clips.append(clip)

        # Batch inference
        all_features = []
        for i in range(0, len(clips), self.batch_clips):
            batch = clips[i : i + self.batch_clips]
            feats = self._inference_batch(batch)
            if feats is not None:
                all_features.append(feats)

        if not all_features:
            return None

        features = np.concatenate(all_features, axis=0)  # [N_clips, 1024]

        if normalize:
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            features = features / (norms + 1e-8)

        logger.info(f"VideoMAE features: {features.shape} from {video_path.name}")
        return features

    def extract_clip(self, frames: list[np.ndarray]) -> Optional[np.ndarray]:
        """Extract VideoMAE features for a single 16-frame clip."""
        if not self.available:
            return None
        if len(frames) < CLIP_FRAMES:
            # Pad by repeating last frame
            frames = frames + [frames[-1]] * (CLIP_FRAMES - len(frames))
        frames = frames[:CLIP_FRAMES]
        feats = self._inference_batch([frames])
        return feats[0] if feats is not None else None

    def _load_video_frames(self, video_path: Path) -> list[np.ndarray]:
        """Load all frames from video, resize to VideoMAE input size."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
            frames.append(frame_resized)
        cap.release()
        return frames

    def _inference_batch(
        self, clips: list[list[np.ndarray]]
    ) -> Optional[np.ndarray]:
        """
        Run VideoMAE inference on a batch of clips.
        Each clip is a list of 16 RGB frames of size (224, 224).
        Returns np.ndarray [batch, 1024].
        """
        try:
            # Process clips through VideoMAE processor
            inputs = self._processor(
                videos=clips,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.device)
            # pixel_values: [B, C, T, H, W] for VideoMAE

            with torch.no_grad():
                outputs = self._model(pixel_values=pixel_values)

            # Use mean-pooled last hidden state as clip feature
            # last_hidden_state: [B, num_patches, 1024]
            hidden = outputs.last_hidden_state   # [B, num_patches, D]
            features = hidden.mean(dim=1)        # [B, D] mean pool over patches
            return features.cpu().float().numpy()

        except Exception as e:
            logger.debug(f"VideoMAE inference error: {e}")
            return None

    def save_features(
        self,
        video_path: Path | str,
        output_path: Path | str,
        stride_frames: int = 8,
    ) -> Optional[Path]:
        """Extract and save VideoMAE features to .npz file."""
        features = self.extract_video(video_path, stride_frames=stride_frames)
        if features is None:
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), videomae_features=features)
        return output_path
