"""
NatyaVeda — Face Extractor
Extracts 68 facial landmarks from RTMW wholebody output (indices 23-90).
Used for Abhinaya (facial expression) and Navarasas (9 emotions) analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Navarasas — 9 principal emotions in Indian classical dance
NAVARASAS = [
    "shringara",    # love / beauty
    "hasya",        # humor / joy
    "karuna",       # compassion / sorrow
    "roudra",       # fury / anger
    "veera",        # heroism / courage
    "bhayanaka",    # fear / terror
    "bibhatsa",     # disgust / aversion
    "adbhuta",      # wonder / amazement
    "shanta",       # peace / tranquility
]

# Key face region indices (within RTMW face kpts 23-90 → local idx 0-67)
FACE_REGIONS = {
    "jawline":       list(range(0, 17)),
    "right_brow":    list(range(17, 22)),
    "left_brow":     list(range(22, 27)),
    "nose_bridge":   list(range(27, 31)),
    "nose_tip":      list(range(31, 36)),
    "right_eye":     list(range(36, 42)),
    "left_eye":      list(range(42, 48)),
    "outer_lip":     list(range(48, 60)),
    "inner_lip":     list(range(60, 68)),
}


@dataclass
class FaceFrame:
    landmarks: np.ndarray     # [68, 3] (x, y, conf)
    frame_idx: int
    timestamp_sec: float

    @property
    def avg_confidence(self) -> float:
        return float(self.landmarks[:, 2].mean())

    def eye_aspect_ratio(self) -> tuple[float, float]:
        """Compute EAR for both eyes (blink / wide-eye detection)."""
        def ear(pts):
            A = np.linalg.norm(pts[1] - pts[5])
            B = np.linalg.norm(pts[2] - pts[4])
            C = np.linalg.norm(pts[0] - pts[3])
            return (A + B) / (2.0 * C + 1e-8)

        r_eye = self.landmarks[36:42, :2]
        l_eye = self.landmarks[42:48, :2]
        return ear(r_eye), ear(l_eye)

    def mouth_aspect_ratio(self) -> float:
        """Compute MAR (open/closed mouth) for expression analysis."""
        outer = self.landmarks[48:60, :2]
        A = np.linalg.norm(outer[2] - outer[10])
        B = np.linalg.norm(outer[4] - outer[8])
        C = np.linalg.norm(outer[0] - outer[6])
        return (A + B) / (2.0 * C + 1e-8)

    def brow_raise(self) -> float:
        """Estimate brow raise (surprise/wonder expression)."""
        r_brow_y = self.landmarks[17:22, 1].mean()
        l_brow_y = self.landmarks[22:27, 1].mean()
        r_eye_y  = self.landmarks[36:42, 1].mean()
        l_eye_y  = self.landmarks[42:48, 1].mean()
        return float(((r_eye_y - r_brow_y) + (l_eye_y - l_brow_y)) / 2.0)

    def expression_features(self) -> np.ndarray:
        """
        Compact expression feature vector for Navarasas classification.
        Returns [12] features.
        """
        r_ear, l_ear = self.eye_aspect_ratio()
        mar = self.mouth_aspect_ratio()
        brow = self.brow_raise()
        # Symmetry
        sym_brow = abs(self.landmarks[17:22, 1].mean() - self.landmarks[22:27, 1].mean())
        sym_eye  = abs(r_ear - l_ear)
        # Landmark spread
        spread = float(np.std(self.landmarks[:, :2]))
        # Normalized jaw width
        jaw_w = float(np.linalg.norm(self.landmarks[0, :2] - self.landmarks[16, :2]))
        return np.array([
            r_ear, l_ear, mar, brow, sym_brow, sym_eye, spread, jaw_w,
            self.landmarks[30, 0],  # nose tip x
            self.landmarks[30, 1],  # nose tip y
            self.avg_confidence,
            float(self.landmarks[36:48, 2].mean()),  # eye confidence
        ], dtype=np.float32)


class FaceExtractor:
    """
    Extracts facial landmarks from RTMW wholebody PoseFrame output.
    RTMW face keypoints occupy indices 23-90 (68 points).
    """

    FACE_SLICE = slice(23, 91)

    def extract_from_pose_frame(self, pose_frame) -> FaceFrame:
        """Pull face landmarks from RTMW PoseFrame."""
        if pose_frame.keypoints.shape[0] < 91:
            return FaceFrame(
                landmarks=np.zeros((68, 3), dtype=np.float32),
                frame_idx=pose_frame.frame_idx,
                timestamp_sec=pose_frame.timestamp_sec,
            )
        face_kpts = pose_frame.keypoints[self.FACE_SLICE].copy()   # [68, 3]
        return FaceFrame(
            landmarks=face_kpts,
            frame_idx=pose_frame.frame_idx,
            timestamp_sec=pose_frame.timestamp_sec,
        )

    def extract_sequence(self, pose_frames: list) -> list[FaceFrame]:
        """Extract face landmarks from a list of PoseFrames."""
        return [self.extract_from_pose_frame(pf) for pf in pose_frames]

    def save_features(self, frames: list[FaceFrame], output_path) -> None:
        import pathlib
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        landmarks  = np.stack([f.landmarks for f in frames])
        expr_feats = np.stack([f.expression_features() for f in frames])
        timestamps = np.array([f.timestamp_sec for f in frames])
        np.savez_compressed(
            str(output_path),
            face_landmarks=landmarks,
            expression_features=expr_feats,
            timestamps=timestamps,
        )
        logger.info(f"Saved face features: {output_path}")
