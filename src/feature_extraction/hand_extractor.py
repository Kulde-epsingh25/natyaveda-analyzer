"""
NatyaVeda — Hand & Finger Extractor
Extracts 21-point hand landmarks per hand (42 total) for mudra recognition.

MediaPipe Hands provides:
  - Wrist (1)
  - Each finger: MCP → PIP → DIP → TIP (4 joints × 5 fingers = 20)
  - Total: 21 landmarks per hand × 2 hands = 42 points

For Indian classical dance, these 42 points encode all Hasta mudras:
  Asamyuta hastas (28 single-hand gestures) and
  Samyuta hastas (24 two-hand gestures)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe 21-point hand landmark names
HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Finger groupings (for per-finger analysis)
FINGER_INDICES = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# MediaPipe hand connections (for visualization)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


# ─────────────────────────────────────────────────────────────────────────────
# Hand Frame Data Class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandFrame:
    """
    Both hands' keypoints for a single frame.

    left_hand  : np.ndarray [21, 3] — (x, y, z) normalized; z is depth relative to wrist
    right_hand : np.ndarray [21, 3] — same
    left_confidence  : float — detection confidence [0,1]
    right_confidence : float — detection confidence [0,1]
    """
    left_hand: np.ndarray          # [21, 3]
    right_hand: np.ndarray         # [21, 3]
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    frame_idx: int = 0
    timestamp_sec: float = 0.0

    @property
    def both_detected(self) -> bool:
        return self.left_confidence > 0.1 and self.right_confidence > 0.1

    def to_feature_vector(self) -> np.ndarray:
        """Concatenate both hands → [126,] vector (42 pts × 3 coords)."""
        return np.concatenate([self.left_hand.flatten(), self.right_hand.flatten()])

    def finger_angles(self, hand: str = "right") -> np.ndarray:
        """
        Compute 5 per-finger flexion angles (MCP→PIP→DIP) for mudra classification.
        Returns [5,] array of angles in radians.
        """
        kpts = self.right_hand if hand == "right" else self.left_hand
        angles = np.zeros(5, dtype=np.float32)
        finger_keys = list(FINGER_INDICES.keys())
        for i, key in enumerate(finger_keys):
            idxs = FINGER_INDICES[key]
            if len(idxs) < 3:
                continue
            # Use MCP → PIP → DIP angle
            a = kpts[idxs[0], :2]
            b = kpts[idxs[1], :2]
            c = kpts[idxs[2], :2]
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angles[i] = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angles

    def mudra_features(self) -> np.ndarray:
        """
        Rich mudra feature vector combining:
          - Raw landmark coords (42×3 = 126)
          - Right finger angles (5)
          - Left finger angles (5)
          - Relative hand positions (3)
          Total: 139 features
        """
        raw = self.to_feature_vector()                    # [126]
        right_angles = self.finger_angles("right")        # [5]
        left_angles = self.finger_angles("left")          # [5]
        rel_pos = (self.left_hand[0, :3] - self.right_hand[0, :3])  # [3] wrist offset
        return np.concatenate([raw, right_angles, left_angles, rel_pos]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Hands Extractor
# ─────────────────────────────────────────────────────────────────────────────

class MediaPipeHandExtractor:
    """
    Extracts 21-point hand landmarks per hand using MediaPipe Hands.
    Provides both 2D (normalized screen) and 3D (world) coordinates.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_3d: bool = True,
    ) -> None:
        self.use_3d = use_3d
        self._hands = None
        self.available = False

        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.available = True
            logger.info("MediaPipe Hands extractor initialized.")
        except ImportError:
            logger.warning("MediaPipe not installed. Hand extractor disabled.")
        except Exception as e:
            logger.warning(f"MediaPipe init failed: {e}")

    def __del__(self):
        if self._hands:
            self._hands.close()

    def extract_frame(
        self, frame: np.ndarray, frame_idx: int = 0, fps: float = 25.0
    ) -> HandFrame:
        """
        Extract hand landmarks from a single BGR frame.
        Returns HandFrame with both hands (zeros if not detected).
        """
        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_conf = 0.0
        right_conf = 0.0

        if not self.available:
            return HandFrame(left_hand, right_hand, left_conf, right_conf, frame_idx, frame_idx / fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lms, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = hand_info.classification[0].label.lower()  # "left" / "right"
                conf = hand_info.classification[0].score

                kpts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

                if label == "left":
                    left_hand = kpts
                    left_conf = float(conf)
                else:
                    right_hand = kpts
                    right_conf = float(conf)

        return HandFrame(
            left_hand=left_hand,
            right_hand=right_hand,
            left_confidence=left_conf,
            right_confidence=right_conf,
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / fps,
        )

    def extract_video(
        self,
        video_path: Path | str,
        sample_fps: float = 25.0,
    ) -> list[HandFrame]:
        """Extract hand poses from every sampled frame of a video."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, int(orig_fps / sample_fps))

        results: list[HandFrame] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                hf = self.extract_frame(frame, frame_idx, orig_fps)
                results.append(hf)
            frame_idx += 1

        cap.release()
        logger.info(
            f"Extracted {len(results)} hand frames from {video_path.name} "
            f"(both hands detected in {sum(1 for r in results if r.both_detected)} frames)"
        )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# RTMW-based hand extractor (uses RTMW wholebody hand keypoints)
# ─────────────────────────────────────────────────────────────────────────────

class RTMWHandExtractor:
    """
    Pulls hand keypoints from RTMW wholebody output (indices 91-132).
    Use this when RTMW pose has already been extracted to avoid double pass.
    """

    def extract_from_pose_frame(self, pose_frame) -> HandFrame:
        """Extract hand data from an existing RTMW PoseFrame."""
        from src.feature_extraction.pose_extractor import LEFT_HAND_SLICE, RIGHT_HAND_SLICE

        if pose_frame.keypoints.shape[0] < 133:
            return HandFrame(
                np.zeros((21, 3), dtype=np.float32),
                np.zeros((21, 3), dtype=np.float32),
                frame_idx=pose_frame.frame_idx,
                timestamp_sec=pose_frame.timestamp_sec,
            )

        left = pose_frame.keypoints[LEFT_HAND_SLICE]    # [21, 3]
        right = pose_frame.keypoints[RIGHT_HAND_SLICE]  # [21, 3]

        left_conf = float(left[:, 2].mean())
        right_conf = float(right[:, 2].mean())

        return HandFrame(
            left_hand=left,
            right_hand=right,
            left_confidence=left_conf,
            right_confidence=right_conf,
            frame_idx=pose_frame.frame_idx,
            timestamp_sec=pose_frame.timestamp_sec,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fused Hand Extractor (MediaPipe + RTMW confidence-weighted fusion)
# ─────────────────────────────────────────────────────────────────────────────

class HandExtractor:
    """
    Confidence-weighted fusion of MediaPipe Hands and RTMW hand keypoints.

    Strategy:
    - If both available: weighted average based on per-frame confidence
    - If only MediaPipe: use MediaPipe
    - If only RTMW: use RTMW wholebody hand portion
    """

    def __init__(
        self,
        use_mediapipe: bool = True,
        use_rtmw: bool = True,
        device: str = "cuda",
    ) -> None:
        self.mp_extractor = MediaPipeHandExtractor() if use_mediapipe else None
        self.rtmw_extractor = RTMWHandExtractor() if use_rtmw else None

    def fuse(
        self,
        mp_frame: Optional[HandFrame],
        rtmw_frame: Optional[HandFrame],
    ) -> HandFrame:
        """Fuse MediaPipe and RTMW hand keypoints by confidence weighting."""
        if mp_frame is None:
            return rtmw_frame
        if rtmw_frame is None:
            return mp_frame

        def _fuse_hand(
            mp_kpts: np.ndarray, mp_conf: float,
            rtmw_kpts: np.ndarray, rtmw_conf: float,
        ) -> tuple[np.ndarray, float]:
            total = mp_conf + rtmw_conf + 1e-8
            fused = (mp_kpts * mp_conf + rtmw_kpts * rtmw_conf) / total
            return fused.astype(np.float32), max(mp_conf, rtmw_conf)

        left, left_conf = _fuse_hand(
            mp_frame.left_hand, mp_frame.left_confidence,
            rtmw_frame.left_hand, rtmw_frame.left_confidence,
        )
        right, right_conf = _fuse_hand(
            mp_frame.right_hand, mp_frame.right_confidence,
            rtmw_frame.right_hand, rtmw_frame.right_confidence,
        )

        return HandFrame(
            left_hand=left,
            right_hand=right,
            left_confidence=left_conf,
            right_confidence=right_conf,
            frame_idx=mp_frame.frame_idx,
            timestamp_sec=mp_frame.timestamp_sec,
        )

    def extract_video(
        self,
        video_path: Path | str,
        pose_frames: Optional[list] = None,  # list[PoseFrame] from RTMW
        sample_fps: float = 25.0,
    ) -> list[HandFrame]:
        """
        Full hand extraction for a video.

        If pose_frames (RTMW) are provided, RTMW hand portion is used directly.
        MediaPipe is run independently for higher finger-tip detail.
        """
        mp_frames: list[HandFrame] = []
        rtmw_frames: list[HandFrame] = []

        if self.mp_extractor and self.mp_extractor.available:
            mp_frames = self.mp_extractor.extract_video(video_path, sample_fps)

        if self.rtmw_extractor and pose_frames:
            rtmw_frames = [
                self.rtmw_extractor.extract_from_pose_frame(pf)
                for pf in pose_frames
            ]

        # Align by frame index and fuse
        if mp_frames and rtmw_frames:
            rtmw_by_idx = {f.frame_idx: f for f in rtmw_frames}
            fused = []
            for mpf in mp_frames:
                rtmwf = rtmw_by_idx.get(mpf.frame_idx)
                fused.append(self.fuse(mpf, rtmwf))
            return fused

        return mp_frames or rtmw_frames

    def save_features(
        self,
        frames: list[HandFrame],
        output_path: Path | str,
    ) -> Path:
        """Save hand feature arrays to .npz file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        left_kpts = np.stack([f.left_hand for f in frames])     # [T, 21, 3]
        right_kpts = np.stack([f.right_hand for f in frames])   # [T, 21, 3]
        left_conf = np.array([f.left_confidence for f in frames])
        right_conf = np.array([f.right_confidence for f in frames])
        timestamps = np.array([f.timestamp_sec for f in frames])

        # Compute mudra feature vectors
        mudra_feats = np.stack([f.mudra_features() for f in frames])  # [T, 139]

        np.savez_compressed(
            str(output_path),
            left_hand=left_kpts,
            right_hand=right_kpts,
            left_confidence=left_conf,
            right_confidence=right_conf,
            timestamps=timestamps,
            mudra_features=mudra_feats,
        )
        logger.info(f"Saved hand features: {output_path} ({left_kpts.shape})")
        return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Mudra Recognizer (rule-based + learned)
# ─────────────────────────────────────────────────────────────────────────────

class MudraRecognizer:
    """
    Classifies Hasta mudras from hand keypoints.
    Uses a combination of geometric rules and a small learned classifier.
    """

    MUDRA_NAMES = [
        "pataka", "tripataka", "ardhapataka", "kartarimukha",
        "mayura", "ardhachandra", "arala", "shukatunda",
        "mushti", "shikhara", "kapittha", "katakamukha",
        "suchi", "chandrakala", "padmakosha", "sarpashirsha",
        "mrigashirsha", "simhamukha", "kangula", "alapadma",
        "chatura", "bhramara", "hamsasya", "hamsapaksha",
        "sandamsha", "mukula", "tamrachuda", "trishula",
    ]

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu") -> None:
        self.classifier = None
        if weights_path and Path(weights_path).exists():
            try:
                import torch
                self.classifier = torch.load(weights_path, map_location=device)
                self.classifier.eval()
                logger.info(f"MudraRecognizer loaded from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load mudra classifier: {e}")

    def predict(self, hand_frame: HandFrame, hand: str = "right") -> tuple[str, float]:
        """
        Predict the mudra for a single hand.

        Returns: (mudra_name, confidence)
        """
        if self.classifier is not None:
            return self._model_predict(hand_frame, hand)
        return self._geometric_predict(hand_frame, hand)

    def _model_predict(self, hand_frame: HandFrame, hand: str) -> tuple[str, float]:
        import torch
        feats = torch.from_numpy(hand_frame.mudra_features()).unsqueeze(0)
        with torch.no_grad():
            logits = self.classifier(feats)
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
        return self.MUDRA_NAMES[idx % len(self.MUDRA_NAMES)], float(probs[idx])

    def _geometric_predict(self, hand_frame: HandFrame, hand: str) -> tuple[str, float]:
        """Simple geometric rules for 5 common mudras."""
        kpts = hand_frame.right_hand if hand == "right" else hand_frame.left_hand
        angles = hand_frame.finger_angles(hand)

        # Alapadma: all fingers spread → all angles close to π
        if np.all(angles > 2.0):
            return "alapadma", 0.75

        # Mushti: all fingers folded → all angles close to 0
        if np.all(angles < 0.5):
            return "mushti", 0.75

        # Pataka: all fingers extended, angles moderate
        if np.all(angles > 1.5) and angles[0] < 1.5:  # thumb bent
            return "pataka", 0.65

        # Suchi: only index extended
        if angles[1] > 2.0 and np.all(angles[[0, 2, 3, 4]] < 1.0):
            return "suchi", 0.70

        # Trishula: index + middle + ring extended
        if angles[1] > 1.8 and angles[2] > 1.8 and angles[3] > 1.8 and angles[4] < 0.8:
            return "trishula", 0.65

        return "unknown", 0.30
