"""
NatyaVeda — Pose Extractor
Primary: RTMW-x (MMPose) — 133 COCO-WholeBody keypoints
Fallback: MoveNet Thunder (TF Hub) — 17 keypoints

RTMW keypoint layout (133 points):
  [0-16]   Body (COCO 17-point): nose, eyes, ears, shoulders, elbows,
            wrists, hips, knees, ankles
  [17-22]  Foot (6): big toe, small toe, heel × 2 feet
  [23-90]  Face (68): facial landmarks (jawline, brows, eyes, nose, lips)
  [91-111] Left hand (21): wrist + 4 fingers × 4 joints + 4 tips
  [112-132] Right hand (21): same layout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RTMW_NUM_KEYPOINTS = 133
MOVENET_NUM_KEYPOINTS = 17

BODY_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FOOT_KPT_NAMES = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

HAND_KPT_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Keypoint group slices
BODY_SLICE = slice(0, 17)
FOOT_SLICE = slice(17, 23)
FACE_SLICE = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PoseFrame:
    """
    All keypoints for a single video frame.

    keypoints: np.ndarray shape [N_KPT, 3] — (x, y, confidence)
               Coordinates are normalized to [0,1] relative to frame dims.
    frame_idx: original frame index in video
    timestamp_sec: time in seconds
    """
    keypoints: np.ndarray                  # [133, 3] or [17, 3]
    frame_idx: int
    timestamp_sec: float
    frame_hw: tuple[int, int]              # (height, width)
    source: str = "rtmw"                   # 'rtmw' | 'movenet'
    avg_body_confidence: float = 0.0

    def __post_init__(self):
        body_kpts = self.keypoints[BODY_SLICE]
        self.avg_body_confidence = float(body_kpts[:, 2].mean())

    @property
    def body(self) -> np.ndarray:
        return self.keypoints[BODY_SLICE]

    @property
    def foot(self) -> np.ndarray:
        if self.keypoints.shape[0] >= 23:
            return self.keypoints[FOOT_SLICE]
        return np.zeros((6, 3))

    @property
    def face(self) -> np.ndarray:
        if self.keypoints.shape[0] >= 91:
            return self.keypoints[FACE_SLICE]
        return np.zeros((68, 3))

    @property
    def left_hand(self) -> np.ndarray:
        if self.keypoints.shape[0] >= 112:
            return self.keypoints[LEFT_HAND_SLICE]
        return np.zeros((21, 3))

    @property
    def right_hand(self) -> np.ndarray:
        if self.keypoints.shape[0] == 133:
            return self.keypoints[RIGHT_HAND_SLICE]
        return np.zeros((21, 3))

    def to_feature_vector(self, include_conf: bool = True) -> np.ndarray:
        """Flatten all keypoints to a 1D feature vector."""
        if include_conf:
            return self.keypoints.flatten()   # [399] for RTMW-133
        return self.keypoints[:, :2].flatten()  # [266] for RTMW-133


# ─────────────────────────────────────────────────────────────────────────────
# RTMW-x Pose Extractor (MMPose)
# ─────────────────────────────────────────────────────────────────────────────

class RTMWExtractor:
    """
    Wraps RTMW-x whole-body pose estimator from MMPose.

    Extracts 133 keypoints per person: body(17) + foot(6) + face(68) + hands(42).
    This is the primary extractor for NatyaVeda.
    """

    MODEL_CONFIGS = {
        "rtmw-x": {
            "config": "td-hm_rtmw-x_8xb2-270e_coco-wholebody-384x288",
            "checkpoint": (
                "https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/"
                "rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288"
                "-f840f204_20231122.pth"
            ),
        },
        "rtmw-l": {
            "config": "td-hm_rtmw-l_8xb2-270e_coco-wholebody-384x288",
            "checkpoint": (
                "https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/"
                "rtmw/rtmw-l_simcc-cocktail14_pt-ucoco_270e-256x192"
                "-59bfd035_20231122.pth"
            ),
        },
    }

    def __init__(
        self,
        model_size: str = "rtmw-x",
        device: str = "cuda",
        bbox_threshold: float = 0.3,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.bbox_threshold = bbox_threshold
        self._model = None
        self._detector = None
        self._load_models()

    def _load_models(self) -> None:
        """Load MMPose RTMW model and RT-DETR person detector."""
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmdet.apis import init_detector, inference_detector

            cfg = self.MODEL_CONFIGS[self.model_size]
            self._model = init_model(cfg["config"], cfg["checkpoint"], device=self.device)
            self._inference_topdown = inference_topdown

            # Person detector (MMDet RT-DETR or YOLO)
            self._detector = init_detector(
                "rtdetr_r50vd_6x_coco",
                "https://download.openmmlab.com/mmdetection/v3.0/rtdetr/"
                "rtdetr_r50vd_6x_coco_20230804_134640-dc40195d.pth",
                device=self.device,
            )
            self._inference_detector = inference_detector

            logger.info(f"RTMW Extractor loaded: {self.model_size}")
            self.available = True
        except (ImportError, Exception) as e:
            logger.warning(f"MMPose/MMDet not available: {e}. RTMW extractor disabled.")
            self.available = False

    def extract_frame(
        self, frame: np.ndarray, frame_idx: int, fps: float = 25.0
    ) -> Optional[PoseFrame]:
        """
        Extract whole-body keypoints from a single BGR frame.
        Returns PoseFrame or None if extraction fails / no person found.
        """
        if not self.available:
            return None

        h, w = frame.shape[:2]
        try:
            # Person detection
            det_result = self._inference_detector(self._detector, frame)
            bboxes = det_result.pred_instances.bboxes.cpu().numpy()
            scores = det_result.pred_instances.scores.cpu().numpy()
            labels = det_result.pred_instances.labels.cpu().numpy()

            # Filter: person class (0), confidence threshold
            person_mask = (labels == 0) & (scores > self.bbox_threshold)
            person_bboxes = bboxes[person_mask]

            if len(person_bboxes) == 0:
                return None

            # Select principal dancer (largest bbox)
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in person_bboxes]
            main_bbox = person_bboxes[np.argmax(areas)]

            # Whole-body pose estimation
            pose_results = self._inference_topdown(self._model, frame, [main_bbox])
            if not pose_results:
                return None

            kpts = pose_results[0].pred_instances.keypoints[0]      # [133, 2]
            kpt_scores = pose_results[0].pred_instances.keypoint_scores[0]  # [133]

            # Normalize to [0, 1]
            kpts_norm = kpts / np.array([w, h])
            keypoints = np.concatenate(
                [kpts_norm, kpt_scores[:, None]], axis=1
            ).astype(np.float32)   # [133, 3]

            return PoseFrame(
                keypoints=keypoints,
                frame_idx=frame_idx,
                timestamp_sec=frame_idx / fps,
                frame_hw=(h, w),
                source="rtmw",
            )

        except Exception as e:
            logger.debug(f"RTMW extraction failed (frame {frame_idx}): {e}")
            return None

    def extract_video(
        self,
        video_path: Path | str,
        sample_fps: float = 25.0,
        min_confidence: float = 0.40,
    ) -> list[PoseFrame]:
        """
        Extract pose from all frames of a video.

        Parameters
        ----------
        video_path : Path
        sample_fps : float
            Frame rate to sample at (use lower to speed up).
        min_confidence : float
            Drop frames with avg body keypoint confidence below this.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, int(orig_fps / sample_fps))

        results: list[PoseFrame] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                pf = self.extract_frame(frame, frame_idx, fps=orig_fps)
                if pf and pf.avg_body_confidence >= min_confidence:
                    results.append(pf)
            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(results)} valid pose frames from {video_path.name}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# MoveNet Thunder Extractor (TF Hub fallback — 17 keypoints)
# ─────────────────────────────────────────────────────────────────────────────

class MoveNetExtractor:
    """
    MoveNet Thunder via TensorFlow Hub.
    17 keypoints: body only (no hands/face).
    Used as fallback or supplementary signal.

    Reference: https://www.tensorflow.org/hub/tutorials/movenet
    """

    MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    INPUT_SIZE = 256  # MoveNet Thunder expects 256×256

    def __init__(self, device: str = "cpu") -> None:
        self._model = None
        self._infer = None
        self.available = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            model = hub.load(self.MODEL_URL)
            self._infer = model.signatures["serving_default"]
            self.available = True
            logger.info("MoveNet Thunder loaded from TF Hub.")
        except ImportError:
            logger.warning("TensorFlow not installed. MoveNet extractor disabled.")
        except Exception as e:
            logger.warning(f"MoveNet load failed: {e}")

    def extract_frame(
        self, frame: np.ndarray, frame_idx: int, fps: float = 25.0
    ) -> Optional[PoseFrame]:
        if not self.available:
            return None
        import tensorflow as tf

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = tf.image.resize_with_pad(
            tf.expand_dims(rgb, axis=0), self.INPUT_SIZE, self.INPUT_SIZE
        )
        input_img = tf.cast(input_img, dtype=tf.int32)

        outputs = self._infer(input=input_img)
        kpts = outputs["output_0"].numpy()[0, 0]  # [17, 3]: y, x, conf

        # MoveNet returns (y, x, conf) — convert to (x, y, conf) normalized [0,1]
        keypoints = np.stack([kpts[:, 1], kpts[:, 0], kpts[:, 2]], axis=1).astype(np.float32)

        return PoseFrame(
            keypoints=keypoints,
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / fps,
            frame_hw=(h, w),
            source="movenet",
        )

    def extract_video(
        self,
        video_path: Path | str,
        sample_fps: float = 25.0,
        min_confidence: float = 0.30,
    ) -> list[PoseFrame]:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, int(orig_fps / sample_fps))
        results = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                pf = self.extract_frame(frame, frame_idx, fps=orig_fps)
                if pf and pf.avg_body_confidence >= min_confidence:
                    results.append(pf)
            frame_idx += 1
        cap.release()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Unified Pose Extractor (selects and fuses both models)
# ─────────────────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    High-level extractor: uses RTMW-x as primary, MoveNet as fallback/fusion.

    When RTMW confidence is low on a frame, MoveNet body keypoints are
    substituted into the body portion of the feature vector.
    """

    def __init__(
        self,
        pose_model: str = "rtmw-x",
        use_movenet_fallback: bool = True,
        device: str = "cuda",
        sample_fps: float = 25.0,
        min_confidence: float = 0.40,
    ) -> None:
        self.sample_fps = sample_fps
        self.min_confidence = min_confidence
        self.use_movenet_fallback = use_movenet_fallback

        self.rtmw = RTMWExtractor(model_size=pose_model, device=device)

        if use_movenet_fallback:
            self.movenet = MoveNetExtractor(device=device)
        else:
            self.movenet = None

    def extract_video(self, video_path: Path | str) -> list[PoseFrame]:
        """Extract poses from video with fallback logic."""
        video_path = Path(video_path)

        if self.rtmw.available:
            frames = self.rtmw.extract_video(
                video_path, self.sample_fps, self.min_confidence
            )
        else:
            logger.warning("RTMW unavailable — using MoveNet only (17 kpts, no hands/face)")
            frames = self.movenet.extract_video(
                video_path, self.sample_fps, self.min_confidence * 0.8
            ) if self.movenet else []

        # Optionally fuse MoveNet body into low-confidence RTMW frames
        if self.use_movenet_fallback and self.movenet and self.movenet.available:
            frames = self._fuse_movenet(video_path, frames)

        return frames

    def _fuse_movenet(
        self, video_path: Path, rtmw_frames: list[PoseFrame]
    ) -> list[PoseFrame]:
        """
        For frames where RTMW body confidence < threshold,
        replace body keypoints with MoveNet predictions.
        """
        low_conf_indices = [
            i for i, pf in enumerate(rtmw_frames)
            if pf.avg_body_confidence < self.min_confidence
        ]

        if not low_conf_indices:
            return rtmw_frames

        logger.info(f"Fusing MoveNet for {len(low_conf_indices)} low-confidence frames")

        cap = cv2.VideoCapture(str(video_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        for i in low_conf_indices:
            pf = rtmw_frames[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, pf.frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            mv_frame = self.movenet.extract_frame(frame, pf.frame_idx, orig_fps)
            if mv_frame:
                # Replace body keypoints only
                pf.keypoints[BODY_SLICE] = mv_frame.keypoints[BODY_SLICE]
        cap.release()
        return rtmw_frames

    def save_features(
        self,
        frames: list[PoseFrame],
        output_path: Path | str,
        include_velocities: bool = True,
        include_accelerations: bool = True,
    ) -> Path:
        """
        Save extracted pose features as compressed numpy archive.

        Saved arrays:
          - keypoints:      [T, 133, 3]
          - velocities:     [T, 133, 2]  (if requested)
          - accelerations:  [T, 133, 2]  (if requested)
          - frame_indices:  [T]
          - timestamps:     [T]
          - confidences:    [T]          (avg body confidence per frame)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        kpts = np.stack([pf.keypoints for pf in frames])     # [T, N_KPT, 3]
        frame_indices = np.array([pf.frame_idx for pf in frames])
        timestamps = np.array([pf.timestamp_sec for pf in frames])
        confidences = np.array([pf.avg_body_confidence for pf in frames])

        save_dict = {
            "keypoints": kpts,
            "frame_indices": frame_indices,
            "timestamps": timestamps,
            "confidences": confidences,
        }

        if include_velocities and len(frames) > 1:
            # First-order temporal difference (pixel/frame)
            coords = kpts[:, :, :2]  # [T, N_KPT, 2]
            vel = np.diff(coords, axis=0, prepend=coords[:1])
            save_dict["velocities"] = vel.astype(np.float32)

        if include_accelerations and len(frames) > 2:
            vel = save_dict.get("velocities", np.diff(kpts[:, :, :2], axis=0, prepend=kpts[:1, :, :2]))
            acc = np.diff(vel, axis=0, prepend=vel[:1])
            save_dict["accelerations"] = acc.astype(np.float32)

        np.savez_compressed(str(output_path), **save_dict)
        logger.info(f"Saved pose features: {output_path} ({kpts.shape})")
        return output_path
