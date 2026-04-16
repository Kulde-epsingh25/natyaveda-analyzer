"""
NatyaVeda — Unified Pose Extractor
Automatically selects the best available pose model based on GPU:

  Tier 1 (GPU ≥8GB)  → RTMW-x via MMPose          — 133 COCO-WholeBody keypoints
  Tier 2 (GPU ≥6GB)  → VitPose-Plus-Huge via HF    — 133 COCO-WholeBody keypoints
  Tier 3 (GPU ≥4GB)  → VitPose-Plus-Large via HF   — 133 COCO-WholeBody keypoints
  Tier 4 (GPU <4GB)  → VitPose-Plus-Base via HF    — 133 COCO-WholeBody keypoints
  Tier 5 (CPU only)  → VitPose-Plus-Base via HF    — 133 COCO-WholeBody keypoints
  Fallback           → MoveNet Thunder via TF Hub   — 17 body keypoints

All tiers produce the same output format: PoseFrame with keypoints [N_KPT, 3].
Body keypoints always use COCO-17 layout at indices 0-16.
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
# COCO WholeBody keypoint layout (133 points)
# ─────────────────────────────────────────────────────────────────────────────
BODY_SLICE      = slice(0, 17)
FOOT_SLICE      = slice(17, 23)
FACE_SLICE      = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE= slice(112, 133)
TOTAL_KEYPOINTS = 133


# ─────────────────────────────────────────────────────────────────────────────
# PoseFrame — universal output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PoseFrame:
    keypoints:   np.ndarray    # [N_KPT, 3] — (x_norm, y_norm, confidence)
    frame_idx:   int
    timestamp_sec: float
    frame_hw:    tuple[int, int]
    source:      str = "unknown"  # 'rtmw' | 'vitpose' | 'movenet'
    n_keypoints: int = 17

    def __post_init__(self):
        self.n_keypoints = self.keypoints.shape[0]
        body = self.keypoints[:min(17, self.n_keypoints)]
        self.avg_body_confidence = float(body[:, 2].mean())

    @property
    def body(self) -> np.ndarray:
        return self.keypoints[BODY_SLICE]

    @property
    def left_hand(self) -> np.ndarray:
        if self.n_keypoints >= 112:
            return self.keypoints[LEFT_HAND_SLICE]
        return np.zeros((21, 3), dtype=np.float32)

    @property
    def right_hand(self) -> np.ndarray:
        if self.n_keypoints >= 133:
            return self.keypoints[RIGHT_HAND_SLICE]
        return np.zeros((21, 3), dtype=np.float32)

    @property
    def face(self) -> np.ndarray:
        if self.n_keypoints >= 91:
            return self.keypoints[FACE_SLICE]
        return np.zeros((68, 3), dtype=np.float32)

    def to_feature_vector(self) -> np.ndarray:
        """Pad to 133 kpts × 3 = 399 features."""
        if self.n_keypoints == TOTAL_KEYPOINTS:
            return self.keypoints.flatten().astype(np.float32)
        padded = np.zeros((TOTAL_KEYPOINTS, 3), dtype=np.float32)
        n = min(self.n_keypoints, TOTAL_KEYPOINTS)
        padded[:n] = self.keypoints[:n]
        return padded.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — RTMW-x via MMPose (primary, 133 keypoints)
# ─────────────────────────────────────────────────────────────────────────────

class RTMWExtractor:
    """
    RTMW-x via MMPose — 133 COCO-WholeBody keypoints.
    Requires: mmpose, mmdet (installed via scripts/install.py with miropsota wheels)
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

    def __init__(self, model_size: str = "rtmw-x", device: str = "cuda") -> None:
        self.model_size = model_size
        self.device = device
        self._pose_model = None
        self._detector = None
        self.available = False
        self._load()

    def _load(self) -> None:
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmdet.apis import init_detector, inference_detector

            cfg = self.MODEL_CONFIGS[self.model_size]
            self._pose_model = init_model(cfg["config"], cfg["checkpoint"], device=self.device)
            self._infer_pose = inference_topdown
            self._detector = init_detector(
                "rtdetr_r50vd_6x_coco",
                "https://download.openmmlab.com/mmdetection/v3.0/rtdetr/"
                "rtdetr_r50vd_6x_coco_20230804_134640-dc40195d.pth",
                device=self.device,
            )
            self._infer_det = inference_detector
            self.available = True
            logger.info("✓ RTMW-%s loaded (133 keypoints)", self.model_size.split("-")[-1])
        except ImportError as e:
            logger.warning("MMPose not available (%s) — falling back to VitPose", e)
        except Exception as e:
            logger.warning("RTMW load error: %s — falling back to VitPose", e)

    def extract_frame(self, frame: np.ndarray, frame_idx: int, fps: float = 25.0) -> Optional[PoseFrame]:
        if not self.available:
            return None
        h, w = frame.shape[:2]
        try:
            det = self._infer_det(self._detector, frame)
            bboxes = det.pred_instances.bboxes.cpu().numpy()
            scores = det.pred_instances.scores.cpu().numpy()
            labels = det.pred_instances.labels.cpu().numpy()
            persons = bboxes[(labels == 0) & (scores > 0.3)]
            if len(persons) == 0:
                return None
            main_box = persons[np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in persons])]
            results = self._infer_pose(self._pose_model, frame, [main_box])
            if not results:
                return None
            kpts = results[0].pred_instances.keypoints[0]          # [133, 2]
            scores_kpt = results[0].pred_instances.keypoint_scores[0]  # [133]
            kpts_norm = kpts / np.array([w, h])
            keypoints = np.concatenate([kpts_norm, scores_kpt[:, None]], axis=1).astype(np.float32)
            return PoseFrame(keypoints=keypoints, frame_idx=frame_idx,
                             timestamp_sec=frame_idx/fps, frame_hw=(h, w), source="rtmw")
        except Exception as e:
            logger.debug("RTMW frame error: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2-4 — VitPose-Plus via HuggingFace (Python 3.12 native, 133 keypoints)
# ─────────────────────────────────────────────────────────────────────────────

class VitPoseExtractor:
    """
    VitPose-Plus via HuggingFace Transformers — 133 COCO-WholeBody keypoints.
    Pure Python, no compilation, Python 3.12 native.
    Downloads model automatically from HuggingFace Hub.

    Models:
      usyd-community/vitpose-plus-huge  — 899M params, best quality
      usyd-community/vitpose-plus-large — 434M params, balanced
      usyd-community/vitpose-plus-base  — 125M params, fast/CPU
    """

    def __init__(
        self,
        model_id: str = "usyd-community/vitpose-plus-base",
        device: str = "cpu",
        fp16: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.fp16 = fp16
        self._model = None
        self._processor = None
        self._detector = None
        self.available = False
        self._load()

    def _load(self) -> None:
        try:
            from transformers import VitPoseForPoseEstimation, AutoProcessor
            import torch

            dtype = torch.float16 if self.fp16 and self.device == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = VitPoseForPoseEstimation.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(self.device)
            self._model.eval()

            # Person detector using HuggingFace DETR (no MMDet needed)
            from transformers import DetrForObjectDetection, DetrImageProcessor
            self._det_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self._det_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50", torch_dtype=dtype
            ).to(self.device)
            self._det_model.eval()

            self.available = True
            model_size = self.model_id.split("-")[-1]
            logger.info("✓ VitPose-Plus-%s loaded (133 keypoints, device=%s)", model_size, self.device)
        except ImportError as e:
            logger.warning("VitPose load error (transformers): %s", e)
        except Exception as e:
            logger.warning("VitPose load error: %s", e)

    def _detect_persons(self, frame_rgb: np.ndarray) -> list[list[float]]:
        """Returns list of [x1, y1, x2, y2] boxes for detected persons."""
        import torch
        from PIL import Image
        pil = Image.fromarray(frame_rgb)
        inputs = self._det_processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._det_model(**inputs)
        h, w = frame_rgb.shape[:2]
        results = self._det_processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.5
        )[0]
        PERSON_CLASS = 1
        boxes = []
        for box, label in zip(results["boxes"], results["labels"]):
            if int(label) == PERSON_CLASS:
                boxes.append(box.tolist())
        return boxes

    def extract_frame(self, frame: np.ndarray, frame_idx: int, fps: float = 25.0) -> Optional[PoseFrame]:
        if not self.available:
            return None
        import torch
        from PIL import Image

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Detect persons
            boxes = self._detect_persons(frame_rgb)
            if not boxes:
                return None
            # Pick largest box (principal dancer)
            main_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            boxes_tensor = torch.tensor([main_box], dtype=torch.float32)

            # VitPose inference
            pil = Image.fromarray(frame_rgb)
            inputs = self._processor(
                images=pil,
                boxes=[boxes_tensor.unsqueeze(0).tolist()],
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            ctx = torch.autocast("cuda", torch.float16) if self.fp16 and self.device == "cuda" \
                  else __import__("contextlib").nullcontext()
            with torch.no_grad(), ctx:
                outputs = self._model(**inputs)

            # Post-process → keypoints in pixel coords
            pose_results = self._processor.post_process_pose_estimation(
                outputs, boxes=[[[main_box]]]
            )[0][0]

            kpts_px = np.array([[kp["x"], kp["y"]] for kp in pose_results["keypoints"]])
            scores  = np.array([kp["score"] for kp in pose_results["keypoints"]])

            # Normalize to [0, 1]
            kpts_norm = kpts_px / np.array([w, h])
            keypoints = np.concatenate([kpts_norm, scores[:, None]], axis=1).astype(np.float32)

            return PoseFrame(
                keypoints=keypoints, frame_idx=frame_idx,
                timestamp_sec=frame_idx/fps, frame_hw=(h, w),
                source=f"vitpose-{self.model_id.split('-')[-1]}"
            )
        except Exception as e:
            logger.debug("VitPose frame error: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 5 Fallback — MoveNet Thunder via TF Hub (17 body keypoints)
# ─────────────────────────────────────────────────────────────────────────────

class MoveNetExtractor:
    """MoveNet Thunder via TF Hub — 17 COCO body keypoints, CPU friendly."""

    MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"

    def __init__(self) -> None:
        self._infer = None
        self.available = False
        self._load()

    def _load(self) -> None:
        try:
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow_hub as hub
            model = hub.load(self.MODEL_URL)
            self._infer = model.signatures["serving_default"]
            self.available = True
            logger.info("✓ MoveNet Thunder loaded (17 keypoints, CPU)")
        except Exception as e:
            logger.warning("MoveNet load error: %s", e)

    def extract_frame(self, frame: np.ndarray, frame_idx: int, fps: float = 25.0) -> Optional[PoseFrame]:
        if not self.available:
            return None
        import tensorflow as tf
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = tf.image.resize_with_pad(tf.expand_dims(rgb, 0), 256, 256)
        input_img = tf.cast(input_img, tf.int32)
        out = self._infer(input=input_img)["output_0"].numpy()[0, 0]  # [17, 3] y,x,conf
        # MoveNet returns (y, x, conf) → convert to (x, y, conf) normalized
        keypoints = np.stack([out[:, 1], out[:, 0], out[:, 2]], axis=1).astype(np.float32)
        return PoseFrame(keypoints=keypoints, frame_idx=frame_idx,
                         timestamp_sec=frame_idx/fps, frame_hw=(h, w), source="movenet")


# ─────────────────────────────────────────────────────────────────────────────
# Unified PoseExtractor — selects tier automatically
# ─────────────────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    Auto-selects the best available pose model based on GPU VRAM.
    Always tries in priority order: RTMW-x → VitPose → MoveNet.

    All models output the same PoseFrame format (399-dim feature vector
    when padded to 133 keypoints).
    """

    def __init__(
        self,
        device_manager=None,
        sample_fps: float = 25.0,
        min_confidence: float = 0.30,
        use_ensemble: bool = True,
        force_movenet_only: bool = False,
    ) -> None:
        self.sample_fps = sample_fps
        self.min_confidence = min_confidence
        self.use_ensemble = use_ensemble
        self.force_movenet_only = force_movenet_only

        # Import here to avoid circular
        if device_manager is None:
            from src.utils.device import DeviceManager
            device_manager = DeviceManager()

        self.dm = device_manager
        self.primary: Optional[object] = None
        self.secondary: Optional[VitPoseExtractor] = None
        self.fallback: Optional[MoveNetExtractor] = None
        self._init_models()

    def _init_models(self) -> None:
        tier = self.dm.pose_model_tier
        device = self.dm.device
        fp16   = self.dm.fp16

        if self.force_movenet_only:
            self.fallback = MoveNetExtractor()
            active = []
            if self.fallback and self.fallback.available:
                active.append("MoveNet(fallback)")
            logger.info("Active pose extractors: %s", " + ".join(active) if active else "NONE")
            return

        # Primary: RTMW-x (MMPose) for GPU ≥8GB
        if tier == "rtmw-x":
            self.primary = RTMWExtractor(model_size="rtmw-x", device=device)
            if not self.primary.available:
                logger.warning("RTMW-x unavailable, falling back to VitPose")
                self.primary = None
                tier = "vitpose-huge"

        # Secondary/primary: VitPose-Plus (GPU ≥2GB or CPU)
        if tier.startswith("vitpose") or (tier == "rtmw-x" and self.use_ensemble):
            model_id = self.dm.vitpose_model_id
            self.secondary = VitPoseExtractor(model_id=model_id, device=device, fp16=fp16)

        # Always load MoveNet as final fallback
        self.fallback = MoveNetExtractor()

        # Log what's actually available
        active = []
        if self.primary and self.primary.available:
            active.append(f"RTMW-x(primary)")
        if self.secondary and self.secondary.available:
            active.append(f"VitPose({self.dm.vitpose_model_id.split('/')[-1]})")
        if self.fallback and self.fallback.available:
            active.append("MoveNet(fallback)")
        logger.info("Active pose extractors: %s", " + ".join(active) if active else "NONE")

    def extract_frame(self, frame: np.ndarray, frame_idx: int, fps: float = 25.0) -> Optional[PoseFrame]:
        """Extract pose from a single frame using best available model."""
        # Try primary (RTMW-x)
        if self.primary and self.primary.available:
            pf = self.primary.extract_frame(frame, frame_idx, fps)
            if pf and pf.avg_body_confidence >= self.min_confidence:
                # Optionally ensemble with VitPose for hand refinement
                if self.use_ensemble and self.secondary and self.secondary.available:
                    pf = self._ensemble(pf, frame, frame_idx, fps)
                return pf

        # Try VitPose
        if self.secondary and self.secondary.available:
            pf = self.secondary.extract_frame(frame, frame_idx, fps)
            if pf and pf.avg_body_confidence >= self.min_confidence:
                return pf

        # Final fallback: MoveNet
        if self.fallback and self.fallback.available:
            return self.fallback.extract_frame(frame, frame_idx, fps)

        return None

    def _ensemble(self, rtmw_pf: PoseFrame, frame: np.ndarray, frame_idx: int, fps: float) -> PoseFrame:
        """
        Confidence-weighted ensemble of RTMW-x and VitPose.
        Uses RTMW for body, blends hand/face if VitPose has higher confidence.
        """
        vitpose_pf = self.secondary.extract_frame(frame, frame_idx, fps)
        if vitpose_pf is None:
            return rtmw_pf

        kpts = rtmw_pf.keypoints.copy()
        # Blend hand regions where VitPose is more confident
        for sl in [LEFT_HAND_SLICE, RIGHT_HAND_SLICE]:
            rtmw_conf = rtmw_pf.keypoints[sl, 2].mean()
            vp_conf   = vitpose_pf.keypoints[sl, 2].mean() if vitpose_pf.n_keypoints >= 133 else 0
            if vp_conf > rtmw_conf:
                kpts[sl] = vitpose_pf.keypoints[sl]

        rtmw_pf.keypoints = kpts
        return rtmw_pf

    def extract_video(self, video_path: Path | str, max_frames: int | None = None) -> list[PoseFrame]:
        """Extract pose from every sampled frame of a video."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        skip = max(1, int(orig_fps / self.sample_fps))
        results, idx = [], 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % skip == 0:
                pf = self.extract_frame(frame, idx, orig_fps)
                if pf and pf.avg_body_confidence >= self.min_confidence:
                    results.append(pf)
                if max_frames and len(results) >= max_frames:
                    break
            idx += 1

        cap.release()
        logger.info("Extracted %d valid frames from %s", len(results), video_path.name)
        return results

    def save_features(self, frames: list[PoseFrame], output_path: Path | str) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kpts = np.stack([pf.to_feature_vector().reshape(TOTAL_KEYPOINTS, 3) for pf in frames])
        ts   = np.array([pf.timestamp_sec for pf in frames])
        conf = np.array([pf.avg_body_confidence for pf in frames])
        fidx = np.array([pf.frame_idx for pf in frames])
        sources = np.array([pf.source for pf in frames])
        vel  = np.diff(kpts[:, :, :2], axis=0, prepend=kpts[:1, :, :2]).astype(np.float32)
        acc  = np.diff(vel,  axis=0, prepend=vel[:1]).astype(np.float32)
        np.savez_compressed(
            str(output_path),
            keypoints=kpts.astype(np.float32),
            velocities=vel, accelerations=acc,
            timestamps=ts, confidences=conf,
            frame_indices=fidx, sources=sources,
        )
        logger.info("Saved pose features: %s %s", output_path, kpts.shape)
        return output_path

    @property
    def status(self) -> str:
        parts = []
        if self.primary and getattr(self.primary, "available", False):
            parts.append("RTMW-x✓")
        if self.secondary and getattr(self.secondary, "available", False):
            parts.append(f"VitPose-{self.dm.vitpose_model_id.split('-')[-1]}✓")
        if self.fallback and getattr(self.fallback, "available", False):
            parts.append("MoveNet✓")
        return " | ".join(parts) if parts else "No models loaded"
