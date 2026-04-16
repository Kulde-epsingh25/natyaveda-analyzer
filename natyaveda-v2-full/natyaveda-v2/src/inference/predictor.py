"""
NatyaVeda — Inference Predictor (Fixed + Prototype Calibration)

Fixes from v1:
  - NameError: 'self' in class body (SkeletonVisualizer) → moved to __init__
  - Prototype-based prediction blending (improves mohiniyattam + kathak)
  - Uncertainty detection for low-confidence + narrow margin predictions
  - Aggregation choices: mean | trimmed | geomean
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]

# Connected body keypoints for skeleton drawing (COCO-17)
BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

FOOT_CONNECTIONS = [
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
    (17, 18), (20, 21),
]

# Hand skeleton connections (MediaPipe-style, 21 keypoints per hand)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]

MUDRA_CLASSES = [
    "pataka", "tripataka", "ardhapataka", "kartarimukha",
    "mayura", "ardhachandra", "arala", "shukatunda",
    "mushti", "shikhara", "kapittha", "katakamukha",
    "suchi", "chandrakala", "padmakosha", "sarpashirsha",
    "mrigashirsha", "simhamukha", "kangula", "alapadma",
    "chatura", "bhramara", "hamsasya", "hamsapaksha",
    "sandamsha", "mukula", "tamrachuda", "trishula",
]


def _face_connections() -> list[tuple[int, int]]:
    """Build 68-point face landmark connections offset into 133-kpt layout."""
    o = 23  # 68 face landmarks are mapped at indices [23..90]
    conn: list[tuple[int, int]] = []

    def chain(start: int, end: int) -> None:
        for i in range(start, end):
            conn.append((o + i, o + i + 1))

    def loop(start: int, end: int) -> None:
        chain(start, end)
        conn.append((o + end, o + start))

    chain(0, 16)      # jaw
    chain(17, 21)     # left eyebrow
    chain(22, 26)     # right eyebrow
    chain(27, 30)     # nose bridge
    chain(31, 35)     # lower nose
    loop(36, 41)      # left eye
    loop(42, 47)      # right eye
    loop(48, 59)      # outer lip
    loop(60, 67)      # inner lip
    return conn


FACE_CONNECTIONS = _face_connections()

DANCE_COLORS = {
    "bharatanatyam": (0, 165, 255),    # orange
    "kathak":        (147, 20, 255),   # purple
    "odissi":        (0, 255, 127),    # spring green
    "kuchipudi":     (255, 191, 0),    # deep sky blue
    "manipuri":      (0, 255, 255),    # cyan
    "mohiniyattam":  (255, 0, 127),    # pink
    "sattriya":      (255, 165, 0),    # blue
    "kathakali":     (0, 0, 255),      # red
}


# ─────────────────────────────────────────────────────────────────────────────
# SkeletonVisualizer — BUG FIXED: self was used in class body, not __init__
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonVisualizer:
    """Draws skeleton + prediction overlay onto video frames."""

    def __init__(
        self,
        show_body:       bool = True,
        show_hands:      bool = True,
        show_face:       bool = False,
        keypoint_radius: int  = 4,
        line_thickness:  int  = 2,
        fast_pose_mode:  bool = False,
        show_all_keypoints: bool = True,
    ) -> None:
        self.show_body       = show_body
        self.show_hands      = show_hands
        self.show_face       = show_face
        self.keypoint_radius = keypoint_radius
        self.line_thickness  = line_thickness
        # FIX: was `force_movenet_only=self.fast_pose_mode` in the CLASS BODY
        # (outside any method) which caused NameError. Now properly in __init__.
        self.fast_pose_mode  = fast_pose_mode
        self.show_all_keypoints = show_all_keypoints
        self.show_feet = True

    def draw(
        self,
        frame:       np.ndarray,
        keypoints:   np.ndarray,
        dance_class: str,
        confidence:  float,
        mudra_name:  Optional[str] = None,
        mudra_confidence: Optional[float] = None,
        probs:       Optional[list] = None,
        draw_skeleton: bool = True,
    ) -> np.ndarray:
        """
        Draws skeleton + prediction text onto frame.
        keypoints: [N_KPT, 3] — (x_norm, y_norm, confidence)
        """
        h, w = frame.shape[:2]
        color = DANCE_COLORS.get(dance_class, (255, 255, 255))
        out = frame.copy()

        # Draw all available keypoints so 133-point whole-body layout is visible.
        if self.show_all_keypoints and keypoints.shape[0] >= 133:
            for i in range(keypoints.shape[0]):
                if keypoints[i, 2] <= 0.2:
                    continue
                px = int(keypoints[i, 0] * w)
                py = int(keypoints[i, 1] * h)
                if i < 17:
                    kp_color = color
                elif i < 23:
                    kp_color = (255, 165, 0)  # feet
                elif i < 91:
                    kp_color = (180, 180, 180)  # face
                else:
                    kp_color = (0, 255, 0)  # hands
                if i < 23:
                    r = 2
                elif i < 91:
                    r = 1
                else:
                    r = 2
                cv2.circle(out, (px, py), r, kp_color, -1)

        # Body skeleton
        if draw_skeleton and self.show_body and keypoints.shape[0] >= 17:
            for a, b in BODY_CONNECTIONS:
                if keypoints[a, 2] > 0.3 and keypoints[b, 2] > 0.3:
                    pa = (int(keypoints[a, 0] * w), int(keypoints[a, 1] * h))
                    pb = (int(keypoints[b, 0] * w), int(keypoints[b, 1] * h))
                    cv2.line(out, pa, pb, color, self.line_thickness)
            for i in range(17):
                if keypoints[i, 2] > 0.3:
                    px = int(keypoints[i, 0] * w)
                    py = int(keypoints[i, 1] * h)
                    cv2.circle(out, (px, py), self.keypoint_radius, color, -1)

        # Feet lines (WholeBody indices 17..22)
        if draw_skeleton and self.show_feet and keypoints.shape[0] >= 23:
            for a, b in FOOT_CONNECTIONS:
                if keypoints[a, 2] > 0.25 and keypoints[b, 2] > 0.25:
                    pa = (int(keypoints[a, 0] * w), int(keypoints[a, 1] * h))
                    pb = (int(keypoints[b, 0] * w), int(keypoints[b, 1] * h))
                    cv2.line(out, pa, pb, (255, 180, 0), 1)

        # Face mesh lines (68 landmarks at indices 23..90)
        if draw_skeleton and self.show_face and keypoints.shape[0] >= 91:
            for a, b in FACE_CONNECTIONS:
                if keypoints[a, 2] > 0.2 and keypoints[b, 2] > 0.2:
                    pa = (int(keypoints[a, 0] * w), int(keypoints[a, 1] * h))
                    pb = (int(keypoints[b, 0] * w), int(keypoints[b, 1] * h))
                    cv2.line(out, pa, pb, (170, 170, 170), 1)

        # Hand keypoints
        if draw_skeleton and self.show_hands and keypoints.shape[0] >= 133:
            for hand_slice in [slice(91, 112), slice(112, 133)]:
                hkpts = keypoints[hand_slice]
                for a, b in HAND_CONNECTIONS:
                    if hkpts[a, 2] > 0.25 and hkpts[b, 2] > 0.25:
                        pa = (int(hkpts[a, 0] * w), int(hkpts[a, 1] * h))
                        pb = (int(hkpts[b, 0] * w), int(hkpts[b, 1] * h))
                        cv2.line(out, pa, pb, (0, 200, 0), 1)
                for i in range(hkpts.shape[0]):
                    if hkpts[i, 2] > 0.3:
                        px = int(hkpts[i, 0] * w)
                        py = int(hkpts[i, 1] * h)
                        cv2.circle(out, (px, py), 2, (0, 255, 0), -1)

        # Prediction overlay (top-left)
        overlay = np.zeros((145, 320, 3), dtype=np.uint8)
        cv2.putText(overlay, f"{dance_class.title()}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(overlay, f"Conf: {confidence*100:.1f}%", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(overlay, f"KPTS: {int(keypoints.shape[0])}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        if mudra_name:
            mudra_text = f"Mudra: {mudra_name}"
            if mudra_confidence is not None:
                mudra_text += f" ({mudra_confidence*100:.1f}%)"
            cv2.putText(overlay, mudra_text, (10, 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
        # Confidence bar
        bar_w = int(confidence * 300)
        cv2.rectangle(overlay, (10, 116), (10 + bar_w, 134), color, -1)
        cv2.rectangle(overlay, (10, 116), (310, 134), (80, 80, 80), 1)

        h2 = min(145, out.shape[0])
        w2 = min(320, out.shape[1])
        out[:h2, :w2] = cv2.addWeighted(out[:h2, :w2], 0.4,
                                         overlay[:h2, :w2], 0.6, 0)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Full inference predictor with prototype-based calibration.

    Prediction strategy:
    1. Extract pose features from video (VitPose or MoveNet)
    2. Split into overlapping temporal windows
    3. Run DanceFormer on each window → get logits + feature embeddings
    4. Optionally blend with prototype similarity scores
    5. Aggregate across windows (mean/trimmed/geomean)
    6. Apply strict confidence check if requested
    """

    def __init__(
        self,
        checkpoint_path:   str,
        pose_model:        str  = "vitpose-base",
        device:            str  = "cuda",
        clip_length:       int  = 64,
        aggregation:       str  = "trimmed",
        strict_confidence: bool = False,
        min_confidence:    float = 0.55,
        min_margin:        float = 0.12,
        prototype_weight:  float = 0.4,   # 0 = pure logits, 1 = pure prototype
    ) -> None:
        self.device         = device
        self.clip_length    = clip_length
        self.aggregation    = aggregation
        self.strict         = strict_confidence
        self.min_conf       = min_confidence
        self.min_margin     = min_margin
        self.proto_weight   = prototype_weight

        self._load_model(checkpoint_path)
        self._load_prototypes(checkpoint_path)
        self._init_extractors(pose_model)

    def _load_model(self, ckpt_path: str) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.models.danceformer import DanceFormer

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt.get("config", {})

        self.model = DanceFormer.from_config({"model": cfg})
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device).eval()
        logger.info("DanceFormer loaded: %s", ckpt_path)

    def _load_prototypes(self, ckpt_path: str) -> None:
        """Load class prototype embeddings if they exist."""
        self.prototypes = None
        proto_path = Path("reports/class_prototypes.npz")
        if proto_path.exists():
            d = np.load(str(proto_path))
            self.prototypes = torch.tensor(d["prototypes"], dtype=torch.float32).to(self.device)
            logger.info("Loaded class prototypes: %s", proto_path)

    def _init_extractors(self, pose_model: str) -> None:
        from src.utils.device import DeviceManager
        from src.feature_extraction.pose_extractor import PoseExtractor
        from src.feature_extraction.hand_extractor import HandExtractor

        dm = DeviceManager()
        logger.info("Device: %s | GPU: %s (%s GB VRAM) | CUDA: %s | Pose tier: %s | fp16: %s",
                    dm.device,
                    dm.config.gpu_info.device_name,
                    f"{dm.config.gpu_info.vram_gb:.1f}",
                    dm.config.gpu_info.cuda_version,
                    dm.pose_model_tier,
                    dm.fp16)

        force_movenet = (pose_model == "movenet-thunder")
        self.pose_extractor = PoseExtractor(
            device_manager=dm,
            sample_fps=25.0,
            force_movenet_only=force_movenet,
        )
        self.hand_extractor = HandExtractor(use_mediapipe=True, use_rtmw=False,
                                            device=dm.device)
        self.visualizer = SkeletonVisualizer(
            show_body=True,
            show_hands=True,
            show_face=True,
            fast_pose_mode=(pose_model == "movenet-thunder")
        )

    def predict_video(
        self,
        video_path:        str,
        output_video_path: Optional[str] = None,
        show_skeleton:     bool = True,
        show_mudras:       bool = True,
    ) -> dict:
        """Run full inference on a video file."""
        video_path = str(video_path)

        # Extract features
        pose_frames = self.pose_extractor.extract_video(video_path)
        if not pose_frames:
            return {"error": "No valid pose frames extracted from video"}

        hand_frames = self.hand_extractor.extract_video(
            video_path, pose_frames=pose_frames, sample_fps=25.0
        )

        # Build keypoint sequence
        kpts = np.stack([pf.to_feature_vector() for pf in pose_frames])  # [T, 399]
        T = len(kpts)
        logger.info("Processing: %s", Path(video_path).name)

        # Fuse hand keypoints into pose (indices 91-132)
        kpts_3d = kpts.reshape(T, 133, 3)
        if hand_frames:
            T_min = min(T, len(hand_frames))
            for i in range(T_min):
                hf = hand_frames[i]
                kpts_3d[i, 91:112]  = hf.left_hand
                kpts_3d[i, 112:133] = hf.right_hand

        # Temporal windows
        stride = max(1, self.clip_length // 2)
        starts = list(range(0, max(1, T - self.clip_length + 1), stride))
        if not starts:
            starts = [0]

        all_probs    = []
        all_features = []
        all_mudra_probs = []

        with torch.no_grad():
            for s in starts:
                e = min(s + self.clip_length, T)
                clip = kpts_3d[s:e].reshape(e - s, 399)

                # Pad if short
                if clip.shape[0] < self.clip_length:
                    pad = np.zeros((self.clip_length - clip.shape[0], 399), np.float32)
                    clip = np.concatenate([clip, pad], axis=0)

                x = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(self.device)
                out = self.model(x)
                logits   = out["dance_logits"]    # [1, 8]
                features = out.get("features")    # [1, D] or None
                mudra_logits = out.get("mudra_logits")  # [1, T, 28] or None

                # Prototype-blended prediction
                if features is not None and self.prototypes is not None:
                    from src.models.danceformer_additions import prototype_predict
                    probs = prototype_predict(features, self.prototypes,
                                             logit_weight=self.proto_weight,
                                             logits=logits)
                else:
                    probs = F.softmax(logits, dim=-1)

                all_probs.append(probs.cpu().numpy()[0])
                if features is not None:
                    all_features.append(features.cpu().numpy()[0])
                if mudra_logits is not None:
                    valid_len = max(1, e - s)
                    mudra_probs = F.softmax(mudra_logits[:, :valid_len, :], dim=-1)
                    all_mudra_probs.append(mudra_probs.mean(dim=1).cpu().numpy()[0])

        # Aggregate
        probs_arr = np.stack(all_probs)  # [n_windows, 8]
        final_probs = self._aggregate(probs_arr)

        pred_idx  = int(final_probs.argmax())
        pred_name = DANCE_CLASSES[pred_idx]
        confidence = float(final_probs[pred_idx])

        final_mudra_probs = None
        mudra_name = None
        mudra_confidence = None
        if all_mudra_probs:
            final_mudra_probs = self._aggregate(np.stack(all_mudra_probs))
            mudra_idx = int(final_mudra_probs.argmax())
            mudra_name = MUDRA_CLASSES[mudra_idx]
            mudra_confidence = float(final_mudra_probs[mudra_idx])

        # Top-2 margin
        sorted_probs = np.sort(final_probs)[::-1]
        top2_margin  = float(sorted_probs[0] - sorted_probs[1])

        logger.info("Prediction: %s (%.1f%%)", pred_name, confidence * 100)

        # Strict mode
        is_uncertain = False
        uncertainty_reasons = []
        raw_pred = pred_name
        if self.strict:
            if confidence < self.min_conf:
                is_uncertain = True
                uncertainty_reasons.append(f"conf={confidence:.2f}<{self.min_conf}")
            if top2_margin < self.min_margin:
                is_uncertain = True
                uncertainty_reasons.append(f"margin={top2_margin:.3f}<{self.min_margin}")
            if is_uncertain:
                pred_name = "unknown"

        result = {
            "dance_form":          pred_name,
            "raw_prediction":      raw_pred,
            "confidence":          confidence,
            "mudra":               mudra_name,
            "mudra_confidence":    mudra_confidence,
            "top2_margin":         top2_margin,
            "is_uncertain":        is_uncertain,
            "uncertainty_reasons": uncertainty_reasons,
            "probabilities":       {DANCE_CLASSES[i]: float(final_probs[i]) for i in range(8)},
            "num_frames_analyzed": T,
            "video":               video_path,
        }

        # Generate annotated video
        if output_video_path:
            self._write_annotated_video(
                video_path, output_video_path,
                kpts_3d, pose_frames, final_probs,
                raw_pred, confidence,
                show_skeleton=show_skeleton,
                mudra_name=(mudra_name if show_mudras else None),
                mudra_confidence=(mudra_confidence if show_mudras else None),
            )

        return result

    def _aggregate(self, probs: np.ndarray) -> np.ndarray:
        """Aggregate window probabilities."""
        if self.aggregation == "trimmed" and len(probs) >= 3:
            # Drop top/bottom 10% of windows per class
            k = max(1, len(probs) // 10)
            s = np.sort(probs, axis=0)
            return s[k:-k].mean(axis=0) if len(s) > 2 * k else probs.mean(axis=0)
        elif self.aggregation == "geomean":
            log_p = np.log(np.clip(probs, 1e-8, 1.0))
            gm = np.exp(log_p.mean(axis=0))
            return gm / gm.sum()
        else:  # "mean"
            return probs.mean(axis=0)

    def _write_annotated_video(self, in_path, out_path, kpts_3d,
                                pose_frames, final_probs, pred_name, confidence,
                                show_skeleton=True, mudra_name=None, mudra_confidence=None):
        """Write annotated output video with skeleton overlay."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kpt_idx = min(frame_idx, len(kpts_3d) - 1)
            annotated = self.visualizer.draw(
                frame, kpts_3d[kpt_idx],
                dance_class=pred_name, confidence=confidence,
                mudra_name=mudra_name,
                mudra_confidence=mudra_confidence,
                probs=final_probs.tolist(),
                draw_skeleton=show_skeleton,
            )
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info("Annotated video saved: %s", out_path)
