"""
NatyaVeda — Inference Predictor
End-to-end prediction on a new video: pose extraction → temporal ensemble → labeled output.
"""

from __future__ import annotations

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

DANCE_COLORS = {
    "bharatanatyam": (0, 165, 255),   # orange
    "kathak":        (0, 255, 0),     # green
    "odissi":        (255, 0, 128),   # pink
    "kuchipudi":     (0, 255, 255),   # yellow
    "manipuri":      (128, 0, 255),   # purple
    "mohiniyattam":  (255, 128, 0),   # blue
    "sattriya":      (0, 200, 200),   # teal
    "kathakali":     (0, 0, 255),     # red
}

# COCO 17-point body skeleton connections
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# Hand skeleton connections (MediaPipe 21-point)
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonVisualizer:
    """
    Renders pose skeleton, hand landmarks, dance label, and confidence overlay
    on a video frame.
    """

    def __init__(
        self,
        show_body: bool = True,
        show_hands: bool = True,
        show_face: bool = False,
        show_label: bool = True,
        show_confidence_bar: bool = True,
        show_mudra: bool = True,
        alpha: float = 0.7,
        skeleton_thickness: int = 2,
        keypoint_radius: int = 4,
    ) -> None:
        self.show_body = show_body
        self.show_hands = show_hands
        self.show_face = show_face
        self.show_label = show_label
        self.show_confidence_bar = show_confidence_bar
        self.show_mudra = show_mudra
        self.alpha = alpha
        self.skeleton_thickness = skeleton_thickness
        self.keypoint_radius = keypoint_radius

    def draw(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,         # [133, 3] — (x_norm, y_norm, conf)
        dance_class: str = "",
        dance_conf: float = 0.0,
        mudra_label: str = "",
        all_probs: Optional[list[float]] = None,
    ) -> np.ndarray:
        """
        Draw all annotations on frame. Returns annotated frame.
        keypoints are in normalized [0,1] coords.
        """
        vis = frame.copy()
        h, w = frame.shape[:2]
        dance_color = DANCE_COLORS.get(dance_class, (255, 255, 255))

        # Overlay panel
        overlay = vis.copy()

        if self.show_body and keypoints.shape[0] >= 17:
            self._draw_body(overlay, keypoints[:17], w, h, dance_color)

        if self.show_hands and keypoints.shape[0] >= 133:
            lh = keypoints[91:112]
            rh = keypoints[112:133]
            self._draw_hand(overlay, lh, w, h, color=(80, 200, 255))   # left: light blue
            self._draw_hand(overlay, rh, w, h, color=(255, 180, 80))   # right: amber

        if self.show_face and keypoints.shape[0] >= 91:
            self._draw_face(overlay, keypoints[23:91], w, h)

        # Blend skeleton overlay
        cv2.addWeighted(overlay, self.alpha, vis, 1 - self.alpha, 0, vis)

        if self.show_label and dance_class:
            self._draw_label(vis, dance_class, dance_conf, dance_color, mudra_label, w, h)

        if self.show_confidence_bar and all_probs is not None:
            self._draw_probability_bars(vis, all_probs, w, h)

        return vis

    # ──────────────────────────────────────────────────────────────────
    def _draw_body(self, frame, kpts, w, h, color):
        """Draw 17-point body skeleton."""
        pts = [(int(kpts[i, 0] * w), int(kpts[i, 1] * h)) for i in range(len(kpts))]
        confs = kpts[:, 2]

        # Connections
        for (a, b) in BODY_SKELETON:
            if a < len(pts) and b < len(pts) and confs[a] > 0.3 and confs[b] > 0.3:
                conf_mean = (confs[a] + confs[b]) / 2.0
                line_color = self._conf_color(conf_mean, color)
                cv2.line(frame, pts[a], pts[b], line_color, self.skeleton_thickness, cv2.LINE_AA)

        # Keypoints
        for i, (pt, conf) in enumerate(zip(pts, confs)):
            if conf > 0.3:
                cv2.circle(frame, pt, self.keypoint_radius, self._conf_color(conf, color), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, self.keypoint_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hand(self, frame, kpts, w, h, color):
        """Draw 21-point hand skeleton."""
        if kpts[:, 2].mean() < 0.15:
            return  # skip low-confidence hand
        pts = [(int(kpts[i, 0] * w), int(kpts[i, 1] * h)) for i in range(len(kpts))]
        confs = kpts[:, 2]

        for (a, b) in HAND_SKELETON:
            if a < len(pts) and b < len(pts) and confs[a] > 0.2 and confs[b] > 0.2:
                cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)

        for i, (pt, conf) in enumerate(zip(pts, confs)):
            if conf > 0.2:
                r = 3 if i == 0 else 2
                cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)

    def _draw_face(self, frame, kpts, w, h):
        """Draw minimal face landmarks (eyes, nose tip, mouth corners)."""
        key_indices = [0, 36, 45, 30, 48, 54]  # jaw, eyes, nose, mouth
        for i in key_indices:
            if i < len(kpts) and kpts[i, 2] > 0.3:
                x = int(kpts[i, 0] * w)
                y = int(kpts[i, 1] * h)
                cv2.circle(frame, (x, y), 2, (200, 200, 200), -1)

    def _draw_label(self, frame, dance_class, conf, color, mudra_label, w, h):
        """Draw dance label, confidence, and mudra in top-left panel."""
        panel_h = 90 if not mudra_label else 115
        cv2.rectangle(frame, (10, 10), (340, panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (340, panel_h), color, 2)

        display_name = dance_class.title()
        cv2.putText(frame, display_name, (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {conf*100:.1f}%", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        if mudra_label and self.show_mudra:
            cv2.putText(frame, f"Mudra: {mudra_label}", (20, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1, cv2.LINE_AA)

    def _draw_probability_bars(self, frame, all_probs, w, h):
        """Draw per-class probability bars in the top-right."""
        bar_w_max = 120
        bar_h = 14
        x0 = w - 200
        y0 = 15

        for i, (dance, prob) in enumerate(zip(DANCE_CLASSES, all_probs)):
            y = y0 + i * (bar_h + 4)
            bar_len = int(prob * bar_w_max)
            color = DANCE_COLORS.get(dance, (200, 200, 200))
            is_top = (prob == max(all_probs))
            bg_color = (40, 40, 40)
            cv2.rectangle(frame, (x0, y), (x0 + bar_w_max, y + bar_h), bg_color, -1)
            if bar_len > 0:
                cv2.rectangle(frame, (x0, y), (x0 + bar_len, y + bar_h), color, -1)
            label = dance[:5]
            font_scale = 0.38
            text_color = (255, 255, 255) if is_top else (160, 160, 160)
            cv2.putText(frame, f"{label} {prob*100:.0f}%",
                        (x0 - 75, y + bar_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def _conf_color(conf: float, base_color: tuple) -> tuple:
        """Fade color toward gray for low-confidence keypoints."""
        gray_val = 100
        t = max(0.0, min(1.0, conf))
        r = int(gray_val + t * (base_color[2] - gray_val))
        g = int(gray_val + t * (base_color[1] - gray_val))
        b = int(gray_val + t * (base_color[0] - gray_val))
        return (b, g, r)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end Predictor
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    End-to-end inference pipeline:
      video → pose extraction → DanceFormer → labeled output video
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        pose_model: str = "rtmw-x",
        device: str = "cuda",
        clip_length: int = 64,
        clip_stride: int = 16,
        confidence_threshold: float = 0.5,
    ) -> None:
        from src.models.danceformer import DanceFormer
        from src.feature_extraction.pose_extractor import PoseExtractor
        from src.feature_extraction.hand_extractor import HandExtractor, MudraRecognizer

        self.device = device
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.confidence_threshold = confidence_threshold

        # Load model
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        self.model = DanceFormer.from_config(ckpt.get("config", {}))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(device).eval()
        logger.info(f"DanceFormer loaded: {checkpoint_path}")

        # Extractors
        self.pose_extractor = PoseExtractor(pose_model=pose_model, device=device)
        self.hand_extractor = HandExtractor(device=device)
        self.mudra_recognizer = MudraRecognizer()
        self.visualizer = SkeletonVisualizer()

    def predict_video(
        self,
        video_path: Path | str,
        output_video_path: Optional[Path | str] = None,
        show_skeleton: bool = True,
        show_mudras: bool = True,
    ) -> dict:
        """
        Run full prediction on a video file.

        Returns:
            dict with prediction results and per-segment timeline.
        """
        video_path = Path(video_path)
        logger.info(f"Processing: {video_path.name}")

        # Step 1: Extract pose features
        pose_frames = self.pose_extractor.extract_video(video_path)
        if not pose_frames:
            logger.warning("No pose frames extracted — check video content")
            return {"error": "No valid pose frames extracted"}

        # Step 2: Extract hand features
        hand_frames = self.hand_extractor.extract_video(
            video_path, pose_frames=pose_frames
        )
        hand_by_idx = {hf.frame_idx: hf for hf in hand_frames}

        # Step 3: Build keypoint array
        kpts_all = np.stack([pf.keypoints for pf in pose_frames])  # [T, 133, 3]
        T = len(kpts_all)

        # Step 4: Sliding-window temporal ensemble
        all_probs = []
        starts = list(range(0, T - self.clip_length + 1, self.clip_stride)) or [0]

        with torch.no_grad():
            for start in starts:
                end = min(start + self.clip_length, T)
                clip = kpts_all[start:end].reshape(1, end - start, -1).astype(np.float32)
                if clip.shape[1] < self.clip_length:
                    pad = np.zeros((1, self.clip_length - clip.shape[1], clip.shape[2]), dtype=np.float32)
                    clip = np.concatenate([clip, pad], axis=1)
                clip_t = torch.from_numpy(clip).to(self.device)
                logits = self.model(clip_t)["dance_logits"]
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                all_probs.append(probs)

        ensemble_probs = np.stack(all_probs).mean(axis=0)
        pred_idx = int(ensemble_probs.argmax())
        pred_dance = DANCE_CLASSES[pred_idx]
        pred_conf  = float(ensemble_probs[pred_idx])

        result = {
            "dance_form": pred_dance,
            "confidence": pred_conf,
            "probabilities": {d: float(p) for d, p in zip(DANCE_CLASSES, ensemble_probs)},
            "num_frames_analyzed": T,
            "video": str(video_path),
        }

        logger.info(f"Prediction: {pred_dance} ({pred_conf*100:.1f}%)")

        # Step 5: Write output video with annotations
        if output_video_path:
            self._write_annotated_video(
                video_path=video_path,
                output_path=Path(output_video_path),
                pose_frames=pose_frames,
                hand_by_idx=hand_by_idx,
                dance_class=pred_dance,
                dance_conf=pred_conf,
                all_probs=ensemble_probs.tolist(),
                show_skeleton=show_skeleton,
                show_mudras=show_mudras,
            )
            result["output_video"] = str(output_video_path)

        return result

    def _write_annotated_video(
        self,
        video_path: Path,
        output_path: Path,
        pose_frames: list,
        hand_by_idx: dict,
        dance_class: str,
        dance_conf: float,
        all_probs: list,
        show_skeleton: bool,
        show_mudras: bool,
    ) -> None:
        """Write skeleton-annotated output video."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        pose_by_idx = {pf.frame_idx: pf for pf in pose_frames}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pf = pose_by_idx.get(frame_idx)
            if pf and show_skeleton:
                # Fuse hand keypoints into full 133-kpt array
                kpts = pf.keypoints.copy()
                hf = hand_by_idx.get(frame_idx)
                if hf and hf.left_confidence > 0.3:
                    kpts[91:112] = hf.left_hand
                if hf and hf.right_confidence > 0.3:
                    kpts[112:133] = hf.right_hand

                mudra_label = ""
                if show_mudras and hf:
                    mudra_name, mudra_conf = self.mudra_recognizer.predict(hf)
                    if mudra_conf > 0.5:
                        mudra_label = mudra_name

                frame = self.visualizer.draw(
                    frame=frame,
                    keypoints=kpts,
                    dance_class=dance_class,
                    dance_conf=dance_conf,
                    mudra_label=mudra_label,
                    all_probs=all_probs,
                )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        logger.info(f"Annotated video saved: {output_path}")
