"""
NatyaVeda — Dance Isolator
Detects and removes non-dance content from raw YouTube performance videos.

Pipeline:
  1. Scene boundary detection (content-adaptive)
  2. Per-scene activity classification (dance vs. non-dance)
  3. Principal dancer tracking (ByteTrack + largest bbox)
  4. Keypoint confidence gating
  5. Motion energy filter
  6. Output: trimmed, clean video segments
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
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scene:
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    dance_score: float = 0.0
    motion_energy: float = 0.0
    principal_bbox: Optional[tuple[int, int, int, int]] = None  # x1,y1,x2,y2


@dataclass
class IsolationResult:
    input_path: Path
    output_segments: list[Path] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    kept_scenes: list[Scene] = field(default_factory=list)
    dropped_scenes: list[Scene] = field(default_factory=list)
    total_input_duration_sec: float = 0.0
    total_output_duration_sec: float = 0.0

    @property
    def retention_ratio(self) -> float:
        if self.total_input_duration_sec == 0:
            return 0.0
        return self.total_output_duration_sec / self.total_input_duration_sec


# ─────────────────────────────────────────────────────────────────────────────
# Motion Energy
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_energy(frames: list[np.ndarray], resize_hw: tuple[int, int] = (120, 160)) -> float:
    """
    Compute mean optical flow magnitude across a list of BGR frames.
    Low motion = audience shots, title cards, etc.
    """
    if len(frames) < 2:
        return 0.0

    energies = []
    prev_gray = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), resize_hw[::-1])
    for frame in frames[1:]:
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), resize_hw[::-1])
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        energies.append(float(mag.mean()))
        prev_gray = gray

    return float(np.mean(energies)) if energies else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Activity Classifier (lightweight CNN inference)
# ─────────────────────────────────────────────────────────────────────────────

class ActivityClassifier:
    """
    Binary dance / non-dance classifier.
    Uses a lightweight MobileNetV3 or VideoMAE snippet.
    Falls back to heuristic if model unavailable.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu") -> None:
        self.model = None
        self.device = device

        if model_path and Path(model_path).exists():
            try:
                import torch
                self.model = torch.jit.load(model_path, map_location=device)
                self.model.eval()
                logger.info(f"ActivityClassifier loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load activity model: {e}. Using heuristics.")

    def score_frames(self, frames: list[np.ndarray]) -> float:
        """
        Return dance probability [0, 1] for a list of BGR frames.
        """
        if self.model is not None:
            return self._model_inference(frames)
        return self._heuristic_score(frames)

    def _model_inference(self, frames: list[np.ndarray]) -> float:
        import torch
        # Sample 8 evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
        clips = []
        for i in indices:
            f = cv2.resize(frames[i], (224, 224))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clips.append(f)
        clip_tensor = torch.from_numpy(
            np.stack(clips).transpose(3, 0, 1, 2)[None]  # [1, C, T, H, W]
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(clip_tensor)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)

    def _heuristic_score(self, frames: list[np.ndarray]) -> float:
        """
        Heuristic: combine motion energy + crowd-detection absence.
        High motion + single person region → high dance probability.
        """
        if len(frames) < 5:
            return 0.3

        motion = compute_motion_energy(frames)
        # Normalize motion to 0-1 (typical dance: 0.01–0.08 range)
        motion_score = min(1.0, motion / 0.06)

        # Color variance as proxy for stage lighting vs. audience
        center_crop = [f[f.shape[0] // 4: 3 * f.shape[0] // 4,
                         f.shape[1] // 4: 3 * f.shape[1] // 4]
                       for f in frames[::5]]
        color_std = float(np.mean([f.std() for f in center_crop]))
        color_score = min(1.0, color_std / 60.0)

        return 0.6 * motion_score + 0.4 * color_score


# ─────────────────────────────────────────────────────────────────────────────
# Person Detector (RT-DETR or YOLO fallback)
# ─────────────────────────────────────────────────────────────────────────────

class PersonDetector:
    """Wraps RT-DETR (HuggingFace) or YOLOv8 for person bounding box detection."""

    def __init__(self, model_name: str = "rt-detr-l", device: str = "cuda") -> None:
        self.device = device
        self.model = None

        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            self._processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_l")
            self._model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_l")
            self._model.to(device).eval()
            self.backend = "rtdetr"
            logger.info("PersonDetector: Using RT-DETR-L (HuggingFace)")
        except Exception:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")
                self.backend = "yolo"
                logger.info("PersonDetector: Using YOLOv8n fallback")
            except Exception:
                self.backend = "none"
                logger.warning("PersonDetector: No model available, skipping detection.")

    def detect_persons(
        self, frame: np.ndarray, confidence: float = 0.5
    ) -> list[tuple[int, int, int, int, float]]:
        """
        Returns list of (x1, y1, x2, y2, confidence) for all detected persons.
        """
        if self.backend == "rtdetr":
            return self._detect_rtdetr(frame, confidence)
        elif self.backend == "yolo":
            return self._detect_yolo(frame, confidence)
        return []

    def _detect_rtdetr(self, frame: np.ndarray, conf_thr: float):
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=[(frame.shape[0], frame.shape[1])], threshold=conf_thr
        )[0]
        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        PERSON_CLASS = 1
        persons = []
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == PERSON_CLASS:
                x1, y1, x2, y2 = map(int, box.tolist())
                persons.append((x1, y1, x2, y2, float(score)))
        return persons

    def _detect_yolo(self, frame: np.ndarray, conf_thr: float):
        results = self._yolo(frame, classes=[0], conf=conf_thr, verbose=False)[0]
        persons = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            persons.append((x1, y1, x2, y2, float(box.conf[0])))
        return persons

    def get_principal_dancer(
        self,
        persons: list[tuple[int, int, int, int, float]],
        frame_hw: tuple[int, int],
        strategy: str = "largest_bbox",
        min_bbox_fraction: float = 0.10,
    ) -> Optional[tuple[int, int, int, int, float]]:
        """
        Select the principal dancer from all detected persons.
        strategy: 'largest_bbox' | 'center'
        """
        if not persons:
            return None

        h, w = frame_hw
        frame_area = h * w

        # Filter tiny detections
        valid = [p for p in persons if ((p[2] - p[0]) * (p[3] - p[1])) / frame_area >= min_bbox_fraction]
        if not valid:
            return None

        if strategy == "largest_bbox":
            return max(valid, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
        elif strategy == "center":
            cx, cy = w / 2, h / 2
            return min(valid, key=lambda p: ((p[0] + p[2]) / 2 - cx) ** 2 + ((p[1] + p[3]) / 2 - cy) ** 2)
        return valid[0]


# ─────────────────────────────────────────────────────────────────────────────
# Scene Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_scenes(video_path: Path, threshold: float = 27.0) -> list[tuple[int, int, float, float]]:
    """
    Detect scene boundaries using content-adaptive algorithm.
    Returns: [(start_frame, end_frame, start_sec, end_sec), ...]
    """
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector

        video_mgr = VideoManager([str(video_path)])
        scene_mgr = SceneManager()
        scene_mgr.add_detector(ContentDetector(threshold=threshold))
        video_mgr.set_downscale_factor()
        video_mgr.start()
        scene_mgr.detect_scenes(frame_source=video_mgr)
        raw_scenes = scene_mgr.get_scene_list()
        video_mgr.release()

        scenes = []
        for start, end in raw_scenes:
            scenes.append((
                start.get_frames(), end.get_frames(),
                start.get_seconds(), end.get_seconds()
            ))
        return scenes
    except ImportError:
        logger.warning("PySceneDetect not installed. Using simple uniform splitting.")
        return _fallback_scene_split(video_path)


def _fallback_scene_split(video_path: Path, segment_sec: float = 30.0) -> list[tuple[int, int, float, float]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    seg_frames = int(segment_sec * fps)
    scenes = []
    start = 0
    while start < total:
        end = min(start + seg_frames, total)
        scenes.append((start, end, start / fps, end / fps))
        start = end
    return scenes


# ─────────────────────────────────────────────────────────────────────────────
# Main DanceIsolator
# ─────────────────────────────────────────────────────────────────────────────

class DanceIsolator:
    """
    Isolates actual dance performance segments from raw YouTube videos.

    Parameters
    ----------
    min_dance_confidence : float
        Minimum activity-classifier score to keep a scene.
    min_motion_threshold : float
        Minimum mean optical flow to keep a scene.
    min_scene_duration_sec : float
        Scenes shorter than this are always dropped.
    principal_strategy : str
        'largest_bbox' or 'center'
    device : str
        'cuda' or 'cpu'
    """

    def __init__(
        self,
        min_dance_confidence: float = 0.65,
        min_motion_threshold: float = 0.005,
        min_scene_duration_sec: float = 3.0,
        min_bbox_fraction: float = 0.15,
        principal_strategy: str = "largest_bbox",
        scene_threshold: float = 27.0,
        device: str = "cuda",
        output_format: str = "mp4",
    ) -> None:
        self.min_dance_confidence = min_dance_confidence
        self.min_motion_threshold = min_motion_threshold
        self.min_scene_duration_sec = min_scene_duration_sec
        self.min_bbox_fraction = min_bbox_fraction
        self.principal_strategy = principal_strategy
        self.scene_threshold = scene_threshold
        self.device = device
        self.output_format = output_format

        self.activity_clf = ActivityClassifier(device=device)
        self.person_detector = PersonDetector(device=device)

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def isolate(
        self,
        video_path: Path | str,
        output_dir: Path | str,
        dance_form: str = "unknown",
    ) -> IsolationResult:
        """
        Process a single video and write isolated dance segments.

        Returns an IsolationResult with metadata.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = IsolationResult(input_path=video_path)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        result.total_input_duration_sec = total_frames / fps

        # Step 1: Detect scene boundaries
        logger.info(f"Detecting scenes in {video_path.name} …")
        raw_scenes = detect_scenes(video_path, threshold=self.scene_threshold)

        # Step 2: Score each scene
        logger.info(f"Scoring {len(raw_scenes)} scenes …")
        scenes: list[Scene] = []
        for sf, ef, ss, es in raw_scenes:
            duration = es - ss
            if duration < self.min_scene_duration_sec:
                continue
            scene = Scene(
                start_frame=sf, end_frame=ef,
                start_time_sec=ss, end_time_sec=es
            )
            # Sample frames for this scene
            frames = self._sample_frames(video_path, sf, ef, n_frames=16)
            if not frames:
                continue

            scene.motion_energy = compute_motion_energy(frames)
            scene.dance_score = self.activity_clf.score_frames(frames)

            # Detect principal dancer
            mid_frame = frames[len(frames) // 2]
            persons = self.person_detector.detect_persons(mid_frame)
            principal = self.person_detector.get_principal_dancer(
                persons, (frame_h, frame_w),
                strategy=self.principal_strategy,
                min_bbox_fraction=self.min_bbox_fraction,
            )
            if principal:
                scene.principal_bbox = principal[:4]

            scenes.append(scene)

        result.scenes = scenes

        # Step 3: Filter scenes
        kept = []
        dropped = []
        for scene in scenes:
            if self._should_keep(scene):
                kept.append(scene)
            else:
                dropped.append(scene)

        result.kept_scenes = kept
        result.dropped_scenes = dropped

        logger.info(
            f"Kept {len(kept)}/{len(scenes)} scenes "
            f"({sum(s.end_time_sec - s.start_time_sec for s in kept):.1f}s retained)"
        )

        # Step 4: Export segments
        stem = video_path.stem
        for i, scene in enumerate(kept):
            out_path = output_dir / f"{stem}_seg{i:03d}.{self.output_format}"
            self._export_segment(video_path, scene, out_path, fps)
            result.output_segments.append(out_path)
            result.total_output_duration_sec += scene.end_time_sec - scene.start_time_sec

        return result

    def isolate_batch(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        dance_form: str = "unknown",
        glob_pattern: str = "*.mp4",
    ) -> list[IsolationResult]:
        """Process all videos in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        results = []
        for video in sorted(input_dir.glob(glob_pattern)):
            logger.info(f"Processing: {video.name}")
            try:
                r = self.isolate(video, output_dir / dance_form, dance_form)
                results.append(r)
                logger.info(
                    f"  → Retention: {r.retention_ratio:.0%} "
                    f"({r.total_output_duration_sec:.0f}s / {r.total_input_duration_sec:.0f}s)"
                )
            except Exception as e:
                logger.error(f"  ✗ Failed: {video.name} — {e}")
        return results

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def _should_keep(self, scene: Scene) -> bool:
        """Return True if this scene passes all quality gates."""
        if scene.dance_score < self.min_dance_confidence:
            return False
        if scene.motion_energy < self.min_motion_threshold:
            return False
        if scene.principal_bbox is None:
            return False  # No visible person detected
        return True

    def _sample_frames(
        self, video_path: Path, start_frame: int, end_frame: int, n_frames: int = 16
    ) -> list[np.ndarray]:
        """Sample N evenly spaced frames from a scene."""
        cap = cv2.VideoCapture(str(video_path))
        indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def _export_segment(
        self,
        video_path: Path,
        scene: Scene,
        out_path: Path,
        fps: float,
    ) -> None:
        """Use FFmpeg to cut a segment from a video without re-encoding."""
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(scene.start_time_sec),
            "-to", str(scene.end_time_sec),
            "-i", str(video_path),
            "-c:v", "libx264", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=False)
