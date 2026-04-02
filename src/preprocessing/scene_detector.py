"""
NatyaVeda — Scene Detector
Wraps PySceneDetect for content-adaptive scene boundary detection.
Used by DanceIsolator to split videos into scenes before quality scoring.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_scenes_pyscene(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len_frames: int = 15,
) -> list[tuple[int, int, float, float]]:
    """
    Detect scene boundaries using PySceneDetect ContentDetector.

    Returns list of (start_frame, end_frame, start_sec, end_sec).
    """
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector

        mgr = VideoManager([str(video_path)])
        sm  = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames))
        mgr.set_downscale_factor()
        mgr.start()
        sm.detect_scenes(frame_source=mgr)
        raw = sm.get_scene_list()
        mgr.release()

        return [
            (s.get_frames(), e.get_frames(), s.get_seconds(), e.get_seconds())
            for s, e in raw
        ]

    except ImportError:
        logger.warning("PySceneDetect not installed — using uniform fallback")
        return _uniform_split(video_path)
    except Exception as e:
        logger.warning(f"Scene detection failed ({e}) — using uniform fallback")
        return _uniform_split(video_path)


def _uniform_split(
    video_path: Path,
    segment_sec: float = 30.0,
) -> list[tuple[int, int, float, float]]:
    """Fallback: split video into fixed-length segments."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    seg   = int(segment_sec * fps)
    scenes = []
    start = 0
    while start < total:
        end = min(start + seg, total)
        scenes.append((start, end, start / fps, end / fps))
        start = end
    return scenes
