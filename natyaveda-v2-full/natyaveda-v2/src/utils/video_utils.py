"""NatyaVeda — Video Utilities (FFmpeg wrappers)"""
from __future__ import annotations
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_video_info(video_path: Path) -> dict:
    """Return dict with fps, duration, width, height."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps":      cap.get(cv2.CAP_PROP_FPS),
        "frames":   int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info


def trim_video(src: Path, dst: Path, start_sec: float, end_sec: float) -> bool:
    """Losslessly trim a video using FFmpeg stream copy."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-to", str(end_sec),
        "-i", str(src), "-c", "copy", str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def resize_video(src: Path, dst: Path, width: int = 640, height: int = 480) -> bool:
    """Resize video to target resolution."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={width}:{height}",
        "-c:a", "copy", str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def extract_frames(video_path: Path, output_dir: Path, fps: float = 5.0) -> int:
    """Extract frames from video at given fps. Returns count of frames saved."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(output_dir / "frame_%05d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return 0
    return len(list(output_dir.glob("*.jpg")))
