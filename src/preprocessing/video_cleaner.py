"""
NatyaVeda — Video Cleaner
Handles video stabilization, noise reduction, and format normalization
before pose extraction.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoCleaner:
    """
    Pre-processes raw video files for downstream pose extraction:
      - Converts to standard MP4 (H.264)
      - Resizes to target resolution
      - Optional ECC-based video stabilization
      - Adjusts frame rate to target fps
    """

    def __init__(
        self,
        target_fps: float = 25.0,
        target_width: int = 640,
        target_height: int = 480,
        stabilize: bool = False,
        output_format: str = "mp4",
    ) -> None:
        self.target_fps    = target_fps
        self.target_width  = target_width
        self.target_height = target_height
        self.stabilize     = stabilize
        self.output_format = output_format

    def clean(self, input_path: Path, output_path: Path) -> bool:
        """Clean and normalize a single video file. Returns True on success."""
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        src = input_path
        if self.stabilize:
            stabilized = output_path.with_suffix(".stab.mp4")
            ok = self._stabilize_ffmpeg(input_path, stabilized)
            if ok:
                src = stabilized

        return self._normalize_ffmpeg(src, output_path)

    def clean_batch(self, input_dir: Path, output_dir: Path, glob: str = "*.mp4") -> list[Path]:
        """Clean all videos in a directory. Returns list of output paths."""
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleaned = []
        for src in sorted(input_dir.glob(glob)):
            dst = output_dir / src.name
            if dst.exists():
                cleaned.append(dst)
                continue
            if self.clean(src, dst):
                logger.info(f"  ✓ Cleaned: {src.name}")
                cleaned.append(dst)
            else:
                logger.warning(f"  ✗ Failed:  {src.name}")
        return cleaned

    def _normalize_ffmpeg(self, src: Path, dst: Path) -> bool:
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vf",
            f"scale={self.target_width}:{self.target_height}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={self.target_width}:{self.target_height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={self.target_fps}",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr.decode()[:300]}")
            return False
        return True

    def _stabilize_ffmpeg(self, src: Path, dst: Path) -> bool:
        transforms = src.with_suffix(".trf")
        cmd1 = ["ffmpeg", "-y", "-i", str(src),
                "-vf", f"vidstabdetect=shakiness=5:accuracy=9:result={transforms}",
                "-f", "null", "-"]
        r1 = subprocess.run(cmd1, capture_output=True)
        if r1.returncode != 0:
            return False
        cmd2 = ["ffmpeg", "-y", "-i", str(src),
                "-vf", f"vidstabtransform=input={transforms}:smoothing=10",
                "-c:v", "libx264", "-crf", "23", str(dst)]
        r2 = subprocess.run(cmd2, capture_output=True)
        transforms.unlink(missing_ok=True)
        return r2.returncode == 0
