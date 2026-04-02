"""
NatyaVeda — Refine Videos Script
Removes non-dance segments (audience, presenters, title cards) from raw YouTube videos.

Usage:
  python scripts/refine_videos.py --input data/raw --output data/refined --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]


def main():
    parser = argparse.ArgumentParser(description="NatyaVeda — Refine Videos (Remove Non-Dance Segments)")
    parser.add_argument("--input",                default="data/raw",     help="Input directory with raw videos")
    parser.add_argument("--output",               default="data/refined", help="Output directory for clean segments")
    parser.add_argument("--dances",   nargs="+",  default=DANCE_CLASSES,  help="Dance forms to process")
    parser.add_argument("--device",               default="cuda",         help="cuda or cpu")
    parser.add_argument("--min-dance-confidence", type=float, default=0.65,
                        help="Min activity score to keep a scene [0-1]")
    parser.add_argument("--min-motion",           type=float, default=0.005,
                        help="Min optical-flow energy to keep a scene")
    parser.add_argument("--min-scene-duration",   type=float, default=3.0,
                        help="Minimum scene duration in seconds")
    parser.add_argument("--min-bbox-fraction",    type=float, default=0.15,
                        help="Dancer must occupy this fraction of the frame")
    parser.add_argument("--remove-audience",      action="store_true", default=True,
                        help="Drop audience/low-motion scenes (default: on)")
    parser.add_argument("--scene-threshold",      type=float, default=27.0,
                        help="PySceneDetect content-change threshold")
    parser.add_argument("--glob",                 default="*.mp4",
                        help="File glob pattern for input videos")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing.dance_isolator import DanceIsolator

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    isolator = DanceIsolator(
        min_dance_confidence=args.min_dance_confidence,
        min_motion_threshold=args.min_motion,
        min_scene_duration_sec=args.min_scene_duration,
        min_bbox_fraction=args.min_bbox_fraction,
        scene_threshold=args.scene_threshold,
        device=args.device,
    )

    grand_total_in  = 0.0
    grand_total_out = 0.0
    total_videos    = 0
    total_segments  = 0

    for dance in args.dances:
        dance_in = input_dir / dance
        if not dance_in.exists():
            logger.warning(f"  Skipping {dance} — folder not found: {dance_in}")
            continue

        videos = list(dance_in.glob(args.glob))
        if not videos:
            logger.warning(f"  Skipping {dance} — no {args.glob} files found in {dance_in}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  [{dance.upper()}]  {len(videos)} videos")
        logger.info(f"{'='*60}")

        results = isolator.isolate_batch(
            input_dir=dance_in,
            output_dir=output_dir,
            dance_form=dance,
            glob_pattern=args.glob,
        )

        dance_in_sec  = sum(r.total_input_duration_sec  for r in results)
        dance_out_sec = sum(r.total_output_duration_sec for r in results)
        dance_segs    = sum(len(r.output_segments)       for r in results)

        grand_total_in  += dance_in_sec
        grand_total_out += dance_out_sec
        total_videos    += len(results)
        total_segments  += dance_segs

        logger.info(
            f"  [{dance.upper()}] Done — "
            f"{dance_segs} segments kept, "
            f"{dance_out_sec/60:.1f} min retained "
            f"({100*dance_out_sec/max(dance_in_sec,1):.0f}% of {dance_in_sec/60:.1f} min input)"
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"  REFINEMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Videos processed : {total_videos}")
    print(f"  Segments kept    : {total_segments}")
    print(f"  Input duration   : {grand_total_in/60:.1f} min")
    print(f"  Output duration  : {grand_total_out/60:.1f} min")
    retention = 100 * grand_total_out / max(grand_total_in, 1)
    print(f"  Retention rate   : {retention:.1f}%")
    print(f"  Output dir       : {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
