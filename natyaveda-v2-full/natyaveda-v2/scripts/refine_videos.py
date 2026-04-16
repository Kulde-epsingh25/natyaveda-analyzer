"""
NatyaVeda -- Refine Videos (fixed for Windows + 0% retention bug)

FAST MODE (recommended first run):
  python scripts/refine_videos.py --input data/raw --output data/refined --fast-mode

NORMAL MODE (uses DETR person detector -- more accurate but slower):
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
    parser = argparse.ArgumentParser(
        description="NatyaVeda Refine Videos -- Remove non-dance segments"
    )
    parser.add_argument("--input",                default="data/raw")
    parser.add_argument("--output",               default="data/refined")
    parser.add_argument("--dances",  nargs="+",   default=DANCE_CLASSES)
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--glob",                 default="*.mp4")

    # Key settings -- fixed defaults based on real testing
    parser.add_argument("--fast-mode",  action="store_true",
                        help="Motion-only mode. No DETR person detector. "
                             "~30s/video. Recommended for first runs.")
    parser.add_argument("--scene-threshold",      type=float, default=40.0,
                        help="Higher = fewer scenes (faster). Default 40 vs original 27.")
    parser.add_argument("--min-dance-confidence", type=float, default=0.35,
                        help="Min activity score to keep a scene. Default 0.35 vs original 0.65.")
    parser.add_argument("--min-motion",           type=float, default=0.003,
                        help="Min optical flow energy. Default 0.003.")
    parser.add_argument("--min-scene-duration",   type=float, default=3.0)
    parser.add_argument("--min-bbox-fraction",    type=float, default=0.05,
                        help="Min fraction of frame dancer must occupy. Default 0.05.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing.dance_isolator import DanceIsolator

    mode_str = "FAST (motion-only, ~30s/video)" if args.fast_mode else "NORMAL (DETR, ~5-17min/video)"
    logger.info("Mode: %s", mode_str)
    logger.info("Scene threshold: %.1f | Min confidence: %.2f | Min motion: %.3f",
                args.scene_threshold, args.min_dance_confidence, args.min_motion)

    isolator = DanceIsolator(
        fast_mode=args.fast_mode,
        min_dance_confidence=args.min_dance_confidence,
        min_motion_threshold=args.min_motion,
        min_scene_duration_sec=args.min_scene_duration,
        min_bbox_fraction=args.min_bbox_fraction,
        scene_threshold=args.scene_threshold,
        device=args.device,
    )

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    total_in = total_out = total_videos = total_segs = 0

    for dance in args.dances:
        dance_in = input_dir / dance
        if not dance_in.exists():
            logger.warning("Skipping %s -- not found", dance)
            continue

        videos = list(dance_in.glob(args.glob))
        if not videos:
            logger.warning("Skipping %s -- no %s files", dance, args.glob)
            continue

        logger.info("\n[%s] %d videos", dance.upper(), len(videos))

        results = isolator.isolate_batch(dance_in, output_dir, dance_form=dance)
        for r in results:
            total_in    += r.total_input_duration_sec
            total_out   += r.total_output_duration_sec
            total_segs  += len(r.output_segments)
        total_videos += len(results)

    print(f"\n{'='*60}")
    print(f"  REFINEMENT COMPLETE")
    print(f"  Videos:   {total_videos}")
    print(f"  Segments: {total_segs}")
    print(f"  Input:    {total_in/60:.1f} min")
    print(f"  Output:   {total_out/60:.1f} min")
    pct = 100 * total_out / max(total_in, 1)
    print(f"  Retained: {pct:.1f}%")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}")

    if pct < 5 and total_videos > 0:
        print("\n  WARNING: Very low retention. Try:")
        print("  python scripts/refine_videos.py --fast-mode --min-dance-confidence 0.2")


if __name__ == "__main__":
    main()