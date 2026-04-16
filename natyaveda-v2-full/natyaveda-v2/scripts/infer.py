"""
NatyaVeda — Inference Script
Usage:
  python scripts/infer.py --video path/to/dance.mp4 --checkpoint weights/danceformer_best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NatyaVeda Dance Inference")
    parser.add_argument("--video",       required=True,  help="Input video path")
    parser.add_argument("--checkpoint",  required=True,  help="Model checkpoint .pt")
    parser.add_argument("--output-video", default=None,  help="Annotated output video path")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pose-model",  default="rtmw-x",
                        choices=["rtmw-x", "rtmw-l", "movenet-thunder"])
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--aggregation", default="trimmed", choices=["mean", "trimmed", "geomean"],
                        help="How to combine probabilities from temporal windows")
    parser.add_argument("--strict", action="store_true",
                        help="Return 'unknown' when confidence or top-2 margin is too low")
    parser.add_argument("--min-confidence", type=float, default=0.55,
                        help="Minimum confidence required when --strict is enabled")
    parser.add_argument("--min-margin", type=float, default=0.12,
                        help="Minimum top-1 vs top-2 probability margin when --strict is enabled")
    parser.add_argument("--no-skeleton", action="store_true")
    parser.add_argument("--no-mudras",   action="store_true")
    parser.add_argument("--json-output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.inference.predictor import Predictor

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        pose_model=args.pose_model,
        device=args.device,
        clip_length=args.clip_length,
        aggregation=args.aggregation,
        strict_confidence=args.strict,
        min_confidence=args.min_confidence,
        min_margin=args.min_margin,
    )

    result = predictor.predict_video(
        video_path=args.video,
        output_video_path=args.output_video,
        show_skeleton=not args.no_skeleton,
        show_mudras=not args.no_mudras,
    )

    # Display results
    print("\n" + "=" * 50)
    print("  NatyaVeda Prediction Result")
    print("=" * 50)

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    dance = result["dance_form"].title()
    conf  = result["confidence"]
    print(f"  Dance Form   : {dance}")
    print(f"  Confidence   : {conf*100:.1f}%")
    if "top2_margin" in result:
        print(f"  Top-2 Margin : {result['top2_margin']:.3f}")
    if result.get("is_uncertain"):
        reasons = ", ".join(result.get("uncertainty_reasons", []))
        print(f"  Uncertain    : YES ({reasons})")
        print(f"  Raw Top-1    : {result.get('raw_prediction', '').title()}")
    print(f"  Frames       : {result['num_frames_analyzed']}")
    print()
    print("  All probabilities:")
    for dance_cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"    {dance_cls:20s} {prob*100:5.1f}% {bar}")

    if args.output_video:
        print(f"\n  Annotated video: {args.output_video}")

    # Save JSON
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  JSON saved: {args.json_output}")


if __name__ == "__main__":
    main()
