"""
Run randomized smoke inference across dance classes until each class has at least
one correct prediction (or per-class attempt budget is exhausted).

Example:
  "d:/New folder (2)/files (3)/.venv/Scripts/python.exe" scripts/random_class_smoke.py \
    --checkpoint weights/danceformer_best.pt --source-root data/raw --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def collect_videos(class_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Randomized per-class smoke test")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--source-root", default="data/raw", help="Root folder containing class subfolders")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--pose-model", default="rtmw-x", choices=["rtmw-x", "rtmw-l", "movenet-thunder"])
    parser.add_argument("--max-attempts-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/smoke_random")
    parser.add_argument(
        "--save-video-mode",
        default="solved",
        choices=["none", "solved", "all"],
        help="Save annotated output videos for none, solved-only, or all attempts",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.inference.predictor import Predictor, DANCE_CLASSES  # pylint: disable=import-error

    random.seed(args.seed)

    source_root = (repo_root / args.source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    pool: Dict[str, List[Path]] = {}
    for cls in DANCE_CLASSES:
        cls_dir = source_root / cls
        if not cls_dir.exists():
            pool[cls] = []
            continue
        vids = collect_videos(cls_dir)
        random.shuffle(vids)
        pool[cls] = vids

    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts_json_dir = out_dir / "attempt_json"
    attempts_json_dir.mkdir(parents=True, exist_ok=True)
    attempts_video_dir = out_dir / "attempt_videos"
    attempts_video_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attempts_json = out_dir / f"random_smoke_attempts_{stamp}.json"
    summary_json = out_dir / f"random_smoke_summary_{stamp}.json"
    summary_csv = out_dir / f"random_smoke_summary_{stamp}.csv"

    predictor = Predictor(
        checkpoint_path=str((repo_root / args.checkpoint).resolve()),
        pose_model=args.pose_model,
        device=args.device,
        clip_length=64,
        aggregation="trimmed",
        strict_confidence=False,
        min_confidence=0.55,
        min_margin=0.12,
    )

    attempts_left = {c: args.max_attempts_per_class for c in DANCE_CLASSES}
    solved = {c: False for c in DANCE_CLASSES}
    solved_record: Dict[str, dict] = {}
    attempts_log: List[dict] = []

    total_round = 0
    while True:
        pending = [c for c in DANCE_CLASSES if not solved[c] and attempts_left[c] > 0 and len(pool[c]) > 0]
        if not pending:
            break

        total_round += 1
        cls = random.choice(pending)
        video = pool[cls].pop()
        attempts_left[cls] -= 1

        print(f"[round {total_round:03d}] class={cls} tries_left={attempts_left[cls]:02d} video={video.name}")

        try:
            result = predictor.predict_video(str(video), output_video_path=None, show_skeleton=False, show_mudras=False)
            pred = result.get("dance_form", "unknown")
            conf = float(result.get("confidence", 0.0))
            ok = pred == cls
            result_json_path = attempts_json_dir / f"round_{total_round:03d}_{cls}_{video.stem}.json"
            result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            row = {
                "round": total_round,
                "expected": cls,
                "video": str(video),
                "predicted": pred,
                "confidence": conf,
                "num_frames": int(result.get("num_frames_analyzed", 0)),
                "correct": ok,
                "error": result.get("error"),
                "result_json": str(result_json_path),
                "output_video": "",
            }

            save_this_video = (
                args.save_video_mode == "all" or
                (args.save_video_mode == "solved" and ok)
            )
            if save_this_video:
                out_video_path = attempts_video_dir / f"round_{total_round:03d}_{cls}_{video.stem}.mp4"
                predictor.predict_video(
                    str(video),
                    output_video_path=str(out_video_path),
                    show_skeleton=True,
                    show_mudras=True,
                )
                row["output_video"] = str(out_video_path)
        except Exception as exc:  # defensive logging for long smoke loops
            row = {
                "round": total_round,
                "expected": cls,
                "video": str(video),
                "predicted": "error",
                "confidence": 0.0,
                "num_frames": 0,
                "correct": False,
                "error": str(exc),
                "result_json": "",
                "output_video": "",
            }
            ok = False

        attempts_log.append(row)

        if ok and not solved[cls]:
            solved[cls] = True
            solved_record[cls] = row
            print(f"  -> solved {cls} with {Path(row['video']).name} (conf={row['confidence']:.3f})")

    summary = {
        "source_root": str(source_root),
        "checkpoint": str((repo_root / args.checkpoint).resolve()),
        "device": args.device,
        "pose_model": args.pose_model,
        "max_attempts_per_class": args.max_attempts_per_class,
        "seed": args.seed,
        "total_attempts": len(attempts_log),
        "solved_count": sum(1 for v in solved.values() if v),
        "all_solved": all(solved.values()),
        "solved": solved,
        "solved_records": solved_record,
    }

    attempts_json.write_text(json.dumps(attempts_log, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "solved", "attempts_used", "video", "predicted", "confidence", "frames"])
        for cls in DANCE_CLASSES:
            rec = solved_record.get(cls, {})
            used = args.max_attempts_per_class - attempts_left[cls]
            writer.writerow([
                cls,
                solved[cls],
                used,
                Path(rec.get("video", "")).name if rec else "",
                rec.get("predicted", ""),
                rec.get("confidence", ""),
                rec.get("num_frames", ""),
            ])

    print("\n=== RANDOM CLASS SMOKE SUMMARY ===")
    for cls in DANCE_CLASSES:
        used = args.max_attempts_per_class - attempts_left[cls]
        state = "OK" if solved[cls] else "NOT SOLVED"
        print(f"{cls:12s} : {state:10s} | attempts={used:02d}")
    print(f"all solved: {summary['all_solved']} | solved_count={summary['solved_count']}/{len(DANCE_CLASSES)}")
    print(f"attempt logs: {attempts_json}")
    print(f"summary json: {summary_json}")
    print(f"summary csv : {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
