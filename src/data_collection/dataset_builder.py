"""
NatyaVeda — Dataset Builder
Utility to inspect, validate, and report on the processed dataset.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]


def build_dataset_report(data_dir: Path) -> dict:
    """Scan a processed data directory and return a summary report."""
    report = {"classes": {}, "total_files": 0, "total_frames": 0, "issues": []}

    for dance in DANCE_CLASSES:
        dance_dir = data_dir / dance
        if not dance_dir.exists():
            report["issues"].append(f"Missing directory: {dance_dir}")
            continue

        files = [f for f in dance_dir.glob("*.npz")
                 if not any(f.name.endswith(s) for s in ("_pose.npz","_hands.npz","_videomae.npz"))]

        frame_counts = []
        for f in files:
            try:
                d = np.load(str(f))
                frame_counts.append(int(d["keypoints"].shape[0]))
                d.close()
            except Exception as e:
                report["issues"].append(f"Corrupt file {f.name}: {e}")

        n_frames = sum(frame_counts)
        report["classes"][dance] = {
            "files": len(files),
            "total_frames": n_frames,
            "avg_frames_per_file": int(np.mean(frame_counts)) if frame_counts else 0,
            "min_frames": int(min(frame_counts)) if frame_counts else 0,
            "max_frames": int(max(frame_counts)) if frame_counts else 0,
        }
        report["total_files"]  += len(files)
        report["total_frames"] += n_frames

    return report


def print_dataset_report(data_dir: Path) -> None:
    report = build_dataset_report(data_dir)
    print(f"\n{'='*65}")
    print(f"  Dataset Report — {data_dir}")
    print(f"{'='*65}")
    print(f"  {'Dance Form':22s} {'Files':>6}  {'Frames':>8}  {'Avg F/file':>10}")
    print(f"  {'-'*55}")
    for dance, info in report["classes"].items():
        print(f"  {dance:22s} {info['files']:>6}  {info['total_frames']:>8}  {info['avg_frames_per_file']:>10}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':22s} {report['total_files']:>6}  {report['total_frames']:>8}")
    if report["issues"]:
        print(f"\n  ⚠️  Issues ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"    - {issue}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
    print_dataset_report(path)
