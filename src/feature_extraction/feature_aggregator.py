"""
NatyaVeda — Feature Aggregator
Merges pose, hand, and VideoMAE features into a unified per-frame feature tensor.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """
    Loads pose (.npz), hand (.npz), and optional VideoMAE (.npz) features
    for a single video and fuses them into the final training-ready array.

    Output merged .npz structure:
      keypoints:      [T, 133, 3]   — full wholebody kpts with fused hands
      velocities:     [T, 133, 2]   — temporal first diff
      accelerations:  [T, 133, 2]   — temporal second diff
      videomae_feats: [T_clips, D]  — VideoMAE tokens (optional)
      label:          int           — dance class index
      dance_form:     str
    """

    def aggregate(
        self,
        pose_path: Path,
        hand_path: Path,
        videomae_path: Path | None = None,
        label: int = 0,
        dance_form: str = "unknown",
        output_path: Path | None = None,
    ) -> dict:
        """Load and merge all feature files. Returns merged dict."""
        pose_path = Path(pose_path)
        hand_path = Path(hand_path)

        pose_data = np.load(str(pose_path))
        hand_data = np.load(str(hand_path))

        kpts      = pose_data["keypoints"].copy()    # [T, 133, 3]
        T_pose    = kpts.shape[0]
        T_hand    = hand_data["left_hand"].shape[0]
        T_min     = min(T_pose, T_hand)

        # Fuse MediaPipe hand keypoints into wholebody array
        lh = hand_data["left_hand"]    # [T', 21, 3]
        rh = hand_data["right_hand"]   # [T', 21, 3]
        lh_conf = hand_data["left_confidence"]   # [T']
        rh_conf = hand_data["right_confidence"]  # [T']

        for t in range(T_min):
            if lh_conf[t] > 0.3:
                kpts[t, 91:112] = lh[t]
            if rh_conf[t] > 0.3:
                kpts[t, 112:133] = rh[t]

        # Velocities & accelerations
        coords = kpts[:, :, :2]  # [T, 133, 2]
        vel    = np.diff(coords, axis=0, prepend=coords[:1]).astype(np.float32)
        acc    = np.diff(vel,    axis=0, prepend=vel[:1]).astype(np.float32)

        merged = {
            "keypoints":     kpts.astype(np.float32),
            "velocities":    vel,
            "accelerations": acc,
            "timestamps":    pose_data["timestamps"],
            "confidences":   pose_data["confidences"],
            "label":         np.int64(label),
            "dance_form":    dance_form,
        }

        # Optional VideoMAE features
        if videomae_path and Path(videomae_path).exists():
            vd = np.load(str(videomae_path))
            merged["videomae_features"] = vd["videomae_features"]
            vd.close()

        pose_data.close()
        hand_data.close()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(output_path), **merged)
            logger.info(f"Saved merged features: {output_path}")

        return merged

    def aggregate_batch(
        self,
        processed_dir: Path,
        output_dir: Path,
        dance_classes: list[str],
    ) -> int:
        """Aggregate all pose+hand pairs in a directory. Returns count of merged files."""
        processed_dir = Path(processed_dir)
        output_dir    = Path(output_dir)
        total = 0

        for idx, dance in enumerate(dance_classes):
            dance_in  = processed_dir / dance
            dance_out = output_dir    / dance
            if not dance_in.exists():
                continue
            dance_out.mkdir(parents=True, exist_ok=True)

            pose_files = sorted(dance_in.glob("*_pose.npz"))
            for pf in pose_files:
                stem = pf.stem.replace("_pose", "")
                hf   = dance_in / f"{stem}_hands.npz"
                vf   = dance_in / f"{stem}_videomae.npz"
                out  = dance_out / f"{stem}.npz"

                if out.exists():
                    total += 1
                    continue
                if not hf.exists():
                    logger.warning(f"No hand file for {stem} — skipping")
                    continue

                self.aggregate(
                    pose_path=pf,
                    hand_path=hf,
                    videomae_path=vf if vf.exists() else None,
                    label=idx,
                    dance_form=dance,
                    output_path=out,
                )
                total += 1

        return total
