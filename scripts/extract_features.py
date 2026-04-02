"""
NatyaVeda — Feature Extraction Pipeline
Runs RTMW + MediaPipe + VideoMAE over all refined videos.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]


def main():
    parser = argparse.ArgumentParser(description="NatyaVeda Feature Extraction")
    parser.add_argument("--input",      default="data/refined")
    parser.add_argument("--output",     default="data/processed")
    parser.add_argument("--pose-model", default="rtmw-x",
                        choices=["rtmw-x", "rtmw-l", "movenet-thunder"])
    parser.add_argument("--hands",      default="mediapipe",
                        choices=["mediapipe", "rtmw"])
    parser.add_argument("--videomae",   action="store_true", help="Extract VideoMAE features")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--fps",        type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dances",     nargs="+", default=DANCE_CLASSES)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.feature_extraction.pose_extractor import PoseExtractor
    from src.feature_extraction.hand_extractor import HandExtractor
    from src.feature_extraction.videomae_extractor import VideoMAEExtractor

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    pose_extractor = PoseExtractor(
        pose_model=args.pose_model,
        device=args.device,
        sample_fps=args.fps,
    )
    hand_extractor = HandExtractor(
        use_mediapipe=(args.hands == "mediapipe"),
        use_rtmw=True,
        device=args.device,
    )
    videomae_extractor = VideoMAEExtractor(device=args.device) if args.videomae else None

    total_videos = 0
    total_done   = 0

    for dance in args.dances:
        dance_in  = input_dir  / dance
        dance_out = output_dir / dance
        if not dance_in.exists():
            logger.warning(f"  {dance}: input dir not found, skipping")
            continue

        dance_out.mkdir(parents=True, exist_ok=True)
        videos = list(dance_in.glob("*.mp4")) + list(dance_in.glob("*.avi"))
        total_videos += len(videos)

        logger.info(f"\n[{dance.upper()}] {len(videos)} videos")

        for vid_path in videos:
            out_stem = vid_path.stem
            pose_out = dance_out / f"{out_stem}_pose.npz"
            hand_out = dance_out / f"{out_stem}_hands.npz"

            if pose_out.exists() and hand_out.exists():
                logger.info(f"  Skip (already extracted): {vid_path.name}")
                total_done += 1
                continue

            logger.info(f"  Extracting: {vid_path.name}")

            try:
                # Pose extraction
                pose_frames = pose_extractor.extract_video(vid_path)
                if not pose_frames:
                    logger.warning(f"    No valid frames — skipping")
                    continue
                pose_extractor.save_features(pose_frames, pose_out)

                # Hand extraction (fused MediaPipe + RTMW)
                hand_frames = hand_extractor.extract_video(
                    vid_path, pose_frames=pose_frames, sample_fps=args.fps
                )
                hand_extractor.save_features(hand_frames, hand_out)

                # VideoMAE (optional)
                if videomae_extractor and videomae_extractor.available:
                    vmae_out = dance_out / f"{out_stem}_videomae.npz"
                    videomae_extractor.save_features(vid_path, vmae_out)

                # Save merged .npz for training
                import numpy as np
                merged_out = dance_out / f"{out_stem}.npz"
                pose_data = np.load(str(pose_out))
                hand_data = np.load(str(hand_out))

                # Fuse hand keypoints into pose keypoints
                kpts = pose_data["keypoints"].copy()   # [T, 133, 3]
                lh   = hand_data["left_hand"]          # [T', 21, 3]
                rh   = hand_data["right_hand"]         # [T', 21, 3]
                T_pose = kpts.shape[0]
                T_hand = lh.shape[0]
                T_min  = min(T_pose, T_hand)

                kpts[:T_min, 91:112] = lh[:T_min]
                kpts[:T_min, 112:133] = rh[:T_min]

                save_dict = {
                    "keypoints":  kpts,
                    "label":      DANCE_CLASSES.index(dance),
                    "dance_form": dance,
                    "timestamps": pose_data["timestamps"],
                    "confidences": pose_data["confidences"],
                }

                if "velocities" in pose_data:
                    save_dict["velocities"] = pose_data["velocities"]
                if "accelerations" in pose_data:
                    save_dict["accelerations"] = pose_data["accelerations"]

                np.savez_compressed(str(merged_out), **save_dict)
                pose_data.close()
                hand_data.close()

                logger.info(f"    ✓ {T_pose} frames → {merged_out.name}")
                total_done += 1

            except Exception as e:
                logger.error(f"    ✗ Failed: {e}")

    print(f"\n✅ Feature extraction complete: {total_done}/{total_videos} videos processed")
    print(f"   Output dir: {output_dir}")


if __name__ == "__main__":
    main()
