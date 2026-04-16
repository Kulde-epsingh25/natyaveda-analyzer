"""
NatyaVeda — Feature Extraction Pipeline (Fixed API + Save All Components)
Runs ViTPose + MediaPipe + VideoMAE and saves all intermediate and merged files.
"""

import argparse
import logging
import sys
import numpy as np
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
    parser.add_argument("--videomae",   action="store_true", help="Extract VideoMAE features")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--pose-model", default=None,
                        choices=["rtmw-x", "rtmw-l", "vitpose-huge", "vitpose-large", "vitpose-base", "movenet-thunder"],
                        help="Override automatic pose tier selection")
    parser.add_argument("--fps",        type=float, default=25.0)
    parser.add_argument("--dances",     nargs="+", default=DANCE_CLASSES)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.feature_extraction.pose_extractor import PoseExtractor
    from src.feature_extraction.hand_extractor import HandExtractor
    from src.feature_extraction.videomae_extractor import VideoMAEExtractor
    from src.utils.device import DeviceManager

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    # 1. Initialize Device Manager (Fixed initialization)
    device_manager = DeviceManager(force_device=args.device, requested_pose_model=args.pose_model)
    force_movenet_only = str(args.pose_model).lower().startswith("movenet") if args.pose_model else False

    # 2. Initialize Extractors with the correct arguments
    pose_extractor = PoseExtractor(
        device_manager=device_manager, 
        sample_fps=args.fps,
        use_ensemble=not force_movenet_only,
        force_movenet_only=force_movenet_only,
    )
    hand_extractor = HandExtractor(
        use_mediapipe=True, 
        use_rtmw=True, 
        device=args.device
    )
    videomae_extractor = VideoMAEExtractor(device=args.device) if args.videomae else None

    for dance in args.dances:
        dance_in  = input_dir  / dance
        dance_out = output_dir / dance
        if not dance_in.exists():
            continue

        dance_out.mkdir(parents=True, exist_ok=True)
        videos = (
            list(dance_in.glob("*.mp4"))
            + list(dance_in.glob("*.avi"))
            + list(dance_in.glob("*.mov"))
            + list(dance_in.glob("*.mkv"))
            + list(dance_in.glob("*.webm"))
        )

        logger.info(f"\n[{dance.upper()}] {len(videos)} videos")

        for vid_path in videos:
            stem = vid_path.stem
            # Define exact paths to save all components
            pose_out   = dance_out / f"{stem}_pose.npz"
            hand_out   = dance_out / f"{stem}_hands.npz"
            vmae_out   = dance_out / f"{stem}_videomae.npz"
            merged_out = dance_out / f"{stem}.npz"

            # Check if final merged file exists
            if merged_out.exists():
                logger.info(f"  Skip (already processed): {vid_path.name}")
                continue

            logger.info(f"  Processing: {vid_path.name}")

            try:
                # 1. Pose Extraction
                pose_frames = pose_extractor.extract_video(vid_path)
                if not pose_frames:
                    logger.warning("    No valid frames — skipping")
                    continue
                pose_extractor.save_features(pose_frames, pose_out)
                logger.info(f"    ✓ Saved {pose_out.name}")

                # 2. Hand Extraction
                hand_frames = hand_extractor.extract_video(vid_path, pose_frames=pose_frames, sample_fps=args.fps)
                hand_extractor.save_features(hand_frames, hand_out)
                logger.info(f"    ✓ Saved {hand_out.name}")

                # 3. VideoMAE (If requested)
                if args.videomae and videomae_extractor:
                    videomae_extractor.save_features(vid_path, vmae_out)
                    logger.info(f"    ✓ Saved {vmae_out.name}")

                # 4. Final Merge (Create the .npz for training)
                p_data = np.load(str(pose_out), allow_pickle=True)
                h_data = np.load(str(hand_out), allow_pickle=True)

                kpts = p_data["keypoints"].copy()
                # Mapping hand data into the 133-keypoint skeleton
                T_min = min(len(kpts), len(h_data["left_hand"]))
                kpts[:T_min, 91:112] = h_data["left_hand"][:T_min]
                kpts[:T_min, 112:133] = h_data["right_hand"][:T_min]

                save_dict = {
                    "keypoints":   kpts,
                    "label":       DANCE_CLASSES.index(dance),
                    "dance_form":  dance,
                    "timestamps":  p_data["timestamps"],
                    "confidences": p_data["confidences"],
                }

                if args.videomae and vmae_out.exists():
                    v_data = np.load(str(vmae_out))
                    save_dict["videomae_features"] = v_data["features"]

                np.savez_compressed(merged_out, **save_dict)
                logger.info(f"    ✅ FINAL MERGED: {merged_out.name}")

            except Exception as e:
                logger.error(f"    ✗ Failed {vid_path.name}: {e}")

    print(f"\n✅ All extractions complete. Check {args.output} for results.")


if __name__ == "__main__":
    main()