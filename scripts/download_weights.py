"""
NatyaVeda — Download Pretrained Weights
Downloads RTMW-x, MoveNet Thunder (via TF Hub), and DanceFormer base checkpoint.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


def download_rtmw():
    """Download RTMW-x wholebody checkpoint via mim."""
    logger.info("Downloading RTMW-x wholebody checkpoint via MMPose mim …")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mim", "download", "mmpose",
                "--config",
                "td-hm_rtmw-x_8xb2-270e_coco-wholebody-384x288",
                "--dest", str(WEIGHTS_DIR),
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info("  ✓ RTMW-x downloaded")
        else:
            logger.warning(f"  mim download failed: {result.stderr[:200]}")
    except Exception as e:
        logger.warning(f"  RTMW download skipped: {e}")


def download_movenet_info():
    """MoveNet is loaded at runtime from TF Hub — just verify TF Hub works."""
    logger.info("Verifying TF Hub connectivity for MoveNet Thunder …")
    try:
        import tensorflow_hub as hub
        logger.info("  ✓ TensorFlow Hub available")
        logger.info("    MoveNet will be downloaded on first use from:")
        logger.info("    https://tfhub.dev/google/movenet/singlepose/thunder/4")
    except ImportError:
        logger.warning("  tensorflow-hub not installed — MoveNet unavailable")


def verify_mediapipe():
    logger.info("Verifying MediaPipe Hands …")
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=2) as hands:
            pass
        logger.info("  ✓ MediaPipe Hands available")
    except ImportError:
        logger.warning("  mediapipe not installed — hand extraction unavailable")


def verify_transformers():
    logger.info("Verifying HuggingFace Transformers (VideoMAE, RT-DETR) …")
    try:
        import transformers
        logger.info(f"  ✓ transformers {transformers.__version__} available")
        logger.info("    VideoMAE-v2 (MCG-NJU/videomae-large) will download on first use")
        logger.info("    RT-DETR-L (PekingU/rtdetr_l) will download on first use")
    except ImportError:
        logger.warning("  transformers not installed")


def create_placeholder_checkpoint():
    """Create a minimal untrained checkpoint for testing inference pipeline."""
    import torch, sys
    sys.path.insert(0, ".")
    try:
        from src.models.danceformer import danceformer_small
        model = danceformer_small()
        ckpt_path = WEIGHTS_DIR / "danceformer_untrained_small.pt"
        torch.save({
            "epoch": 0,
            "state_dict": model.state_dict(),
            "val_f1": 0.0,
            "config": {
                "embed_dim": 128,
                "use_videomae_fusion": False,
                "num_transformer_layers": 4,
                "num_attention_heads": 4,
                "ff_dim": 512,
                "max_sequence_length": 64,
                "num_dance_classes": 8,
                "num_mudra_classes": 28,
                "videomae_proj_dim": 1024,
            },
        }, str(ckpt_path))
        logger.info(f"  ✓ Placeholder checkpoint saved: {ckpt_path}")
    except Exception as e:
        logger.warning(f"  Could not create placeholder: {e}")


if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  NatyaVeda — Downloading Weights & Verifying Setup")
    logger.info("=" * 55)
    download_rtmw()
    download_movenet_info()
    verify_mediapipe()
    verify_transformers()
    create_placeholder_checkpoint()
    logger.info("\nDone. Run `python scripts/verify_setup.py` to validate the full pipeline.")
