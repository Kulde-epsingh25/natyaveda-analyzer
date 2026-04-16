"""
NatyaVeda v2 — Download Weights
Downloads all model weights: RTMW-x, VitPose-Plus, MoveNet verification.
"""
import logging, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

def download_rtmw():
    logger.info("Downloading RTMW-x via mim ...")
    try:
        r = subprocess.run([sys.executable, "-m", "mim", "download", "mmpose",
             "--config", "td-hm_rtmw-x_8xb2-270e_coco-wholebody-384x288",
             "--dest", str(WEIGHTS_DIR)], capture_output=True, text=True, timeout=600)
        if r.returncode == 0: logger.info("  ✓ RTMW-x downloaded")
        else: logger.warning("  mim failed — RTMW-x downloads on first use")
    except Exception as e: logger.warning("  RTMW-x error: %s", e)

def prefetch_vitpose(model_id="usyd-community/vitpose-plus-base"):
    logger.info("Pre-fetching %s from HuggingFace ...", model_id)
    try:
        from transformers import VitPoseForPoseEstimation, AutoProcessor
        AutoProcessor.from_pretrained(model_id)
        VitPoseForPoseEstimation.from_pretrained(model_id)
        logger.info("  ✓ Cached: %s", model_id)
    except Exception as e: logger.warning("  Pre-fetch failed: %s", e)

def prefetch_detr():
    logger.info("Pre-fetching DETR person detector ...")
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        logger.info("  ✓ DETR-ResNet-50 cached")
    except Exception as e: logger.warning("  DETR pre-fetch failed: %s", e)

def check_movenet():
    logger.info("Verifying TF Hub (MoveNet) ...")
    try:
        import tensorflow_hub as hub; logger.info("  ✓ TF Hub OK")
    except ImportError: logger.warning("  tensorflow-hub not installed")

def create_placeholder():
    try:
        import torch
        from src.models.danceformer import danceformer_small
        model = danceformer_small()
        path = WEIGHTS_DIR / "danceformer_untrained_small.pt"
        torch.save({"epoch":0,"val_f1":0.0,"state_dict":model.state_dict(),
            "config":{"embed_dim":128,"use_videomae_fusion":False,
            "num_transformer_layers":4,"num_attention_heads":4,"ff_dim":512,
            "max_sequence_length":64,"num_dance_classes":8,"num_mudra_classes":28,
            "videomae_proj_dim":1024}}, str(path))
        logger.info("  ✓ Placeholder checkpoint saved")
    except Exception as e: logger.warning("  Placeholder failed: %s", e)

if __name__ == "__main__":
    import torch
    logger.info("="*55)
    logger.info("  NatyaVeda v2 — Downloading Weights")
    logger.info("="*55)
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        logger.info("GPU: %s (%.1fGB)", torch.cuda.get_device_name(0), vram)
        if vram >= 6: prefetch_vitpose("usyd-community/vitpose-plus-huge")
    else: logger.info("CPU mode")
    download_rtmw()
    prefetch_vitpose()
    prefetch_detr()
    check_movenet()
    create_placeholder()
    logger.info("Done. Run: python scripts/verify_setup.py")
