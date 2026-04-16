"""
NatyaVeda v2 -- Environment Verification (Final Clean Version)
"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "  ✅ "
FAIL = "  ❌ "
WARN = "  ⚠️ "
results = []

def check(name, fn, warn_ok=False):
    try:
        msg = fn()
        results.append((PASS if not warn_ok else WARN, f"{name} — {msg}"))
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))

# Core packages
check("setuptools < 81", lambda: __import__("setuptools").__version__)
check("PyTorch", lambda: __import__("torch").__version__)
check("TorchVision", lambda: __import__("torchvision").__version__)

def check_numpy():
    import numpy as np
    v = np.__version__
    if int(v.split(".")[0]) >= 2:
        raise RuntimeError(f"numpy {v} is >=2.0 -- must be <2.0")
    return v
check("NumPy < 2.0", check_numpy)

check("OpenCV Headless", lambda: __import__("cv2").__version__)
check("scikit-learn", lambda: __import__("sklearn").__version__)
check("Transformers", lambda: __import__("transformers").__version__)
check("HuggingFace Hub", lambda: __import__("huggingface_hub").__version__)
check("MediaPipe", lambda: __import__("mediapipe").__version__)
check("TensorFlow", lambda: __import__("tensorflow").__version__)
check("TF Hub", lambda: __import__("tensorflow_hub").__version__)
check("PySceneDetect", lambda: __import__("scenedetect").__version__)
check("yt-dlp", lambda: __import__("yt_dlp").version.__version__)
check("einops", lambda: __import__("einops").__version__)
check("wandb", lambda: __import__("wandb").__version__)
check("MMEngine", lambda: __import__("mmengine").__version__)
check("MMCV", lambda: __import__("mmcv").__version__)
check("MMDet", lambda: __import__("mmdet").__version__)
check("MMPose (RTMW-x)", lambda: __import__("mmpose").__version__)

def check_vitpose():
    from transformers import VitPoseForPoseEstimation
    return "Available via transformers"
check("VitPose-Plus (HF)", check_vitpose)

def check_cuda():
    import torch
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"{torch.cuda.get_device_name(0)} ({vram:.1f} GB)"
    raise RuntimeError("No GPU detected by PyTorch")
check("CUDA GPU", check_cuda)

def check_dm():
    from src.utils.device import DeviceManager
    return DeviceManager().summary()
check("DeviceManager", check_dm)

def check_df():
    import torch
    from src.models.danceformer import danceformer_small
    m = danceformer_small()
    o = m(torch.randn(2, 32, 399))
    return f"dance{list(o['dance_logits'].shape)}"
check("DanceFormer", check_df)

print("\n" + "="*65)
print("  NatyaVeda Analyzer v2 — Environment Check")
print("="*65)
for icon, msg in results:
    print(f"{icon} {msg}")
print("="*65)