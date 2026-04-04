"""
NatyaVeda v2 — Environment Verification
Checks all dependencies, GPU tier selection, and runs a forward pass.
"""
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

PASS = "  ✅"
WARN = "  ⚠️ "
FAIL = "  ❌"
results = []


def check(name, fn, warn_ok=False):
    try:
        msg = fn()
        results.append((PASS, f"{name}" + (f" — {msg}" if msg else "")))
    except Exception as e:
        icon = WARN if warn_ok else FAIL
        results.append((icon, f"{name}: {e}"))


# ── Core packages ───────────────────────────────────────────────
check("setuptools < 81",  lambda: (
    __import__("setuptools"),
    v := __import__("setuptools").__version__,
    (_ for _ in ()).throw(RuntimeError(f"version {v} ≥ 81 — will break mmcv!")) if int(v.split(".")[0]) >= 81 else v
)[-1])

check("PyTorch",          lambda: __import__("torch").__version__)
check("TorchVision",      lambda: __import__("torchvision").__version__)
check("NumPy <2.0",       lambda: (v := __import__("numpy").__version__,
                                   _ if int(v.split(".")[0]) < 2 else None, v)[2])
check("OpenCV Headless",  lambda: __import__("cv2").__version__)
check("scikit-learn",     lambda: __import__("sklearn").__version__)
check("Transformers",     lambda: __import__("transformers").__version__)
check("HuggingFace Hub",  lambda: __import__("huggingface_hub").__version__)
check("MediaPipe",        lambda: __import__("mediapipe").__version__)
check("TensorFlow",       lambda: __import__("tensorflow").__version__)
check("TF Hub",           lambda: __import__("tensorflow_hub").__version__)
check("PySceneDetect",    lambda: __import__("scenedetect").__version__)
check("yt-dlp",           lambda: __import__("yt_dlp").version.__version__)
check("einops",           lambda: __import__("einops").__version__)
check("wandb",            lambda: __import__("wandb").__version__)

# ── MMPose stack ────────────────────────────────────────────────
check("MMEngine",         lambda: __import__("mmengine").__version__)
check("MMCV",             lambda: __import__("mmcv").__version__)
check("MMDet",            lambda: __import__("mmdet").__version__)
check("MMPose (RTMW-x)",  lambda: __import__("mmpose").__version__)

# ── VitPose ─────────────────────────────────────────────────────
def check_vitpose():
    from transformers import VitPoseForPoseEstimation
    return "VitPose-Plus available via transformers"
check("VitPose-Plus (HF)", check_vitpose)

# ── GPU ─────────────────────────────────────────────────────────
def check_cuda():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"{name} ({vram:.1f} GB)"
    r = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
    if r.returncode == 0:
        raise RuntimeError("GPU present but container lacks --gpus flag → run: docker run --gpus all")
    return "No GPU — CPU mode active"
check("CUDA GPU", check_cuda, warn_ok=True)

# ── DeviceManager ───────────────────────────────────────────────
def check_device_manager():
    from src.utils.device import DeviceManager
    dm = DeviceManager()
    return dm.summary()
check("DeviceManager", check_device_manager)

# ── DanceFormer model ───────────────────────────────────────────
def check_danceformer():
    import torch
    from src.models.danceformer import danceformer_small
    m = danceformer_small()
    x = torch.randn(2, 32, 399)
    o = m(x)
    assert o["dance_logits"].shape == (2, 8)
    assert o["mudra_logits"].shape == (2, 32, 28)
    return f"dance{list(o['dance_logits'].shape)} mudra{list(o['mudra_logits'].shape)}"
check("DanceFormer forward pass", check_danceformer)

# ── Print results ────────────────────────────────────────────────
print("\n" + "="*65)
print("  NatyaVeda Analyzer v2 — Environment Check")
print("="*65)
for icon, msg in results:
    print(f"{icon} {msg}")

passed = sum(1 for icon, _ in results if "✅" in icon)
warned = sum(1 for icon, _ in results if "⚠️" in icon)
failed = sum(1 for icon, _ in results if "❌" in icon)

print("="*65)
print(f"  {passed} passed  |  {warned} warnings  |  {failed} failed")

if failed == 0:
    print("  ✅ All checks passed — ready to run!")
elif failed <= 2:
    print("  ⚠️  Minor issues — check warnings above")
else:
    print("  ❌ Critical failures — run: python scripts/install.py")
print("="*65 + "\n")
