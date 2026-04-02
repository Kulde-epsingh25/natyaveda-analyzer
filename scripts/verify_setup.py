"""
NatyaVeda — Setup Verification
Checks all dependencies, model components, and runs a quick forward pass.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

results = []


def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))


# ── Core dependencies
check("PyTorch",          lambda: __import__("torch"))
check("NumPy",            lambda: __import__("numpy"))
check("OpenCV",           lambda: __import__("cv2"))
check("scikit-learn",     lambda: __import__("sklearn"))
check("Transformers (HF)",lambda: __import__("transformers"))
check("MediaPipe",        lambda: __import__("mediapipe"))
check("PyYAML",           lambda: __import__("yaml"))
check("yt-dlp",           lambda: __import__("yt_dlp"))
check("einops",           lambda: __import__("einops"))

# ── MMPose (optional but primary)
check("MMEngine",         lambda: __import__("mmengine"))
check("MMCV",             lambda: __import__("mmcv"))
check("MMDet",            lambda: __import__("mmdet"))
check("MMPose",           lambda: __import__("mmpose"))

# ── TensorFlow / TF Hub (for MoveNet)
check("TensorFlow",       lambda: __import__("tensorflow"))
check("TF Hub",           lambda: __import__("tensorflow_hub"))

# ── Scene detection
check("PySceneDetect",    lambda: __import__("scenedetect"))

# ── Project modules
check("DanceFormer model", lambda: (
    sys.path.insert(0, "."),
    __import__("src.models.danceformer", fromlist=["DanceFormer"])
))

# ── Quick model forward pass
def _quick_forward():
    import torch
    sys.path.insert(0, ".")
    from src.models.danceformer import danceformer_small
    model = danceformer_small()
    kpts = torch.randn(1, 32, 399)
    out = model(kpts)
    assert out["dance_logits"].shape == (1, 8)
    assert out["mudra_logits"].shape == (1, 32, 28)

check("DanceFormer forward pass", _quick_forward)

# ── CUDA
def _cuda_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — GPU training unavailable (CPU mode works)")

check("CUDA GPU", _cuda_check)

# ── Print summary
print("\n" + "=" * 60)
print("  NatyaVeda — Environment Check")
print("=" * 60)
for icon, msg in results:
    print(f"{icon} {msg}")

n_pass = sum(1 for icon, _ in results if "✅" in icon)
n_fail = sum(1 for icon, _ in results if "❌" in icon)
print("=" * 60)
print(f"  {n_pass} passed  |  {n_fail} failed")
if n_fail == 0:
    print("  ✅ All checks passed — ready to train!")
elif n_fail <= 2:
    print("  ⚠️  Some optional dependencies missing — core pipeline should work.")
else:
    print("  ❌ Critical dependencies missing — run: pip install -e '.[dev]'")
