"""
NatyaVeda — GPU Diagnostic Script
Run this to check GPU status, diagnose container issues,
and see exactly which pose model tier will be used.

Usage: python scripts/check_gpu.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_gpu():
    print("\n" + "="*60)
    print("  NatyaVeda — GPU Diagnostic Report")
    print("="*60)

    # 1. nvidia-smi
    print("\n  [1] nvidia-smi output:")
    r = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if r.returncode == 0:
        for line in r.stdout.strip().split("\n"):
            print(f"      {line}")
    else:
        print("      ❌ nvidia-smi not found or no GPU")

    # 2. PyTorch CUDA
    print("\n  [2] PyTorch CUDA status:")
    try:
        import torch
        print(f"      PyTorch version : {torch.__version__}")
        print(f"      CUDA available  : {torch.cuda.is_available()}")
        print(f"      CUDA version    : {torch.version.cuda}")
        print(f"      cuDNN version   : {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            print(f"      GPU count       : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram = props.total_memory / (1024**3)
                print(f"      GPU {i}           : {props.name} ({vram:.1f} GB)")
                print(f"      Compute cap     : {props.major}.{props.minor}")
        else:
            print("      ⚠️  CUDA not available to PyTorch")
    except ImportError:
        print("      ❌ PyTorch not installed")

    # 3. CUDA libraries
    print("\n  [3] CUDA library availability:")
    for lib in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"]:
        if Path(lib).exists():
            print(f"      ✓ nvcc found: {lib}")
            break
    else:
        print("      ⚠️  nvcc not found (optional for inference)")

    # 4. Container GPU check
    print("\n  [4] Container GPU passthrough check:")
    proc_path = Path("/proc/driver/nvidia/version")
    if proc_path.exists():
        print(f"      ✓ NVIDIA driver visible at /proc/driver/nvidia/")
        print(f"        {proc_path.read_text().strip()[:80]}")
    else:
        print("      ⚠️  /proc/driver/nvidia not found")
        r2 = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
        if r2.returncode == 0:
            print("      ✓ GPU is present but container lacks --gpus flag!")
            print("      → See fix instructions below")
        else:
            print("      → No GPU present on this machine")

    # 5. DeviceManager
    print("\n  [5] NatyaVeda DeviceManager selection:")
    try:
        from src.utils.device import DeviceManager, print_gpu_fix_instructions
        dm = DeviceManager()
        print(f"      {dm.summary()}")
        print(f"      Pose model      : {dm.pose_model_tier}")
        print(f"      VitPose model   : {dm.vitpose_model_id}")
        print(f"      fp16 enabled    : {dm.fp16}")

        # Check if GPU present but not accessible
        import torch
        r3 = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
        if r3.returncode == 0 and not torch.cuda.is_available():
            print_gpu_fix_instructions()
    except Exception as e:
        print(f"      Could not load DeviceManager: {e}")

    # 6. Recommendation
    print("\n  [6] Recommendation:")
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram >= 8:
                print(f"      ✅ {vram:.1f}GB VRAM — Full RTMW-x (133 kpts) + fp16 enabled")
            elif vram >= 4:
                print(f"      ✅ {vram:.1f}GB VRAM — VitPose-Large (133 kpts)")
            else:
                print(f"      ✅ {vram:.1f}GB VRAM — VitPose-Base (133 kpts)")
        else:
            r4 = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
            if r4.returncode == 0:
                print("      ⚠️  GPU present but inaccessible — restart container with --gpus all")
            else:
                print("      ℹ️  CPU mode — VitPose-Base + MoveNet will be used")
                print("         Training will work; extraction will be slower")
    except ImportError:
        pass

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    check_gpu()
