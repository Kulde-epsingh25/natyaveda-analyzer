"""
NatyaVeda Analyzer v2 — One-Command Installer
Handles every known installation error automatically:
  - Pins setuptools<81 (fixes pkg_resources for mmcv + tf-hub)
  - Detects GPU/CUDA and picks correct PyTorch index
  - Installs MMCV via miropsota prebuilt wheels (Python 3.12 compatible)
  - Installs opencv-python-headless (no libGL needed)
  - Installs system libs (libGL, ninja) via apt if available
  - Validates every package after install

Usage:
    python scripts/install.py           # auto-detect GPU
    python scripts/install.py --cpu     # force CPU-only install
    python scripts/install.py --cuda 121 # force CUDA 12.1
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"\n  ▶ {cmd}")
    kwargs = dict(shell=True, text=True)
    if capture:
        kwargs["capture_output"] = True
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed (exit {result.returncode})")
        if capture:
            print(result.stderr)
    return result


def pip(packages: str, extra_args: str = "") -> bool:
    result = run(f'"{sys.executable}" -m pip install {packages} {extra_args} -q')
    return result.returncode == 0


def can_import(module: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True
    )
    return result.returncode == 0


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# GPU Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_gpu() -> dict:
    """
    Detect GPU and CUDA availability through multiple methods:
    1. nvidia-smi (definitive — GPU must be present on host)
    2. torch.cuda (checks if CUDA libs are accessible)
    3. /proc/driver/nvidia (Linux GPU driver check)
    Returns dict with: has_gpu, cuda_version, vram_gb, device_name
    """
    info = {"has_gpu": False, "cuda_version": None, "vram_gb": 0, "device_name": "CPU"}

    # Method 1: nvidia-smi (most reliable)
    r = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
        shell=True, capture_output=True, text=True
    )
    if r.returncode == 0 and r.stdout.strip():
        lines = r.stdout.strip().split("\n")
        first = lines[0].split(",")
        info["has_gpu"] = True
        info["device_name"] = first[0].strip()
        try:
            vram_str = first[1].strip().replace(" MiB", "").replace(" MB", "")
            info["vram_gb"] = int(vram_str) / 1024
        except (ValueError, IndexError):
            info["vram_gb"] = 8  # assume 8GB if parse fails

    # Method 2: CUDA version from nvcc or nvidia-smi
    r2 = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if r2.returncode == 0:
        for line in r2.stdout.split("\n"):
            if "CUDA Version:" in line:
                try:
                    version = line.split("CUDA Version:")[1].strip().split()[0]
                    # Convert "12.1" → "121", "11.8" → "118"
                    info["cuda_version"] = version.replace(".", "")[:3]
                except (IndexError, ValueError):
                    info["cuda_version"] = "121"  # safe default
                break

    # Method 3: Check /proc (Linux containers with GPU passthrough)
    if not info["has_gpu"] and Path("/proc/driver/nvidia/version").exists():
        info["has_gpu"] = True
        if not info["cuda_version"]:
            info["cuda_version"] = "121"

    return info


def pick_torch_index(gpu_info: dict, force_cpu: bool, force_cuda: str | None) -> tuple[str, str]:
    """Returns (torch_index_url, cuda_tag) e.g. ('https://...cu121', 'cu121')"""
    if force_cpu:
        return "https://download.pytorch.org/whl/cpu", "cpu"

    if force_cuda:
        tag = f"cu{force_cuda}"
        return f"https://download.pytorch.org/whl/{tag}", tag

    if gpu_info["has_gpu"] and gpu_info["cuda_version"]:
        cv = gpu_info["cuda_version"]
        # Round to nearest supported: 118, 121, 124
        supported = {"118": "cu118", "121": "cu121", "124": "cu124"}
        tag = supported.get(cv, "cu121")  # default cu121
        return f"https://download.pytorch.org/whl/{tag}", tag

    return "https://download.pytorch.org/whl/cpu", "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────────────────────────────────────

def install_system_deps() -> None:
    """Install system packages needed for video/pose processing."""
    if platform.system() != "Linux":
        print("  ℹ️  Skipping apt install (not Linux)")
        return

    # Check if apt is available
    if subprocess.run("which apt-get", shell=True, capture_output=True).returncode != 0:
        print("  ℹ️  apt-get not available, skipping system deps")
        return

    print("  Installing system libraries via apt...")
    run(
        "apt-get update -qq && apt-get install -y --no-install-recommends "
        "libgl1 libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 "
        "ninja-build gcc g++ ffmpeg git",
        check=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Install steps
# ─────────────────────────────────────────────────────────────────────────────

def step_setuptools() -> None:
    """MUST be first — pin setuptools<81 to restore pkg_resources."""
    print_section("Step 1/8 — Pin setuptools (fixes mmcv + tf-hub + mim)")
    pip('"setuptools>=69.5.1,<81" wheel pip --upgrade')
    v = subprocess.run(
        [sys.executable, "-c", "import setuptools; print(setuptools.__version__)"],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"  ✓ setuptools {v}")


def step_pytorch(index_url: str, cuda_tag: str) -> None:
    print_section(f"Step 2/8 — PyTorch 2.2.0 ({cuda_tag})")

    if can_import("torch"):
        v = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True
        ).stdout.strip()
        print(f"  ✓ PyTorch {v} already installed — skipping")
        return

    pip(f"torch==2.2.0 torchvision==0.17.0 --index-url {index_url}")

    # Verify
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"],
        capture_output=True, text=True
    )
    print(f"  ✓ {result.stdout.strip()}")


def step_mmpose(cuda_tag: str) -> None:
    """
    Install MMCV via miropsota prebuilt wheels (Python 3.12 compatible).
    Then install mmdet and mmpose normally.
    """
    print_section("Step 3/8 — MMPose Ecosystem (RTMW-x 133 keypoints)")

    # Step A: mmengine (pure Python, always works)
    print("  Installing mmengine...")
    pip("mmengine>=0.10.0")

    # Step B: MMCV via miropsota prebuilt (bypasses Python 3.12 build issue)
    # Format: mmcv==2.2.0+pt2.2.0cu121
    if cuda_tag == "cpu":
        mmcv_variant = "mmcv==2.2.0+pt2.2.0cpu"
    else:
        mmcv_variant = f"mmcv==2.2.0+pt2.2.0{cuda_tag}"

    print(f"  Installing {mmcv_variant} via prebuilt wheels...")
    r = pip(
        f'"{mmcv_variant}"',
        extra_args='--extra-index-url https://miropsota.github.io/torch_packages_builder'
    )

    if not r:
        print("  ⚠️  miropsota wheel failed, trying mmcv-lite (no CUDA ops, still functional)...")
        # mmcv-lite has no custom CUDA ops — pose estimation still works via CPU ops
        pip("mmcv-lite>=2.1.0")

    # Step C: mmdet and mmpose
    print("  Installing mmdet and mmpose...")
    pip("mmdet==3.3.0")
    pip("mmpose==1.3.1")

    # Download RTMW-x weights via mim
    print("  Downloading RTMW-x weights via mim...")
    r = subprocess.run(
        f'"{sys.executable}" -m mim download mmpose '
        f'--config td-hm_rtmw-x_8xb2-270e_coco-wholebody-384x288 '
        f'--dest weights/',
        shell=True, capture_output=True, text=True
    )
    if r.returncode == 0:
        print("  ✓ RTMW-x weights downloaded")
    else:
        print("  ⚠️  RTMW-x weights will download on first use")

    # Verify
    for pkg in ["mmengine", "mmdet", "mmpose"]:
        ok = "✓" if can_import(pkg) else "✗"
        print(f"  {ok} {pkg}")


def step_vitpose() -> None:
    """Install VitPose-Plus via HuggingFace transformers (Python 3.12 native)."""
    print_section("Step 4/8 — VitPose-Plus + HuggingFace (secondary pose)")
    pip("transformers>=4.40.0 huggingface-hub>=0.22.0 timm>=1.0.0 accelerate>=0.27.0 einops>=0.7.0")
    print("  ✓ transformers (VitPose-Plus, VideoMAE-v2, DETR available)")


def step_tensorflow() -> None:
    """Install tensorflow-cpu + tf-hub for MoveNet Thunder (CPU fallback)."""
    print_section("Step 5/8 — TensorFlow CPU + TF Hub (MoveNet fallback)")
    # tensorflow-cpu: no CUDA conflict, works identically for inference
    pip("tensorflow-cpu>=2.15.0 tensorflow-hub>=0.16.0")
    ok = "✓" if can_import("tensorflow") else "✗"
    print(f"  {ok} tensorflow")
    ok = "✓" if can_import("tensorflow_hub") else "✗"
    print(f"  {ok} tensorflow_hub")


def step_mediapipe_cv() -> None:
    """Install MediaPipe + opencv-headless (no libGL required)."""
    print_section("Step 6/8 — MediaPipe + OpenCV Headless")
    # opencv-python-headless: identical to opencv-python but no libGL dependency
    pip("mediapipe>=0.10.0 opencv-python-headless>=4.9.0 Pillow>=10.0.0 imageio>=2.31.0 albumentations>=1.3.0")
    for pkg in ["mediapipe", "cv2"]:
        ok = "✓" if can_import(pkg) else "✗"
        print(f"  {ok} {pkg}")


def step_remaining() -> None:
    """Install all remaining packages from requirements.txt."""
    print_section("Step 7/8 — All Remaining Dependencies")
    pip(
        "numpy>=1.24.0 scipy>=1.11.0 pandas>=2.0.0 scikit-learn>=1.3.0 "
        "yt-dlp>=2024.1.0 scenedetect>=0.6.3 ffmpeg-python>=0.2.0 "
        "wandb>=0.16.0 tensorboard>=2.15.0 pyyaml>=6.0 omegaconf>=2.3.0 "
        "rich>=13.0.0 tqdm>=4.65.0 matplotlib>=3.7.0 seaborn>=0.12.0 "
        "pytest>=7.4.0 pytest-cov>=4.1.0 black>=23.0.0 isort>=5.12.0 flake8>=6.0.0"
    )


def step_project() -> None:
    """Install NatyaVeda package itself."""
    print_section("Step 8/8 — Install NatyaVeda Project")
    root = Path(__file__).parent.parent
    pip(f'-e "{root}[dev]" --no-deps')
    ok = "✓" if can_import("src") else "✗"
    print(f"  {ok} natyaveda package")


# ─────────────────────────────────────────────────────────────────────────────
# Final validation
# ─────────────────────────────────────────────────────────────────────────────

def validate() -> None:
    print_section("VALIDATION")

    checks = [
        ("torch",          "PyTorch"),
        ("torchvision",    "TorchVision"),
        ("mmengine",       "MMEngine"),
        ("mmdet",          "MMDet"),
        ("mmpose",         "MMPose (RTMW-x 133 kpts)"),
        ("transformers",   "Transformers (VitPose-Plus)"),
        ("mediapipe",      "MediaPipe (hand mudra)"),
        ("cv2",            "OpenCV Headless"),
        ("tensorflow",     "TensorFlow (MoveNet)"),
        ("tensorflow_hub", "TF Hub"),
        ("scenedetect",    "PySceneDetect"),
        ("sklearn",        "scikit-learn"),
        ("wandb",          "Weights & Biases"),
        ("yt_dlp",         "yt-dlp"),
    ]

    passed = 0
    failed = []
    for module, name in checks:
        ok = can_import(module)
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
        if ok:
            passed += 1
        else:
            failed.append(name)

    # DanceFormer model test
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'.');  "
         "import torch; from src.models.danceformer import danceformer_small; "
         "m=danceformer_small(); x=torch.randn(2,32,399); o=m(x); "
         "assert o['dance_logits'].shape==(2,8); print('OK')"],
        capture_output=True, text=True
    )
    ok = result.returncode == 0 and "OK" in result.stdout
    print(f"  {'✅' if ok else '❌'} DanceFormer forward pass")
    if ok:
        passed += 1
    else:
        failed.append("DanceFormer")

    # GPU check
    gpu_result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'CUDA:{torch.cuda.is_available()} "
         "| Devices:{torch.cuda.device_count()} "
         "| Name:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"\n  GPU Status: {gpu_result}")

    print(f"\n  {'='*50}")
    print(f"  {passed}/{len(checks)+1} checks passed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("  ✅ EVERYTHING INSTALLED — ready to run!")
    print(f"  {'='*50}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NatyaVeda Installer")
    parser.add_argument("--cpu",  action="store_true", help="Force CPU-only install")
    parser.add_argument("--cuda", default=None,        help="Force CUDA version e.g. 121 or 118")
    parser.add_argument("--skip-system", action="store_true", help="Skip apt-get installs")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  NatyaVeda Analyzer v2 — Full Installer")
    print(f"  Python {sys.version.split()[0]} | {platform.system()}")
    print("="*60)

    # Detect GPU
    print("\n  Detecting GPU...")
    gpu = detect_gpu()
    if gpu["has_gpu"]:
        print(f"  ✅ GPU detected: {gpu['device_name']} ({gpu['vram_gb']:.1f} GB VRAM)")
        print(f"     CUDA version: {gpu['cuda_version']}")
    else:
        print("  ⚠️  No GPU detected — installing CPU mode")
        print("     Pose extraction will use VitPose-Base + MoveNet on CPU")

    index_url, cuda_tag = pick_torch_index(gpu, args.cpu, args.cuda)
    print(f"  Using torch index: {index_url}")

    # Run install steps
    if not args.skip_system:
        install_system_deps()

    step_setuptools()
    step_pytorch(index_url, cuda_tag)
    step_mmpose(cuda_tag)
    step_vitpose()
    step_tensorflow()
    step_mediapipe_cv()
    step_remaining()
    step_project()
    validate()


if __name__ == "__main__":
    main()

