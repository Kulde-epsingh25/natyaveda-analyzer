r"""
NatyaVeda Analyzer v2 -- Bulletproof Installer
Fixes applied from final logs:
  - 'Ghost Buster' added to forcefully delete locked ~umpy Windows folders.
  - Ironclad NumPy Lock: "numpy<2.0.0" is injected into EVERY pip command.
  - Uninstalls conflicting ml-dtypes (TensorFlow conflict fix).
"""

from __future__ import annotations
import subprocess
import sys
import platform
import shutil
import site
from pathlib import Path


def ghost_buster() -> None:
    """Force-deletes corrupted temporary folders (~umpy, -umpy) left by failed pip runs."""
    for sp in site.getsitepackages():
        sp_path = Path(sp)
        if sp_path.exists():
            # Delete folders starting with ~ (like ~umpy) or - (like -umpy)
            for prefix in ["~*", "-*"]:
                for ghost_folder in sp_path.glob(prefix):
                    try:
                        shutil.rmtree(ghost_folder)
                    except Exception:
                        pass


def run(cmd: str) -> bool:
    print(f"\n  >> {cmd[:110]}")
    return subprocess.run(cmd, shell=True, text=True).returncode == 0


def pip(packages: str, extra: str = "", lock_numpy: bool = True) -> bool:
    """Runs pip, but mathematically prevents numpy 2.x from sneaking in."""
    if lock_numpy and "numpy" not in packages:
        packages += ' "numpy>=1.24.0,<2.0.0"'
    return run(f'"{sys.executable}" -m pip install {packages} {extra} -q')


def uninstall(packages: str) -> None:
    run(f'"{sys.executable}" -m pip uninstall -y {packages} -q')


def can_import(mod: str) -> bool:
    return subprocess.run([sys.executable, "-c", f"import {mod}"],
                          capture_output=True).returncode == 0


def section(title: str) -> None:
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ── GPU Detection ─────────────────────────────────────────────────────────────

def detect_gpu() -> dict:
    info = {"has_gpu": False, "cuda_version": "121", "vram_gb": 4.0, "tag": "cu121"}
    r = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                       shell=True, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        parts = r.stdout.strip().split("\n")[0].split(",")
        info["has_gpu"] = True
        info["name"] = parts[0].strip()
        try:
            info["vram_gb"] = int(parts[1].strip().replace(" MiB","").replace(" MB","")) / 1024
        except Exception:
            info["vram_gb"] = 4.0

    r2 = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if r2.returncode == 0:
        for line in r2.stdout.split("\n"):
            if "CUDA Version:" in line:
                cv = line.split("CUDA Version:")[1].strip().split()[0].replace(".", "")[:3]
                info["cuda_version"] = cv
                info["tag"] = {"118": "cu118", "121": "cu121", "124": "cu124"}.get(cv, "cu121")
                break
    return info


# ── Step 0: Clean Environment ────────────────────────────────────────────────

def step_clean_environment() -> None:
    section("Step 0 -- Nuke Ghost Folders & Conflicting Packages")
    print("  [0a] Busting ghost folders (~umpy)...")
    ghost_buster()
    
    print("  [0b] Removing conflicts (OpenCV, ml-dtypes)...")
    uninstall("opencv-python opencv-contrib-python opencv-python-headless ml-dtypes")
    
    pip('"opencv-python-headless<4.10"', extra="--force-reinstall")
    print(f"  opencv-python-headless: {'OK' if can_import('cv2') else 'FAIL'}")


# ── Step 1: Lock core versions ───────────────────────────────────────────────

def step_core_lock() -> None:
    section("Step 1 -- Atomic Lock (numpy, setuptools, protobuf, click)")
    pip(
        '"numpy>=1.24.0,<2.0.0" '
        '"setuptools>=69.5.1,<81" '
        '"protobuf>=3.11,<4.0" '
        '"click<8.3.0" '
        'wheel',
        extra="--force-reinstall",
        lock_numpy=False # Already included in string
    )


# ── Step 2: PyTorch ───────────────────────────────────────────────────────────

def step_pytorch(tag: str) -> None:
    section(f"Step 2 -- PyTorch 2.2.0 ({tag})")
    if can_import("torch"):
        r = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True)
        print(f"  Already installed: {r.stdout.strip()} -- skipping")
        return
    pip(f"torch==2.2.0 torchvision==0.17.0",
        extra=f"--index-url https://download.pytorch.org/whl/{tag}")


# ── Step 3: MMPose ────────────────────────────────────────────────────────────

def step_mmpose(tag: str) -> None:
    section("Step 3 -- MMPose Ecosystem (RTMW-x 133 keypoints)")

    print("  [3a] chumpy...")
    pip("chumpy", extra="--no-build-isolation")

    print("  [3b] mmengine...")
    pip("mmengine>=0.10.0")

    print("  [3c] mmcv via prebuilt wheel...")
    variant = f"mmcv==2.2.0+pt2.2.0{tag}" if tag != "cpu" else "mmcv==2.2.0+pt2.2.0cpu"
    ok = pip(f'"{variant}"',
             extra="--extra-index-url https://miropsota.github.io/torch_packages_builder")
    if not ok:
        pip("mmcv-lite==2.1.0")

    print("  [3d] mmdet & tools...")
    pip("mmdet==3.3.0", extra="--no-deps")
    pip("pycocotools shapely terminaltables scipy matplotlib") # NumPy lock will automatically append here!

    print("  [3e] mmpose...")
    pip("mmpose==1.3.1")


# ── Step 4: VitPose / HuggingFace ────────────────────────────────────────────

def step_vitpose() -> None:
    section("Step 4 -- VitPose-Plus + HuggingFace (transformers 4.49)")
    pip('"transformers>=4.49.0,<4.52.0" "huggingface-hub>=0.23.0,<1.0.0" '
        'timm>=1.0.0 accelerate>=0.27.0 einops>=0.7.0 safetensors>=0.4.0')


# ── Step 5: TensorFlow CPU ────────────────────────────────────────────────────

def step_tensorflow() -> None:
    section("Step 5 -- TensorFlow-CPU 2.15 + TF Hub (MoveNet)")
    pip('"tensorflow-cpu==2.15.0" "keras<3.0" "tensorflow-hub>=0.16.0" "ml-dtypes~=0.2.0"')


# ── Step 6: MediaPipe ─────────────────────────────────────────────────────────

def step_mediapipe() -> None:
    section("Step 6 -- MediaPipe (hand mudra landmarks)")
    pip('"mediapipe==0.10.11" "Pillow>=10.0.0" "imageio>=2.31.0" "albumentations>=1.3.0"')


# ── Step 7: Remaining deps ────────────────────────────────────────────────────

def step_remaining() -> None:
    section("Step 7 -- Remaining Dependencies")
    pip(
        '"wandb>=0.16.0,<0.17" '
        '"tensorboard>=2.15.0" '
        '"scipy>=1.11.0" "pandas>=2.0.0" "scikit-learn>=1.3.0" '
        '"yt-dlp>=2024.1.0" "scenedetect>=0.6.3" "ffmpeg-python>=0.2.0" '
        '"pyyaml>=6.0" "omegaconf>=2.3.0" "rich>=13.0.0" "tqdm>=4.65.0" '
        '"matplotlib>=3.7.0" "seaborn>=0.12.0" '
        '"pytest>=7.4.0" "pytest-cov>=4.1.0" "black>=23.0.0" "isort>=5.12.0" "flake8>=6.0.0"'
    )


# ── Step 8: Project link ──────────────────────────────────────────────────────

def step_project() -> None:
    section("Step 8 -- Install NatyaVeda Project")
    root = Path(__file__).parent.parent.absolute()
    pip(f'-e "{root}[dev]"', extra="--no-deps")


# ── Step 9: Final numpy re-lock ──────────────────────────────────────────────

def step_relock_numpy() -> None:
    section("Step 9 -- Re-lock numpy (Sanity Check)")
    pip('"numpy>=1.24.0,<2.0.0"', extra="--force-reinstall", lock_numpy=False)


# ── Validation ────────────────────────────────────────────────────────────────

def validate() -> None:
    section("VALIDATION")
    checks = [
        ("numpy",          "NumPy"),
        ("torch",          "PyTorch"),
        ("torchvision",    "TorchVision"),
        ("mmengine",       "MMEngine"),
        ("mmdet",          "MMDet"),
        ("mmpose",         "MMPose"),
        ("transformers",   "Transformers (VitPose)"),
        ("mediapipe",      "MediaPipe"),
        ("cv2",            "OpenCV Headless"),
        ("tensorflow",     "TensorFlow"),
        ("sklearn",        "scikit-learn"),
        ("wandb",          "W&B"),
        ("yt_dlp",         "yt-dlp"),
    ]
    passed = failed = 0
    for mod, name in checks:
        ok = can_import(mod)
        print(f"  {'OK' if ok else 'FAIL'} {name}")
        if ok: passed += 1
        else:  failed += 1

    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'.');"
         "import torch; from src.models.danceformer import danceformer_small;"
         "m=danceformer_small(); o=m(torch.randn(2,32,399));"
         "assert o['dance_logits'].shape==(2,8); print('PASS')"],
        capture_output=True, text=True
    )
    ok = "PASS" in r.stdout
    print(f"  {'OK' if ok else 'FAIL'} DanceFormer")
    if ok: passed += 1
    else:  failed += 1

    r2 = subprocess.run(
        [sys.executable, "-c",
         "import torch; "
         "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"],
        capture_output=True, text=True
    )
    print(f"\n  {r2.stdout.strip()}")
    print(f"\n  {passed}/{passed+failed} passed")
    if failed == 0:
        print("  ALL DONE -- ready to run!")
    else:
        print(f"  {failed} failed -- check above")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cpu",  action="store_true")
    p.add_argument("--cuda", default=None)
    args = p.parse_args()

    gpu = detect_gpu()
    tag = "cpu" if args.cpu else (f"cu{args.cuda}" if args.cuda else gpu.get("tag", "cu121"))

    print(f"\n  NatyaVeda Installer | Python {sys.version.split()[0]} | {platform.system()}")
    
    step_clean_environment() # 0
    step_core_lock()         # 1
    step_pytorch(tag)        # 2
    step_mmpose(tag)         # 3
    step_vitpose()           # 4
    step_tensorflow()        # 5
    step_mediapipe()         # 6
    step_remaining()         # 7
    step_project()           # 8
    step_relock_numpy()      # 9
    validate()

if __name__ == "__main__":
    main()