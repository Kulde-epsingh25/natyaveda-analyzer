# ⚡ NatyaVeda Analyzer v2 — Install & Run Commands
### Every command you need. Python 3.10/3.11/3.12 · GPU + CPU · MMPose + VitPose

---

## FIRST THING — Check if you have a GPU

```bash
nvidia-smi
```

```
# If you see this → you have a GPU (GPU will be auto-used)
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...   Driver Version: ...   CUDA Version: 12.1                  |
| GPU Name           | VRAM                                                   |

# If you see "command not found" → CPU mode (still works, just slower)
```

---

## PART 1 — One-Command Full Install (recommended)

This single command handles everything in the correct order:
- Pins `setuptools<81` (fixes MMPose + TF Hub)
- Detects your GPU and picks correct PyTorch CUDA version
- Installs MMCV via prebuilt wheels (Python 3.12 compatible)
- Installs VitPose-Plus from HuggingFace
- Installs `opencv-python-headless` (no libGL needed)

```bash
# Navigate into the project
cd natyaveda-v2

# Run the installer (auto-detects GPU)
python scripts/install.py
```

**Force CPU mode (if no GPU or container issues):**
```bash
python scripts/install.py --cpu
```

**Force specific CUDA version:**
```bash
python scripts/install.py --cuda 121   # CUDA 12.1
python scripts/install.py --cuda 118   # CUDA 11.8
```

---

## PART 2 — Manual Install (if one-command fails)

Run these in exact order. Do NOT skip or reorder.

### 2.1 — Virtual environment
```bash
python3 -m venv natyaveda_env
source natyaveda_env/bin/activate       # Linux/macOS
# natyaveda_env\Scripts\Activate.ps1   # Windows PowerShell
pip install --upgrade pip
```

### 2.2 — CRITICAL: Pin setuptools FIRST
```bash
# This MUST run before everything else
# setuptools 82+ removed pkg_resources — breaks MMPose, mim, and TF Hub
pip install "setuptools>=69.5.1,<81" wheel
```

### 2.3 — System libraries (Ubuntu/WSL2)
```bash
sudo apt-get update && sudo apt-get install -y \
    libgl1 libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    ninja-build gcc g++ ffmpeg git
```

### 2.4 — PyTorch (pick one)
```bash
# CUDA 12.1 (recommended if you have GPU)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2.5 — MMPose (primary 133-keypoint extractor)
```bash
# Step A: mmengine (pure Python, always works)
pip install mmengine>=0.10.0

# Step B: MMCV via prebuilt wheels (Python 3.12 compatible — no build needed)
# For CUDA 12.1:
pip install "mmcv==2.2.0+pt2.2.0cu121" \
    --extra-index-url https://miropsota.github.io/torch_packages_builder

# For CUDA 11.8:
pip install "mmcv==2.2.0+pt2.2.0cu118" \
    --extra-index-url https://miropsota.github.io/torch_packages_builder

# For CPU:
pip install "mmcv==2.2.0+pt2.2.0cpu" \
    --extra-index-url https://miropsota.github.io/torch_packages_builder

# Step C: mmdet + mmpose
pip install mmdet==3.3.0 mmpose==1.3.1
```

**If MMCV prebuilt wheel fails:**
```bash
# Fallback — mmcv-lite (no custom CUDA ops, pose still works)
pip install mmcv-lite>=2.1.0
pip install mmdet==3.3.0 mmpose==1.3.1
```

### 2.6 — VitPose-Plus + HuggingFace (secondary/fallback, Python 3.12 native)
```bash
pip install transformers>=4.40.0 huggingface-hub>=0.22.0 timm>=1.0.0 accelerate>=0.27.0 einops>=0.7.0
```

### 2.7 — TensorFlow CPU + TF Hub (MoveNet fallback)
```bash
# Use tensorflow-cpu — avoids CUDA library conflicts in containers
pip install tensorflow-cpu>=2.15.0 tensorflow-hub>=0.16.0
```

### 2.8 — MediaPipe + OpenCV Headless
```bash
# IMPORTANT: Use opencv-python-headless NOT opencv-python
# Headless has no libGL.so.1 dependency — works in all containers
pip install mediapipe>=0.10.0 opencv-python-headless>=4.9.0 Pillow>=10.0.0
```

### 2.9 — Remaining packages
```bash
pip install -r requirements.txt
```

### 2.10 — Install project
```bash
pip install -e ".[dev]"
```

---

## PART 3 — GPU Container Fix

If `nvidia-smi` shows your GPU but PyTorch says `CUDA: False`:

### VS Code Dev Container
```bash
# In .devcontainer/devcontainer.json — already included in v2:
"runArgs": ["--gpus", "all"]
```
Then: **Ctrl+Shift+P → "Rebuild Container"**

### Docker run command
```bash
# Add --gpus all
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/weights:/app/weights \
    natyaveda
```

### Docker Compose
```yaml
# In docker-compose.yml:
services:
  natyaveda:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## PART 4 — Download Weights & Verify

```bash
# Download RTMW-x weights + prefetch VitPose from HuggingFace
python scripts/download_weights.py

# Check GPU status and which pose tier will be used
python scripts/check_gpu.py

# Verify all packages
python scripts/verify_setup.py
```

**Expected output after full install:**
```
✅ setuptools < 81
✅ PyTorch
✅ NumPy <2.0
✅ OpenCV Headless
✅ Transformers (VitPose-Plus)
✅ MediaPipe
✅ TensorFlow
✅ TF Hub
✅ MMEngine
✅ MMCV
✅ MMDet
✅ MMPose (RTMW-x 133 kpts)
✅ VitPose-Plus (HF)
✅ CUDA GPU — RTX 3090 (24.0 GB)   ← or CPU mode if no GPU
✅ DeviceManager — Device: cuda | Pose: rtmw-x | fp16: True
✅ DanceFormer forward pass
```

---

## PART 5 — Pipeline (same as before)

```bash
# 1. Download YouTube videos
python scripts/download_data.py \
    --dances bharatanatyam kathak odissi kuchipudi manipuri mohiniyattam sattriya kathakali \
    --output data/raw --max-per-dance 100

# 2. Refine (remove non-dance segments)
python scripts/refine_videos.py \
    --input data/raw --output data/refined --device cuda

# 3. Extract features (auto-selects RTMW-x or VitPose based on GPU)
python scripts/extract_features.py \
    --input data/refined --output data/processed \
    --videomae --device cuda

# 4. Build splits
python scripts/build_splits.py \
    --input data/processed --output data/splits \
    --train 0.80 --val 0.10 --test 0.10 --seed 42

# 5. Train
python scripts/train.py \
    --config config/config.yaml \
    --data data/splits \
    --model danceformer-large \
    --epochs 100 --device cuda

# 6. Evaluate
python scripts/evaluate.py \
    --checkpoint weights/danceformer_best.pt \
    --test-data data/splits --report-dir reports --device cuda

# 7. Infer on a new video
python scripts/infer.py \
    --video /path/to/dance.mp4 \
    --checkpoint weights/danceformer_best.pt \
    --output-video outputs/analyzed.mp4 --device cuda
```

---

## PART 6 — Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific classes
pytest tests/test_all.py::TestDeviceManager -v
pytest tests/test_all.py::TestPoseFrame -v
pytest tests/test_all.py::TestVitPose -v
pytest tests/test_all.py::TestHandFrame -v
pytest tests/test_all.py::TestDanceFormer -v
pytest tests/test_all.py::TestTraining -v
pytest tests/test_all.py::TestIsolator -v
pytest tests/test_all.py::TestVisualizer -v
pytest tests/test_all.py::TestTaxonomy -v
pytest tests/test_all.py::TestKeypointUtils -v

# Zero-data quick sanity check (no GPU, no video files needed)
pytest tests/test_all.py -v -k "TestDanceFormer or TestPoseEmbed or TestTaxonomy"
```

---

## QUICK REFERENCE — FULL PIPELINE ONE BLOCK

```bash
# ── Setup ────────────────────────────────────────────────────
python3 -m venv natyaveda_env && source natyaveda_env/bin/activate
python scripts/install.py          # one command — handles everything

# ── Verify ───────────────────────────────────────────────────
python scripts/check_gpu.py
python scripts/download_weights.py
python scripts/verify_setup.py

# ── Data ─────────────────────────────────────────────────────
python scripts/download_data.py --dances bharatanatyam kathak odissi kuchipudi manipuri mohiniyattam sattriya kathakali --output data/raw --max-per-dance 100
python scripts/refine_videos.py --input data/raw --output data/refined --device cuda
python scripts/extract_features.py --input data/refined --output data/processed --videomae --device cuda
python scripts/build_splits.py --input data/processed --output data/splits --train 0.80 --val 0.10 --test 0.10 --seed 42

# ── Train ─────────────────────────────────────────────────────
python scripts/train.py --config config/config.yaml --data data/splits --model danceformer-large --epochs 100 --device cuda

# ── Evaluate ──────────────────────────────────────────────────
python scripts/evaluate.py --checkpoint weights/danceformer_best.pt --test-data data/splits --report-dir reports --device cuda

# ── Infer ─────────────────────────────────────────────────────
python scripts/infer.py --video /path/to/dance.mp4 --checkpoint weights/danceformer_best.pt --output-video outputs/analyzed.mp4 --device cuda

# ── Test ──────────────────────────────────────────────────────
pytest tests/ -v
```

---

## ERROR → FIX TABLE

| Error | Fix |
|-------|-----|
| `No module named 'pkg_resources'` | `pip install "setuptools<81"` |
| `mmcv build fails` | Use prebuilt: `pip install "mmcv==2.2.0+pt2.2.0cu121" --extra-index-url https://miropsota.github.io/torch_packages_builder` |
| `libGL.so.1 not found` | `sudo apt-get install -y libgl1 libgl1-mesa-glx` OR use `opencv-python-headless` |
| `CUDA available: False` (GPU present) | Add `--gpus all` to docker run, OR `"runArgs":["--gpus","all"]` in devcontainer.json |
| `CUDA out of memory` | Add `--batch 16` to train, or set `accumulation_steps: 2` in config |
| `numpy 2.x conflict` | `pip install "numpy==1.26.4" --force-reinstall` |
| `tf-hub pkg_resources error` | `pip install "setuptools<81"` then `pip install tensorflow-hub==0.16.0` |
| `mim command not found` | `pip install openmim` then check `setuptools<81` is installed |
| `No valid pose frames` | Lower `min_body_confidence: 0.20` in `config.yaml` |
| `wandb prompt` | Add `--no-wandb` flag to train command |
