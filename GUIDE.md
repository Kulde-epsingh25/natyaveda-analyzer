# 📖 NatyaVeda Analyzer — Complete Setup & Usage Guide

> This guide walks you through every step: from a blank machine to a fully trained and tested Indian classical dance classifier.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installing Dependencies](#2-installing-dependencies)
   - 2.1 [Python & Virtual Environment](#21-python--virtual-environment)
   - 2.2 [PyTorch](#22-pytorch)
   - 2.3 [MMPose Ecosystem (RTMW-x)](#23-mmpose-ecosystem-rtmw-x)
   - 2.4 [TensorFlow & TF Hub (MoveNet)](#24-tensorflow--tf-hub-movenet)
   - 2.5 [MediaPipe (Hand Landmarks)](#25-mediapipe-hand-landmarks)
   - 2.6 [HuggingFace (VideoMAE, RT-DETR)](#26-huggingface-videomae-rt-detr)
   - 2.7 [All Other Dependencies](#27-all-other-dependencies)
   - 2.8 [System Tools (FFmpeg, Git)](#28-system-tools-ffmpeg-git)
3. [Clone & Install the Project](#3-clone--install-the-project)
4. [Download Pretrained Weights](#4-download-pretrained-weights)
5. [Verify Setup](#5-verify-setup)
6. [Where to Store Raw Video Data](#6-where-to-store-raw-video-data)
7. [Step-by-Step Pipeline](#7-step-by-step-pipeline)
   - Step 1: Download Videos from YouTube
   - Step 2: Refine & Clean Videos
   - Step 3: Extract Features
   - Step 4: Build Dataset Splits
   - Step 5: Train the Model
   - Step 6: Evaluate
   - Step 7: Run Inference on New Video
8. [Command Reference (Quick List)](#8-command-reference-quick-list)
9. [Running Tests](#9-running-tests)
10. [Docker Alternative](#10-docker-alternative)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04 / Windows 10 WSL2 / macOS 12 | Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB (RTX 3060) | 16–24 GB (RTX 3090 / A100) |
| Storage | 50 GB free | 200 GB free |
| CUDA | 11.8 | 12.1 |
| FFmpeg | 4.4+ | 6.x |

> **CPU-only mode:** The full pipeline works on CPU — but feature extraction will be very slow (hours vs. minutes). Training is not practical without a GPU.

---

## 2. Installing Dependencies

### 2.1 Python & Virtual Environment

```bash
# Check your Python version first
python3 --version   # must be 3.10 or 3.11

# Create a dedicated virtual environment
python3 -m venv natyaveda_env

# Activate it
# Linux / macOS:
source natyaveda_env/bin/activate

# Windows (PowerShell):
natyaveda_env\Scripts\Activate.ps1

# Upgrade pip inside the environment
pip install --upgrade pip setuptools wheel
```

---

### 2.2 PyTorch

Install PyTorch **matching your CUDA version**. Check your CUDA version first:

```bash
nvidia-smi   # look for "CUDA Version: XX.X" in top-right corner
```

**If you have CUDA 11.8:**
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

**If you have CUDA 12.1:**
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (no GPU):**
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

**Verify PyTorch:**
```bash
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
# Expected: 2.2.0   CUDA: True
```

---

### 2.3 MMPose Ecosystem (RTMW-x)

MMPose provides the primary whole-body pose estimator (133 keypoints).

```bash
# Step 1 — Install mim (MMPose package manager)
pip install openmim

# Step 2 — Install MMPose dependencies in correct order
mim install mmengine
mim install "mmcv>=2.1.0"
mim install "mmdet>=3.3.0"
mim install "mmpose>=1.3.0"

# Verify MMPose installation
python3 -c "import mmpose; print('MMPose version:', mmpose.__version__)"
```

> ⚠️ **Important:** Always install `mmengine → mmcv → mmdet → mmpose` in this order. Installing out of order causes version conflicts.

---

### 2.4 TensorFlow & TF Hub (MoveNet)

MoveNet Thunder is used as a fallback pose estimator for low-confidence frames.

```bash
pip install tensorflow==2.15.0 tensorflow-hub==0.16.0

# Verify TensorFlow
python3 -c "import tensorflow as tf; print('TF:', tf.__version__)"
# Expected: TF: 2.15.0
```

> 💡 TF and PyTorch can coexist in the same environment. MoveNet downloads automatically on first use from TF Hub — no manual download needed.

---

### 2.5 MediaPipe (Hand Landmarks)

MediaPipe provides 21-point per-hand finger joint landmarks for mudra classification.

```bash
pip install mediapipe==0.10.14

# Verify MediaPipe
python3 -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"
```

---

### 2.6 HuggingFace (VideoMAE, RT-DETR)

VideoMAE-v2 provides holistic video tokens. RT-DETR provides the person detector used by DanceIsolator.

```bash
pip install transformers==4.40.0 huggingface-hub==0.22.0 timm==1.0.0 accelerate==0.27.0

# These models download automatically on first use:
#   VideoMAE-v2:  MCG-NJU/videomae-large       (~1.8 GB)
#   RT-DETR-L:    PekingU/rtdetr_l              (~240 MB)
#
# To pre-download manually (optional, saves time during first run):
python3 -c "
from transformers import VideoMAEModel, VideoMAEImageProcessor
VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-large')
VideoMAEModel.from_pretrained('MCG-NJU/videomae-large')
print('VideoMAE downloaded.')
"
```

---

### 2.7 All Other Dependencies

```bash
# Core data and ML utilities
pip install numpy>=1.24.0 scipy>=1.11.0 pandas>=2.0.0 scikit-learn>=1.3.0 einops>=0.7.0

# Computer vision
pip install opencv-python>=4.9.0 Pillow>=10.0.0

# Video download and scene detection
pip install yt-dlp>=2024.1.0 "scenedetect[opencv]>=0.6.3"

# Training logging and config
pip install wandb>=0.16.0 pyyaml>=6.0 rich>=13.0.0 tqdm>=4.65.0

# Visualization
pip install matplotlib>=3.7.0 seaborn>=0.12.0

# Testing
pip install pytest>=7.4.0 pytest-cov>=4.1.0
```

---

### 2.8 System Tools (FFmpeg, Git)

FFmpeg is required for video trimming, conversion, and frame extraction.

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install -y ffmpeg git

# Verify
ffmpeg -version | head -1
# Expected: ffmpeg version 6.x.x ...
```

**macOS (Homebrew):**
```bash
brew install ffmpeg git
```

**Windows (WSL2):**
```bash
sudo apt update && sudo apt install -y ffmpeg git
```

---

## 3. Clone & Install the Project

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/natyaveda-analyzer.git
cd natyaveda-analyzer

# Install the project in editable mode with dev tools
pip install -e ".[dev]"

# Confirm the package is installed
python3 -c "import src; print('NatyaVeda package OK')"
```

---

## 4. Download Pretrained Weights

```bash
# This downloads RTMW-x via mim and creates a placeholder DanceFormer checkpoint
python scripts/download_weights.py
```

What this does:
- Downloads **RTMW-x** wholebody checkpoint (~200 MB) via `mim` into `weights/`
- Confirms **MoveNet** will download from TF Hub on first use
- Confirms **VideoMAE-v2** and **RT-DETR** will download from HuggingFace on first use
- Creates `weights/danceformer_untrained_small.pt` — a placeholder for testing the pipeline before training

After the command you should see:
```
✓ RTMW-x downloaded
✓ TensorFlow Hub available
✓ MediaPipe Hands available
✓ transformers available
✓ Placeholder checkpoint saved: weights/danceformer_untrained_small.pt
```

---

## 5. Verify Setup

```bash
python scripts/verify_setup.py
```

Expected output:
```
============================================================
  NatyaVeda — Environment Check
============================================================
  ✅ PyTorch
  ✅ NumPy
  ✅ OpenCV
  ✅ scikit-learn
  ✅ Transformers (HF)
  ✅ MediaPipe
  ✅ PyYAML
  ✅ yt-dlp
  ✅ einops
  ✅ MMEngine
  ✅ MMCV
  ✅ MMDet
  ✅ MMPose
  ✅ TensorFlow
  ✅ TF Hub
  ✅ PySceneDetect
  ✅ DanceFormer model
  ✅ DanceFormer forward pass
  ✅ CUDA GPU
============================================================
  19 passed  |  0 failed
  ✅ All checks passed — ready to train!
```

> If some checks show ❌, re-run the relevant install step from Section 2.

---

## 6. Where to Store Raw Video Data

```
natyaveda-analyzer/
└── data/
    ├── raw/                    ← PUT YOUR DOWNLOADED VIDEOS HERE
    │   ├── bharatanatyam/
    │   │   ├── bharatanatyam_abc123.mp4
    │   │   └── bharatanatyam_def456.mp4
    │   ├── kathak/
    │   │   └── kathak_xyz789.mp4
    │   ├── odissi/
    │   ├── kuchipudi/
    │   ├── manipuri/
    │   ├── mohiniyattam/
    │   ├── sattriya/
    │   └── kathakali/
    │
    ├── refined/                ← Cleaned segments go here (auto-created)
    ├── processed/              ← Extracted features go here (auto-created)
    └── splits/                 ← Train/Val/Test splits (auto-created)
```

### Rules for Raw Video Files

| Rule | Details |
|------|---------|
| **Folder name = dance class** | Files inside `data/raw/kathak/` are labeled as Kathak |
| **Minimum duration** | 60 seconds (configured in `config/config.yaml`) |
| **Recommended duration** | 3–10 minutes per video |
| **Format** | `.mp4` preferred; `.avi`, `.mkv` also accepted |
| **Resolution** | 480p minimum; 720p or 1080p preferred |
| **Source** | YouTube performance recordings, recitals, stage concerts |
| **Avoid** | Tutorials, lessons, fusion/Bollywood, multi-angle cuts |

### Option A — Let the pipeline download automatically (recommended)

See Step 1 in Section 7 — the downloader handles everything.

### Option B — Bring your own videos

If you already have videos, just place them in the correct subfolder:
```bash
# Example: copy your existing Bharatanatyam videos
cp /your/videos/bharatanatyam/*.mp4 data/raw/bharatanatyam/

# Example: copy Kathak videos
cp /your/videos/kathak/*.mp4 data/raw/kathak/
```

---

## 7. Step-by-Step Pipeline

> Run these commands **in order**, from the project root directory with your virtual environment active.

---

### ✅ Step 1: Download Videos from YouTube

This searches YouTube for performance videos per dance form, filters by duration and relevance keywords, and downloads them.

```bash
# Download for ALL 8 dance forms (recommended for full training)
python scripts/download_data.py \
    --dances bharatanatyam kathak odissi kuchipudi manipuri mohiniyattam sattriya kathakali \
    --output data/raw \
    --max-per-dance 100 \
    --min-duration 60 \
    --max-duration 600 \
    --config config/config.yaml

# ── OR — Download for specific dance forms only
python scripts/download_data.py \
    --dances bharatanatyam kathak \
    --output data/raw \
    --max-per-dance 50

# ── OR — Dry run to see what would be downloaded (no actual download)
python scripts/download_data.py \
    --dances bharatanatyam \
    --dry-run
```

**What happens:**
- Searches YouTube for each configured query (e.g., "Bharatanatyam solo stage performance")
- Filters out tutorials, lessons, Bollywood fusion, and short clips
- Downloads highest quality MP4 (720p+) into `data/raw/<dance_form>/`
- Also writes a `.info.json` metadata file per video

**Expected output:**
```
[BHARATANATYAM] Starting download …
  Searching: 'Bharatanatyam solo stage performance'
  ✓ Downloaded: bharatanatyam_abc123.mp4
  ✓ Downloaded: bharatanatyam_def456.mp4
  ...
[BHARATANATYAM] Downloaded 48 videos.
```

**After this step, check:**
```bash
ls data/raw/bharatanatyam/   # should show .mp4 files
ls data/raw/kathak/
```

---

### ✅ Step 2: Refine & Clean Videos

This is the most important preprocessing step. It removes non-dance segments (audience shots, title cards, presenters) from each downloaded video.

```bash
# Refine all dance forms
python scripts/refine_videos.py \
    --input data/raw \
    --output data/refined \
    --min-dance-confidence 0.65 \
    --remove-audience \
    --min-scene-duration 3.0 \
    --device cuda

# ── OR — Refine specific dance forms
python scripts/refine_videos.py \
    --input data/raw \
    --output data/refined \
    --dances bharatanatyam kathak \
    --device cuda

# ── CPU fallback (slower)
python scripts/refine_videos.py \
    --input data/raw \
    --output data/refined \
    --device cpu
```

**What happens internally:**

```
Each video goes through 5 stages:
  Stage 1: PySceneDetect splits video into scenes at cut points
  Stage 2: Activity classifier scores each scene (dance vs. non-dance)
  Stage 3: RT-DETR detects persons; ByteTrack finds the lead dancer
  Stage 4: Frames with < 40% avg keypoint confidence are dropped
  Stage 5: Very-low-motion scenes (title cards, audience) are dropped

Output: Cleaned .mp4 segments in data/refined/<dance_form>/
```

**Expected output:**
```
Processing: bharatanatyam_abc123.mp4
  Detecting scenes … 24 scenes found
  Scoring 24 scenes …
  Kept 17/24 scenes (312.4s retained)
  → Retention: 78% (312s / 401s)
```

**After this step, check:**
```bash
ls data/refined/bharatanatyam/    # should show *_seg000.mp4, *_seg001.mp4 etc.
```

---

### ✅ Step 3: Extract Features

Runs RTMW-x + MediaPipe Hands + VideoMAE-v2 on every refined video and saves pose keypoint arrays.

```bash
# Full extraction — all models, all dance forms
python scripts/extract_features.py \
    --input data/refined \
    --output data/processed \
    --pose-model rtmw-x \
    --hands mediapipe \
    --videomae \
    --device cuda \
    --fps 25

# ── Without VideoMAE (faster, slightly lower accuracy)
python scripts/extract_features.py \
    --input data/refined \
    --output data/processed \
    --pose-model rtmw-x \
    --hands mediapipe \
    --device cuda

# ── Use MoveNet only (if MMPose is not installed)
python scripts/extract_features.py \
    --input data/refined \
    --output data/processed \
    --pose-model movenet-thunder \
    --hands mediapipe \
    --device cpu

# ── Extract only specific dances (useful for resuming)
python scripts/extract_features.py \
    --input data/refined \
    --output data/processed \
    --dances bharatanatyam kathak odissi \
    --device cuda
```

**What is saved per video:**

Each `.mp4` in `data/refined/` produces one `.npz` in `data/processed/`:

```
data/processed/bharatanatyam/bharatanatyam_abc123_seg000.npz
  ├── keypoints:     [T, 133, 3]   — 133 COCO-WholeBody keypoints (x, y, conf)
  ├── velocities:    [T, 133, 2]   — temporal first differences
  ├── accelerations: [T, 133, 2]   — temporal second differences
  ├── label:         0             — class index (0 = bharatanatyam)
  ├── dance_form:    "bharatanatyam"
  ├── timestamps:    [T]           — seconds per frame
  └── confidences:   [T]           — avg body keypoint confidence per frame
```

**Expected output:**
```
[BHARATANATYAM] 34 videos
  Extracting: bharatanatyam_abc123_seg000.mp4
    ✓ 412 frames → bharatanatyam_abc123_seg000.npz
  Extracting: bharatanatyam_abc123_seg001.mp4
    ✓ 389 frames → bharatanatyam_abc123_seg001.npz
```

**After this step, check:**
```bash
ls data/processed/bharatanatyam/    # should show .npz files
python3 -c "
import numpy as np
d = np.load('data/processed/bharatanatyam/bharatanatyam_abc123_seg000.npz')
print('keypoints shape:', d['keypoints'].shape)   # e.g. (412, 133, 3)
print('label:', d['label'])                        # 0
"
```

---

### ✅ Step 4: Build Dataset Splits

Splits the processed .npz files into train / val / test sets with stratification by dance form.

```bash
# Build 80/10/10 stratified splits
python scripts/build_splits.py \
    --input data/processed \
    --output data/splits \
    --train 0.80 \
    --val 0.10 \
    --test 0.10 \
    --seed 42

# ── Optional: 5-fold cross-validation splits
python scripts/build_splits.py \
    --input data/processed \
    --output data/splits \
    --folds 5 \
    --seed 42
```

**Expected output:**
```
Building 80/10/10 splits …
  Total .npz files found: 1847
  Train: 1477 files
  Val:    185 files
  Test:   185 files

Per-class distribution:
  bharatanatyam  — train: 210, val: 26, test: 26
  kathak         — train: 198, val: 25, test: 25
  odissi         — train: 187, val: 23, test: 23
  ...
Splits saved to data/splits/
```

**After this step, check:**
```bash
ls data/splits/
# train/  val/  test/  split_info.json
cat data/splits/split_info.json
```

---

### ✅ Step 5: Train the Model

```bash
# Train the large model (recommended — best accuracy)
python scripts/train.py \
    --config config/config.yaml \
    --data data/splits \
    --model danceformer-large \
    --epochs 100 \
    --device cuda \
    --output weights

# ── Train the base model (balanced speed/accuracy)
python scripts/train.py \
    --config config/config.yaml \
    --data data/splits \
    --model danceformer-base \
    --epochs 80 \
    --device cuda

# ── Train the small model (fast, for testing the pipeline)
python scripts/train.py \
    --config config/config.yaml \
    --data data/splits \
    --model danceformer-small \
    --epochs 30 \
    --device cuda

# ── Resume from a checkpoint
python scripts/train.py \
    --config config/config.yaml \
    --data data/splits \
    --model danceformer-large \
    --resume weights/danceformer_epoch050.pt \
    --device cuda

# ── Train without W&B logging (offline)
python scripts/train.py \
    --config config/config.yaml \
    --data data/splits \
    --model danceformer-large \
    --no-wandb \
    --device cuda
```

**Training progress output:**
```
============================================================
  NatyaVeda Analyzer — Training
============================================================
  Model    : danceformer-large
  Data     : data/splits
  Device   : cuda
  Epochs   : 100
  Batch    : 32
  LR       : 0.0001
  Params   : 8,234,504

Epoch   1/100 | Train loss=2.1842 acc=0.148 | Val loss=2.0981 acc=0.162 f1=0.141
Epoch   5/100 | Train loss=1.7431 acc=0.341 | Val loss=1.6823 acc=0.372 f1=0.358
Epoch  10/100 | Train loss=1.2104 acc=0.551 | Val loss=1.1843 acc=0.573 f1=0.561
  ✅ Best model saved (val F1=0.5614)
...
Epoch  50/100 | Train loss=0.4123 acc=0.871 | Val loss=0.4891 acc=0.863 f1=0.859
  ✅ Best model saved (val F1=0.8591)
...
Epoch 100/100 | Train loss=0.2841 acc=0.921 | Val loss=0.3742 acc=0.907 f1=0.904
```

**Best checkpoint is saved automatically to:**
```
weights/danceformer_best.pt
```

**Recommended training time:**

| Model | GPU | ~Time for 100 epochs |
|-------|-----|---------------------|
| danceformer-small | RTX 3060 8GB | ~2 hours |
| danceformer-base  | RTX 3060 8GB | ~5 hours |
| danceformer-large | RTX 3090 24GB | ~8 hours |
| danceformer-large | A100 40GB | ~3 hours |

---

### ✅ Step 6: Evaluate on Test Set

```bash
# Full evaluation with TTA (Test-Time Augmentation)
python scripts/evaluate.py \
    --checkpoint weights/danceformer_best.pt \
    --test-data data/splits \
    --report-dir reports \
    --device cuda

# ── Without TTA (faster, slightly lower accuracy)
python scripts/evaluate.py \
    --checkpoint weights/danceformer_best.pt \
    --test-data data/splits \
    --report-dir reports \
    --no-tta

# ── CPU evaluation
python scripts/evaluate.py \
    --checkpoint weights/danceformer_best.pt \
    --test-data data/splits \
    --report-dir reports \
    --device cpu
```

**Expected output:**
```
======================================================================
  NatyaVeda Dance Classifier — Test Results
======================================================================
  Accuracy       : 0.9032 (90.32%)
  F1 (weighted)  : 0.9018
  F1 (macro)     : 0.8974
  Test clips     : 185

                    precision  recall  f1-score  support
  bharatanatyam       0.94     0.96      0.95       26
  kathak              0.91     0.89      0.90       25
  odissi              0.93     0.92      0.92       23
  kuchipudi           0.88     0.87      0.88       22
  manipuri            0.85     0.86      0.85       21
  mohiniyattam        0.87     0.85      0.86       20
  sattriya            0.82     0.84      0.83       19
  kathakali           0.95     0.93      0.94       24

  Confusion Matrix:
  bhar  kath  odis  kuch  mani  mohi  satt  kath
  25     0     0     1     0     0     0     0   bhar
   1    22     0     1     0     0     1     0   kath
   ...

Reports saved to reports/
  - reports/evaluation_report.json
  - reports/confusion_matrix.png
```

---

### ✅ Step 7: Run Inference on a New Video

```bash
# Basic inference — print prediction to console
python scripts/infer.py \
    --video /path/to/your/dance_video.mp4 \
    --checkpoint weights/danceformer_best.pt \
    --device cuda

# Full inference — also generate annotated output video
python scripts/infer.py \
    --video /path/to/your/dance_video.mp4 \
    --checkpoint weights/danceformer_best.pt \
    --output-video outputs/analyzed_dance.mp4 \
    --device cuda

# Save JSON results
python scripts/infer.py \
    --video /path/to/your/dance_video.mp4 \
    --checkpoint weights/danceformer_best.pt \
    --output-video outputs/analyzed_dance.mp4 \
    --json-output outputs/result.json \
    --device cuda

# Disable mudra labels (faster)
python scripts/infer.py \
    --video /path/to/your/dance_video.mp4 \
    --checkpoint weights/danceformer_best.pt \
    --no-mudras \
    --device cuda
```

**Expected output:**
```
==================================================
  NatyaVeda — Prediction Result
==================================================
  Dance Form   : Bharatanatyam
  Confidence   : 94.2%
  Frames       : 1247

  All probabilities:
    bharatanatyam         94.2% ████████████████████████████
    kathakali              2.1% ▌
    kuchipudi              1.3% ▍
    kathak                 0.9% ▎
    odissi                 0.8% ▎
    manipuri               0.4% ▏
    mohiniyattam           0.2% ▏
    sattriya               0.1% ▏

  Annotated video: outputs/analyzed_dance.mp4
```

---

## 8. Command Reference (Quick List)

Copy-paste this as a reference. Run these **in order** for a fresh setup.

```bash
# ─────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ─────────────────────────────────────────────────
python3 -m venv natyaveda_env
source natyaveda_env/bin/activate
pip install --upgrade pip setuptools wheel

# PyTorch (adjust URL for your CUDA version)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# MMPose (must be in this order)
pip install openmim
mim install mmengine
mim install "mmcv>=2.1.0"
mim install "mmdet>=3.3.0"
mim install "mmpose>=1.3.0"

# All other dependencies
pip install tensorflow==2.15.0 tensorflow-hub==0.16.0
pip install mediapipe==0.10.14
pip install transformers==4.40.0 huggingface-hub==0.22.0 timm==1.0.0
pip install numpy scipy pandas scikit-learn einops opencv-python Pillow
pip install yt-dlp "scenedetect[opencv]" ffmpeg-python
pip install wandb pyyaml rich tqdm matplotlib seaborn
pip install pytest pytest-cov

# Clone and install project
git clone https://github.com/YOUR_USERNAME/natyaveda-analyzer.git
cd natyaveda-analyzer
pip install -e ".[dev]"

# ─────────────────────────────────────────────────
# WEIGHTS & VERIFICATION
# ─────────────────────────────────────────────────
python scripts/download_weights.py
python scripts/verify_setup.py

# ─────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────
# Step 1 — Download YouTube videos
python scripts/download_data.py \
    --dances bharatanatyam kathak odissi kuchipudi manipuri mohiniyattam sattriya kathakali \
    --output data/raw \
    --max-per-dance 100

# Step 2 — Refine (remove non-dance segments)
python scripts/refine_videos.py \
    --input data/raw \
    --output data/refined \
    --device cuda

# Step 3 — Extract pose + hand + VideoMAE features
python scripts/extract_features.py \
    --input data/refined \
    --output data/processed \
    --pose-model rtmw-x \
    --hands mediapipe \
    --videomae \
    --device cuda

# Step 4 — Build train/val/test splits
python scripts/build_splits.py \
    --input data/processed \
    --output data/splits \
    --train 0.80 --val 0.10 --test 0.10 --seed 42

# Step 5 — Train
python scripts/train.py \
    --config config/config.yaml \
    --data data/splits \
    --model danceformer-large \
    --epochs 100 \
    --device cuda

# Step 6 — Evaluate
python scripts/evaluate.py \
    --checkpoint weights/danceformer_best.pt \
    --test-data data/splits \
    --report-dir reports \
    --device cuda

# Step 7 — Infer on new video
python scripts/infer.py \
    --video /path/to/dance.mp4 \
    --checkpoint weights/danceformer_best.pt \
    --output-video outputs/analyzed.mp4 \
    --device cuda
```

---

## 9. Running Tests

### Run All Tests

```bash
# From project root, with virtual environment active
pytest tests/ -v
```

### Run with Coverage Report

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run Specific Test Classes

```bash
# Test only the DanceFormer model
pytest tests/test_all.py::TestDanceFormer -v

# Test only pose extraction
pytest tests/test_all.py::TestPoseFrame -v

# Test only hand/mudra extraction
pytest tests/test_all.py::TestHandFrame -v

# Test only dance isolation logic
pytest tests/test_all.py::TestDanceIsolator -v

# Test only training pipeline components
pytest tests/test_all.py::TestTrainingPipeline -v

# Test only the visualizer
pytest tests/test_all.py::TestVisualizer -v

# Test only pose embedding
pytest tests/test_all.py::TestPosePatchEmbedding -v
```

### Run a Single Test by Name

```bash
# Run only the DanceFormer forward pass test
pytest tests/test_all.py::TestDanceFormer::test_forward_pass_small -v

# Run only the parameter count test
pytest tests/test_all.py::TestDanceFormer::test_parameter_count_large -v
```

### Expected Test Output

```
============================================ test session starts ============================================
platform linux -- Python 3.11.x, pytest-7.4.x
collected 40 items

tests/test_all.py::TestPoseFrame::test_rtmw_pose_frame_creation       PASSED    [ 2%]
tests/test_all.py::TestPoseFrame::test_feature_vector_shape            PASSED    [ 5%]
tests/test_all.py::TestPoseFrame::test_feature_vector_no_conf          PASSED    [ 7%]
tests/test_all.py::TestPoseFrame::test_avg_confidence                  PASSED    [10%]
tests/test_all.py::TestPoseFrame::test_movenet_pose_frame              PASSED    [12%]
tests/test_all.py::TestHandFrame::test_feature_vector_shape            PASSED    [15%]
tests/test_all.py::TestHandFrame::test_mudra_feature_vector_shape      PASSED    [17%]
tests/test_all.py::TestHandFrame::test_finger_angles_shape             PASSED    [20%]
tests/test_all.py::TestHandFrame::test_both_detected                   PASSED    [22%]
tests/test_all.py::TestDanceFormer::test_forward_pass_small            PASSED    [25%]
tests/test_all.py::TestDanceFormer::test_forward_pass_base             PASSED    [27%]
tests/test_all.py::TestDanceFormer::test_padding_mask                  PASSED    [30%]
tests/test_all.py::TestDanceFormer::test_parameter_count_small         PASSED    [32%]
tests/test_all.py::TestDanceFormer::test_parameter_count_large         PASSED    [35%]
tests/test_all.py::TestDanceFormer::test_save_load                     PASSED    [37%]
tests/test_all.py::TestDanceFormer::test_from_config                   PASSED    [40%]
tests/test_all.py::TestDanceFormer::test_predict_method                PASSED    [42%]
tests/test_all.py::TestDanceFormer::test_no_videomae                   PASSED    [45%]
tests/test_all.py::TestDanceFormer::test_videomae_fusion               PASSED    [47%]
...
============================================ 40 passed in 18.32s ============================================
```

### Quick Sanity Check (no GPU, no data needed)

```bash
# This runs only the model architecture tests — no video files or GPU required
pytest tests/test_all.py -v -k "TestDanceFormer or TestPoseFrame or TestHandFrame"
```

---

## 10. Docker Alternative

If you want to skip the dependency installation entirely, use Docker:

```bash
# Build the image (takes ~15 minutes first time, downloads all dependencies)
docker build -t natyaveda .

# Verify environment inside Docker
docker run --gpus all natyaveda python scripts/verify_setup.py

# Run the full pipeline inside Docker
# Mount your local data/ and weights/ directories into the container
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/weights:/app/weights \
    -v $(pwd)/outputs:/app/outputs \
    natyaveda \
    python scripts/train.py \
        --config config/config.yaml \
        --data data/splits \
        --model danceformer-large \
        --epochs 100 \
        --device cuda

# Run inference inside Docker
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/weights:/app/weights \
    -v $(pwd)/outputs:/app/outputs \
    -v /path/to/your/videos:/app/videos \
    natyaveda \
    python scripts/infer.py \
        --video /app/videos/my_dance.mp4 \
        --checkpoint /app/weights/danceformer_best.pt \
        --output-video /app/outputs/analyzed.mp4
```

---

## 11. Troubleshooting

### "CUDA out of memory" during training

```bash
# Reduce batch size in config or via CLI
python scripts/train.py --batch 16 ...
# Or use gradient accumulation (effectively batch 32 with 16 in memory)
# In config/config.yaml: training.gradient.accumulation_steps: 2
```

### "No valid pose frames extracted"

This means RTMW or MoveNet found no persons in the video. Check:
```bash
# 1. Is the video playable?
ffprobe /path/to/video.mp4

# 2. Is the dancer visible?
# Try lowering the confidence threshold
python scripts/extract_features.py --input data/refined --output data/processed --device cuda
# Edit config.yaml: feature_extraction.keypoint_gating.min_avg_confidence: 0.25
```

### "mim install mmpose fails"

```bash
# Install specific compatible versions
pip install mmengine==0.10.4
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html
pip install mmdet==3.3.0
pip install mmpose==1.3.1
```

### "ModuleNotFoundError: No module named 'src'"

```bash
# Make sure you're in the project root and installed in editable mode
cd natyaveda-analyzer
pip install -e .
```

### "yt-dlp: ERROR: Sign in to confirm your age"

```bash
# Export YouTube cookies from your browser (install yt-dlp browser extension)
# Then pass them to the downloader:
python scripts/download_data.py \
    --dances bharatanatyam \
    --cookies-file cookies.txt \
    --output data/raw
```

### W&B login prompt during training

```bash
# Option 1: Log in once
wandb login

# Option 2: Disable W&B entirely
python scripts/train.py ... --no-wandb
```

---

*For questions or issues, open a GitHub Issue at:*
`https://github.com/YOUR_USERNAME/natyaveda-analyzer/issues`
