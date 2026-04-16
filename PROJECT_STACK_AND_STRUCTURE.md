# NatyaVeda v2 - Project Stack, Versions, and Structure

## 1. Project Identity

- Name: natyaveda-analyzer
- Version: 2.0.0
- License: Apache-2.0
- Python requirement: >= 3.10
- Main domain: Indian classical dance classification and analysis

Source of truth:
- setup.py
- config/config.yaml
- requirements.txt
- Dockerfile
- docker-compose.yml

## 2. Runtime and Environment

### 2.1 Local runtime

- Python environment: venv
- Active Python version used in this workspace: 3.10.11
- Typical interpreter path in this workspace:
	- d:/New folder (2)/files (3)/.venv/Scripts/python.exe

### 2.2 Container runtime

- Base image: nvidia/cuda:12.1.1-runtime-ubuntu22.04
- OS in container: Ubuntu 22.04
- Python in container: 3.10
- Compose orchestrator: docker compose
- GPU profile image tag: natyaveda:cuda121
- CPU profile image tag: natyaveda:cpu

## 3. Dependency Stack (Pinned and Core)

### 3.1 Pinned or constrained from requirements.txt

- numpy < 2.0.0
- setuptools < 81
- protobuf == 3.20.3
- opencv-python-headless < 4.10
- torch == 2.2.0
- torchvision == 0.17.0
- mmengine >= 0.10.0
- mmdet == 3.3.0
- mmpose == 1.3.1
- mediapipe == 0.10.11
- tensorflow-cpu == 2.15.0
- transformers >= 4.40.0, < 4.50.0
- wandb == 0.16.6
- scenedetect == 0.6.3
- yt-dlp >= 2024.1.0
- scipy
- pandas
- scikit-learn
- matplotlib
- seaborn
- rich
- tqdm

### 3.2 Core libraries from setup.py

- huggingface-hub >= 0.22.0
- timm >= 1.0.0
- accelerate >= 0.27.0
- einops >= 0.7.0
- Pillow >= 10.0.0
- ffmpeg-python >= 0.2.0
- pyyaml >= 6.0
- omegaconf >= 2.3.0
- tensorboard >= 2.15.0

### 3.3 Dev extras

- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- black >= 23.0.0
- isort >= 5.12.0
- flake8 >= 6.0.0

## 4. Model and Training Configuration

From config/config.yaml:

- Model: danceformer-large
- Pose keypoints (primary): 133 (RTMW-x)
- Backup pose model: MoveNet Thunder (17 keypoints)
- Video backbone: MCG-NJU/videomae-large
- Transformer layers: 8
- Attention heads: 8
- FF dim: 1024
- Batch size: 32
- Epochs: 100
- Optimizer: AdamW
- Learning rate: 1e-4
- Scheduler: cosine warmup (10 warmup epochs)
- Mixed precision mode: auto

Loss configuration:

- Focal dance classification
- Supervised contrastive loss
- Center loss
- Inter-class centroid push loss
- Angular loss
- Mudra-weighted supervision
- Triplet loss

## 5. Data and Pipeline Stages

### 5.1 Data directories

- Raw videos: data/raw
- Refined videos: data/refined
- Processed features: data/processed
- Splits: data/splits
- Model weights: weights
- Reports: reports
- Outputs: outputs

### 5.2 End-to-end workflow

1. Collect videos
2. Refine dance segments
3. Extract pose and video features
4. Build train/val/test splits
5. Train DanceFormer
6. Train triplet embedding model
7. Compute class prototypes
8. Evaluate and generate confusion matrix
9. Run inference and create annotated output videos

## 6. Docker Services Available

- natyaveda-cpu
- natyaveda-gpu
- setup-cpu
- setup-gpu
- pipeline-cpu
- pipeline-gpu
- infer-cpu
- infer-gpu
- train-gpu
- triplet-train-gpu
- extract-gpu
- evaluate-gpu
- verify-cpu

## 7. Project Structure (Current)

```text
natyaveda-v2/
|-- .devcontainer/
|-- .github/
|-- config/
|   |-- config.yaml
|   `-- dance_classes.yaml
|-- data/
|   |-- raw/
|   |-- refined/
|   |-- processed/
|   `-- splits/
|-- docs/
|-- notebooks/
|-- outputs/
|-- reports/
|-- scripts/
|   |-- build_splits.py
|   |-- check_gpu.py
|   |-- compute_prototypes.py
|   |-- download_data.py
|   |-- download_weights.py
|   |-- evaluate.py
|   |-- extract_features.py
|   |-- infer.py
|   |-- refine_videos.py
|   |-- train.py
|   |-- train_triplet_embedding.py
|   `-- verify_setup.py
|-- src/
|   |-- data_collection/
|   |-- feature_extraction/
|   |-- inference/
|   |-- models/
|   |-- preprocessing/
|   |-- training/
|   |-- utils/
|   `-- __init__.py
|-- tests/
|-- weights/
|-- docker-compose.yml
|-- Dockerfile
|-- README.md
|-- requirements.txt
|-- setup.py
`-- PROJECT_STACK_AND_STRUCTURE.md
```

## 8. Notes for Reproducibility

- Keep setuptools < 81 to avoid dependency breakages.
- Keep protobuf == 3.20.3 for MediaPipe compatibility.
- Keep numpy < 2.0.0 to avoid downstream binary mismatches.
- Recompute prototypes after each new training cycle.
- If refined data changes, regenerate processed features and splits before training.

## 9. Key Findings (Latest Runs)

### 9.1 Model quality snapshot

- Best recent evaluation (test set):
	- Accuracy: 92.66%
	- Weighted F1: 0.9257
	- Macro F1: 0.9210
- Strong classes:
	- Bharatanatyam, Odissi, Manipuri, Sattriya (high precision/recall)
- Remaining confusion hotspots:
	- Mohiniyattam <-> Kuchipudi
	- Kathak <-> Kathakali

### 9.2 Data findings

- Class imbalance was a major contributor to unstable cluster separation.
- Adding data across all classes improved balance significantly, but backlog in feature extraction still existed (mostly Sattriya and Kathakali at last check).
- Important: changes in refined data do not affect training until processed features and splits are rebuilt.

### 9.3 Pose and hand signal findings

- Hand detection density is still a bottleneck for fine-grained separation.
- Previously observed issue:
	- very low both-hand detection frames in some Mohiniyattam clips.
- Applied mitigation already in config:
	- MediaPipe hand detection/tracking confidence lowered to 0.30.

### 9.4 Embedding and prototype findings

- Prototype calibration is now integrated in inference and evaluation workflow.
- Inter-class cosine distance matrix is useful for early diagnosis of confusable pairs.
- Pipeline improvement:
	- compute prototypes after training
	- use prototype blending during inference/evaluation

### 9.5 Engineering fixes that improved stability

- Predictor crash fixed (self usage placement issue in predictor visualizer initialization path).
- Mixed-precision overflow issue in custom loss masking fixed by dtype-safe min/max values.
- Evaluation prototype blending dtype mismatch fixed (feature/prototype dtype alignment).
- t-SNE compatibility in prototype script fixed across sklearn variants (n_iter/max_iter handling).

## 10. High-Value Content for Final Project Report

Include these sections in the report for maximum technical value:

1. Problem framing and dance-pair confusion risk
2. Dataset evolution timeline (raw -> refined -> processed -> splits)
3. Class balance before vs after augmentation
4. Model architecture and why multi-level pose extraction is needed
5. Loss design rationale:
	 - focal + contrastive + center + inter-class push + angular + triplet
6. Quantitative results table:
	 - accuracy, weighted F1, macro F1, per-class precision/recall/F1
7. Confusion matrix with commentary on top misclassification pairs
8. Embedding-space analysis:
	 - prototype distances
	 - cluster visualization
9. Ablation notes (if available):
	 - without prototype blending
	 - without hand-focused adjustments
10. Reproducibility checklist:
	 - exact package constraints
	 - config values
	 - command order
11. Known limitations and future work:
	 - hand detector robustness in fast mudra sequences
	 - targeted augmentation for confused class pairs

## 11. Suggested Report Artifacts (File Outputs)

- reports/evaluation_report.json
- reports/confusion_matrix.png
- reports/class_prototypes.npz
- reports/cluster_visualization.png
- outputs/analyzed_after_fix.mp4 (or latest annotated inference outputs)


