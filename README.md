# NatyaVeda Analyzer (v2)

AI-powered Indian classical dance recognition and analysis.

This README is optimized for one goal: get the project running quickly and correctly.

## What This Repo Contains

- End-to-end pipeline: video collection, preprocessing, feature extraction, training, evaluation, and inference
- Support for 8 Indian classical dance forms
- Scripts for local runs and Docker-based runs

## Repository Layout

```text
natyaveda-v2/
|-- config/                    # YAML configs
|-- data/                      # raw/refined/processed/splits datasets
|-- docs/                      # project and design docs
|-- archive/
|   |-- versions/              # archived version marker files
|   `-- notes/                 # archived loose notes
|-- experiments/
|   `-- scratch/               # temporary notebooks/scripts
|-- figures/                   # report images
|-- notebooks/                 # reusable analysis notebooks
|-- outputs/                   # inference outputs and generated artifacts
|-- reports/                   # evaluation and metrics reports
|-- scripts/                   # CLI entry points
|-- src/                       # Python package source
|-- tests/                     # test suite
|-- weights/                   # model checkpoints
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- setup.py
`-- README.md
```

## Prerequisites

- Python 3.10+
- Git
- FFmpeg (recommended)
- CUDA GPU optional (CPU also supported)

## Quick Setup (Windows PowerShell)

Run from project root (folder containing this README):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
python scripts/verify_setup.py
```

## Quick Setup (Linux/macOS)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
python scripts/verify_setup.py
```

## First Run Checklist

1. Confirm config files exist in `config/`.
2. Put model checkpoints in `weights/` (if not downloading automatically).
3. Keep training/evaluation split inputs under `data/splits/`.

## Common Commands

### Download data

```bash
python scripts/download_data.py --output data/raw
```

### Refine videos

```bash
python scripts/refine_videos.py --input data/raw --output data/refined
```

### Extract features

```bash
python scripts/extract_features.py --input data/refined --output data/processed
```

### Build splits

```bash
python scripts/build_splits.py --input data/processed --output data/splits
```

### Train

```bash
python scripts/train.py --config config/config.yaml --data data/splits
```

### Evaluate

```bash
python scripts/evaluate.py --config config/config.yaml --test-data data/splits
```

### Inference

```bash
python scripts/infer.py --video path/to/video.mp4 --checkpoint weights/danceformer_best.pt
```

## Docker (Optional)

```bash
docker compose --profile cpu build natyaveda-cpu
docker compose --profile cpu run --rm setup-cpu
```

For GPU:

```bash
docker compose --profile gpu build natyaveda-gpu
docker compose --profile gpu run --rm setup-gpu
```

## Important Notes

- Use `data/splits` explicitly for training and evaluation.
- If MediaPipe protobuf errors appear, pin protobuf to `3.20.3`.
- Keep temporary experiments in `experiments/scratch/` so the root stays clean.

## Where To Read More

- Detailed layout notes: `docs/PROJECT_LAYOUT.md`
- Additional docs: `docs/`

## License

Apache 2.0 (as configured in this repository).
