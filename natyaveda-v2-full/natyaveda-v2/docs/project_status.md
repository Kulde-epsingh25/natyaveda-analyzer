# NatyaVeda Analyzer v2 - Project Status Report

## Scope

This document captures the current state of the NatyaVeda Analyzer v2 project after the recent debugging and hardening work in this workspace. It is meant to be a practical status report, not a marketing summary. It describes:

- what the project does
- how the system is structured
- what problems we found
- what we changed to fix them
- what is still unresolved
- what should be checked before calling the project production-ready

This report reflects the repository state and the issues observed while running training, inference, and Docker-related workflows in the current environment.

## Project Overview

NatyaVeda Analyzer is a dance classification and analysis pipeline for Indian classical dance videos. The system combines:

- pose extraction
- hand landmark extraction
- optional video appearance features
- temporal classification with DanceFormer
- per-video and per-segment inference
- dataset split generation
- training and evaluation tooling
- Docker-based reproducible execution

The intended output is a dance-form prediction with confidence, optional mudra support, and annotated video output.

## Core Pipeline

The project follows this flow:

1. Raw video input
2. Video refinement and clip filtering
3. Pose extraction
4. Hand extraction
5. Optional VideoMAE feature extraction
6. Feature merging into .npz clips
7. Train/validation/test split generation
8. DanceFormer training
9. Evaluation and prototype caching
10. Inference with temporal voting and optional overlays

## Repository Structure

Important locations in the repo:

- `src/models/danceformer.py` - model definition and model variants
- `src/training/trainer.py` - dataset, losses, trainer, EMA, and training loop
- `src/inference/predictor.py` - end-to-end video inference and annotated output
- `scripts/train.py` - training entry point
- `scripts/evaluate.py` - evaluation entry point
- `scripts/infer.py` - inference CLI
- `scripts/build_splits.py` - split generation
- `scripts/extract_features.py` - feature extraction pipeline
- `config/config.yaml` - training and model configuration
- `docker-compose.yml` - container orchestration
- `Dockerfile` - image build logic
- `scripts/docker_pipeline.sh` - container pipeline runner

## What We Found

### 1. Training used `data/splits`, not `data/processed`

The training and evaluation code are designed to consume the split directory structure. Earlier usage patterns that pointed training at `data/processed` could cause fallback behavior and make train/val logic unreliable.

Observed behavior:

- `DanceDataset` expects `train`, `val`, and `test` subdirectories when available.
- using `data/splits` is the correct path for training and evaluation.
- this is important for reliable accuracy measurement and proper separation of classes.

Fix applied:

- updated Docker pipeline commands and documentation to use `data/splits` for training and evaluation.

### 2. Inference needed stronger correctness controls

The original inference behavior could return a single top class even when the prediction was ambiguous.

Observed behavior:

- one clip could dominate the result
- a small confidence gap between the top two classes was not surfaced clearly
- outputs could look decisive even when the model was not really confident

Fixes applied:

- added robust temporal aggregation in `src/inference/predictor.py`
- added trimmed mean aggregation as the default
- added optional `mean` and `geomean` aggregation modes
- added strict confidence gating
- added top-2 margin reporting
- added `unknown` output when strict mode is enabled and the result is too uncertain

This does not make predictions perfect, but it reduces confident wrong answers.

### 3. Cluster separation in embedding space was too weak

The user wanted cleaner separation between dance classes so they are easier to identify.

Observed behavior:

- embeddings could overlap between visually similar classes
- confusing pairs such as Bharatanatyam and Mohiniyattam could appear close
- the previous loss mix did not push class clusters apart aggressively enough

Fixes applied:

- strengthened center separation loss
- added intra-class variance penalty
- added hard-negative centroid separation
- added batch-hard triplet loss
- increased separation-oriented loss weights in `config/config.yaml`

Result:

- cleaner class clusters during training are now more likely
- the model should learn more distinct class boundaries
- embedding visualization is still recommended to confirm this quantitatively

### 4. Mixed precision training was incompatible with the installed PyTorch version

Observed error:

- `torch.amp.GradScaler` was not available in the installed `torch 2.2.0+cu121`

Fix applied:

- updated `src/training/trainer.py` to fall back to `torch.cuda.amp.GradScaler` when needed
- kept compatibility with newer PyTorch releases too

This restored training startup in the current environment.

### 5. Docker build context was too large

Observed behavior:

- Docker build sent a very large context because `data`, `weights`, `outputs`, `reports`, and other heavy folders were being included
- this caused Docker to become unstable and fail with EOF during image transfer

Fix applied:

- tightened `.dockerignore` to exclude heavy runtime artifacts and non-build assets
- added Docker Compose services for CPU and GPU workflows
- added a reusable pipeline script for container runs

Residual issue:

- Docker Desktop itself was unstable during the session and sometimes remained in a starting state
- the repo changes reduce the build load, but local Docker health still matters

### 6. Docker Desktop / WSL issues blocked container validation

Observed behavior:

- `docker compose build` initially failed because Docker Desktop was not running
- later runs hit Docker API / daemon instability
- Docker Desktop status remained in `starting` for a while

What was done:

- confirmed compose config was valid
- documented Windows troubleshooting in the README previously
- reduced build context to make container startup less fragile

What remains:

- if Docker Desktop or WSL is unhealthy on the host machine, container builds can still fail even though the repo configuration is correct

### 7. TensorFlow / TF Hub and MediaPipe emit expected warnings

Observed warnings:

- TensorFlow oneDNN notices
- deprecated `pkg_resources` usage in TensorFlow Hub
- slow image processor warnings from Hugging Face
- weight-loading warnings for DETR and timm checkpoints

Interpretation:

- these are common runtime warnings in the current dependency stack
- they do not necessarily mean the pipeline is broken
- they are noisy and could be reduced later, but they were not the primary blocker

### 8. Hand detection was weak on the smoke test clip

Observed result from inference:

- pose extraction succeeded
- the clip produced 378 valid frames
- hand extraction produced 384 hand frames, but both hands were detected in 0 frames

Meaning:

- the current clip likely does not provide reliable hand visibility
- mudra signal is weak for that sample
- hand-based reasoning cannot be trusted for this clip

This is a data-quality issue rather than a code crash.

## Changes Made So Far

### Training stability and separation

In `src/training/trainer.py`:

- added deterministic seeding
- added weighted sampler safety improvements
- added mixed-precision scaling compatibility
- strengthened center separation objective
- added hard triplet loss
- added intra-class variance penalty
- added hard-negative centroid separation
- improved gradient accumulation handling

### Inference reliability

In `src/inference/predictor.py` and `scripts/infer.py`:

- added trimmed probability aggregation
- added geomean aggregation option
- added strict confidence mode
- added top-2 margin output
- added `unknown` fallback when uncertainty is high
- exposed those controls in the CLI

### Docker and orchestration

In `Dockerfile`, `docker-compose.yml`, `.dockerignore`, and `scripts/docker_pipeline.sh`:

- containerized the project for CPU and GPU use
- added setup and pipeline services
- reduced build context size
- added a reusable all-in-one pipeline script
- documented Windows-specific Docker issues

### Configuration

In `config/config.yaml`:

- increased separation-oriented training weights
- added new loss configuration knobs
- made cluster separation easier to tune from config

## Current Verified State

The following were verified in the workspace:

- `config/config.yaml` parses successfully
- inference CLI help works with the new options
- inference runs successfully on the smoke test video
- strict mode correctly returns `unknown` for low-confidence predictions
- training code imports successfully with the current PyTorch version
- trainer code now supports the installed PyTorch API level

## Example Inference Result Observed

On the smoke test clip, the updated inference pipeline produced:

- raw top prediction: Bharatanatyam
- confidence: 57.1%
- top-2 margin: 0.150
- strict output: `unknown`
- main competing class: Mohiniyattam at 42.1%

This is a useful example of the strict mode working correctly: it prevents a borderline prediction from being reported as fully reliable.

## Still Open Problems

### 1. Class confusion still exists

Some classes remain visually and temporally similar, especially:

- Bharatanatyam vs Mohiniyattam
- Kathak vs other spin-heavy forms
- overlapping styles with low-quality or partial-body clips

The new losses should help, but they do not guarantee clean separation without better data.

### 2. Hand signal quality is inconsistent

For some clips, especially low-resolution or occluded videos:

- MediaPipe hands may fail
- both-hand detection may be absent
- mudra classification becomes unreliable

This limits fine-grained analysis on some videos.

### 3. Docker Desktop health is still external to the repo

The repository is more robust now, but builds still depend on:

- Docker Desktop running correctly
- WSL backend health on Windows
- GPU passthrough when GPU mode is used

### 4. Long training quality is not yet fully proven

The code is now more stable and more separation-aware, but there is still a need to verify:

- actual validation F1 improvement after a full run
- whether the new separation losses overfit
- whether strict inference thresholds should be tuned per class

### 5. The dataset quality itself may still be the main bottleneck

Even with good code, results can be limited by:

- noisy clips
- wrong labels
- mixed dance styles in one video
- unbalanced class counts
- partial dancer visibility

## Recommended Next Steps

1. Run a full training job on `data/splits` and compare validation F1 against the previous checkpoint.
2. Generate embedding plots with UMAP or t-SNE to verify that class clusters are actually separating better.
3. Tune `min-confidence` and `min-margin` on a validation set rather than a single test clip.
4. Review classes with the highest confusion and inspect their source clips.
5. Clean or rebalance the dataset if the same confusion pairs continue to dominate errors.
6. Re-run Docker after confirming Docker Desktop and WSL are healthy.

## Bottom Line

The project is now significantly more stable and more accuracy-aware than before:

- inference is more honest about uncertainty
- training is more separation-driven
- Docker is more complete
- PyTorch compatibility issues are fixed

The remaining accuracy ceiling is likely driven more by data quality and class similarity than by the current training/inference code.
