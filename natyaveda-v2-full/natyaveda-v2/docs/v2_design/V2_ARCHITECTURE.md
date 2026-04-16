# NatyaVeda V2 Architecture (Design Only)

Status: Draft design. No implementation in this document.

## Goals

- Preserve compatibility with current training format while enabling richer multimodal signals.
- Decouple extraction modules so body, hand, face, and video streams can evolve independently.
- Add quality controls and confidence-aware filtering for robust large-scale ingestion.
- Support phased rollout (v1-compatible -> hybrid -> full v2).

## High-Level Pipeline

1. Video Ingestion
- Input: refined dance clips.
- Responsibilities: file validation, metadata collection, clip duration checks.

2. Core Landmark Extraction
- Body/wholebody keypoints (133-point schema baseline).
- Hand refinements.
- Face landmarks from the same temporal index.

3. Deep Feature Branches
- Face expression branch (temporal expression embedding).
- Hand/mudra branch (shape + transition embedding).
- Motion branch (velocity, acceleration, periodicity, rhythm profile).
- Video branch (VideoMAE or equivalent visual backbone features).

4. Temporal Alignment and Fusion
- Frame-index alignment across all branches.
- Missing-value masking and interpolation policy.
- Cross-stream fusion tensor assembly.

5. Packaging and Quality Report
- Save v2 feature package with metadata, masks, confidence curves.
- Produce per-video quality report with rejection and warning reasons.

## Proposed Module Layout

- src/feature_extraction_v2/
  - orchestrator.py
  - alignment.py
  - quality.py
  - io.py
  - body_branch.py
  - hand_branch.py
  - face_branch.py
  - motion_branch.py
  - video_branch.py

- src/schemas/
  - feature_package_v2.py
  - quality_report_v2.py

- scripts/
  - extract_features_v2.py
  - validate_features_v2.py

## Design Principles

- Deterministic outputs for identical input + config.
- Explicit schema versioning in every output file.
- Fail-soft behavior: partial branch failure should still produce a package with masks.
- Traceability: each saved artifact includes source model IDs, settings, and timestamps.

## Key Non-Functional Requirements

- Throughput target: maintain practical processing speed on 4GB-8GB GPUs.
- Memory safety: bounded in-memory buffering for long clips.
- Observability: structured logs + quality counters by dance form.

## Out of Scope (for initial v2)

- End-to-end retraining redesign.
- Real-time inference UI changes.
- Multi-person choreography graph modeling.
