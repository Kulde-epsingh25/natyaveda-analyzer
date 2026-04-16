# NatyaVeda V2 Roadmap (Design)

Status: Planning artifact. No execution performed.

## Phase 0: Design Freeze

- Finalize schema and naming conventions.
- Define metrics for quality and acceptance.
- Approve fallback behavior per branch.

Exit criteria:
- Architecture and schema documents approved.

## Phase 1: Extraction Skeleton

- Build v2 orchestrator without changing v1 pipeline.
- Add branch interfaces with mock outputs.
- Implement schema writer and validator.

Exit criteria:
- v2 files generated from a small sample set.
- Validation passes on all sample outputs.

## Phase 2: Deep Face and Hand Features

- Add expression embeddings and region-level facial descriptors.
- Add mudra-oriented hand embeddings and transition metrics.
- Attach confidence traces and per-branch quality scoring.

Exit criteria:
- Stable extraction on all dance forms including makeup-heavy performances.

## Phase 3: Temporal Fusion and Robustness

- Implement stream alignment, masks, and interpolation.
- Improve handling of occlusion, profile angles, and fast head movement.
- Add failure analytics and recovery policy.

Exit criteria:
- Reduced drop rate on difficult clips.
- Quality score correlates with manual inspection.

## Phase 4: Training Integration

- Add v2 dataset loader with v1 fallback mode.
- Enable ablation: body only, body+hands, full multimodal.
- Track performance by dance form and style variation.

Exit criteria:
- Reproducible training experiments comparing v1 and v2.

## Phase 5: Production Hardening

- Improve throughput and memory usage.
- Add resumable checkpoints and partial artifact recovery.
- Publish model/data card updates.

Exit criteria:
- Long-run extraction reliability and stable outputs.
