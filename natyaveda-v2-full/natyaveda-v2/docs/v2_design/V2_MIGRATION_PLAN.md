# NatyaVeda V2 Migration Plan (No-Code Plan)

Status: Migration strategy only.

## Objectives

- Keep current workflows working while gradually enabling v2 outputs.
- Avoid breaking existing training scripts and dataset consumers.
- Provide clear rollback options at every stage.

## Migration Strategy

1. Dual Output Period
- Continue generating existing .npz outputs.
- In parallel, generate .v2.npz outputs for selected dance forms.

2. Validator Gate
- Introduce schema validator as a required gate before training on v2 artifacts.
- Reject or quarantine malformed v2 packages.

3. Loader Upgrade
- Update dataset loader to detect schema_version.
- Route v1 files through legacy parser and v2 files through multimodal parser.

4. Progressive Rollout by Dance Form
- Start with stable forms already well covered.
- Then rollout to difficult sets (for example, Kathakali makeup-heavy clips).

5. Benchmark and Signoff
- Compare accuracy, robustness, and failure rates against v1 baseline.
- Sign off only when metrics are equal or better and extraction stability is proven.

## Rollback Plan

- If v2 branch fails quality thresholds, training automatically falls back to v1 parser.
- Keep v1 extraction untouched until v2 has sustained stability across full dataset.

## Risks and Mitigations

- Risk: Face branch instability under makeup/occlusion.
- Mitigation: Confidence-aware masking + temporal smoothing + fallback descriptors.

- Risk: Increased extraction runtime.
- Mitigation: Configurable branch toggles and tiered hardware profiles.

- Risk: Schema drift over iterations.
- Mitigation: strict schema_version tagging and validator CI checks.

## Success Criteria

- No regression in existing v1 workflows.
- v2 outputs available and validated for all target classes.
- Measurable model quality improvements on expression- and mudra-heavy clips.
