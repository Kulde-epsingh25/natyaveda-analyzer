# NatyaVeda V2 Feature Schema (Draft)

Status: Design schema. Not yet enforced in code.

## File Naming

- Main package: <video_stem>.v2.npz
- Optional debug package: <video_stem>.v2.debug.npz

## Required Fields

- schema_version: str
- dance_form: str
- label: int
- frame_timestamps: float32[T]
- valid_mask: uint8[T]
- source_models: object/dict-like metadata

## Core Streams

1. keypoints_133
- Shape: float32[T, 133, 3]
- Semantics: x_norm, y_norm, confidence

2. hand_features
- Shape: float32[T, H]
- H is configurable and documented in metadata

3. face_landmarks_68
- Shape: float32[T, 68, 3]

4. face_expression_features
- Shape: float32[T, E]
- E is configurable and documented in metadata

5. motion_features
- Shape: float32[T, M]
- Includes velocity, acceleration, rhythm descriptors

6. video_features
- Shape: float32[Tv, V]
- Tv may differ from T; alignment map required

## Alignment Fields

- video_to_pose_index: int32[Tv]
- stream_presence_mask: uint8[T, S]
- interpolation_flags: uint8[T, S]

## Quality Fields

- body_confidence: float32[T]
- hand_confidence: float32[T]
- face_confidence: float32[T]
- quality_score_global: float32[1]
- quality_flags: object/dict-like metadata

## Compatibility Fields

- keypoints: float32[T, 133, 3]
- timestamps: float32[T]
- confidences: float32[T]

These fields preserve compatibility with existing consumers while new consumers read the full v2 schema.

## Validation Rules

- T must be > 0 for accepted package.
- Keypoint coordinates expected in [0, 1] unless explicitly flagged.
- Confidence values must be in [0, 1].
- All stream arrays must have consistent dtype and finite values.
