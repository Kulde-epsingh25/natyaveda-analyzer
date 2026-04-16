# NatyaVeda DanceFormer — Model Card

## Model Description

**DanceFormer** is a Transformer-based classifier for Indian classical dance recognition from video. It ingests whole-body skeleton keypoints (133 COCO-WholeBody points including full hands/fingers and face) extracted from raw video, and outputs both a dance form prediction (8 classes) and per-frame mudra/hasta gesture labels (28 classes).

## Model Architecture Summary

| Component | Details |
|-----------|---------|
| Input | 133 keypoints × 3 coords (x,y,conf) per frame = 399-dim |
| Temporal window | 64 frames (≈2.5 sec at 25fps) |
| Embedding | Group-aware pose patch embedding (body, feet, hands, face split) |
| Backbone | Transformer Encoder, 8 layers, 8 heads, dim=256, ff=1024 |
| Output heads | Dance form (8-way) + Mudra (28-way per frame) |
| Optional fusion | VideoMAE-v2 Large cross-attention tokens |
| Parameters | ~8M (large), ~2M (small) |

## Feature Extraction Stack

| Source | Keypoints | Use |
|--------|-----------|-----|
| **RTMW-x** (MMPose) | 133 pts whole-body | Primary extraction |
| **MoveNet Thunder** (TF Hub) | 17 pts body | Fallback for low-confidence frames |
| **MediaPipe Hands** | 21 pts × 2 hands | Finger-level mudra detail |
| **VideoMAE-v2** (HuggingFace) | 1024-dim clip tokens | Holistic appearance context |

## Intended Uses

- Classifying Indian classical dance forms in recorded videos
- Mudra/hasta gesture recognition for educational annotation
- Dance performance analysis and feedback systems
- Digital archiving of Indian classical dance heritage

## Out-of-Scope Uses

- Real-time performance coaching (latency not optimized)
- Folk, contemporary, or western dance forms (not trained on)
- Medical / clinical gait analysis

## Data

Training data is collected from YouTube performance videos for 8 dance forms. A rigorous 5-stage **DanceIsolator** pipeline removes audience shots, title cards, presenter segments, and low-motion frames before feature extraction.

**No personal data is stored** — only anonymized skeleton keypoint sequences.

## Evaluation

Test-set evaluation uses 5-fold cross-validation with stratification by dance form and performer gender. Metrics: weighted F1 (primary), per-class precision/recall, confusion matrix.

## Limitations

- Performance may degrade on unusual camera angles, extreme costume coverage of joints, or very fast movements causing motion blur
- Kathakali and Bharatanatyam may be confused due to similar Abhinaya postures
- Manipuri and Mohiniyattam have similar gentle, flowing body styles — lowest individual F1 scores

## Citation

```bibtex
@software{natyaveda2024,
  title     = {NatyaVeda Analyzer: AI-Powered Indian Classical Dance Recognition},
  year      = {2024},
  url       = {https://github.com/YOUR_USERNAME/natyaveda-analyzer},
  license   = {Apache-2.0}
}
```

## Key References

1. **RTMW**: Jiang et al., "RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation" (2024) — https://arxiv.org/abs/2407.08634
2. **AlphaPose**: Fang et al., "AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time" (2022) — https://arxiv.org/abs/2211.03375
3. **MoveNet**: Google, TF Hub Single-pose Thunder — https://www.tensorflow.org/hub/tutorials/movenet
4. **VideoMAE v2**: Wang et al., "VideoMAE V2: Scaling Video Masked Autoencoders" (2023) — MCG-NJU/videomae-large
5. **MediaPipe Hands**: Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking"
