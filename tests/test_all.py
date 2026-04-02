"""
NatyaVeda — Test Suite
Run with: pytest tests/ -v --tb=short
"""

import numpy as np
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# test_pose_extractor.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPoseFrame:
    """Test PoseFrame data container."""

    def test_rtmw_pose_frame_creation(self):
        from src.feature_extraction.pose_extractor import PoseFrame
        kpts = np.random.rand(133, 3).astype(np.float32)
        kpts[:, 2] = np.random.uniform(0.5, 1.0, 133)
        pf = PoseFrame(keypoints=kpts, frame_idx=0, timestamp_sec=0.0, frame_hw=(480, 640))
        assert pf.keypoints.shape == (133, 3)
        assert pf.body.shape == (17, 3)
        assert pf.foot.shape == (6, 3)
        assert pf.face.shape == (68, 3)
        assert pf.left_hand.shape == (21, 3)
        assert pf.right_hand.shape == (21, 3)

    def test_feature_vector_shape(self):
        from src.feature_extraction.pose_extractor import PoseFrame
        kpts = np.random.rand(133, 3).astype(np.float32)
        pf = PoseFrame(keypoints=kpts, frame_idx=0, timestamp_sec=0.0, frame_hw=(480, 640))
        fv = pf.to_feature_vector(include_conf=True)
        assert fv.shape == (399,), f"Expected 399, got {fv.shape}"

    def test_feature_vector_no_conf(self):
        from src.feature_extraction.pose_extractor import PoseFrame
        kpts = np.random.rand(133, 3).astype(np.float32)
        pf = PoseFrame(keypoints=kpts, frame_idx=0, timestamp_sec=0.0, frame_hw=(480, 640))
        fv = pf.to_feature_vector(include_conf=False)
        assert fv.shape == (266,), f"Expected 266, got {fv.shape}"

    def test_avg_confidence(self):
        from src.feature_extraction.pose_extractor import PoseFrame
        kpts = np.zeros((133, 3), dtype=np.float32)
        kpts[:17, 2] = 0.8   # body confidence = 0.8
        pf = PoseFrame(keypoints=kpts, frame_idx=0, timestamp_sec=0.0, frame_hw=(480, 640))
        assert abs(pf.avg_body_confidence - 0.8) < 1e-5

    def test_movenet_pose_frame(self):
        from src.feature_extraction.pose_extractor import PoseFrame
        kpts = np.random.rand(17, 3).astype(np.float32)
        pf = PoseFrame(keypoints=kpts, frame_idx=5, timestamp_sec=0.2, frame_hw=(720, 1280))
        assert pf.keypoints.shape == (17, 3)
        assert pf.frame_idx == 5


# ─────────────────────────────────────────────────────────────────────────────
# test_hand_extractor.py
# ─────────────────────────────────────────────────────────────────────────────

class TestHandFrame:
    """Test HandFrame data container and mudra features."""

    def _make_hand_frame(self, lconf=0.9, rconf=0.9):
        from src.feature_extraction.hand_extractor import HandFrame
        left  = np.random.rand(21, 3).astype(np.float32)
        right = np.random.rand(21, 3).astype(np.float32)
        return HandFrame(left, right, lconf, rconf, frame_idx=0)

    def test_feature_vector_shape(self):
        hf = self._make_hand_frame()
        fv = hf.to_feature_vector()
        assert fv.shape == (126,), f"Expected 126, got {fv.shape}"

    def test_mudra_feature_vector_shape(self):
        hf = self._make_hand_frame()
        mf = hf.mudra_features()
        assert mf.shape == (139,), f"Expected 139 (126+5+5+3), got {mf.shape}"

    def test_finger_angles_shape(self):
        hf = self._make_hand_frame()
        angles = hf.finger_angles("right")
        assert angles.shape == (5,)
        assert np.all(angles >= 0) and np.all(angles <= np.pi + 1e-5)

    def test_both_detected(self):
        from src.feature_extraction.hand_extractor import HandFrame
        left  = np.zeros((21, 3), dtype=np.float32)
        right = np.zeros((21, 3), dtype=np.float32)
        hf = HandFrame(left, right, left_confidence=0.0, right_confidence=0.0)
        assert not hf.both_detected

        hf2 = HandFrame(left, right, left_confidence=0.8, right_confidence=0.7)
        assert hf2.both_detected


# ─────────────────────────────────────────────────────────────────────────────
# test_danceformer.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDanceFormer:
    """Test DanceFormer model forward pass, shapes, and utilities."""

    @pytest.fixture
    def model_small(self):
        from src.models.danceformer import danceformer_small
        return danceformer_small()

    @pytest.fixture
    def model_base(self):
        from src.models.danceformer import danceformer_base
        return danceformer_base()

    def test_forward_pass_small(self, model_small):
        model = model_small
        B, T = 2, 64
        kpts = torch.randn(B, T, 399)
        out = model(kpts)
        assert "dance_logits" in out
        assert "mudra_logits" in out
        assert out["dance_logits"].shape == (B, 8)
        assert out["mudra_logits"].shape == (B, T, 28)
        assert out["features"].shape == (B, 128)

    def test_forward_pass_base(self, model_base):
        model = model_base
        B, T = 4, 48
        kpts = torch.randn(B, T, 399)
        vel  = torch.randn(B, T, 399)
        acc  = torch.randn(B, T, 399)
        out = model(kpts, vel, acc)
        assert out["dance_logits"].shape == (B, 8)
        assert out["mudra_logits"].shape == (B, T, 28)

    def test_padding_mask(self, model_small):
        model = model_small
        B, T = 2, 64
        kpts = torch.randn(B, T, 399)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, 50:] = True   # last 14 frames padded
        out = model(kpts, padding_mask=mask)
        assert out["dance_logits"].shape == (B, 8)

    def test_parameter_count_small(self, model_small):
        count = model_small.count_parameters()
        assert count > 0
        # Small model should be under 5M params
        assert count < 5_000_000, f"Small model too large: {count:,} params"

    def test_parameter_count_large(self):
        from src.models.danceformer import danceformer_large
        model = danceformer_large()
        count = model.count_parameters()
        # Large model should be between 5M–50M params
        assert 5_000_000 < count < 50_000_000, f"Unexpected param count: {count:,}"

    def test_save_load(self, model_small, tmp_path):
        model = model_small
        save_path = tmp_path / "test_model.pt"
        model.save(save_path)
        assert save_path.exists()

    def test_from_config(self):
        from src.models.danceformer import DanceFormer
        config = {
            "pose_embed_dim": 128,
            "num_transformer_layers": 4,
            "num_attention_heads": 4,
            "ff_dim": 512,
            "transformer_dropout": 0.1,
            "max_sequence_length": 64,
            "num_dance_classes": 8,
            "num_mudra_classes": 28,
            "videomae_fusion": False,
        }
        model = DanceFormer.from_config({"model": config})
        assert model is not None

    def test_predict_method(self, model_small):
        model = model_small
        kpts = torch.randn(1, 64, 399)
        result = model.predict(kpts)
        assert result["dance_class"] in [
            "bharatanatyam", "kathak", "odissi", "kuchipudi",
            "manipuri", "mohiniyattam", "sattriya", "kathakali",
        ]
        assert 0.0 <= result["dance_confidence"] <= 1.0
        assert len(result["dance_probabilities"]) == 8
        assert abs(sum(result["dance_probabilities"]) - 1.0) < 1e-4

    def test_no_videomae(self):
        from src.models.danceformer import DanceFormer
        model = DanceFormer(use_videomae_fusion=False, num_transformer_layers=2)
        kpts = torch.randn(2, 32, 399)
        out = model(kpts)
        assert out["dance_logits"].shape == (2, 8)

    def test_videomae_fusion(self):
        from src.models.danceformer import DanceFormer
        model = DanceFormer(
            use_videomae_fusion=True, videomae_dim=1024,
            embed_dim=128, num_transformer_layers=2, num_heads=4, ff_dim=256
        )
        kpts = torch.randn(2, 32, 399)
        video_tokens = torch.randn(2, 8, 1024)
        out = model(kpts, video_tokens=video_tokens)
        assert out["dance_logits"].shape == (2, 8)


# ─────────────────────────────────────────────────────────────────────────────
# test_pose_encoder.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPosePatchEmbedding:
    """Test pose embedding module."""

    def test_output_shape(self):
        from src.models.danceformer import PosePatchEmbedding
        embed = PosePatchEmbedding(embed_dim=256)
        kpts = torch.randn(4, 64, 399)
        out = embed(kpts)
        assert out.shape == (4, 64, 256), f"Expected (4,64,256), got {out.shape}"

    def test_with_velocities(self):
        from src.models.danceformer import PosePatchEmbedding
        embed = PosePatchEmbedding(embed_dim=128)
        kpts = torch.randn(2, 32, 399)
        vel  = torch.randn(2, 32, 399)
        acc  = torch.randn(2, 32, 399)
        out = embed(kpts, vel, acc)
        assert out.shape == (2, 32, 128)


# ─────────────────────────────────────────────────────────────────────────────
# test_dance_isolator.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDanceIsolator:
    """Test dance isolation logic (no video files needed — unit tests on helpers)."""

    def test_motion_energy_zero_for_static(self):
        import cv2
        from src.preprocessing.dance_isolator import compute_motion_energy
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
        energy = compute_motion_energy([frame, frame, frame])
        assert energy < 0.001, f"Static frames should have near-zero motion, got {energy}"

    def test_motion_energy_high_for_motion(self):
        from src.preprocessing.dance_isolator import compute_motion_energy
        frames = []
        for i in range(5):
            f = np.zeros((240, 320, 3), dtype=np.uint8)
            f[:, i*30:(i+1)*30] = 200   # moving white band
            frames.append(f)
        energy = compute_motion_energy(frames)
        assert energy > 0.001, f"Moving frames should have nonzero motion"

    def test_activity_classifier_heuristic(self):
        from src.preprocessing.dance_isolator import ActivityClassifier
        clf = ActivityClassifier(model_path=None)
        # High-motion, high-variance frames → higher dance score
        frames = [np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8) for _ in range(10)]
        score = clf.score_frames(frames)
        assert 0.0 <= score <= 1.0

    def test_scene_keeps_high_score(self):
        from src.preprocessing.dance_isolator import DanceIsolator, Scene
        isolator = DanceIsolator(min_dance_confidence=0.6, min_motion_threshold=0.005)
        good_scene = Scene(0, 100, 0.0, 4.0, dance_score=0.9, motion_energy=0.02,
                           principal_bbox=(10, 10, 200, 400))
        bad_scene  = Scene(100, 200, 4.0, 8.0, dance_score=0.2, motion_energy=0.001,
                           principal_bbox=None)
        assert isolator._should_keep(good_scene) is True
        assert isolator._should_keep(bad_scene) is False


# ─────────────────────────────────────────────────────────────────────────────
# test_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingPipeline:
    """Integration-style tests for training components."""

    def test_focal_loss_shape(self):
        from src.training.trainer import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 8)
        labels = torch.randint(0, 8, (8,))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0   # scalar
        assert float(loss) >= 0

    def test_focal_loss_perfect_prediction(self):
        from src.training.trainer import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        # Perfect logits → very low focal loss
        logits = torch.eye(8) * 20   # high score on diagonal
        labels = torch.arange(8)
        loss = loss_fn(logits, labels)
        assert float(loss) < 0.1

    def test_dataset_normalization(self, tmp_path):
        """Test that normalization centers keypoints at hip."""
        from src.training.trainer import DanceDataset
        import os

        # Create minimal fake data
        T = 100
        kpts = np.random.rand(T, 133, 3).astype(np.float32)
        # Set hip keypoints (11, 12) to known positions
        kpts[:, 11, :2] = 0.4  # left hip
        kpts[:, 12, :2] = 0.6  # right hip
        kpts[:, 5, :2]  = 0.4  # left shoulder
        kpts[:, 6, :2]  = 0.6  # right shoulder

        dance_dir = tmp_path / "train" / "kathak"
        dance_dir.mkdir(parents=True)
        np.savez_compressed(str(dance_dir / "test_video.npz"), keypoints=kpts)

        ds = DanceDataset(str(tmp_path), split="train", clip_length=64, augment=False)
        assert len(ds) > 0
        sample = ds[0]
        assert sample["keypoints"].shape == (64, 399)
        assert sample["label"].item() == 1  # kathak is index 1

    def test_dance_class_indices(self):
        from src.training.trainer import DANCE_CLASSES, DANCE_TO_IDX
        assert len(DANCE_CLASSES) == 8
        assert DANCE_TO_IDX["bharatanatyam"] == 0
        assert DANCE_TO_IDX["kathakali"] == 7
        for i, d in enumerate(DANCE_CLASSES):
            assert DANCE_TO_IDX[d] == i


# ─────────────────────────────────────────────────────────────────────────────
# test_visualizer.py
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualizer:
    """Test skeleton drawing utilities."""

    def test_draw_on_frame(self):
        from src.inference.predictor import SkeletonVisualizer
        viz = SkeletonVisualizer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        kpts  = np.random.rand(133, 3).astype(np.float32)
        kpts[:, 2] = 0.9  # high confidence
        result = viz.draw(
            frame=frame,
            keypoints=kpts,
            dance_class="bharatanatyam",
            dance_conf=0.87,
            mudra_label="pataka",
            all_probs=[0.87, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02],
        )
        assert result.shape == (480, 640, 3)
        # Frame should differ from original (annotations added)
        assert not np.array_equal(result, frame)

    def test_draw_without_hands(self):
        from src.inference.predictor import SkeletonVisualizer
        viz = SkeletonVisualizer(show_hands=False)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        kpts  = np.random.rand(133, 3).astype(np.float32)
        result = viz.draw(frame, kpts, "kathak", 0.75)
        assert result.shape == (480, 640, 3)
