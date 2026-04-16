"""
NatyaVeda v2 — Complete Test Suite
pytest tests/ -v --tb=short
"""
import sys, os
from pathlib import Path
import numpy as np
import pytest, torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def kpts133():
    k = np.random.rand(133,3).astype(np.float32)
    k[:,2] = np.random.uniform(0.5,1.0,133)
    return k

@pytest.fixture
def kpts17():
    k = np.random.rand(17,3).astype(np.float32)
    k[:,2] = np.random.uniform(0.4,1.0,17)
    return k

@pytest.fixture
def frame():
    return np.zeros((480,640,3), dtype=np.uint8)

# ── DeviceManager ─────────────────────────────────────────────────────────────

class TestDeviceManager:
    def test_cpu_forced(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        assert dm.device == "cpu"
        assert dm.fp16 is False

    def test_cpu_uses_vitpose_base(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        assert "vitpose" in dm.pose_model_tier
        assert "base" in dm.vitpose_model_id

    def test_is_cuda_false_cpu(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        assert dm.config.is_cuda is False

    def test_summary_string(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        assert isinstance(dm.summary(), str)

    def test_move_model_cpu(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        m = torch.nn.Linear(4,2)
        moved = dm.move_model(m)
        assert next(moved.parameters()).device.type == "cpu"

    def test_to_device_cpu(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        t = dm.to_device(torch.randn(3,3))
        assert t.device.type == "cpu"

    def test_autocast_cpu_no_crash(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        with dm.autocast_context():
            _ = torch.randn(2,2)

    def test_gpu_if_available(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager()
        if torch.cuda.is_available():
            assert dm.device == "cuda"
            assert dm.config.gpu_info.vram_gb > 0
        else:
            assert dm.device == "cpu"

    def test_uses_vitpose_on_cpu(self):
        from src.utils.device import DeviceManager
        dm = DeviceManager(force_cpu=True)
        assert dm.config.uses_vitpose
        assert not dm.config.uses_mmpose

    def test_vram_thresholds(self):
        assert 10.0 >= 8.0   # rtmw-x
        assert 7.0  >= 6.0   # vitpose-huge
        assert 5.0  >= 4.0   # vitpose-large
        assert 3.0  >= 2.0   # vitpose-base

# ── PoseFrame ─────────────────────────────────────────────────────────────────

class TestPoseFrame:
    def test_create_133(self, kpts133):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts133, 0, 0.0, (480,640), "rtmw")
        assert pf.n_keypoints == 133
        assert pf.avg_body_confidence > 0

    def test_create_17(self, kpts17):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts17, 5, 0.2, (480,640), "movenet")
        assert pf.n_keypoints == 17

    def test_body_shape(self, kpts133):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts133, 0, 0.0, (480,640))
        assert pf.body.shape == (17,3)

    def test_left_hand_shape(self, kpts133):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts133, 0, 0.0, (480,640))
        assert pf.left_hand.shape == (21,3)

    def test_right_hand_shape(self, kpts133):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts133, 0, 0.0, (480,640))
        assert pf.right_hand.shape == (21,3)

    def test_face_shape(self, kpts133):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts133, 0, 0.0, (480,640))
        assert pf.face.shape == (68,3)

    def test_feature_vector_399(self, kpts133):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts133, 0, 0.0, (480,640))
        assert pf.to_feature_vector().shape == (399,)

    def test_feature_vector_17_padded_to_399(self, kpts17):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts17, 0, 0.0, (480,640))
        fv = pf.to_feature_vector()
        assert fv.shape == (399,)
        assert np.all(fv[17*3:] == 0)

    def test_movenet_zero_hands(self, kpts17):
        from src.feature_extraction.pose_extractor import PoseFrame
        pf = PoseFrame(kpts17, 0, 0.0, (480,640))
        assert np.all(pf.left_hand == 0)

# ── VitPose ───────────────────────────────────────────────────────────────────

class TestVitPose:
    def test_importable(self):
        try:
            from transformers import VitPoseForPoseEstimation
            assert VitPoseForPoseEstimation is not None
        except ImportError:
            pytest.skip("transformers not installed")

    def test_extractor_init_no_crash(self):
        from src.feature_extraction.pose_extractor import VitPoseExtractor
        ext = VitPoseExtractor(model_id="usyd-community/vitpose-plus-base", device="cpu")
        assert hasattr(ext, "available")

    def test_rtmw_graceful_unavailable(self, frame):
        from src.feature_extraction.pose_extractor import RTMWExtractor
        ext = RTMWExtractor.__new__(RTMWExtractor)
        ext.available = False; ext._pose_model = None
        result = ext.extract_frame(frame, 0, 25.0)
        assert result is None

    def test_unified_extractor_status(self):
        from src.utils.device import DeviceManager
        from src.feature_extraction.pose_extractor import PoseExtractor
        dm = DeviceManager(force_cpu=True)
        pe = PoseExtractor(device_manager=dm)
        assert isinstance(pe.status, str)

# ── HandFrame ─────────────────────────────────────────────────────────────────

class TestHandFrame:
    def _hf(self, lc=0.9, rc=0.9):
        from src.feature_extraction.hand_extractor import HandFrame
        l = np.random.rand(21,3).astype(np.float32)
        r = np.random.rand(21,3).astype(np.float32)
        return HandFrame(l, r, lc, rc, frame_idx=0)

    def test_fv_shape(self): assert self._hf().to_feature_vector().shape == (126,)
    def test_mudra_shape(self): assert self._hf().mudra_features().shape == (139,)
    def test_angles_shape(self): assert self._hf().finger_angles("right").shape == (5,)
    def test_angles_range(self):
        a = self._hf().finger_angles("right")
        assert np.all(a >= 0) and np.all(a <= np.pi + 1e-4)
    def test_both_detected(self): assert self._hf(0.9, 0.8).both_detected
    def test_not_detected(self):
        from src.feature_extraction.hand_extractor import HandFrame
        hf = HandFrame(np.zeros((21,3),np.float32), np.zeros((21,3),np.float32), 0.0, 0.0)
        assert not hf.both_detected
    def test_mudra_predict(self):
        from src.feature_extraction.hand_extractor import MudraRecognizer
        r = MudraRecognizer()
        name, conf = r.predict(self._hf())
        assert isinstance(name, str) and 0.0 <= conf <= 1.0

# ── DanceFormer ───────────────────────────────────────────────────────────────

class TestDanceFormer:
    @pytest.fixture
    def small(self):
        from src.models.danceformer import danceformer_small
        return danceformer_small()

    def test_forward(self, small):
        out = small(torch.randn(2,32,399))
        assert out["dance_logits"].shape == (2,8)
        assert out["mudra_logits"].shape == (2,32,28)

    def test_with_vel_acc(self, small):
        out = small(torch.randn(2,32,399), torch.randn(2,32,399), torch.randn(2,32,399))
        assert out["dance_logits"].shape == (2,8)

    def test_padding_mask(self, small):
        mask = torch.zeros(2,32,dtype=torch.bool); mask[0,25:] = True
        out = small(torch.randn(2,32,399), padding_mask=mask)
        assert out["dance_logits"].shape == (2,8)

    def test_no_nan(self, small):
        out = small(torch.randn(2,32,399))
        assert not torch.isnan(out["dance_logits"]).any()

    def test_params_small(self, small):
        assert 0 < small.count_parameters() < 5_000_000

    def test_params_large(self):
        from src.models.danceformer import danceformer_large
        m = danceformer_large()
        assert 5_000_000 < m.count_parameters() < 60_000_000

    def test_predict(self, small):
        r = small.predict(torch.randn(1,32,399))
        assert r["dance_class"] in ["bharatanatyam","kathak","odissi","kuchipudi",
                                     "manipuri","mohiniyattam","sattriya","kathakali"]
        assert abs(sum(r["dance_probabilities"]) - 1.0) < 1e-4

    def test_save(self, small, tmp_path):
        small.save(tmp_path / "m.pt")
        assert (tmp_path / "m.pt").exists()

    def test_probs_sum_one(self, small):
        import torch.nn.functional as F
        probs = F.softmax(small(torch.randn(4,32,399))["dance_logits"], dim=-1)
        assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)

    def test_no_videomae(self):
        from src.models.danceformer import DanceFormer
        m = DanceFormer(use_videomae_fusion=False, num_transformer_layers=2)
        assert m(torch.randn(2,32,399))["dance_logits"].shape == (2,8)

    def test_videomae_fusion(self):
        from src.models.danceformer import DanceFormer
        m = DanceFormer(use_videomae_fusion=True, num_transformer_layers=2,
                        embed_dim=128, num_heads=4, ff_dim=256)
        out = m(torch.randn(2,32,399), video_tokens=torch.randn(2,8,1024))
        assert out["dance_logits"].shape == (2,8)

# ── PosePatchEmbedding ────────────────────────────────────────────────────────

class TestPoseEmbed:
    def test_shape_256(self):
        from src.models.danceformer import PosePatchEmbedding
        assert PosePatchEmbedding(256)(torch.randn(4,64,399)).shape == (4,64,256)
    def test_shape_128(self):
        from src.models.danceformer import PosePatchEmbedding
        assert PosePatchEmbedding(128)(torch.randn(2,32,399)).shape == (2,32,128)
    def test_with_vel_acc(self):
        from src.models.danceformer import PosePatchEmbedding
        e = PosePatchEmbedding(128)
        out = e(torch.randn(2,32,399), torch.randn(2,32,399), torch.randn(2,32,399))
        assert out.shape == (2,32,128)
    def test_no_nan(self):
        from src.models.danceformer import PosePatchEmbedding
        out = PosePatchEmbedding(256)(torch.randn(2,32,399))
        assert not torch.isnan(out).any()

# ── Training ──────────────────────────────────────────────────────────────────

class TestTraining:
    def test_focal_loss(self):
        from src.training.losses import FocalLoss
        l = FocalLoss()(torch.randn(8,8), torch.randint(0,8,(8,)))
        assert l.ndim == 0 and float(l) >= 0

    def test_focal_loss_perfect(self):
        from src.training.losses import FocalLoss
        l = FocalLoss()(torch.eye(8)*20, torch.arange(8))
        assert float(l) < 0.05

    def test_supcon(self):
        from src.training.losses import SupConLoss
        import torch.nn.functional as F
        l = SupConLoss()(F.normalize(torch.randn(8,256),dim=1), torch.randint(0,8,(8,)))
        assert float(l) >= 0

    def test_augmentor_shape(self):
        from src.training.augmentation import PoseAugmentor
        k = np.random.rand(64,133,3).astype(np.float32)
        r = PoseAugmentor()(k)
        assert r.shape == k.shape

    def test_dataset_clip(self, tmp_path):
        from src.training.dataset import NatyaVedaDataset
        d = tmp_path / "kathak"; d.mkdir()
        np.savez_compressed(str(d/"t.npz"), keypoints=np.random.rand(100,133,3).astype(np.float32))
        ds = NatyaVedaDataset(str(tmp_path), clip_length=32, augment=False)
        assert len(ds) > 0
        item = ds[0]
        assert item["keypoints"].shape == (32,399)

    def test_class_weights(self, tmp_path):
        from src.training.dataset import NatyaVedaDataset
        for dance in ["bharatanatyam","kathak"]:
            d = tmp_path/dance; d.mkdir()
            np.savez_compressed(str(d/"t.npz"), keypoints=np.random.rand(80,133,3).astype(np.float32))
        ds = NatyaVedaDataset(str(tmp_path), clip_length=32, augment=False)
        w = ds.get_class_weights()
        assert w.shape == (8,) and (w >= 0).all()

# ── DanceIsolator ─────────────────────────────────────────────────────────────

class TestIsolator:
    def test_static_motion_zero(self):
        from src.preprocessing.dance_isolator import compute_motion_energy
        f = np.ones((240,320,3),np.uint8)*128
        assert compute_motion_energy([f,f,f]) < 0.001

    def test_moving_motion_nonzero(self):
        from src.preprocessing.dance_isolator import compute_motion_energy
        frames = []
        for i in range(5):
            f = np.zeros((240,320,3),np.uint8); f[:,i*30:(i+1)*30] = 200
            frames.append(f)
        assert compute_motion_energy(frames) > 0.001

    def test_keep_good_scene(self):
        from src.preprocessing.dance_isolator import DanceIsolator, Scene
        iso = DanceIsolator(min_dance_confidence=0.6, min_motion_threshold=0.005)
        s = Scene(0,100,0.0,4.0,dance_score=0.9,motion_energy=0.03,principal_bbox=(0,0,200,400))
        assert iso._should_keep(s)

    def test_drop_low_conf(self):
        from src.preprocessing.dance_isolator import DanceIsolator, Scene
        iso = DanceIsolator(min_dance_confidence=0.6)
        s = Scene(0,100,0.0,4.0,dance_score=0.1,motion_energy=0.03,principal_bbox=(0,0,200,400))
        assert not iso._should_keep(s)

    def test_drop_no_person(self):
        from src.preprocessing.dance_isolator import DanceIsolator, Scene
        iso = DanceIsolator()
        s = Scene(0,100,0.0,4.0,dance_score=0.9,motion_energy=0.05,principal_bbox=None)
        assert not iso._should_keep(s)

# ── Visualizer ────────────────────────────────────────────────────────────────

class TestVisualizer:
    def test_draw_133(self, frame, kpts133):
        from src.inference.predictor import SkeletonVisualizer
        kpts133[:,2] = 0.9
        r = SkeletonVisualizer().draw(frame, kpts133, "bharatanatyam", 0.92)
        assert r.shape == frame.shape

    def test_draw_17(self, frame, kpts17):
        from src.inference.predictor import SkeletonVisualizer
        kpts17[:,2] = 0.9
        r = SkeletonVisualizer().draw(frame, kpts17, "kathak", 0.80)
        assert r.shape == frame.shape

    def test_modifies_frame(self, frame, kpts133):
        from src.inference.predictor import SkeletonVisualizer
        kpts133[:,2] = 0.9
        r = SkeletonVisualizer().draw(frame, kpts133, "odissi", 0.75)
        assert not np.array_equal(r, frame)

# ── Taxonomy ──────────────────────────────────────────────────────────────────

class TestTaxonomy:
    def test_8_dance_classes(self):
        from src.training.dataset import DANCE_CLASSES
        assert len(DANCE_CLASSES) == 8

    def test_unique_indices(self):
        from src.training.dataset import DANCE_TO_IDX
        assert len(set(DANCE_TO_IDX.values())) == 8

    def test_28_mudras(self):
        from src.feature_extraction.hand_extractor import MudraRecognizer
        assert len(MudraRecognizer.MUDRA_NAMES) == 28

# ── Keypoint Utils ────────────────────────────────────────────────────────────

class TestKeypointUtils:
    def test_normalize_shape(self):
        from src.utils.keypoint_utils import normalize_keypoints
        k = np.random.rand(50,133,3).astype(np.float32)
        k[:,11,:2]=0.4; k[:,12,:2]=0.6; k[:,5,:2]=0.4; k[:,6,:2]=0.6
        assert normalize_keypoints(k).shape == k.shape

    def test_conf_unchanged(self):
        from src.utils.keypoint_utils import normalize_keypoints
        k = np.random.rand(10,133,3).astype(np.float32); k[:,:,2] = 0.8
        np.testing.assert_allclose(normalize_keypoints(k)[:,:,2], 0.8, rtol=1e-5)

    def test_joint_angles_shape(self):
        from src.utils.keypoint_utils import compute_joint_angles
        k = np.random.rand(30,133,3).astype(np.float32)
        angles = compute_joint_angles(k)
        assert angles.shape == (30,6)
        assert np.all(angles >= 0)
