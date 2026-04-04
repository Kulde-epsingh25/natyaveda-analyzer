"""
NatyaVeda — Smart Device Manager
Auto-detects GPU, selects the best available device, picks the correct model
size for available VRAM, and provides graceful CPU fallback everywhere.

Usage:
    from src.utils.device import DeviceManager
    dm = DeviceManager()
    print(dm.device)          # 'cuda' or 'cpu'
    print(dm.pose_model_tier) # 'rtmw-x' | 'vitpose-huge' | 'vitpose-base' | 'movenet'
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    available: bool = False
    device_name: str = "CPU"
    vram_gb: float = 0.0
    cuda_version: str = ""
    device_index: int = 0
    compute_capability: tuple = (0, 0)


@dataclass
class DeviceConfig:
    """
    Selected device configuration with appropriate model tiers.
    Pose model tiers (in priority order):
      'rtmw-x'        — MMPose RTMW-x, 133 keypoints, GPU ≥8GB
      'vitpose-huge'  — HuggingFace VitPose-Plus-Huge, 133 keypoints, GPU ≥6GB
      'vitpose-large' — HuggingFace VitPose-Plus-Large, 133 keypoints, GPU ≥4GB
      'vitpose-base'  — HuggingFace VitPose-Plus-Base, 133 keypoints, GPU ≥2GB or CPU
      'movenet'       — TF Hub MoveNet Thunder, 17 keypoints, CPU only
    """
    device: str = "cpu"
    pose_model_tier: str = "movenet"
    vitpose_model_id: str = "usyd-community/vitpose-plus-base"
    batch_size_video: int = 4
    fp16_enabled: bool = False
    gpu_info: GPUInfo = field(default_factory=GPUInfo)

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def uses_mmpose(self) -> bool:
        return self.pose_model_tier == "rtmw-x"

    @property
    def uses_vitpose(self) -> bool:
        return self.pose_model_tier.startswith("vitpose")


class DeviceManager:
    """
    Singleton-style device manager. Call once at startup.

    Priority:
      1. GPU with VRAM ≥8GB   → RTMW-x (MMPose, 133 kpts) + fp16
      2. GPU with VRAM ≥6GB   → VitPose-Huge (HF, 133 kpts) + fp16
      3. GPU with VRAM ≥4GB   → VitPose-Large (HF, 133 kpts)
      4. GPU with VRAM ≥2GB   → VitPose-Base (HF, 133 kpts)
      5. CPU                  → VitPose-Base (HF, CPU mode) + MoveNet fallback
    """

    def __init__(self, force_cpu: bool = False, force_device: str | None = None) -> None:
        self.force_cpu = force_cpu
        self.force_device = force_device
        self.config = self._detect_and_configure()
        self._log_config()

    def _detect_gpu(self) -> GPUInfo:
        info = GPUInfo()
        if self.force_cpu:
            return info

        if not torch.cuda.is_available():
            # Try to diagnose WHY cuda is unavailable
            self._diagnose_cuda()
            return info

        info.available = True
        info.device_index = 0
        info.device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info.vram_gb = props.total_memory / (1024 ** 3)
        info.compute_capability = (props.major, props.minor)
        info.cuda_version = torch.version.cuda or ""
        return info

    def _diagnose_cuda(self) -> None:
        """Log helpful diagnostics when CUDA is not available."""
        # Check if nvidia-smi sees a GPU (GPU present but not passed to container)
        r = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            logger.warning(
                "⚠️  GPU detected by nvidia-smi but NOT by PyTorch CUDA.\n"
                "    This usually means the container was started WITHOUT --gpus flag.\n"
                "    FIX: docker run --gpus all ...\n"
                "    OR:  In VS Code devcontainer.json add: \"runArgs\": [\"--gpus\", \"all\"]\n"
                "    GPU found by nvidia-smi:\n%s", r.stdout.strip()
            )
        else:
            logger.info("No GPU detected — running in CPU mode.")

    def _detect_and_configure(self) -> DeviceConfig:
        gpu = self._detect_gpu()
        cfg = DeviceConfig(gpu_info=gpu)

        if self.force_device:
            cfg.device = self.force_device
        elif gpu.available:
            cfg.device = "cuda"
        else:
            cfg.device = "cpu"

        # Pick pose model tier based on available VRAM
        if gpu.available:
            vram = gpu.vram_gb
            if vram >= 8.0:
                cfg.pose_model_tier  = "rtmw-x"
                cfg.vitpose_model_id = "usyd-community/vitpose-plus-huge"
                cfg.batch_size_video = 16
                cfg.fp16_enabled     = True
            elif vram >= 6.0:
                cfg.pose_model_tier  = "vitpose-huge"
                cfg.vitpose_model_id = "usyd-community/vitpose-plus-huge"
                cfg.batch_size_video = 8
                cfg.fp16_enabled     = True
            elif vram >= 4.0:
                cfg.pose_model_tier  = "vitpose-large"
                cfg.vitpose_model_id = "usyd-community/vitpose-plus-large"
                cfg.batch_size_video = 4
                cfg.fp16_enabled     = False
            else:
                cfg.pose_model_tier  = "vitpose-base"
                cfg.vitpose_model_id = "usyd-community/vitpose-plus-base"
                cfg.batch_size_video = 2
                cfg.fp16_enabled     = False
        else:
            # CPU mode — use lightweight VitPose-Base via HuggingFace
            cfg.pose_model_tier  = "vitpose-base"
            cfg.vitpose_model_id = "usyd-community/vitpose-plus-base"
            cfg.batch_size_video = 2
            cfg.fp16_enabled     = False

        return cfg

    def _log_config(self) -> None:
        gpu = self.config.gpu_info
        if gpu.available:
            logger.info(
                "Device: %s | GPU: %s (%.1f GB VRAM) | CUDA: %s | "
                "Pose tier: %s | fp16: %s",
                self.config.device, gpu.device_name, gpu.vram_gb,
                gpu.cuda_version, self.config.pose_model_tier,
                self.config.fp16_enabled
            )
        else:
            logger.info(
                "Device: CPU | Pose tier: %s | Model: %s",
                self.config.pose_model_tier, self.config.vitpose_model_id
            )

    # ── Public API ──────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def pose_model_tier(self) -> str:
        return self.config.pose_model_tier

    @property
    def vitpose_model_id(self) -> str:
        return self.config.vitpose_model_id

    @property
    def fp16(self) -> bool:
        return self.config.fp16_enabled

    def move_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move a PyTorch model to the configured device, with optional fp16."""
        model = model.to(self.device)
        if self.fp16 and self.config.is_cuda:
            model = model.half()
        return model

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the configured device."""
        return tensor.to(self.device)

    def autocast_context(self):
        """Return an autocast context for mixed precision inference."""
        if self.fp16 and self.config.is_cuda:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        import contextlib
        return contextlib.nullcontext()

    def summary(self) -> str:
        gpu = self.config.gpu_info
        if gpu.available:
            return (
                f"Device: CUDA ({gpu.device_name}, {gpu.vram_gb:.1f}GB) | "
                f"Pose: {self.pose_model_tier} | fp16: {self.fp16}"
            )
        return f"Device: CPU | Pose: {self.pose_model_tier} ({self.vitpose_model_id})"


# ─────────────────────────────────────────────────────────────────────────────
# GPU fix instructions helper
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_fix_instructions() -> None:
    """Print instructions when GPU is physically present but not accessible."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  GPU detected by system but NOT accessible to PyTorch        ║
║  This is a container configuration issue — fix:             ║
╠══════════════════════════════════════════════════════════════╣
║  DOCKER:                                                     ║
║    docker run --gpus all -v $(pwd):/app your_image           ║
║                                                              ║
║  VS CODE DEVCONTAINER:                                       ║
║    In .devcontainer/devcontainer.json add:                  ║
║      "runArgs": ["--gpus", "all"]                           ║
║    Then: Ctrl+Shift+P → "Rebuild Container"                 ║
║                                                              ║
║  DOCKER COMPOSE:                                             ║
║    In docker-compose.yml under your service:                ║
║      deploy:                                                 ║
║        resources:                                            ║
║          reservations:                                       ║
║            devices:                                          ║
║              - driver: nvidia                                ║
║                count: all                                    ║
║                capabilities: [gpu]                           ║
╚══════════════════════════════════════════════════════════════╝
""")
