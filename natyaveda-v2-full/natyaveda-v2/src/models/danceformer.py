"""
NatyaVeda — DanceFormer
Transformer-based classifier for Indian classical dance recognition.

Architecture:
  1. Pose Patch Embedding: project 399-dim keypoint vector → 256-dim
  2. Temporal Transformer Encoder: 8 layers × 8 heads
  3. [CLS] Token Pooling
  4. Dance Head: 8-class dance form classification
  5. Mudra Head: 28-class hasta gesture classification (auxiliary)

Input:
  - Keypoints: [B, T, 133, 3] or pre-flattened [B, T, 399]
  - Optionally: VideoMAE tokens [B, T', 1024] fused via cross-attention

Output:
  - dance_logits: [B, 8]
  - mudra_logits: [B, T, 28]
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable absolute positional embeddings for temporal sequences."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pos_embed(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Pose Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PosePatchEmbedding(nn.Module):
    """
    Embeds per-frame keypoint vectors into a latent space.

    The spatial structure of keypoints is preserved through:
    - Group-aware linear projections (body, hands, face, feet projected separately)
    - Concatenation and final linear projection to embed_dim
    """

    BODY_DIM = 17 * 3      # 51
    FOOT_DIM = 6 * 3       # 18
    FACE_DIM = 68 * 3      # 204
    LEFT_HAND_DIM = 21 * 3 # 63
    RIGHT_HAND_DIM = 21 * 3 # 63
    TOTAL_DIM = 399        # full RTMW feature per frame

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Group-aware sub-projections
        self.body_proj = nn.Linear(self.BODY_DIM, 64)
        self.foot_proj = nn.Linear(self.FOOT_DIM, 16)
        self.face_proj = nn.Linear(self.FACE_DIM, 64)
        self.lhand_proj = nn.Linear(self.LEFT_HAND_DIM, 48)
        self.rhand_proj = nn.Linear(self.RIGHT_HAND_DIM, 48)

        # 64+16+64+48+48 = 240 → embed_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(240, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Velocity + acceleration channels added as residual
        self.vel_proj = nn.Linear(self.TOTAL_DIM, embed_dim)
        self.acc_proj = nn.Linear(self.TOTAL_DIM, embed_dim)

    def forward(
        self,
        keypoints: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        accelerations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        keypoints : [B, T, 399]   — per-frame flattened keypoints
        velocities : [B, T, 399]  — temporal first differences
        accelerations : [B, T, 399] — temporal second differences

        Returns
        -------
        [B, T, embed_dim]
        """
        B, T, _ = keypoints.shape

        # Split by group
        body   = keypoints[..., :51]
        foot   = keypoints[..., 51:69]
        face   = keypoints[..., 69:273]
        lhand  = keypoints[..., 273:336]
        rhand  = keypoints[..., 336:399]

        # Group sub-projections
        b_feat = F.gelu(self.body_proj(body))    # [B, T, 64]
        f_feat = F.gelu(self.foot_proj(foot))    # [B, T, 16]
        fc_feat = F.gelu(self.face_proj(face))   # [B, T, 64]
        lh_feat = F.gelu(self.lhand_proj(lhand)) # [B, T, 48]
        rh_feat = F.gelu(self.rhand_proj(rhand)) # [B, T, 48]

        # Concatenate → [B, T, 240]
        fused = torch.cat([b_feat, f_feat, fc_feat, lh_feat, rh_feat], dim=-1)
        out = self.fusion_proj(fused)  # [B, T, embed_dim]

        # Add velocity and acceleration residuals
        if velocities is not None:
            out = out + self.vel_proj(velocities) * 0.5
        if accelerations is not None:
            out = out + self.acc_proj(accelerations) * 0.25

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTransformerEncoder(nn.Module):
    """
    Standard Transformer encoder operating over time dimension.
    Each frame is a token; self-attention captures temporal dependencies.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        norm_first: bool = True,    # Pre-LN (more stable)
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, D]
        src_key_padding_mask: [B, T] — True for positions to ignore
        Returns: [B, T, D]
        """
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(out)


# ─────────────────────────────────────────────────────────────────────────────
# VideoMAE Token Fusion (optional)
# ─────────────────────────────────────────────────────────────────────────────

class VideoMAEFusion(nn.Module):
    """
    Cross-attention fusion of VideoMAE-v2 global tokens with pose sequence.
    VideoMAE tokens provide holistic appearance context beyond skeleton.
    """

    def __init__(self, pose_dim: int = 256, video_dim: int = 1024) -> None:
        super().__init__()
        self.video_proj = nn.Linear(video_dim, pose_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=pose_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(pose_dim)

    def forward(
        self,
        pose_tokens: torch.Tensor,    # [B, T, pose_dim]
        video_tokens: torch.Tensor,   # [B, T', video_dim]
    ) -> torch.Tensor:
        v = self.video_proj(video_tokens)          # [B, T', pose_dim]
        attn_out, _ = self.cross_attn(
            query=pose_tokens, key=v, value=v
        )
        return self.norm(pose_tokens + attn_out)   # [B, T, pose_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Classification Heads
# ─────────────────────────────────────────────────────────────────────────────

class DanceHead(nn.Module):
    """8-way Indian classical dance form classifier."""

    DANCE_CLASSES = [
        "bharatanatyam", "kathak", "odissi", "kuchipudi",
        "manipuri", "mohiniyattam", "sattriya", "kathakali",
    ]

    def __init__(self, in_dim: int = 256, num_classes: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] pooled representation → [B, num_classes] logits"""
        return self.head(x)


class MudraHead(nn.Module):
    """28-class per-frame mudra (hasta) classifier (auxiliary task)."""

    def __init__(
        self, in_dim: int = 256, num_classes: int = 28, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, num_classes] per-frame mudra logits"""
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main DanceFormer
# ─────────────────────────────────────────────────────────────────────────────

class DanceFormer(nn.Module):
    """
    NatyaVeda DanceFormer — Full Indian classical dance recognition model.

    Parameters
    ----------
    embed_dim : int
        Latent dimension (default: 256)
    num_transformer_layers : int
        Number of Transformer encoder layers (default: 8)
    num_heads : int
        Number of attention heads (default: 8)
    ff_dim : int
        Feed-forward intermediate dimension (default: 1024)
    max_seq_len : int
        Maximum input sequence length in frames (default: 128)
    num_dance_classes : int
        Number of dance form classes (default: 8)
    num_mudra_classes : int
        Number of mudra/hasta classes (default: 28)
    use_videomae_fusion : bool
        Whether to use VideoMAE token cross-attention (default: True)
    videomae_dim : int
        Dimension of VideoMAE feature vectors (default: 1024)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_transformer_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        num_dance_classes: int = 8,
        num_mudra_classes: int = 28,
        use_videomae_fusion: bool = True,
        videomae_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.use_videomae_fusion = use_videomae_fusion

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Pose embedding
        self.pose_embed = PosePatchEmbedding(embed_dim=embed_dim, dropout=dropout)

        # Positional encoding
        self.pos_enc = LearnablePositionalEncoding(max_len=max_seq_len + 1, d_model=embed_dim)

        # Temporal Transformer
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=embed_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Optional VideoMAE cross-attention
        if use_videomae_fusion:
            self.videomae_fusion = VideoMAEFusion(pose_dim=embed_dim, video_dim=videomae_dim)
        else:
            self.videomae_fusion = None

        # Classification heads
        self.dance_head = DanceHead(embed_dim, num_dance_classes, dropout)
        self.mudra_head = MudraHead(embed_dim, num_mudra_classes, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        keypoints: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        accelerations: Optional[torch.Tensor] = None,
        video_tokens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        keypoints : [B, T, 399]
        velocities : [B, T, 399] (optional)
        accelerations : [B, T, 399] (optional)
        video_tokens : [B, T', 1024] VideoMAE tokens (optional)
        padding_mask : [B, T] bool tensor, True = pad (optional)

        Returns
        -------
        dict with:
            dance_logits : [B, 8]
            mudra_logits : [B, T, 28]
            features     : [B, D] — [CLS] embedding
        """
        B, T, _ = keypoints.shape

        # Pose embedding [B, T, D]
        x = self.pose_embed(keypoints, velocities, accelerations)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
        x = torch.cat([cls, x], dim=1)            # [B, T+1, D]

        # Positional encoding
        x = self.pos_enc(x)

        # Extend padding mask to account for [CLS]
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Temporal Transformer
        x = self.temporal_encoder(x, src_key_padding_mask=padding_mask)  # [B, T+1, D]

        # Optional VideoMAE fusion (on frame tokens, not CLS)
        if self.use_videomae_fusion and self.videomae_fusion is not None and video_tokens is not None:
            x[:, 1:] = self.videomae_fusion(x[:, 1:], video_tokens)

        # [CLS] token → dance classification
        cls_out = x[:, 0]                         # [B, D]
        dance_logits = self.dance_head(cls_out)   # [B, 8]

        # Per-frame tokens → mudra classification
        frame_tokens = x[:, 1:]                   # [B, T, D]
        mudra_logits = self.mudra_head(frame_tokens)  # [B, T, 28]

        return {
            "dance_logits": dance_logits,
            "mudra_logits": mudra_logits,
            "features": cls_out,
        }

    @torch.no_grad()
    def predict(
        self,
        keypoints: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        accelerations: Optional[torch.Tensor] = None,
    ) -> dict[str, any]:
        """Convenience method for inference with softmax probabilities."""
        self.eval()
        out = self.forward(keypoints, velocities, accelerations)

        dance_probs = F.softmax(out["dance_logits"], dim=-1)
        dance_pred = int(dance_probs.argmax(-1).item())

        mudra_probs = F.softmax(out["mudra_logits"], dim=-1)
        mudra_pred = mudra_probs.argmax(-1)  # [B, T]

        return {
            "dance_class": DanceHead.DANCE_CLASSES[dance_pred],
            "dance_confidence": float(dance_probs[0, dance_pred]),
            "dance_probabilities": dance_probs[0].cpu().numpy().tolist(),
            "dance_classes": DanceHead.DANCE_CLASSES,
            "mudra_predictions": mudra_pred[0].cpu().numpy().tolist(),
            "features": out["features"],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict) -> "DanceFormer":
        """Instantiate from config dict (e.g., from YAML)."""
        m = config.get("model", config)
        return cls(
            embed_dim=m.get("pose_embed_dim", 256),
            num_transformer_layers=m.get("num_transformer_layers", 8),
            num_heads=m.get("num_attention_heads", 8),
            ff_dim=m.get("ff_dim", 1024),
            dropout=m.get("transformer_dropout", 0.1),
            max_seq_len=m.get("max_sequence_length", 128),
            num_dance_classes=m.get("num_dance_classes", 8),
            num_mudra_classes=m.get("num_mudra_classes", 28),
            use_videomae_fusion=m.get("videomae_fusion", True),
            videomae_dim=m.get("videomae_proj_dim", 1024),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "embed_dim": self.embed_dim,
                "use_videomae_fusion": self.use_videomae_fusion,
            },
        }, str(path))
        logger.info(f"DanceFormer saved: {path}")

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "DanceFormer":
        path = Path(path)
        ckpt = torch.load(str(path), map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        logger.info(f"DanceFormer loaded: {path} ({model.count_parameters():,} params)")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Model variants
# ─────────────────────────────────────────────────────────────────────────────

def danceformer_small() -> DanceFormer:
    """Lightweight model for fast inference / edge deployment."""
    return DanceFormer(embed_dim=128, num_transformer_layers=4, num_heads=4, ff_dim=512)


def danceformer_base() -> DanceFormer:
    """Balanced model — good accuracy, moderate compute."""
    return DanceFormer(embed_dim=256, num_transformer_layers=6, num_heads=8, ff_dim=1024)


def danceformer_large() -> DanceFormer:
    """Full model — highest accuracy (default for training)."""
    return DanceFormer(embed_dim=256, num_transformer_layers=8, num_heads=8, ff_dim=1024)


MODEL_REGISTRY = {
    "danceformer-small": danceformer_small,
    "danceformer-base": danceformer_base,
    "danceformer-large": danceformer_large,
}
