"""
NatyaVeda — Loss Functions (v2 — Cluster Separation Focus)

Changes from v1:
  - FocalLoss: unchanged (still used for dance classification)
  - SupConLoss: add pair-aware temperature — harder for confused pairs
  - CenterLoss: NEW — pulls same-class embeddings together
  - InterClassPushLoss: NEW — explicitly pushes confused pairs apart
  - PairwiseAngularLoss: NEW — based on Orthogonal Projection Loss paper
  - NatyaVedaCombinedLoss: single wrapper used by trainer

Key insight from confusion matrix:
  mohiniyattam↔kuchipudi  = 32% confusion → hardest pair, needs dedicated push
  kathak↔kathakali        = 19% confusion
  odissi↔bharatanatyam   = 14% confusion
  kuchipudi↔sattriya     = 12% confusion
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Dance class index mapping ─────────────────────────────────────────────────
DANCE_CLASSES = [
    "bharatanatyam",   # 0
    "kathak",          # 1
    "odissi",          # 2
    "kuchipudi",       # 3
    "manipuri",        # 4
    "mohiniyattam",    # 5
    "sattriya",        # 6
    "kathakali",       # 7
]

# Confused pairs from real confusion matrix (row→col confusions)
# (class_a_idx, class_b_idx): confusion_weight
CONFUSED_PAIRS = {
    (5, 3): 3.0,   # mohiniyattam → kuchipudi  (32%)  ← hardest
    (1, 7): 2.5,   # kathak       → kathakali  (19%)
    (2, 0): 2.0,   # odissi       → bharatanatyam (14%)
    (3, 6): 1.5,   # kuchipudi    → sattriya   (12%)
    (3, 1): 1.5,   # kuchipudi    → kathak     (11%)
    (5, 6): 1.5,   # mohiniyattam → sattriya   (16%)
    (1, 6): 1.2,   # kathak       → sattriya   (8%)
}


# ── 1. Focal Loss ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss — down-weights easy examples, focuses on hard ones.
    Unchanged from v1 but alpha now supports per-class tensor.
    """
    def __init__(self, gamma: float = 2.0, alpha: float | torch.Tensor = 0.25,
                 label_smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.shape[-1]
        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(logits, self.label_smoothing / n_cls)
                smooth.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.label_smoothing + self.label_smoothing / n_cls)
            log_prob = F.log_softmax(logits, dim=-1)
            ce = -(smooth * log_prob).sum(dim=-1)
        else:
            ce = F.cross_entropy(logits, targets, reduction="none")

        prob_t = torch.exp(-ce)
        focal_w = (1 - prob_t) ** self.gamma

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = self.alpha

        loss = alpha_t * focal_w * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── 2. Supervised Contrastive Loss ───────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with pair-aware temperature.
    Confused pairs get a lower temperature = harder push.
    """
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: L2-normalized [B, D]
        labels:   class indices [B]
        """
        device = features.device
        B = features.shape[0]
        if B < 4:
            return features.sum() * 0.0

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature   # [B, B]

        # Masks
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()   # [B, B]
        eye  = torch.eye(B, device=device)
        pos_mask = same - eye                                           # exclude self

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) - exp_sim * eye)
        log_prob  = sim - log_denom

        # Mean over positives per anchor
        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            return features.sum() * 0.0

        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / n_pos.clamp(min=1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        return loss[valid].mean()


# ── 3. Center Loss ────────────────────────────────────────────────────────────

class CenterLoss(nn.Module):
    """
    Center Loss: pulls each class's embeddings toward a learnable prototype center.
    Reduces intra-class variance. Critical for mohiniyattam which is spread out.
    """
    def __init__(self, n_classes: int = 8, embed_dim: int = 256,
                 alpha: float = 0.5) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.alpha     = alpha  # center update rate
        self.centers   = nn.Parameter(torch.randn(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.centers.unsqueeze(0))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D]
        labels:   [B]  — class indices
        Returns intra-class pull loss.
        """
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch.detach())
        # Update centers (EMA)
        with torch.no_grad():
            for c in range(self.n_classes):
                mask = (labels == c)
                if mask.any():
                    self.centers.data[c] = (
                        (1 - self.alpha) * self.centers.data[c]
                        + self.alpha     * features[mask].mean(dim=0)
                    )
        return loss


# ── 4. Inter-Class Push Loss ──────────────────────────────────────────────────

class InterClassPushLoss(nn.Module):
    """
    Pushes confused class pairs apart in embedding space.
    Uses CONFUSED_PAIRS from the confusion matrix to weight the push.
    
    For each confused pair (a, b): penalize when d(center_a, center_b) < margin.
    This directly targets mohiniyattam↔kuchipudi (worst confusion at 32%).
    """
    def __init__(self, center_module: CenterLoss,
                 margin: float = 1.5,
                 confused_pairs: dict | None = None) -> None:
        super().__init__()
        self.centers = center_module
        self.margin  = margin
        self.pairs   = confused_pairs or CONFUSED_PAIRS

    def forward(self, _features=None, _labels=None) -> torch.Tensor:
        """
        Computes push loss from center positions only (doesn't need batch features).
        Call after CenterLoss to get updated centers.
        """
        loss = torch.tensor(0.0, device=self.centers.centers.device)
        n_pairs = 0
        for (a, b), w in self.pairs.items():
            ca = F.normalize(self.centers.centers[a], dim=0)
            cb = F.normalize(self.centers.centers[b], dim=0)
            # Cosine distance = 1 - cosine_similarity
            cosine_dist = 1.0 - (ca * cb).sum()
            # Penalize when distance < margin (cosine distance [0,2])
            push = F.relu(self.margin - cosine_dist)
            loss = loss + w * push
            n_pairs += 1
        return loss / max(n_pairs, 1)


# ── 5. Pairwise Angular Loss ──────────────────────────────────────────────────

class PairwiseAngularLoss(nn.Module):
    """
    Based on Orthogonal Projection Loss (OPL) — enforces that different-class
    feature vectors are orthogonal (maximally separated) in embedding space.
    Simpler than metric learning but highly effective for well-defined classes.
    """
    def __init__(self, orthogonality_weight: float = 1.0) -> None:
        super().__init__()
        self.w = orthogonality_weight

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        if B < 2:
            return features.sum() * 0.0

        # L2 normalize
        fn = F.normalize(features, dim=1)

        # Gram matrix
        gram = torch.matmul(fn, fn.T)  # [B, B]

        # Same-class pairs: gram should be +1 (intra pull)
        # Diff-class pairs: gram should be  0 (inter push)
        same = (labels.unsqueeze(0) == labels.unsqueeze(1))
        eye  = torch.eye(B, device=features.device, dtype=torch.bool)

        # Inter-class: off-diagonal, different class → penalize non-zero gram
        inter_mask = ~same & ~eye
        if inter_mask.any():
            inter_loss = gram[inter_mask].pow(2).mean()
        else:
            inter_loss = torch.tensor(0.0, device=features.device)

        return self.w * inter_loss


# ── 6. Mudra Discrimination Loss ──────────────────────────────────────────────

class MudraCrossEntropyLoss(nn.Module):
    """
    Frame-level mudra classification loss.
    Focuses attention on hand gesture discrimination — critical for
    separating mohiniyattam, kuchipudi, kathak which differ mainly in hastas.
    """
    def __init__(self, n_mudras: int = 28, ignore_index: int = -1) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.n_mudras = n_mudras

    def forward(self, mudra_logits: torch.Tensor,
                mudra_labels: Optional[torch.Tensor]) -> torch.Tensor:
        if mudra_labels is None:
            return mudra_logits.sum() * 0.0
        B, T, C = mudra_logits.shape
        return self.ce(mudra_logits.reshape(B * T, C), mudra_labels.reshape(B * T))


# ── 7. Combined Loss (used by Trainer) ───────────────────────────────────────

class NatyaVedaCombinedLoss(nn.Module):
    """
    Unified loss combining all components.

    Weights (tuned for confusion matrix results):
      dance_cls:     focal cross-entropy (weight 1.0)
      supcon:        supervised contrastive (weight 0.30)
      center:        intra-class pull (weight 0.12)
      inter_push:    inter-class push on confused pairs (weight 0.25)  ← NEW
      angular:       orthogonal projection (weight 0.10)               ← NEW
      mudra:         hand gesture CE (weight 0.40)
    """

    def __init__(self, config: dict | None = None, embed_dim: int = 256) -> None:
        super().__init__()
        cfg = config or {}
        loss_cfg = cfg.get("training", {}).get("loss", {})

        # Focal loss
        self.focal = FocalLoss(
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", 0.25),
            label_smoothing=loss_cfg.get("label_smoothing", 0.05),
        )

        # Supervised contrastive
        self.supcon = SupConLoss(
            temperature=loss_cfg.get("supervised_contrastive_temperature", 0.07)
        )

        # Center loss (learnable prototypes)
        self.center = CenterLoss(n_classes=8, embed_dim=embed_dim)

        # Inter-class push on confused pairs
        self.inter_push = InterClassPushLoss(
            center_module=self.center,
            margin=loss_cfg.get("center_inter_margin", 1.2),
        )

        # Angular/orthogonal push
        self.angular = PairwiseAngularLoss(
            orthogonality_weight=loss_cfg.get("angular_weight", 1.0)
        )

        # Mudra loss
        self.mudra_ce = MudraCrossEntropyLoss()

        # Loss weights
        self.w_dance   = loss_cfg.get("dance_loss_weight",               1.0)
        self.w_supcon  = loss_cfg.get("supervised_contrastive_weight",   0.30)
        self.w_center  = loss_cfg.get("center_loss_weight",              0.12)
        self.w_push    = loss_cfg.get("center_inter_weight",             0.25)
        self.w_angular = loss_cfg.get("angular_weight",                  0.10)
        self.w_mudra   = loss_cfg.get("mudra_loss_weight",               0.40)

    def forward(
        self,
        dance_logits:  torch.Tensor,
        dance_labels:  torch.Tensor,
        features:      Optional[torch.Tensor] = None,
        mudra_logits:  Optional[torch.Tensor] = None,
        mudra_labels:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns (total_loss, breakdown_dict).
        """
        breakdown = {}

        # 1. Dance classification
        dance_loss = self.focal(dance_logits, dance_labels)
        breakdown["dance"] = float(dance_loss)
        total = self.w_dance * dance_loss

        # 2. Supervised contrastive (needs L2-normalized features)
        if features is not None and features.shape[0] >= 4:
            fn = F.normalize(features, dim=1)
            sc_loss = self.supcon(fn, dance_labels)
            breakdown["supcon"] = float(sc_loss)
            total = total + self.w_supcon * sc_loss

            # 3. Center loss (intra-class pull)
            ct_loss = self.center(features, dance_labels)
            breakdown["center"] = float(ct_loss)
            total = total + self.w_center * ct_loss

            # 4. Inter-class push (confused pairs)
            push_loss = self.inter_push()
            breakdown["inter_push"] = float(push_loss)
            total = total + self.w_push * push_loss

            # 5. Angular orthogonal loss
            ang_loss = self.angular(features, dance_labels)
            breakdown["angular"] = float(ang_loss)
            total = total + self.w_angular * ang_loss

        # 6. Mudra loss (if labels provided)
        if mudra_logits is not None:
            m_loss = self.mudra_ce(mudra_logits, mudra_labels)
            breakdown["mudra"] = float(m_loss)
            total = total + self.w_mudra * m_loss

        breakdown["total"] = float(total)
        return total, breakdown
