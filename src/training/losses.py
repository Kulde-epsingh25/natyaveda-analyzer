"""NatyaVeda — Loss Functions"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss — pulls same-class embeddings together."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, D] L2-normalized; labels: [B]
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return loss.mean()
