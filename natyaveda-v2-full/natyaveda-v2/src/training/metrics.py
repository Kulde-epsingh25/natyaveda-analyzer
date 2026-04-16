"""NatyaVeda — Training Metrics"""
from __future__ import annotations
import numpy as np

DANCE_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "kathakali",
]


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )
    return {
        "accuracy":   accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision":   precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":      recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class":   classification_report(
            y_true, y_pred, target_names=DANCE_CLASSES,
            output_dict=True, zero_division=0
        ),
    }


def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Return fraction where true label is in top-k predictions."""
    top_k = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(labels[i] in top_k[i] for i in range(len(labels)))
    return correct / max(len(labels), 1)
