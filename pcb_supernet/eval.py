"""Evaluation utilities for PCB defect models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from sklearn.metrics import f1_score


@dataclass
class BenchmarkResult:
    accuracy: float
    f1: float
    inference_ms: float
    memory_mb: float


def _sigmoid_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, float]:
    model.eval()
    correct = torch.zeros(len(class_names), device=device)
    total = torch.zeros(len(class_names), device=device)
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = _sigmoid_to_binary(logits)
            correct += (preds == targets).sum(dim=0)
            total += torch.ones_like(targets).sum(dim=0)
            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
    per_class_accuracy = (correct / total.clamp(min=1)).cpu().tolist()
    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        **{name: acc for name, acc in zip(class_names, per_class_accuracy)},
        "f1_macro": f1,
    }
