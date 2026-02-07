"""Baseline model utilities (VGG16, ResNet50, YOLOv10)."""
from __future__ import annotations

from typing import Any

import torch

from pcb_supernet.models import TransferLearningClassifier


def build_vgg16(num_classes: int) -> TransferLearningClassifier:
    return TransferLearningClassifier("vgg16", num_classes=num_classes)


def build_resnet50(num_classes: int) -> TransferLearningClassifier:
    return TransferLearningClassifier("resnet50", num_classes=num_classes)


def build_yolov10(model_path: str | None = None) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLOv10 baseline requires the 'ultralytics' package. "
            "Install it before using YOLOv10 baselines."
        ) from exc
    return YOLO(model_path or "yolov10n.pt")


def yolo_inference(model: Any, image: str) -> Any:
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model(image)
