"""Dataset utilities for PCB defect experiments."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_data_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_class_names(data_yaml: Dict) -> List[str]:
    names = data_yaml.get("names")
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names.keys())]
    if isinstance(names, list):
        return names
    raise ValueError("data.yaml missing 'names' field for class definitions.")


def assert_class_consistency(data_yaml_paths: List[Path]) -> List[str]:
    class_sets = []
    for path in data_yaml_paths:
        data_yaml = load_data_yaml(path)
        class_sets.append(extract_class_names(data_yaml))

    reference = class_sets[0]
    for idx, names in enumerate(class_sets[1:], start=1):
        if names != reference:
            raise ValueError(
                "Class mismatch detected between datasets. "
                f"Dataset index {idx} has {names}, expected {reference}."
            )
    return reference


def build_train_transforms(image_size: int, use_mosaic: bool) -> A.Compose:
    transform_list = [
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=(90, 270), p=0.3),
        A.RandomCrop(height=image_size, width=image_size, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.3),
        A.Resize(image_size, image_size),
        A.Normalize(),
    ]
    if use_mosaic:
        transform_list.insert(0, A.Resize(image_size, image_size))
    return A.Compose(transform_list)


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(),
        ]
    )


class PCBClassificationDataset(Dataset):
    """Classification dataset built from YOLO-style labels."""

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        class_names: List[str],
        transform: A.Compose,
        use_mosaic: bool = False,
        mosaic_prob: float = 0.3,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.transform = transform
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.images = sorted(
            [p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        )

    def __len__(self) -> int:
        return len(self.images)

    def _load_label(self, label_path: Path) -> np.ndarray:
        num_classes = len(self.class_names)
        multi_hot = np.zeros(num_classes, dtype=np.float32)
        if not label_path.exists():
            return multi_hot
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                class_idx = int(parts[0])
                if class_idx < num_classes:
                    multi_hot[class_idx] = 1.0
        return multi_hot

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.images[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        image = self._read_image(image_path)
        label = self._load_label(label_path)
        return image, label

    def _apply_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = [idx] + random.sample(range(len(self.images)), 3)
        images = []
        labels = []
        for i in indices:
            image, label = self._load_sample(i)
            images.append(image)
            labels.append(label)
        mosaic_image = np.zeros_like(images[0])
        h, w, _ = mosaic_image.shape
        h_half, w_half = h // 2, w // 2
        mosaic_image[:h_half, :w_half] = cv2.resize(images[0], (w_half, h_half))
        mosaic_image[:h_half, w_half:] = cv2.resize(images[1], (w_half, h_half))
        mosaic_image[h_half:, :w_half] = cv2.resize(images[2], (w_half, h_half))
        mosaic_image[h_half:, w_half:] = cv2.resize(images[3], (w_half, h_half))
        combined_label = np.clip(np.sum(labels, axis=0), 0, 1)
        return mosaic_image, combined_label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_mosaic and random.random() < self.mosaic_prob:
            image, label = self._apply_mosaic(idx)
        else:
            image, label = self._load_sample(idx)
        transformed = self.transform(image=image)
        image_tensor = torch.tensor(transformed["image"]).permute(2, 0, 1).float()
        label_tensor = torch.tensor(label).float()
        return image_tensor, label_tensor
