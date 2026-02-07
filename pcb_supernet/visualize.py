"""Visualization helpers for PCB defect models."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as F

from pcb_supernet.data import build_eval_transforms


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, class_idx: int) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM requires a forward and backward pass.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-6)
        return cam.cpu().numpy()


def test_and_visualize(
    model: torch.nn.Module,
    test_dir: Path,
    class_names: List[str],
    device: torch.device,
    image_size: int = 224,
    samples: int = 5,
) -> None:
    """Visualize predictions with Grad-CAM for random samples."""
    model.eval()
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {test_dir}")
    chosen = random.sample(images, min(samples, len(images)))
    transform = build_eval_transforms(image_size)

    fig, axes = plt.subplots(len(chosen), 2, figsize=(8, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")
    grad_cam = GradCAM(model, target_layer)

    for idx, image_path in enumerate(chosen):
        image = plt.imread(image_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        resized = transform(image=image)["image"]
        tensor = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        tensor.requires_grad = True
        logits = model(tensor)
        pred_idx = torch.sigmoid(logits).argmax(dim=1).item()
        model.zero_grad(set_to_none=True)
        logits[:, pred_idx].backward(retain_graph=True)
        cam = grad_cam.generate(pred_idx)
        cam = np.uint8(255 * cam)
        cam = np.stack([cam] * 3, axis=-1)
        cam = F.resize(torch.tensor(cam).permute(2, 0, 1), [image.shape[0], image.shape[1]])
        cam = cam.permute(1, 2, 0).numpy()
        overlay = (0.6 * image + 0.4 * cam).astype(np.uint8)

        axes[idx, 0].imshow(image)
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title(f"Original: {image_path.name}")

        axes[idx, 1].imshow(overlay)
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title(f"Predicted: {class_names[pred_idx]}")

    plt.tight_layout()
    plt.show()
