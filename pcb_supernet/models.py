"""Model definitions for PCB defect detection/classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torchvision import models


@dataclass
class BackboneOutput:
    features: torch.Tensor
    pooled: torch.Tensor


def _build_backbone(backbone_name: str, freeze: bool) -> Tuple[nn.Module, int]:
    if backbone_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = model.features
        out_channels = 512
        if freeze:
            for param in features.parameters():
                param.requires_grad = False
        return features, out_channels
    if backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        out_channels = 2048
        if freeze:
            for param in layers.parameters():
                param.requires_grad = False
        return layers, out_channels
    raise ValueError(f"Unsupported backbone: {backbone_name}")


class PCBMultiHeadFusion(nn.Module):
    """Late-fusion model combining VGG16 and ResNet50 features."""

    def __init__(
        self,
        num_classes: int,
        attention_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.5,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()
        self.vgg_backbone, vgg_channels = _build_backbone("vgg16", freeze_backbones)
        self.res_backbone, res_channels = _build_backbone("resnet50", freeze_backbones)

        self.vgg_proj = nn.Conv2d(vgg_channels, attention_dim, kernel_size=1)
        self.res_proj = nn.Conv2d(res_channels, attention_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(attention_dim, attention_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, num_classes),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        vgg_feat = self.vgg_backbone(x)
        res_feat = self.res_backbone(x)
        vgg_proj = self.vgg_proj(vgg_feat)
        res_proj = self.res_proj(res_feat)
        vgg_tokens = vgg_proj.flatten(2).transpose(1, 2)
        res_tokens = res_proj.flatten(2).transpose(1, 2)
        tokens = torch.cat([vgg_tokens, res_tokens], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens)
        pooled = attended.mean(dim=1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self._encode(x)
        return self.classifier(fused)


class TransferLearningClassifier(nn.Module):
    """Wrapper for VGG16 or ResNet50 fine-tuning."""

    def __init__(self, backbone_name: str, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        backbone, out_channels = _build_backbone(backbone_name, freeze=False)
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.classifier(pooled)
