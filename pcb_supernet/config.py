"""Configuration objects for PCB defect experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    root_dirs: List[Path]
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    use_mosaic: bool = True
    mosaic_prob: float = 0.3
    class_names: Optional[List[str]] = None


@dataclass
class ModelConfig:
    num_classes: int
    dropout: float = 0.5
    attention_heads: int = 4
    attention_dim: int = 256
    freeze_backbones: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    log_interval: int = 10
    mixed_precision: bool = False
    resume_checkpoint: Optional[Path] = None


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
