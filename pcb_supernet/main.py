"""Experiment entrypoint for PCB SuperNet comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from pcb_supernet.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from pcb_supernet.data import (
    PCBClassificationDataset,
    assert_class_consistency,
    build_eval_transforms,
    build_train_transforms,
)
from pcb_supernet.eval import evaluate_classifier
from pcb_supernet.models import PCBMultiHeadFusion, TransferLearningClassifier
from pcb_supernet.train import Trainer


def build_dataloaders(config: DataConfig):
    train_datasets = []
    val_datasets = []
    for root in config.root_dirs:
        train_dir = root / "train"
        val_dir = root / "val"
        train_datasets.append(
            PCBClassificationDataset(
                train_dir / "images",
                train_dir / "labels",
                config.class_names,
                build_train_transforms(config.image_size, config.use_mosaic),
                use_mosaic=config.use_mosaic,
                mosaic_prob=config.mosaic_prob,
            )
        )
        val_datasets.append(
            PCBClassificationDataset(
                val_dir / "images",
                val_dir / "labels",
                config.class_names,
                build_eval_transforms(config.image_size),
            )
        )

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def run_experiment(config: ExperimentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(config.data)

    model = PCBMultiHeadFusion(
        num_classes=config.model.num_classes,
        attention_dim=config.model.attention_dim,
        attention_heads=config.model.attention_heads,
        dropout=config.model.dropout,
        freeze_backbones=config.model.freeze_backbones,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.epochs
    )
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        device,
        mixed_precision=config.training.mixed_precision,
    )

    if config.training.resume_checkpoint:
        trainer.load_checkpoint(config.training.resume_checkpoint)

    for epoch in range(config.training.epochs):
        metrics = trainer.train_epoch(train_loader, log_interval=config.training.log_interval)
        print(f"Epoch {epoch + 1}: loss={metrics['loss']:.4f}")

    metrics = evaluate_classifier(model, val_loader, device, config.data.class_names)
    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    trainer.save_checkpoint(config.output_dir / "pcb_supernet.pt")


def build_transfer_baseline(backbone: str, num_classes: int) -> TransferLearningClassifier:
    return TransferLearningClassifier(backbone_name=backbone, num_classes=num_classes)


def parse_class_names(dataset_roots: List[Path]) -> List[str]:
    data_yaml_paths = [root / "data.yaml" for root in dataset_roots]
    return assert_class_consistency(data_yaml_paths)


def main() -> None:
    dataset_roots = [
        Path("/path/to/HRIPCB_UPDATE"),
        Path("/path/to/DeepPCB"),
        Path("/path/to/deeppcb"),
    ]
    class_names = parse_class_names(dataset_roots)
    data_cfg = DataConfig(root_dirs=dataset_roots, class_names=class_names)
    model_cfg = ModelConfig(num_classes=len(class_names))
    train_cfg = TrainingConfig()
    exp_cfg = ExperimentConfig(data=data_cfg, model=model_cfg, training=train_cfg)
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
