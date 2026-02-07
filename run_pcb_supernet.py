"""Orchestrate PCB SuperNet experiments."""
from __future__ import annotations

from pathlib import Path

import kagglehub

from pcb_supernet.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from pcb_supernet.main import parse_class_names, run_experiment


def download_datasets() -> list[Path]:
    hripcb_path = Path(kagglehub.dataset_download("ibrahimgergesmoussa/hripcb-update"))
    deeppcb_path = Path(kagglehub.dataset_download("dangdinh123/deeppcb"))
    deeppcb_alt_path = Path(kagglehub.dataset_download("kcnttngotruongan/deeppcb"))
    return [hripcb_path, deeppcb_path, deeppcb_alt_path]


def main() -> None:
    dataset_roots = download_datasets()
    class_names = parse_class_names(dataset_roots)

    data_cfg = DataConfig(root_dirs=dataset_roots, class_names=class_names)
    model_cfg = ModelConfig(num_classes=len(class_names))
    train_cfg = TrainingConfig()
    exp_cfg = ExperimentConfig(data=data_cfg, model=model_cfg, training=train_cfg)

    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
