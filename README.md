# DEEP

## PCB SuperNet Overview

This repository contains a modular PyTorch implementation for PCB defect detection/classification
with the proposed **PCB-MultiHead Fusion** super-model and baselines (VGG16, ResNet50, YOLOv10).
The codebase is structured to keep configuration, dataset, model, training, evaluation, and
visualization components isolated for reproducible research.

### Key Modules

- `pcb_supernet/config.py`: Experiment configuration objects.
- `pcb_supernet/data.py`: Dataset parsing, augmentation, and class consistency checks.
- `pcb_supernet/models.py`: VGG16/ResNet50 transfer learning and the multi-head fusion model.
- `pcb_supernet/train.py`: Training loop with optional mixed precision.
- `pcb_supernet/eval.py`: Metrics reporting utilities.
- `pcb_supernet/visualize.py`: `test_and_visualize()` with Grad-CAM.
- `pcb_supernet/baselines.py`: Baseline model builders.
- `run_pcb_supernet.py`: Example entrypoint to download datasets and run the fusion model.

### Notebook Entry Point

The primary workflow is provided in `pcb_supernet_notebook.ipynb`, which mirrors the research
pipeline (dataset download, class consistency checks, training, baselines, and Grad-CAM
visualization).

### Quick Start

```bash
python run_pcb_supernet.py
```

Refer to `REPORT.md` to fill in the benchmarking tables after running experiments.
