"""Training utilities for PCB defect models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.cuda import amp


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: torch.device,
        mixed_precision: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = amp.GradScaler(enabled=mixed_precision)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader: Iterable, log_interval: int = 10) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(dataloader, start=1):
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item()
            if batch_idx % log_interval == 0:
                avg_loss = running_loss / batch_idx
                print(f"Batch {batch_idx}: loss={avg_loss:.4f}")
        if self.scheduler:
            self.scheduler.step()
        return {"loss": running_loss / max(len(dataloader), 1)}

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
