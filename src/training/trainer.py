"""Supervised Deep Learning Trainer with transfer learning warmup, cosine annealing, and checkpointing."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.classifier import EuroSATClassifier, count_parameters
from src.utils import ensure_dirs, setup_logger

logger = setup_logger("eurosat.trainer")


class Trainer:
    """Encapsulates the training loop, validation loop, metric tracking, and checkpoint saving."""

    def __init__(
        self,
        model: EuroSATClassifier,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_cfg = config["training"]

        label_smoothing = float(self.train_cfg.get("label_smoothing", 0.05))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        lr = float(self.train_cfg["learning_rate"])
        wd = float(self.train_cfg.get("weight_decay", 1e-4))
        opt_name = self.train_cfg.get("optimizer", "adamw").lower()

        if opt_name == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        epochs = int(self.train_cfg["epochs"])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)

        self.checkpoint_dir = Path(self.train_cfg["checkpoint_dir"])
        self.metrics_dir = Path(config["evaluation"]["metrics_dir"])
        ensure_dirs(self.checkpoint_dir, self.metrics_dir)

    def train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch and return (average_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        return running_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation split and return (average_loss, accuracy)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        return running_loss / max(total, 1), correct / max(total, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Execute full two-stage training (head warmup + full fine-tuning) with early stopping & checkpointing."""
        epochs = int(self.train_cfg["epochs"])
        freeze_epochs = int(self.config["model"].get("freeze_backbone_epochs", 0))
        patience = int(self.train_cfg.get("early_stopping_patience", 5))
        best_ckpt_path = self.checkpoint_dir / self.train_cfg["best_model_name"]

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        best_weights = copy.deepcopy(self.model.state_dict())

        history: List[Dict[str, Any]] = []
        total_params, _ = count_parameters(self.model)
        logger.info(
            f"Starting training on {self.device} | Architecture: {self.model.architecture} | "
            f"Total Parameters: {total_params:,} | Epochs: {epochs}"
        )

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            t_ep = time.time()

            # Transfer learning warmup: freeze feature extractor for initial epochs if configured
            if freeze_epochs > 0 and epoch <= freeze_epochs:
                self.model.set_backbone_trainable(False)
                stage = "head_warmup"
            else:
                self.model.set_backbone_trainable(True)
                stage = "full_finetune"

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            ep_time = time.time() - t_ep
            record = {
                "epoch": epoch,
                "stage": stage,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
                "lr": round(current_lr, 7),
                "epoch_time_sec": round(ep_time, 2),
            }
            history.append(record)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({stage}) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
                f"LR: {current_lr:.2e} ({ep_time:.1f}s)"
            )

            # Checkpointing on best validation accuracy (tie-broken by validation loss)
            improved = (val_acc > best_val_acc) or (
                abs(val_acc - best_val_acc) < 1e-6 and val_loss < best_val_loss
            )
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_weights = copy.deepcopy(self.model.state_dict())

                torch.save(
                    {
                        "epoch": epoch,
                        "architecture": self.model.architecture,
                        "num_classes": self.model.num_classes,
                        "class_names": self.config["data"]["class_names"],
                        "model_input_size": self.config["data"]["model_input_size"],
                        "val_acc": best_val_acc,
                        "val_loss": best_val_loss,
                        "model_state_dict": best_weights,
                        "config": self.config,
                    },
                    best_ckpt_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                    break

        total_time = time.time() - start_time
        self.model.load_state_dict(best_weights)

        # Save training history CSV and JSON
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.metrics_dir / "training_history.csv", index=False)

        summary = {
            "architecture": self.model.architecture,
            "device": str(self.device),
            "total_parameters": total_params,
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "best_val_acc": round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "total_training_time_sec": round(total_time, 2),
            "checkpoint_path": str(best_ckpt_path.as_posix()),
            "history": history,
        }
        with open(self.metrics_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Training finished in {total_time:.1f}s | Best Epoch: {best_epoch} | "
            f"Best Val Accuracy: {best_val_acc*100:.2f}%"
        )
        return summary
