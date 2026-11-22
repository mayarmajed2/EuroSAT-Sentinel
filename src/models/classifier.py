"""Deep Learning CNN and Transfer Learning architectures for EuroSAT land-use classification."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models


class CustomEuroSATCNN(nn.Module):
    """Lightweight 4-Stage Residual/BatchNorm CNN built from scratch as an ablation reference."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4 (Target layer for Grad-CAM)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class EuroSATClassifier(nn.Module):
    """Unified Transfer Learning & CNN Classifier supporting ResNet-18, MobileNetV3-Small, and CustomCNN."""

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if self.architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        elif self.architecture == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(256, num_classes),
            )
        elif self.architecture == "custom_cnn":
            self.backbone = CustomEuroSATCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. Choose from: 'resnet18', 'mobilenet_v3_small', 'custom_cnn'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """Freeze or unfreeze feature extractor layers while always keeping the classification head trainable."""
        for name, param in self.backbone.named_parameters():
            if self.architecture == "resnet18" and name.startswith("fc."):
                param.requires_grad = True
            elif self.architecture == "mobilenet_v3_small" and name.startswith("classifier."):
                param.requires_grad = True
            elif self.architecture == "custom_cnn" and name.startswith("classifier."):
                param.requires_grad = True
            else:
                param.requires_grad = trainable

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the final spatial convolutional layer for Grad-CAM activation mapping."""
        if self.architecture == "resnet18":
            return self.backbone.layer4[-1].conv2
        if self.architecture == "mobilenet_v3_small":
            return self.backbone.features[-1][0]
        if self.architecture == "custom_cnn":
            return self.backbone.features[-3]
        raise ValueError(f"No Grad-CAM layer configured for {self.architecture}")


def build_model(config: Dict[str, Any]) -> EuroSATClassifier:
    """Instantiate the configured Deep Learning model."""
    model_cfg = config["model"]
    num_classes = int(config["data"]["num_classes"])
    return EuroSATClassifier(
        architecture=model_cfg.get("architecture", "resnet18"),
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.3)),
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
