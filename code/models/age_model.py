from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = ("resnet18", "resnet50", "mobilenet_v3_small", "convnext_tiny")


class CoralLayer(nn.Module):
    """Rank-consistent ordinal head from Cao, Mirjalili, and Raschka (2020)."""

    def __init__(self, in_features: int, num_thresholds: int):
        super().__init__()
        self.coral_weights = nn.Linear(in_features, 1, bias=False)
        self.coral_bias = nn.Parameter(
            torch.arange(num_thresholds, 0, -1, dtype=torch.float32) / num_thresholds
        )

    def forward(self, features):
        return self.coral_weights(features) + self.coral_bias


class AgeEstimator(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        num_outputs: int = 101,
        pretrained: bool = True,
        method: str = "dldl",
    ):
        super().__init__()
        make_head = lambda in_features: (
            CoralLayer(in_features, num_outputs)
            if method == "coral"
            else nn.Linear(in_features, num_outputs)
        )
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            self.model.fc = make_head(self.model.fc.in_features)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            self.model.fc = make_head(self.model.fc.in_features)
        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            final_layer = self.model.classifier[-1]
            self.model.classifier[-1] = make_head(final_layer.in_features)
        elif backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = models.convnext_tiny(weights=weights)
            final_layer = self.model.classifier[-1]
            self.model.classifier[-1] = make_head(final_layer.in_features)
        else:
            choices = ", ".join(SUPPORTED_BACKBONES)
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose one of: {choices}")

        self.backbone = backbone
        self.num_outputs = num_outputs
        self.method = method

    def forward(self, images):
        return self.model(images)

    def head_parameters(self):
        if self.backbone.startswith("resnet"):
            return self.model.fc.parameters()
        return self.model.classifier[-1].parameters()

    def backbone_parameters(self):
        head_parameter_ids = {id(parameter) for parameter in self.head_parameters()}
        return (parameter for parameter in self.parameters() if id(parameter) not in head_parameter_ids)
