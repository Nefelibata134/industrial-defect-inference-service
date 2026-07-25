from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn


def build_segmentation_model(
    model_name: str,
    *,
    class_count: int,
    pretrained: bool = True,
) -> nn.Module:
    if model_name != "unet_resnet18":
        raise ValueError(f"unsupported segmentation model: {model_name}")
    if class_count <= 0:
        raise ValueError("class_count must be positive")

    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=class_count,
        activation=None,
    )
