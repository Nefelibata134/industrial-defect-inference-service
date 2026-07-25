import pytest
import torch

from industrial_defect.loss import BCEDiceLoss, soft_dice_loss
from industrial_defect.model import build_segmentation_model


def test_unet_resnet18_preserves_segmentation_shape() -> None:
    model = build_segmentation_model(
        "unet_resnet18",
        class_count=4,
        pretrained=False,
    )

    logits = model(torch.zeros((2, 3, 64, 128)))

    assert logits.shape == (2, 4, 64, 128)


def test_bce_dice_loss_rewards_correct_logits() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    correct_logits = torch.where(targets == 1, 10.0, -10.0)
    incorrect_logits = -correct_logits
    criterion = BCEDiceLoss()

    correct_loss = criterion(correct_logits, targets)
    incorrect_loss = criterion(incorrect_logits, targets)

    assert correct_loss < incorrect_loss


def test_soft_dice_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        soft_dice_loss(
            torch.zeros((1, 4, 8, 8)),
            torch.zeros((1, 4, 4, 4)),
        )
