import pytest
import torch

from industrial_defect.metrics import SegmentationMetrics


def test_segmentation_metrics_for_perfect_prediction() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.where(targets == 1, 10.0, -10.0)
    meter = SegmentationMetrics(class_count=1)

    meter.update(logits, targets)
    metrics = meter.compute()

    assert metrics["macro_dice"] == pytest.approx(1.0)
    assert metrics["macro_iou"] == pytest.approx(1.0)
    assert metrics["macro_precision"] == pytest.approx(1.0)
    assert metrics["macro_recall"] == pytest.approx(1.0)
    assert metrics["image_false_negative_rate"] == pytest.approx(0.0)


def test_segmentation_metrics_detect_image_false_negative() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    logits = torch.full_like(targets, -10.0)
    meter = SegmentationMetrics(class_count=1)

    meter.update(logits, targets)
    metrics = meter.compute()

    assert metrics["macro_dice"] == pytest.approx(0.0)
    assert metrics["macro_recall"] == pytest.approx(0.0)
    assert metrics["image_false_negative_rate"] == pytest.approx(1.0)
