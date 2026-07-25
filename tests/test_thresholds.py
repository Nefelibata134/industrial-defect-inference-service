import pytest
import torch

from industrial_defect.thresholds import (
    ThresholdSweepMetrics,
    select_best_thresholds,
)


def test_threshold_sweep_reports_per_class_dice() -> None:
    targets = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.tensor([[[[0.0, -10.0]]]])
    meter = ThresholdSweepMetrics(class_count=1, thresholds=(0.25, 0.75))

    meter.update(logits, targets)
    results = meter.compute()

    assert results[0]["per_class_dice"] == pytest.approx([1.0])
    assert results[0]["macro_dice"] == pytest.approx(1.0)
    assert results[1]["per_class_dice"] == pytest.approx([0.0])
    assert results[1]["macro_dice"] == pytest.approx(0.0)


def test_threshold_sweep_accumulates_multiple_batches() -> None:
    meter = ThresholdSweepMetrics(class_count=1, thresholds=(0.5,))
    targets = torch.tensor([[[[1.0]]]])

    meter.update(torch.tensor([[[[10.0]]]]), targets)
    meter.update(torch.tensor([[[[-10.0]]]]), targets)
    results = meter.compute()

    assert results[0]["per_class_dice"] == pytest.approx([2.0 / 3.0])
    assert results[0]["per_class_recall"] == pytest.approx([0.5])


def test_threshold_sweep_rejects_duplicate_thresholds() -> None:
    with pytest.raises(ValueError, match="unique"):
        ThresholdSweepMetrics(class_count=1, thresholds=(0.5, 0.5))


def test_select_best_thresholds_uses_per_class_scores() -> None:
    results = [
        {"threshold": 0.2, "per_class_dice": [0.7, 0.1]},
        {"threshold": 0.5, "per_class_dice": [0.4, 0.8]},
    ]

    best = select_best_thresholds(results)

    assert best == [
        {"class_id": 1, "threshold": 0.2, "dice": 0.7},
        {"class_id": 2, "threshold": 0.5, "dice": 0.8},
    ]
