import pytest
import torch

from industrial_defect.error_analysis import (
    aggregate_image_presence,
    compute_image_class_statistics,
)


def test_compute_image_class_statistics_for_partial_overlap() -> None:
    targets = torch.tensor([[[[1, 1, 0], [0, 1, 0]]]], dtype=torch.bool)
    predictions = torch.tensor([[[[1, 0, 1], [0, 1, 0]]]], dtype=torch.bool)

    statistics = compute_image_class_statistics(predictions, targets)

    assert statistics["intersection"].tolist() == [[2]]
    assert statistics["predicted_pixels"].tolist() == [[3]]
    assert statistics["target_pixels"].tolist() == [[3]]
    assert statistics["false_positive_pixels"].tolist() == [[1]]
    assert statistics["false_negative_pixels"].tolist() == [[1]]
    assert statistics["union"].tolist() == [[4]]
    assert statistics["dice"].item() == pytest.approx(2 / 3)
    assert statistics["iou"].item() == pytest.approx(1 / 2)
    assert statistics["precision"].item() == pytest.approx(2 / 3)
    assert statistics["recall"].item() == pytest.approx(2 / 3)
    assert statistics["image_true_positive"].item() is True
    assert statistics["image_has_overlap"].item() is True


def test_empty_prediction_and_target_are_true_negative_with_undefined_overlap() -> None:
    empty = torch.zeros((1, 1, 2, 2), dtype=torch.bool)

    statistics = compute_image_class_statistics(empty, empty)

    assert statistics["image_true_negative"].item() is True
    assert torch.isnan(statistics["dice"]).item()
    assert torch.isnan(statistics["iou"]).item()
    assert torch.isnan(statistics["precision"]).item()
    assert torch.isnan(statistics["recall"]).item()


def test_wrong_predicted_class_is_class_false_negative_and_false_positive() -> None:
    targets = torch.tensor([[[[1]], [[0]]]], dtype=torch.bool)
    predictions = torch.tensor([[[[0]], [[1]]]], dtype=torch.bool)

    statistics = compute_image_class_statistics(predictions, targets)

    assert statistics["image_false_negative"].tolist() == [[True, False]]
    assert statistics["image_false_positive"].tolist() == [[False, True]]
    assert statistics["image_true_positive"].tolist() == [[False, False]]
    assert statistics["image_has_overlap"].tolist() == [[False, False]]

    counts = aggregate_image_presence(statistics)

    assert counts["true_positive"].tolist() == [0, 0]
    assert counts["false_positive"].tolist() == [0, 1]
    assert counts["false_negative"].tolist() == [1, 0]
    assert counts["true_negative"].tolist() == [0, 0]
    assert counts["has_overlap"].tolist() == [0, 0]


def test_compute_image_class_statistics_validates_input_contract() -> None:
    boolean_masks = torch.zeros((1, 1, 2, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="same shape"):
        compute_image_class_statistics(boolean_masks, boolean_masks[:, :, :, :1])
    with pytest.raises(ValueError, match="NCHW"):
        compute_image_class_statistics(boolean_masks[0], boolean_masks[0])
    with pytest.raises(TypeError, match="torch.bool"):
        compute_image_class_statistics(boolean_masks.float(), boolean_masks.float())
