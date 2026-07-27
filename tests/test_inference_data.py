from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_defect.inference_data import (
    load_inference_batch,
    preprocess_image,
    probability_to_logit_threshold,
)


def test_probability_to_logit_threshold() -> None:
    assert probability_to_logit_threshold(0.8) == pytest.approx(np.log(4.0))


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.1])
def test_probability_to_logit_threshold_rejects_invalid_values(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        probability_to_logit_threshold(probability)


def test_preprocess_image_returns_normalized_nchw_tensor() -> None:
    image = np.full((4, 8, 3), 255, dtype=np.uint8)
    tensor = preprocess_image(image, (4, 2))

    assert tensor.shape == (3, 2, 4)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)


def test_load_inference_batch_reads_requested_split(tmp_path: Path) -> None:
    images_dir = tmp_path / "dataset" / "train_images"
    images_dir.mkdir(parents=True)
    cv2.imwrite(str(images_dir / "a.jpg"), np.zeros((4, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(images_dir / "b.jpg"), np.ones((4, 8, 3), dtype=np.uint8))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_id,split\n"
        "a.jpg,val\n"
        "b.jpg,train\n",
        encoding="utf-8",
    )

    batch, image_ids = load_inference_batch(
        tmp_path / "dataset",
        manifest,
        split="val",
        batch_size=1,
        image_size=(8, 4),
    )

    assert batch.shape == (1, 3, 4, 8)
    assert image_ids == ["a.jpg"]
