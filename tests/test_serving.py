import cv2
import numpy as np
import pytest

from industrial_defect.serving import (
    ServiceSettings,
    decode_image,
    serialize_segmentation,
)


def encode_image(width: int = 1600, height: int = 256) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_decode_image_accepts_expected_dimensions() -> None:
    image = decode_image(encode_image(), ServiceSettings())

    assert image.shape == (256, 1600, 3)


def test_decode_image_rejects_unexpected_dimensions() -> None:
    with pytest.raises(ValueError, match="image must have shape"):
        decode_image(encode_image(100, 100), ServiceSettings())


def test_service_settings_reject_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="request_timeout_seconds must be positive"):
        ServiceSettings(request_timeout_seconds=0.0)


def test_serialize_segmentation_returns_per_class_rle() -> None:
    logits = np.full((1, 4, 2, 4), -10.0, dtype=np.float32)
    logits[0, 1, 0, 0:2] = 10.0

    result = serialize_segmentation(
        logits,
        threshold=0.8,
        output_size=(4, 2),
    )

    assert len(result) == 4
    assert result[0]["detected"] is False
    assert result[1]["detected"] is True
    assert result[1]["defect_pixels"] == 2
    assert result[1]["components"] == [{"x": 0, "y": 0, "width": 2, "height": 1, "pixels": 2}]
    assert result[1]["rle"] == "1 1 3 1"
