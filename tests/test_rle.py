import numpy as np
import pytest

from industrial_defect.rle import decode_rle, encode_rle


def test_empty_rle_decodes_to_zero_mask() -> None:
    mask = decode_rle("", shape=(3, 4))

    assert mask.dtype == np.uint8
    assert mask.shape == (3, 4)
    assert not mask.any()


def test_known_column_major_rle() -> None:
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0:2, 0] = 1
    mask[1:3, 2] = 1

    encoded = encode_rle(mask)
    decoded = decode_rle(encoded, shape=mask.shape)

    assert encoded == "1 2 8 2"
    np.testing.assert_array_equal(decoded, mask)


def test_non_binary_values_are_encoded_as_foreground() -> None:
    mask = np.asarray([[0, 2], [255, 0]], dtype=np.uint8)

    decoded = decode_rle(encode_rle(mask), shape=mask.shape)

    np.testing.assert_array_equal(decoded, mask > 0)


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        ("1 2 4", "start-length pairs"),
        ("0 1", "positive"),
        ("1 -2", "positive"),
        ("12 2", "exceeds mask size"),
        ("5 2 3 1", "sorted"),
        ("2 3 4 2", "must not overlap"),
    ],
)
def test_invalid_rle_is_rejected(encoded: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_rle(encoded, shape=(3, 4))


def test_encode_rejects_non_2d_mask() -> None:
    with pytest.raises(ValueError, match="mask must be 2D"):
        encode_rle(np.zeros((1, 3, 4), dtype=np.uint8))
