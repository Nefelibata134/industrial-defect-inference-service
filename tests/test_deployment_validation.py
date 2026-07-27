import pytest

from industrial_defect.deployment_validation import validate_backend_parity


def test_fp32_parity_requires_zero_mask_mismatch() -> None:
    result = validate_backend_parity(
        "fp32",
        max_abs_error=1e-5,
        mean_abs_error=1e-6,
        mask_mismatch_rate=1e-6,
    )

    assert result["passed"] is False


def test_fp16_parity_accepts_bounded_numeric_drift() -> None:
    result = validate_backend_parity(
        "fp16",
        max_abs_error=1.3,
        mean_abs_error=0.02,
        mask_mismatch_rate=1.3e-5,
    )

    assert result["passed"] is True


@pytest.mark.parametrize("precision", ["int8", "tf32"])
def test_parity_rejects_unknown_precision(precision: str) -> None:
    with pytest.raises(ValueError):
        validate_backend_parity(
            precision,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            mask_mismatch_rate=0.0,
        )
