from __future__ import annotations

from typing import Any


def validate_backend_parity(
    precision: str,
    *,
    max_abs_error: float,
    mean_abs_error: float,
    mask_mismatch_rate: float,
) -> dict[str, Any]:
    if precision == "fp32":
        limits = {
            "max_abs_error": 1e-3,
            "mean_abs_error": 1e-4,
            "mask_mismatch_rate": 0.0,
        }
    elif precision == "fp16":
        limits = {
            "max_abs_error": 2.0,
            "mean_abs_error": 5e-2,
            "mask_mismatch_rate": 1e-4,
        }
    else:
        raise ValueError("precision must be fp32 or fp16")

    passed = (
        max_abs_error <= limits["max_abs_error"]
        and mean_abs_error <= limits["mean_abs_error"]
        and mask_mismatch_rate <= limits["mask_mismatch_rate"]
    )
    return {
        "passed": passed,
        "limits": limits,
    }
