import pytest

from industrial_defect.benchmarking import latency_summary


def test_latency_summary_reports_percentiles_and_throughput() -> None:
    summary = latency_summary([1.0, 2.0, 3.0], batch_size=2)

    assert summary["mean_ms"] == 2.0
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 3.0
    assert summary["p50_ms"] == 2.0
    assert summary["throughput_images_per_second"] == 1000.0


@pytest.mark.parametrize(
    ("values", "batch_size", "message"),
    [
        ([], 1, "non-empty"),
        ([1.0], 0, "positive"),
    ],
)
def test_latency_summary_rejects_invalid_input(
    values: list[float],
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        latency_summary(values, batch_size=batch_size)
