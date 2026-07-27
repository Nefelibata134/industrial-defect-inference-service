import pytest

from industrial_defect.tensorrt_engine import batch_shapes


def test_batch_shapes_builds_nchw_profile() -> None:
    assert batch_shapes(
        (1, 4, 8),
        channels=3,
        height=256,
        width=1024,
    ) == (
        (1, 3, 256, 1024),
        (4, 3, 256, 1024),
        (8, 3, 256, 1024),
    )


@pytest.mark.parametrize(
    "profile",
    [
        (0, 4, 8),
        (4, 1, 8),
        (1, 8, 4),
    ],
)
def test_batch_shapes_rejects_invalid_profile(
    profile: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError):
        batch_shapes(
            profile,
            channels=3,
            height=256,
            width=1024,
        )
