"""Input validation and the exception hierarchy."""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.exceptions import InvalidImageError, PhytoVisionError
from phytovision.validation import validate_rgb_image


def test_validate_accepts_rgb() -> None:
    validate_rgb_image(np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4), dtype=np.uint8),  # grayscale
        np.zeros((4, 4, 4), dtype=np.uint8),  # RGBA
        np.zeros((0, 0, 3), dtype=np.uint8),  # empty
        "not an array",
    ],
)
def test_validate_rejects_bad_inputs(bad) -> None:
    with pytest.raises(InvalidImageError):
        validate_rgb_image(bad)


def test_invalid_image_is_catchable_as_both_bases() -> None:
    assert issubclass(InvalidImageError, PhytoVisionError)
    assert issubclass(InvalidImageError, ValueError)
