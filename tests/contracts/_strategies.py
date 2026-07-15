"""Hypothesis strategies for the contract and property suites."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from phytovision.regions.base import region_from_mask
from phytovision.types import Image, Region

_UNIT_FLOATS = st.floats(
    min_value=0.0, max_value=1.0, width=32, allow_nan=False, allow_infinity=False
)


def rgb_images(min_size: int = 16, max_size: int = 40) -> st.SearchStrategy[Image]:
    """Valid H x W x 3 float32 images with values in [0, 1]."""
    shape = st.tuples(st.integers(min_size, max_size), st.integers(min_size, max_size), st.just(3))
    return hnp.arrays(np.float32, shape, elements=_UNIT_FLOATS)


@st.composite
def images_with_region(draw, min_size: int = 16, max_size: int = 36) -> tuple[Image, Region]:
    """An image and a same-shaped region with at least one foreground pixel."""
    height = draw(st.integers(min_size, max_size))
    width = draw(st.integers(min_size, max_size))
    image = draw(hnp.arrays(np.float32, (height, width, 3), elements=_UNIT_FLOATS))
    mask = draw(hnp.arrays(np.bool_, (height, width)))
    if not mask.any():
        mask = mask.copy()
        mask[height // 2, width // 2] = True
    return image, region_from_mask(region_id=0, label="plant", mask=mask)
