"""Metamorphic invariants: transforms that must not change the water-stress verdict.

Water stress does not depend on orientation or mild illumination, so a flip, a rotation, or a wider
background border should leave the verdict essentially unchanged, and a uniform brightness scale
should not move a chromatic ratio feature at all. These encode domain expectations that plain
example tests do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.pipeline import Pipeline

_SCORE_TOL = 0.1

_GEOMETRIC = {
    "fliplr": np.fliplr,
    "flipud": np.flipud,
    "rot90": lambda a: np.rot90(a),
}


def _stress(image):
    return Pipeline.default().analyze(np.ascontiguousarray(image.astype(np.float32))).stress


@pytest.mark.parametrize("image_fixture", ["healthy_image", "stressed_image"])
@pytest.mark.parametrize("transform", list(_GEOMETRIC.values()), ids=list(_GEOMETRIC))
def test_geometric_transforms_preserve_the_verdict(image_fixture, transform, request) -> None:
    image = request.getfixturevalue(image_fixture)
    base = _stress(image)
    moved = _stress(transform(image))
    assert moved.label == base.label
    assert moved.score == pytest.approx(base.score, abs=_SCORE_TOL)


@pytest.mark.parametrize("factor", [0.6, 0.85])
def test_gcc_is_invariant_to_uniform_brightness(healthy_image, plant_region, factor) -> None:
    # The green chromatic coordinate is a ratio, so scaling every channel by the same factor (no
    # clipping) leaves it exactly unchanged. A stronger claim than a label staying put.
    from phytovision.phenotyping.colour import ColourFeatures

    base = ColourFeatures().extract(healthy_image, plant_region).values["colour.gcc_mean"]
    scaled_image = np.clip(healthy_image * factor, 0.0, 1.0).astype(np.float32)
    scaled = ColourFeatures().extract(scaled_image, plant_region).values["colour.gcc_mean"]

    assert scaled == pytest.approx(base, abs=1e-6)


def test_background_border_does_not_change_the_verdict(healthy_image) -> None:
    base = _stress(healthy_image)
    bordered = np.pad(healthy_image, ((16, 16), (16, 16), (0, 0)), constant_values=0.1)
    padded = _stress(bordered)
    assert padded.label == base.label
    assert padded.score == pytest.approx(base.score, abs=_SCORE_TOL)
