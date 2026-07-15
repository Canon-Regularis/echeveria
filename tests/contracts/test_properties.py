"""Property-based invariants over generated inputs (Hypothesis)."""

from __future__ import annotations

import numpy as np
import pytest
from _strategies import rgb_images
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from phytovision.exceptions import InvalidImageError
from phytovision.models.base import StressModel
from phytovision.models.conformal import conformal_quantile
from phytovision.models.stress.ensemble import EnsembleStressModel
from phytovision.types import PlantFeatures, StressAssessment
from phytovision.validation import validate_rgb_image

_UNIT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(image=rgb_images())
@settings(max_examples=30, deadline=None)
def test_valid_rgb_images_are_accepted(image) -> None:
    validate_rgb_image(image)  # must not raise


_MALFORMED_SHAPES = st.one_of(
    st.tuples(st.integers(1, 8), st.integers(1, 8)),  # 2-D, not H x W x 3
    st.tuples(
        st.integers(1, 8), st.integers(1, 8), st.integers(1, 8).filter(lambda c: c != 3)
    ),  # wrong channel count
)


@given(array=hnp.arrays(np.float32, _MALFORMED_SHAPES, elements=_UNIT))
@settings(max_examples=30, deadline=None)
def test_malformed_arrays_are_rejected(array) -> None:
    with pytest.raises(InvalidImageError):
        validate_rgb_image(array)


def test_empty_and_non_ndarray_inputs_are_rejected() -> None:
    with pytest.raises(InvalidImageError):
        validate_rgb_image(np.zeros((0, 4, 3), dtype=np.float32))
    with pytest.raises(InvalidImageError):
        validate_rgb_image([[0.0, 0.0, 0.0]])  # not an ndarray


@given(
    scores=st.lists(_UNIT, min_size=1, max_size=30),
    alpha_a=st.floats(0.01, 0.98, allow_nan=False),
    alpha_b=st.floats(0.01, 0.98, allow_nan=False),
)
@settings(max_examples=50, deadline=None)
def test_conformal_quantile_is_monotone_in_alpha(scores, alpha_a, alpha_b) -> None:
    low, high = sorted((alpha_a, alpha_b))
    # A smaller alpha means higher coverage, so the threshold is at least as large.
    assert conformal_quantile(scores, low) >= conformal_quantile(scores, high)


class _Constant(StressModel):
    name = "constant"

    def __init__(self, score: float) -> None:
        self._score = score

    def predict(self, features: PlantFeatures) -> StressAssessment:
        return StressAssessment(self._score, 0.5, "healthy", self.name)


@given(
    scores=st.lists(_UNIT, min_size=1, max_size=6),
    weights=st.lists(st.floats(0.0, 10.0, allow_nan=False), min_size=1, max_size=6),
)
@settings(max_examples=50, deadline=None)
def test_ensemble_score_stays_in_unit_interval(scores, weights) -> None:
    count = min(len(scores), len(weights))
    scores, weights = scores[:count], weights[:count]
    assume(sum(weights) > 0.0)  # the ensemble requires a positive weight sum
    ensemble = EnsembleStressModel([_Constant(s) for s in scores], weights=weights)
    result = ensemble.predict(PlantFeatures(values={}, region_count=1)).score
    assert 0.0 <= result <= 1.0
