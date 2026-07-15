"""Every registered segmenter upholds the PlantSegmenter contract (registry-driven)."""

from __future__ import annotations

import pytest
from _invariants import assert_valid_mask
from _strategies import rgb_images
from hypothesis import given, settings

from phytovision.registries import SEGMENTERS

_NAMES = SEGMENTERS.names()


def test_expected_segmenters_are_registered() -> None:
    # Registry-driven parametrize auto-enrolls a new segmenter. This guards the other direction: the
    # known built-ins must stay registered (catch an accidental de-registration).
    assert {"exg-otsu", "lab-chroma"} <= set(_NAMES)


@pytest.mark.parametrize("name", _NAMES)
def test_segmenter_returns_a_valid_mask(name, healthy_image) -> None:
    mask = SEGMENTERS.create(name).segment(healthy_image)
    assert_valid_mask(mask, healthy_image)


@pytest.mark.parametrize("name", _NAMES)
def test_segmenter_finds_a_proper_foreground(name, healthy_image) -> None:
    # On a structured blob (a plant on a dark background) a segmenter finds a proper subset,
    # not the whole frame. The whole-frame result is only the emergency fallback.
    mask = SEGMENTERS.create(name).segment(healthy_image)
    assert not mask.all(), f"{name} captured the entire frame on a structured image"


@pytest.mark.parametrize("name", _NAMES)
@given(image=rgb_images())
@settings(max_examples=25, deadline=None)
def test_segmenter_contract_holds_on_random_images(name, image) -> None:
    # The mask contract must hold on any valid RGB image, not just the fixtures.
    assert_valid_mask(SEGMENTERS.create(name).segment(image), image)
