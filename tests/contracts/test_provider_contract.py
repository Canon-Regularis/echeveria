"""Every registered region provider upholds the RegionProvider contract (registry-driven).

The leaf-instance provider needs a working leaf segmenter injected (its default one is a placeholder
that is not yet trained), so it is built with the test segmenter; others build with no arguments.
"""

from __future__ import annotations

import pytest
from _invariants import assert_valid_region_set

from phytovision.regions.base import RegionProvider
from phytovision.regions.leaf_instance import LeafInstanceRegionProvider
from phytovision.registries import REGION_PROVIDERS

_NAMES = REGION_PROVIDERS.names()


def _build(name: str, leaf_segmenter) -> RegionProvider:
    if name == "leaf-instance":
        return LeafInstanceRegionProvider(leaf_segmenter)
    return REGION_PROVIDERS.create(name)


def test_expected_providers_are_registered() -> None:
    # Registry-driven parametrize auto-enrolls new providers; this guards against losing a built-in.
    assert {"whole-plant", "leaf-instance"} <= set(_NAMES)


@pytest.mark.parametrize("name", _NAMES)
def test_provider_returns_a_valid_region_set(
    name, healthy_image, plant_mask, leaf_segmenter
) -> None:
    provider = _build(name, leaf_segmenter)
    region_set = provider.regions(healthy_image, plant_mask)
    assert_valid_region_set(region_set, plant_mask.shape)
