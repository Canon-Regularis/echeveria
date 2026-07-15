"""Fixtures shared by the contract suite. The top-level tests/conftest.py fixtures are inherited."""

from __future__ import annotations

import pytest

from phytovision.pipeline import Pipeline
from phytovision.types import PlantFeatures


@pytest.fixture
def plant_features(healthy_image) -> PlantFeatures:
    """A real feature vector, exactly what a stress model receives from the pipeline."""
    return Pipeline.default().analyze(healthy_image).plant_features
