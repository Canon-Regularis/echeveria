"""A rule-based drought-stage classifier. Literature-motivated priors, not a fitted model.

It reduces the plant features to three bounded marker scores and reads the pattern the Sedum drought
study describes: pigment change (greenness loss, yellowing, anthocyanin reddening) appears before
turgor loss (shape solidity, curling), which appears before necrosis (browning). These are RGB
proxies, not measured biochemistry, and the thresholds are priors, so treat the stage as indicative.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from phytovision._num import clip01, feature_value, normalize01
from phytovision.models.drought.base import DroughtStageModel
from phytovision.types import PlantFeatures

_WELL_WATERED = "well-watered"
_EARLY = "early-stress"
_MODERATE = "moderate"
_SEVERE = "severe"


class RuleBasedDroughtStage(DroughtStageModel):
    name: ClassVar[str] = "rule-based-drought-v0"

    def stage(self, features: PlantFeatures) -> dict[str, object]:
        return stage_from_values(features.values)


def stage_from_values(values: Mapping[str, float | None]) -> dict[str, object]:
    """Classify a drought stage from a feature-value mapping (no ``PlantFeatures`` needed)."""
    pigment = pigment_marker(values)
    turgor = turgor_marker(values)
    necrosis = necrosis_marker(values)

    # Ordered so any elevated marker escalates the stage; nothing elevated stays well-watered.
    if necrosis >= 0.40 or turgor >= 0.60:
        name = _SEVERE
    elif turgor >= 0.30 or pigment >= 0.50 or necrosis >= 0.20:
        name = _MODERATE
    elif pigment >= 0.25 or turgor >= 0.20 or necrosis >= 0.10:
        name = _EARLY
    else:
        name = _WELL_WATERED

    markers = {"pigment": pigment, "turgor_loss": turgor, "necrosis": necrosis}
    return {
        "stage": name,
        "basis": _basis(pigment, turgor, necrosis),
        "markers": {key: round(value, 3) for key, value in markers.items()},
    }


def _basis(pigment: float, turgor: float, necrosis: float) -> str:
    """Name the markers actually elevated, so the basis never contradicts the reported scores."""
    # Floors match the earliest stage each marker can trigger, so a driver is named iff it counts.
    candidates = [
        (pigment, "pigment loss", 0.25),
        (turgor, "turgor loss", 0.20),
        (necrosis, "browning", 0.10),
    ]
    drivers = [label for value, label, floor in sorted(candidates, reverse=True) if value >= floor]
    return " and ".join(drivers) if drivers else "no strong drought markers"


def pigment_marker(values: Mapping[str, float | None]) -> float:
    """Pigment-stress score in [0,1]: greenness loss plus yellowing plus anthocyanin reddening."""
    # Missing greenness defaults to the healthy high end, so an absent feature adds no false stress.
    greenness_loss = 1.0 - normalize01(feature_value(values, "colour.gcc_mean", 0.42), 0.28, 0.42)
    yellowing = normalize01(feature_value(values, "colour.yellow_fraction", 0.0), 0.02, 0.50)
    reddening = normalize01(feature_value(values, "colour.red_fraction", 0.0), 0.02, 0.50)
    return clip01(0.50 * greenness_loss + 0.35 * yellowing + 0.15 * reddening)


def turgor_marker(values: Mapping[str, float | None]) -> float:
    """Turgor-loss score in [0,1]: low shape solidity, plus concavity and radial variation."""
    solidity_loss = 1.0 - normalize01(feature_value(values, "geometry.solidity", 0.95), 0.40, 0.95)
    concavity = normalize01(feature_value(values, "morphology.concavity", 0.0), 0.0, 0.50)
    radial = normalize01(feature_value(values, "morphology.radial_variation", 0.0), 0.0, 0.50)
    return clip01(0.50 * solidity_loss + 0.30 * concavity + 0.20 * radial)


def necrosis_marker(values: Mapping[str, float | None]) -> float:
    """Necrosis score in [0,1]: the browning fraction."""
    return normalize01(feature_value(values, "colour.brown_fraction", 0.0), 0.02, 0.50)
