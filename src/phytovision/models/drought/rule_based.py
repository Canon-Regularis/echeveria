"""A rule-based drought-stage classifier. Literature-motivated priors, not a fitted model.

It reduces the plant features to three bounded marker scores and reads the pattern the Sedum drought
study describes: pigment change (greenness loss, yellowing, anthocyanin reddening) appears before
turgor loss (shape solidity, curling), which appears before necrosis (browning). These are RGB
proxies, not measured biochemistry, and the thresholds are priors, so treat the stage as indicative.

The module also derives three physiology proxies (water potential, stomatal conductance,
transpiration) from the same markers. They are second-order composites of the markers above, so they
add interpretive grounding, not independent signal. Do not feed them to the stress model or the
forecaster as new inputs; that would count the same pixels twice. Each is a crude RGB proxy, never a
measured quantity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from phytovision._num import clip01, feature_value, normalize01
from phytovision.explainability.physiology import physiology_note
from phytovision.models.drought.base import DroughtStageModel
from phytovision.types import PlantFeatures

_WELL_WATERED = "well-watered"
_EARLY = "early-stress"
_MODERATE = "moderate"
_SEVERE = "severe"

# The marker score at which each driver first counts, so the early-stage branch and the basis text
# read from one definition and cannot disagree about whether a marker is elevated.
_PIGMENT_EARLY = 0.25
_TURGOR_EARLY = 0.20
_NECROSIS_EARLY = 0.10


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
    elif pigment >= _PIGMENT_EARLY or turgor >= _TURGOR_EARLY or necrosis >= _NECROSIS_EARLY:
        name = _EARLY
    else:
        name = _WELL_WATERED

    markers = {"pigment": pigment, "turgor_loss": turgor, "necrosis": necrosis}
    return {
        "stage": name,
        "basis": _basis(pigment, turgor, necrosis),
        "markers": {key: round(value, 3) for key, value in markers.items()},
        "physiology": physiology_proxies(values),
        "physiology_basis": _physiology_basis(),
    }


def _basis(pigment: float, turgor: float, necrosis: float) -> str:
    """Name the markers actually elevated, so the basis never contradicts the reported scores."""
    # Floors match the earliest stage each marker can trigger, so a driver is named iff it counts.
    candidates = [
        (pigment, "pigment loss", _PIGMENT_EARLY),
        (turgor, "turgor loss", _TURGOR_EARLY),
        (necrosis, "browning", _NECROSIS_EARLY),
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


def water_potential_proxy(values: Mapping[str, float | None]) -> float:
    """An ordinal water-deficit index in [0,1], not a leaf water potential in MPa.

    Higher means a larger inferred deficit, so a more negative leaf water potential; it rises under
    a dry-down. Turgor loss is the most direct visible correlate of a falling potential, so it
    leads, with pigment loss second. Empty values read as no deficit (well-watered).
    """
    return clip01(0.60 * turgor_marker(values) + 0.40 * pigment_marker(values))


def stomatal_conductance_proxy(values: Mapping[str, float | None]) -> float:
    """A relative stomatal-conductance index in [0,1], not mmol m-2 s-1.

    Higher means more inferred opening; it falls under a dry-down as guard-cell turgor drops and the
    stomata close. The closure term is dominated by turgor loss, with pigment and necrosis next.
    Empty values read as fully open.
    """
    closure = (
        0.60 * turgor_marker(values)
        + 0.25 * pigment_marker(values)
        + 0.15 * necrosis_marker(values)
    )
    return clip01(1.0 - closure)


def transpiration_proxy(values: Mapping[str, float | None]) -> float:
    """A relative transpiration index in [0,1], not mm/day or a flux.

    Higher means more inferred water loss; it falls under a dry-down. It scales the conductance
    proxy by a green-canopy factor in [0.40, 1.0], so it never exceeds the conductance proxy. No
    vapour-pressure deficit or driving gradient is observable from one RGB frame.
    """
    greenness = normalize01(feature_value(values, "colour.gcc_mean", 0.42), 0.28, 0.42)
    return clip01(stomatal_conductance_proxy(values) * (0.40 + 0.60 * greenness))


def physiology_proxies(values: Mapping[str, float | None]) -> dict[str, float]:
    """The three physiology proxies as a compact block. Each is a crude RGB proxy, not measured."""
    return {
        "water_potential_proxy": round(water_potential_proxy(values), 3),
        "stomatal_conductance_proxy": round(stomatal_conductance_proxy(values), 3),
        "transpiration_proxy": round(transpiration_proxy(values), 3),
    }


def _physiology_basis() -> str:
    """A caveat naming each proxy's mechanism, so the physiology notes travel with the values."""
    keys = (
        "physiology.water_potential_proxy",
        "physiology.stomatal_conductance_proxy",
        "physiology.transpiration_proxy",
    )
    notes = [physiology_note(key) or "crude RGB proxy" for key in keys]
    return "; ".join(notes) + ". Crude RGB proxies, not measurements."
