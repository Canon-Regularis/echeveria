"""Physiological meaning for feature keys, used to ground explanations.

Each entry names the drought-physiology mechanism an RGB feature stands in for, so a reason can read
"yellowing raises the estimate (chlorophyll degradation)" instead of just naming the feature. The
mapping is literature-motivated (see MODEL_CARD.md), and the phrases describe RGB proxies, not
measured biochemistry. Keys not listed simply carry no note; the explanation stays as-is.
"""

from __future__ import annotations

# Feature key -> short physiological mechanism it proxies.
_PHYSIOLOGY: dict[str, str] = {
    "colour.gcc_mean": "canopy greenness (chlorophyll)",
    "colour.exg_mean": "canopy greenness (chlorophyll)",
    "colour.greenness_ratio": "canopy greenness (chlorophyll)",
    "colour.yellow_fraction": "chlorophyll degradation",
    "colour.brown_fraction": "necrosis / tissue death",
    "colour.red_fraction": "anthocyanin accumulation",
    "colour.saturation_mean": "pigment saturation loss",
    "colour.saturation_std": "pigment saturation loss",
    "geometry.solidity": "turgor loss / leaf deformation",
    "morphology.solidity": "turgor loss / leaf deformation",
    "morphology.concavity": "turgor loss / leaf deformation",
    "morphology.radial_variation": "turgor loss / leaf deformation",
    "morphology.roughness": "turgor loss / leaf deformation",
    "texture.entropy": "surface texture change",
    "texture.glcm_contrast": "surface texture change",
    "plant.wilted_leaf_ratio": "senescing leaf fraction",
    # Physiology proxies (drought-head readings, not features). Each phrase states the direction so
    # it travels wherever the note renders, and marks the value a proxy, never a measured quantity.
    "physiology.water_potential_proxy": (
        "water deficit / inferred low leaf water potential (ordinal RGB proxy, higher is drier, "
        "not MPa)"
    ),
    "physiology.stomatal_conductance_proxy": (
        "inferred stomatal opening / conductance (relative RGB proxy, higher is more open, no flux)"
    ),
    "physiology.transpiration_proxy": (
        "inferred transpiration / water loss (relative RGB proxy, higher is more loss, not a flux)"
    ),
}


def physiology_note(feature_key: str) -> str | None:
    """The physiological mechanism a feature proxies, or None if unmapped."""
    return _PHYSIOLOGY.get(feature_key)
