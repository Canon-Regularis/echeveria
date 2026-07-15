"""Counterfactual explanations: the smallest single-feature change that flips the predicted bucket.

A gradient-free line search over each declared-bounded feature (``feature_contract.FEATURE_BOUNDS``)
keeps proposed changes within physically plausible ranges. It is model-agnostic: it only calls
``predict``, so it works for the heuristic, the gradient-boosted model, and ensembles alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phytovision.feature_contract import FEATURE_BOUNDS
from phytovision.models.base import StressModel
from phytovision.types import PlantFeatures


@dataclass(frozen=True)
class Counterfactual:
    """One feature change that flips the verdict, from ``current_value`` to ``target_value``."""

    feature: str
    current_value: float
    target_value: float
    target_label: str


def counterfactuals(
    model: StressModel, features: PlantFeatures, *, top_k: int = 3, steps: int = 41
) -> list[Counterfactual]:
    """The smallest single-feature changes (within declared bounds) that flip the predicted bucket.

    Only the declared-bounded features are searched, since unbounded ones have no plausible range.
    Ranked by change size relative to the feature's range, so the easiest change comes first. The
    search is a fixed grid, so it can miss a flip band narrower than the grid spacing; a returned
    change is always a real flip, but not necessarily the tightest one.
    """
    current_label = model.predict(features).label
    ranked: list[tuple[float, Counterfactual]] = []
    for key, (low, high) in FEATURE_BOUNDS.items():
        current_value = features.values.get(key)
        if current_value is None:
            continue
        flip = _nearest_flip(
            model, features, key, float(current_value), low, high, current_label, steps
        )
        if flip is None:
            continue
        span = high - low
        cost = abs(flip.target_value - flip.current_value) / span if span > 0 else float("inf")
        ranked.append((cost, flip))
    ranked.sort(key=lambda item: item[0])
    return [flip for _, flip in ranked[:top_k]]


def _nearest_flip(
    model: StressModel,
    features: PlantFeatures,
    key: str,
    current_value: float,
    low: float,
    high: float,
    current_label: str,
    steps: int,
) -> Counterfactual | None:
    best: Counterfactual | None = None
    best_distance = float("inf")
    for candidate in np.linspace(low, high, steps):
        value = float(candidate)
        label = model.predict(_with_value(features, key, value)).label
        if label != current_label:
            distance = abs(value - current_value)
            if distance < best_distance:
                best_distance = distance
                best = Counterfactual(key, current_value, value, label)
    return best


def _with_value(features: PlantFeatures, key: str, value: float) -> PlantFeatures:
    values = dict(features.values)
    values[key] = value
    return PlantFeatures(
        values=values, region_count=features.region_count, per_region=features.per_region
    )
