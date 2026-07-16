"""Early-warning detection: pigment change before the verdict crosses to stressed.

The Sedum drought study finds pigment changes (greenness loss, yellowing, anthocyanin reddening)
precede collapse. This flags a plant whose pigment-stress proxy is rising over time while its latest
stress score is still below the stressed cut-off, so deterioration is caught before the verdict
flips. All inputs are RGB proxies, not measured pigment content.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from phytovision.models.drought.rule_based import pigment_marker
from phytovision.temporal._fit import slope as fit_slope
from phytovision.temporal.history import Observation

_STRESSED_THRESHOLD = 0.66  # matches bucket_label's stressed cut-off
_RISING_TOLERANCE = 0.01  # pigment slope per step below this is treated as flat


@dataclass(frozen=True, slots=True)
class EarlyWarning:
    plant_id: str
    flagged: bool
    pigment_slope: float  # least-squares slope of the pigment-stress proxy per observation
    latest_score: float
    n: int
    note: str


def pigment_early_warning(plant_id: str, series: Sequence[Observation]) -> EarlyWarning:
    """Flag a plant whose pigment stress is rising while its score is still below stressed."""
    ordered = sorted(series, key=lambda obs: obs.timestamp)
    if not ordered:
        return EarlyWarning(plant_id, False, 0.0, 0.0, 0, "no observations")

    pigment = [pigment_marker(obs.features) for obs in ordered]
    latest_score = ordered[-1].stress_score
    if len(pigment) < 2:
        return EarlyWarning(
            plant_id, False, 0.0, latest_score, len(pigment), "need two observations"
        )

    slope = fit_slope(pigment)
    already_stressed = latest_score >= _STRESSED_THRESHOLD
    flagged = slope > _RISING_TOLERANCE and not already_stressed
    if flagged:
        note = "pigment deteriorating while the verdict is still below stressed"
    elif already_stressed:
        note = "already stressed; not an early warning"
    else:
        note = "pigment stable"
    return EarlyWarning(plant_id, flagged, slope, latest_score, len(pigment), note)
