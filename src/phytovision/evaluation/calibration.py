"""Calibration diagnostics for a score against binary events, using numpy only.

The stress score is not a calibrated probability, so these tools show how far it is from one. A
reliability curve bins the score and compares each bin's mean score to the observed event rate; the
Brier score is the mean squared error against the 0/1 events; the expected calibration error is the
size-weighted average gap between predicted and observed. The events are a proxy unless they come
from measured ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phytovision.exceptions import ContractViolationError


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Per-bin calibration: mean predicted score, observed event rate, and the count in each bin."""

    mean_score: tuple[float, ...]
    observed_rate: tuple[float, ...]
    counts: tuple[int, ...]


def _as_arrays(scores: Sequence[float], events: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(scores, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64)
    if score.shape != event.shape:
        raise ContractViolationError("scores and events must be the same length")
    if score.size == 0:
        raise ContractViolationError("calibration needs at least one observation")
    return score, event


def reliability_curve(
    scores: Sequence[float], events: Sequence[float], n_bins: int = 10
) -> ReliabilityCurve:
    """Bin the scores into ``n_bins`` equal-width bins over [0, 1] and summarise each bin.

    An empty bin reports NaN for its mean score and observed rate, and a count of zero.
    """
    if n_bins < 1:
        raise ContractViolationError("n_bins must be at least one")
    score, event = _as_arrays(scores, events)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(score, edges[1:-1]), 0, n_bins - 1)

    mean_score: list[float] = []
    observed_rate: list[float] = []
    counts: list[int] = []
    for bin_id in range(n_bins):
        in_bin = index == bin_id
        count = int(in_bin.sum())
        counts.append(count)
        mean_score.append(float(score[in_bin].mean()) if count else float("nan"))
        observed_rate.append(float(event[in_bin].mean()) if count else float("nan"))
    return ReliabilityCurve(tuple(mean_score), tuple(observed_rate), tuple(counts))


def brier_score(scores: Sequence[float], events: Sequence[float]) -> float:
    """Mean squared error of the score against the 0/1 events; lower is better calibrated."""
    score, event = _as_arrays(scores, events)
    return float(np.mean((score - event) ** 2))


def expected_calibration_error(
    scores: Sequence[float], events: Sequence[float], n_bins: int = 10
) -> float:
    """Size-weighted mean gap between each bin's mean score and its observed event rate."""
    score, _ = _as_arrays(scores, events)
    curve = reliability_curve(scores, events, n_bins)
    total = score.size
    error = 0.0
    for mean, observed, count in zip(
        curve.mean_score, curve.observed_rate, curve.counts, strict=True
    ):
        if count:
            error += (count / total) * abs(mean - observed)
    return error
