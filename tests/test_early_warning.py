"""Temporal early-warning: flag rising pigment stress before the verdict crosses to stressed."""

from __future__ import annotations

from phytovision.temporal import FeatureHistory, pigment_early_warning, plant_early_warnings
from phytovision.temporal.history import Observation


def _obs(timestamp: str, score: float, **features: float) -> Observation:
    return Observation("p", timestamp, score, dict(features))


def test_flags_rising_pigment_while_still_below_stressed() -> None:
    series = [
        _obs("2026-03-01", 0.20, **{"colour.gcc_mean": 0.42}),
        _obs("2026-03-02", 0.30, **{"colour.gcc_mean": 0.36}),
        _obs("2026-03-03", 0.40, **{"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.20}),
    ]
    warning = pigment_early_warning("p", series)
    assert warning.flagged
    assert warning.pigment_slope > 0
    assert warning.latest_score < 0.66


def test_not_flagged_when_pigment_is_stable() -> None:
    series = [_obs(f"2026-03-0{i + 1}", 0.2, **{"colour.gcc_mean": 0.42}) for i in range(3)]
    warning = pigment_early_warning("p", series)
    assert not warning.flagged
    assert warning.note == "pigment stable"


def test_not_flagged_when_already_stressed() -> None:
    # A late verdict is not an early warning even though pigment is rising.
    series = [
        _obs("2026-03-01", 0.50, **{"colour.gcc_mean": 0.40}),
        _obs("2026-03-02", 0.70, **{"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.30}),
    ]
    warning = pigment_early_warning("p", series)
    assert not warning.flagged
    assert "already stressed" in warning.note


def test_sorts_before_fitting() -> None:
    # Out of order input; the chronological trend (greenness falling) must still be detected.
    series = [
        _obs("2026-03-03", 0.40, **{"colour.gcc_mean": 0.30}),
        _obs("2026-03-01", 0.20, **{"colour.gcc_mean": 0.42}),
    ]
    warning = pigment_early_warning("p", series)
    assert warning.pigment_slope > 0
    assert warning.flagged


def test_empty_and_single_series() -> None:
    assert not pigment_early_warning("p", []).flagged
    single = pigment_early_warning("p", [_obs("2026-03-01", 0.2, **{"colour.gcc_mean": 0.4})])
    assert not single.flagged
    assert single.n == 1


def test_plant_early_warnings_over_a_history() -> None:
    history = FeatureHistory()
    history.add(_obs("2026-03-01", 0.20, **{"colour.gcc_mean": 0.42}))
    history.add(
        _obs("2026-03-02", 0.40, **{"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.2})
    )
    warnings = plant_early_warnings(history)
    assert set(warnings) == {"p"}
    assert warnings["p"].flagged
