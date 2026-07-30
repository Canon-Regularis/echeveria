"""Survival analysis: base-deps event derivation always runs; lifelines fits are gated."""

from __future__ import annotations

import importlib.util
import math
import sys

import pytest

from phytovision.evaluation.survival import survival_concordance
from phytovision.models.base import STRESSED_THRESHOLD
from phytovision.models.survival import (
    KaplanMeierSurvival,
    SurvivalDataset,
    SurvivalRecord,
    WeibullAFTSurvival,
    crossing_index,
    derive_records,
    early_covariates,
    fit_cohort_survival,
    observed_event,
)
from phytovision.registries import SURVIVAL_MODELS
from phytovision.simulation import DryDownParams, cohort_history, simulate_cohort
from phytovision.temporal._fit import fit_line
from phytovision.temporal.history import FeatureHistory, Observation

_HAS_LIFELINES = importlib.util.find_spec("lifelines") is not None
_needs_lifelines = pytest.mark.skipif(
    not _HAS_LIFELINES, reason="needs the stats extra (lifelines)"
)


def _history(**params: object):
    return cohort_history(simulate_cohort(60, DryDownParams(**params), seed=7))  # type: ignore[arg-type]


def _tiny_dataset() -> SurvivalDataset:
    return SurvivalDataset(
        (
            SurvivalRecord("a", 3, 1, {"baseline_stress": 0.2, "early_slope": 0.05}),
            SurvivalRecord("b", 5, 0, {"baseline_stress": 0.1, "early_slope": 0.02}),
        )
    )


# --- base-deps derivation (always runs) ---


def test_observed_event_covers_crossing_censoring_and_immediate() -> None:
    assert observed_event([0.1, 0.3, 0.7, 0.8]) == (2, 1)  # crosses at index 2
    assert observed_event([0.1, 0.2, 0.3]) == (2, 0)  # never crosses, censored at the last index
    assert observed_event([0.7, 0.8]) == (0, 1)  # already stressed at the first image
    assert crossing_index([0.1, 0.2]) is None


def test_covariates_are_a_pure_function_of_the_scores() -> None:
    scores = [0.2, 0.25, 0.32, 0.5, 0.7]
    covariates = early_covariates(scores, warmup=3)
    assert covariates["baseline_stress"] == pytest.approx(sum(scores[:3]) / 3)
    assert covariates["early_slope"] == pytest.approx(fit_line(scores[:3])[0])
    # A pure readout of the scores: the same scores always give the same covariates.
    assert early_covariates(list(scores), warmup=3) == covariates
    # A single-point window has no slope.
    assert early_covariates([0.4], warmup=3)["early_slope"] == 0.0
    # A two-point window still yields a finite slope.
    assert math.isfinite(early_covariates([0.4, 0.6], warmup=3)["early_slope"])


def test_derive_records_caps_the_covariate_window_at_the_crossing() -> None:
    # For a plant that wilts inside the warmup window, the "early" covariates must stop at the
    # crossing so a post-event observation cannot leak the outcome into the held-out concordance.
    # wilter crosses the 0.66 cut at step 1, so its baseline is just the pre-crossing 0.20; without
    # the `min(warmup, step_position) if event` cap the window would run to 3 and average in
    # 0.80/0.90 to give 0.633. A censored plant never crosses, so it keeps the full warmup.
    history = FeatureHistory()
    for i, score in enumerate([0.20, 0.80, 0.90, 0.95]):
        history.add(Observation("wilter", f"t{i}", score))
    for i, score in enumerate([0.10, 0.15]):
        history.add(Observation("censored", f"t{i}", score))
    records = {record.plant_id: record for record in derive_records(history).records}

    assert records["wilter"].event_observed == 1
    assert records["wilter"].covariates["baseline_stress"] == pytest.approx(0.20)  # not 0.633
    assert records["wilter"].covariates["early_slope"] == pytest.approx(0.0)
    assert records["censored"].event_observed == 0
    assert records["censored"].covariates["baseline_stress"] == pytest.approx(0.125)
    assert records["censored"].covariates["early_slope"] == pytest.approx(0.05)


@_needs_lifelines
def test_survival_concordance_ranks_a_missing_median_as_longest_surviving() -> None:
    # A plant with no in-window median gets a sentinel past the longest duration, so it reads as
    # longest-surviving, not shorter than a finite median. Durations 3/5/7 (all events) with medians
    # {a: None, b: 4, c: 6}: only the b-c pair is concordant, so C = 1/3. A sentinel flipped below
    # the finite medians would rank the missing-median plant shortest and return 1.0 instead.
    dataset = SurvivalDataset(
        (
            SurvivalRecord("a", 3, 1, {"baseline_stress": 0.2, "early_slope": 0.0}),
            SurvivalRecord("b", 5, 1, {"baseline_stress": 0.2, "early_slope": 0.0}),
            SurvivalRecord("c", 7, 1, {"baseline_stress": 0.2, "early_slope": 0.0}),
        )
    )
    c_index = survival_concordance({"a": None, "b": 4.0, "c": 6.0}, dataset)
    assert c_index == pytest.approx(1 / 3)


def test_derive_records_shape_and_censoring_counts() -> None:
    history = _history(n_steps=20, base_decline_rate=0.16, decline_rate_spread=0.5)
    dataset = derive_records(history)
    assert len(dataset) == len(history.plant_ids)
    assert all(record.duration >= 1 for record in dataset.records)
    assert dataset.covariate_names() == ("baseline_stress", "early_slope")
    # The event count equals the number of plants that actually crossed the cut.
    crossings = sum(
        1
        for pid in history.plant_ids
        if any(o.stress_score >= STRESSED_THRESHOLD for o in history.series_for(pid))
    )
    assert dataset.n_events == crossings


def test_derive_records_drops_single_observation_plants() -> None:
    from phytovision.temporal.history import FeatureHistory, Observation

    history = FeatureHistory()
    history.add(Observation("lonely", "2026-03-01", 0.5, {}))
    history.add(Observation("pair", "2026-03-01", 0.2, {}))
    history.add(Observation("pair", "2026-03-02", 0.8, {}))
    dataset = derive_records(history)
    assert dataset.plant_ids() == ["pair"]  # the single-observation plant is dropped


def test_event_derivation_matches_the_latent_without_reading_it() -> None:
    # With no observation noise the observed score equals the latent, so the derived crossing must
    # equal the simulator's latent event_step, proving the observed derivation tracks the truth.
    cohort = simulate_cohort(40, DryDownParams(n_steps=20, observation_noise=0.0), seed=5)
    for plant in cohort.series:
        scores = [o.stress_score for o in plant.observations]
        step, event = observed_event(scores)
        if plant.censored:
            assert event == 0
            assert step == len(scores) - 1
        else:
            assert event == 1
            assert step == plant.event_step


def test_missing_extra_raises_a_clear_error(monkeypatch) -> None:
    # Force `import lifelines` to fail, even when installed, to cover the clear-error contract.
    monkeypatch.setitem(sys.modules, "lifelines", None)
    with pytest.raises(ImportError, match="stats extra"):
        KaplanMeierSurvival().fit(_tiny_dataset())


def test_registry_wiring() -> None:
    # A subset check, so a throwaway model another test registers cannot break this.
    assert {"cox-ph", "kaplan-meier", "weibull-aft"} <= set(SURVIVAL_MODELS.names())
    assert isinstance(SURVIVAL_MODELS.create("weibull-aft"), WeibullAFTSurvival)
    assert isinstance(SURVIVAL_MODELS.create("kaplan-meier"), KaplanMeierSurvival)


def test_finite_or_none_maps_non_finite_and_non_numeric_to_none() -> None:
    from phytovision.models.survival.base import _finite_or_none

    assert _finite_or_none(3.5) == 3.5
    assert _finite_or_none(None) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none("not a number") is None  # a non-numeric value is treated as no value


def test_exclusion_reason_names_short_and_prevalent_apart() -> None:
    from phytovision.models.survival import exclusion_reason

    assert exclusion_reason([0.3]) == "insufficient-observations"
    assert exclusion_reason([0.9, 0.92, 0.95]) == "already-stressed-at-first-observation"
    assert exclusion_reason([0.2, 0.4, 0.6]) is None  # a normal declining plant is kept


def test_all_prevalent_cohort_reports_the_real_reason() -> None:
    # The raise happens in derive_records before any lifelines use, so this needs no stats extra. An
    # all-prevalent cohort must not claim "no plant has two or more observations": every plant here
    # has three.
    from phytovision.exceptions import InsufficientDataError
    from phytovision.temporal.history import FeatureHistory, Observation

    history = FeatureHistory()
    for plant in ("a", "b"):
        for i, score in enumerate([0.80, 0.85, 0.90]):
            history.add(Observation(plant, f"2026-03-0{i + 1}", score, {}))
    with pytest.raises(InsufficientDataError, match="already over the stressed cut"):
        fit_cohort_survival(history, "weibull-aft")


# --- lifelines fits (gated on the stats extra) ---


@_needs_lifelines
def test_a_faster_cohort_wilts_sooner() -> None:
    fast = KaplanMeierSurvival().fit(derive_records(_history(n_steps=20, base_decline_rate=0.20)))
    slow = KaplanMeierSurvival().fit(derive_records(_history(n_steps=20, base_decline_rate=0.08)))
    assert fast.cohort_median() is not None and slow.cohort_median() is not None
    assert fast.cohort_median() < slow.cohort_median()  # faster decline, shorter survival


@_needs_lifelines
def test_kaplan_meier_curve_starts_at_one_and_is_non_increasing() -> None:
    fit = KaplanMeierSurvival().fit(
        derive_records(_history(n_steps=20, base_decline_rate=0.14, decline_rate_spread=0.5))
    )
    curve = fit.cohort_curve()
    assert curve.survival[0] == pytest.approx(1.0)
    assert all(a >= b - 1e-9 for a, b in zip(curve.survival, curve.survival[1:], strict=False))
    for low, mid, high in zip(curve.lower, curve.survival, curve.upper, strict=True):
        assert low <= mid + 1e-9 <= high + 1e-9


@_needs_lifelines
def test_fully_censored_cohort_is_handled() -> None:
    history = cohort_history(
        simulate_cohort(30, DryDownParams(n_steps=8, base_decline_rate=0.04), seed=3)
    )
    dataset = derive_records(history)
    assert dataset.n_events == 0
    for name in ("kaplan-meier", "weibull-aft"):
        fit = fit_cohort_survival(history, name)
        assert fit.cohort_median is None
        assert all(plant.median is None for plant in fit.per_plant.values())
        assert fit.curve.survival[0] == pytest.approx(1.0)


@_needs_lifelines
def test_weibull_aft_beats_chance_and_individualizes() -> None:
    history = _history(n_steps=20, base_decline_rate=0.16, decline_rate_spread=0.06)
    fit = fit_cohort_survival(history, "weibull-aft")
    assert fit.concordance_index is not None and fit.concordance_index >= 0.55
    medians = [p.median for p in fit.per_plant.values() if p.median is not None]
    assert len(set(medians)) > 1  # per-plant covariate model, not one broadcast value
    assert all(p.basis == "weibull-aft" for p in fit.per_plant.values())


@_needs_lifelines
def test_weibull_aft_band_brackets_the_median() -> None:
    fit = fit_cohort_survival(
        _history(n_steps=20, base_decline_rate=0.16, decline_rate_spread=0.5), "weibull-aft"
    )
    bracketed = [
        p
        for p in fit.per_plant.values()
        if p.median is not None and p.lower is not None and p.upper is not None
    ]
    assert bracketed
    for plant in bracketed:
        assert plant.lower <= plant.median <= plant.upper


@_needs_lifelines
def test_covariate_model_falls_back_to_km_on_a_tiny_cohort() -> None:
    history = cohort_history(
        simulate_cohort(2, DryDownParams(n_steps=12, base_decline_rate=0.2), seed=1)
    )
    prediction = WeibullAFTSurvival().fit(derive_records(history)).predict(derive_records(history))
    assert {p.basis for p in prediction.values()} == {"cohort-km"}


@_needs_lifelines
def test_cox_medians_are_finite_or_none_never_infinite() -> None:
    fit = fit_cohort_survival(
        _history(n_steps=20, base_decline_rate=0.16, decline_rate_spread=0.5), "cox-ph"
    )
    for plant in fit.per_plant.values():
        assert plant.median is None or math.isfinite(plant.median)
    assert fit.concordance_index is not None and fit.concordance_index >= 0.55


@_needs_lifelines
def test_cox_predicts_a_single_record_cohort() -> None:
    # lifelines squeezes a one-row prediction to a bare scalar with no .iloc; reading it back used
    # to raise AttributeError, which breaks a leave-one-out survival fold. It now reads cleanly.
    dataset = derive_records(_history(n_steps=20, base_decline_rate=0.16, decline_rate_spread=0.5))
    model = SURVIVAL_MODELS.create("cox-ph").fit(dataset)
    prediction = model.predict(dataset.subset([0]))
    assert len(prediction) == 1
    (plant,) = prediction.values()
    assert plant.median is None or math.isfinite(plant.median)


@_needs_lifelines
def test_covariate_predict_fallback_is_scoped_to_one_call(monkeypatch) -> None:
    # A fitted covariate model that raises at predict time must degrade only that call to the cohort
    # baseline, keeping the trained fitter: the old path nulled it and refitted on the predict-time
    # argument, so every later prediction returned that (often held-out) cohort's median.
    dataset = derive_records(_history(n_steps=20, base_decline_rate=0.16, decline_rate_spread=0.5))
    model = WeibullAFTSurvival().fit(dataset)
    assert model._fitter is not None  # a real covariate fit, not the fallback
    original = model._fitter.predict_median

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ValueError("degenerate covariate profile")

    monkeypatch.setattr(model._fitter, "predict_median", _boom)
    degraded = model.predict(dataset)
    assert {plant.basis for plant in degraded.values()} == {"cohort-km"}  # this call degraded
    assert model._fitter is not None  # but the trained fitter is kept, not nulled

    monkeypatch.setattr(model._fitter, "predict_median", original)  # the library recovers
    recovered = model.predict(dataset)
    assert {plant.basis for plant in recovered.values()} == {model.name}  # trained model runs again


@_needs_lifelines
def test_refitting_clears_the_previous_cohorts_fallback() -> None:
    # A cached Kaplan-Meier fallback from an earlier cohort must not survive a re-fit: a later
    # predict-time degradation would otherwise report that old cohort's median for every plant,
    # under a "cohort-km" basis that reveals nothing about which cohort it came from.
    first = derive_records(_history(n_steps=20, base_decline_rate=0.30, decline_rate_spread=0.5))
    second = derive_records(_history(n_steps=20, base_decline_rate=0.08, decline_rate_spread=0.5))
    model = WeibullAFTSurvival().fit(first)
    model._fallback = KaplanMeierSurvival().fit(first)  # as a predict-time degrade would cache

    model.fit(second)
    assert model._fallback is None  # the stale baseline is gone
    assert model._train_dataset is second  # and the degrade target is the current cohort
