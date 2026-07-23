"""Derive survival data from observed stress trajectories, and fit a cohort in one call.

The event is derived from the observed score crossing the stressed cut, never from the simulator's
latent truth, so this runs on any manifest. The covariates are the two things an early set of images
reveals: the baseline stress level and the early slope. Both are RGB proxies read from the same
stress score that defines the crossing; the honest framing is "given the first few images, when does
this plant wilt".
"""

from __future__ import annotations

from phytovision.exceptions import InsufficientDataError
from phytovision.models.base import STRESSED_THRESHOLD
from phytovision.models.survival.base import (
    SurvivalDataset,
    SurvivalFit,
    SurvivalRecord,
)
from phytovision.models.survival.kaplan_meier import KaplanMeierSurvival
from phytovision.temporal._fit import fit_line
from phytovision.temporal.history import FeatureHistory

_DEFAULT_WARMUP = 3


def crossing_index(scores: list[float]) -> int | None:
    """The first index where the score reaches the stressed cut, or None if it never does."""
    for index, score in enumerate(scores):
        if score >= STRESSED_THRESHOLD:
            return index
    return None


def exclusion_reason(scores: list[float]) -> str | None:
    """Why a plant is excluded from the survival cohort, or None when it is kept.

    Two things disqualify a plant, and they are different facts about it: fewer than two
    observations leaves no slope to fit, and a prevalent plant, already over the stressed cut at its
    first frame, has no pre-event window to build an honest covariate from. Naming them apart lets a
    surface say "already stressed" rather than falsely blaming an observation count.
    """
    if len(scores) < 2:
        return "insufficient-observations"
    if scores[0] >= STRESSED_THRESHOLD:
        return "already-stressed-at-first-observation"
    return None


def observed_event(scores: list[float]) -> tuple[int, int]:
    """The observed (step_position, event_observed): the crossing step and its flag, else censored.

    A crossing yields ``(crossing_index, 1)``; a series that never crosses yields
    ``(last_index, 0)``, so the censored duration is the full observed window.
    """
    crossing = crossing_index(scores)
    if crossing is not None:
        return crossing, 1
    return len(scores) - 1, 0


def early_covariates(scores: list[float], warmup: int = _DEFAULT_WARMUP) -> dict[str, float]:
    """Two observable baseline covariates: the early stress level and the early slope.

    The window is the first ``warmup`` observations. The level is their mean, which damps the
    observation noise a single reading would carry; the slope is the least-squares slope over the
    window, or zero when the window holds a single point.
    """
    window = scores[: max(1, min(warmup, len(scores)))]
    baseline = sum(window) / len(window)
    slope = fit_line(window)[0] if len(window) >= 2 else 0.0
    return {"baseline_stress": baseline, "early_slope": slope}


def derive_records(history: FeatureHistory, warmup: int = _DEFAULT_WARMUP) -> SurvivalDataset:
    """Reduce each plant's observed trajectory to a survival record.

    Plants with fewer than two observations are dropped, because a slope needs two points. The
    modelling duration is ``step_position + 1``, a strictly positive observation count.
    """
    records: list[SurvivalRecord] = []
    for plant_id in history.plant_ids:
        scores = [observation.stress_score for observation in history.series_for(plant_id)]
        # A short series (no slope) and a prevalent plant (no pre-event window) are both excluded;
        # exclusion_reason names which, so a surface never conflates the two.
        if exclusion_reason(scores) is not None:
            continue
        step_position, event = observed_event(scores)
        # Cap the covariate window at the crossing for a plant that wilts within the warmup window,
        # so the "early" covariates cannot include post-event observations and leak the outcome into
        # the held-out concordance. A censored plant never crosses, so it keeps the full warmup.
        covariate_warmup = min(warmup, step_position) if event else warmup
        records.append(
            SurvivalRecord(
                plant_id=plant_id,
                duration=step_position + 1,
                event_observed=event,
                covariates=early_covariates(scores, covariate_warmup),
            )
        )
    return SurvivalDataset(tuple(records))


def fit_cohort_survival(
    history: FeatureHistory, model: str = "weibull-aft", warmup: int = _DEFAULT_WARMUP
) -> SurvivalFit:
    """Derive the cohort, fit the chosen model, and pack a ``SurvivalFit``.

    A Kaplan-Meier baseline always provides the cohort survival curve, the cohort median, and its
    confidence interval, so the curve shown is a real Kaplan-Meier estimate regardless of the chosen
    per-plant model. The chosen model provides the per-plant estimates and the concordance index.
    """
    from phytovision.registries import SURVIVAL_MODELS

    dataset = derive_records(history, warmup)
    if not dataset.records:  # every plant was excluded: name the actual reason, not a guessed one
        reasons = {
            exclusion_reason([obs.stress_score for obs in history.series_for(plant_id)])
            for plant_id in history.plant_ids
        }
        if reasons == {"already-stressed-at-first-observation"}:
            raise InsufficientDataError(
                "every plant was already over the stressed cut at its first observation, so there "
                "is no pre-event window to model"
            )
        raise InsufficientDataError(
            "survival needs at least one plant with two or more observations"
        )
    baseline = KaplanMeierSurvival().fit(dataset)

    chosen = SURVIVAL_MODELS.create(model)
    chosen.fit(dataset)
    per_plant = chosen.predict(dataset)
    concordance = chosen.concordance_index(dataset)
    return SurvivalFit(
        model_name=chosen.name,
        curve=baseline.cohort_curve(),
        cohort_median=baseline.cohort_median(),
        cohort_median_ci=baseline.cohort_median_ci(),
        concordance_index=concordance,
        per_plant=per_plant,
        note=chosen.note,
    )
