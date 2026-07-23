"""Survival analysis of time-to-wilt.

The public surface is the data contracts, the model ABC and its three concrete models, the
observed-event derivation, and the one-call cohort fit. The models register in ``SURVIVAL_MODELS``
and import lifelines lazily, so this package imports on a base install; fitting needs ``stats``.
"""

from __future__ import annotations

from phytovision.models.survival.base import (
    SURVIVAL_DISCLAIMER,
    PlantSurvival,
    SurvivalCurve,
    SurvivalDataset,
    SurvivalFit,
    SurvivalModel,
    SurvivalRecord,
)
from phytovision.models.survival.cohort import (
    crossing_index,
    derive_records,
    early_covariates,
    exclusion_reason,
    fit_cohort_survival,
    observed_event,
)
from phytovision.models.survival.cox import CoxPHSurvival
from phytovision.models.survival.kaplan_meier import KaplanMeierSurvival
from phytovision.models.survival.weibull_aft import WeibullAFTSurvival

__all__ = [
    "SURVIVAL_DISCLAIMER",
    "CoxPHSurvival",
    "KaplanMeierSurvival",
    "PlantSurvival",
    "SurvivalCurve",
    "SurvivalDataset",
    "SurvivalFit",
    "SurvivalModel",
    "SurvivalRecord",
    "WeibullAFTSurvival",
    "crossing_index",
    "derive_records",
    "early_covariates",
    "exclusion_reason",
    "fit_cohort_survival",
    "observed_event",
]
