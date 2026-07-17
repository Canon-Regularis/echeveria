"""A Cox proportional-hazards survival model (lifelines, the ``stats`` extra).

Cox is a second discriminative model, so the leaderboard ranks two covariate models against the
Kaplan-Meier baseline. It is not the default because its median comes from a Breslow baseline
bounded by the largest observed duration, so a low-risk plant whose survival never reaches one half
inside the window has an infinite median, which surfaces as None (no median within the window).
"""

from __future__ import annotations

from typing import Any, ClassVar

from phytovision.models.survival.base import _EXTRA_HINT
from phytovision.models.survival.covariate import CovariateSurvival


class CoxPHSurvival(CovariateSurvival):
    name: ClassVar[str] = "cox-ph"
    note: ClassVar[str] = "Cox proportional-hazards over observable baseline covariates"
    penalizer: ClassVar[float] = 0.1

    def _make_fitter(self) -> Any:
        try:
            from lifelines import CoxPHFitter
        except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
            raise ImportError(_EXTRA_HINT) from exc
        return CoxPHFitter(penalizer=self._penalizer)
