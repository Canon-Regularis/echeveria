"""A Weibull accelerated-failure-time survival model (lifelines, the ``stats`` extra).

The accelerated-failure-time form is parametric, so it extrapolates a median past the last observed
step and keeps the per-plant median finite on the short series that phenotyping produces, where the
Cox step baseline would return infinity. That is why it is the default covariate model.
"""

from __future__ import annotations

from typing import Any, ClassVar

from phytovision.models.survival.base import _EXTRA_HINT
from phytovision.models.survival.covariate import CovariateSurvival


class WeibullAFTSurvival(CovariateSurvival):
    name: ClassVar[str] = "weibull-aft"
    note: ClassVar[str] = "Weibull accelerated-failure-time over observable baseline covariates"
    penalizer: ClassVar[float] = 0.01

    def _make_fitter(self) -> Any:
        try:
            from lifelines import WeibullAFTFitter
        except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
            raise ImportError(_EXTRA_HINT) from exc
        return WeibullAFTFitter(penalizer=self._penalizer)
