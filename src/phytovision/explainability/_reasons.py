"""Shared reason-building for the contribution-based explainers."""

from __future__ import annotations

from phytovision.models.base import ContributionModel
from phytovision.types import PlantFeatures, Reason


def build_reasons(
    model: ContributionModel,
    features: PlantFeatures,
    contributions: dict[str, float],
    top_k: int,
) -> tuple[Reason, ...]:
    """Turn a signed per-feature contribution dict into the top-k human-readable reasons."""
    reasons: list[Reason] = []
    for key, contribution in contributions.items():
        if contribution == 0.0:
            continue
        value = features.values.get(key)
        increases = contribution > 0.0
        verb = "raises" if increases else "lowers"
        reasons.append(
            Reason(
                feature=key,
                direction="increases" if increases else "decreases",
                contribution=contribution,
                value=float(value) if value is not None else float("nan"),
                description=f"{model.feature_label(key)} {verb} the estimate",
            )
        )
    reasons.sort(key=lambda r: abs(r.contribution), reverse=True)
    return tuple(reasons[:top_k])
