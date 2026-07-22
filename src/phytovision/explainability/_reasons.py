"""Shared reason-building for the contribution-based explainers."""

from __future__ import annotations

from phytovision._num import as_float
from phytovision.explainability.physiology import physiology_note
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
        # A schema-drifted feature carries no live value, yet a SHAP attribution over the full
        # trained vector can still score it. Skip it rather than cite an unmeasured feature with a
        # NaN value, which would otherwise reach the JSON digest as invalid NaN. The contributions
        # path already drops these upstream; guarding here covers the SHAP path too.
        if value is None:
            continue
        increases = contribution > 0.0
        verb = "raises" if increases else "lowers"
        # Cite the physiological mechanism when the feature has one, so the reason is grounded.
        note = physiology_note(key)
        description = f"{model.feature_label(key)} {verb} the estimate"
        if note is not None:
            description = f"{description} ({note})"
        reasons.append(
            Reason(
                feature=key,
                direction="increases" if increases else "decreases",
                contribution=contribution,
                value=as_float(value, float("nan")),
                description=description,
            )
        )
    reasons.sort(key=lambda r: abs(r.contribution), reverse=True)
    return tuple(reasons[:top_k])
