"""Area-weighted plant-level aggregation."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np

from phytovision.phenotyping.aggregation.base import FeatureAggregator
from phytovision.types import FeatureVector, PlantFeatures, RegionSet

logger = logging.getLogger(__name__)

_EPS = 1e-9
_DEFAULT_SENESCENCE_KEYS = ("colour.yellow_fraction", "colour.brown_fraction")


class PlantLevelAggregator(FeatureAggregator):
    """Reduce per-region features to a plant vector.

    Reduction is driven by the extractor-supplied ``reduction_policy`` (``"sum"`` vs ``"mean"`` per
    key); without a policy, every feature is area-weighted-averaged. Traits deciding whether
    a leaf is "wilted" are configurable (``senescence_keys``) rather than hardcoded.
    """

    def __init__(
        self,
        wilt_senescence_threshold: float = 0.30,
        senescence_keys: Sequence[str] = _DEFAULT_SENESCENCE_KEYS,
    ) -> None:
        self.wilt_senescence_threshold = wilt_senescence_threshold
        self.senescence_keys = tuple(senescence_keys)

    def aggregate(
        self,
        regions: RegionSet,
        features: Sequence[FeatureVector],
        reduction_policy: Mapping[str, str] | None = None,
    ) -> PlantFeatures:
        policy = reduction_policy or {}
        area_by_id = {region.id: float(region.area_px) for region in regions}
        total_area = sum(area_by_id.values())
        image_area = float(regions.image_shape[0] * regions.image_shape[1])
        # Coverage is the union of the regions, not the sum of their areas: overlapping leaf masks
        # would otherwise double-count shared pixels and push coverage above 1.
        union_area = float(np.logical_or.reduce([region.mask for region in regions]).sum())

        values: dict[str, float | None] = {}
        for key in sorted({k for fv in features for k in fv.values}):
            vals, weights = [], []
            for fv in features:
                if key in fv.values:
                    vals.append(fv.values[key])
                    weights.append(area_by_id[fv.region_id])
            values[key] = self._reduce(key, vals, weights, policy)

        # Plant-level traits that exist regardless of region count.
        values["plant.region_count"] = float(len(regions))
        values["plant.total_area_px"] = total_area
        values["plant.canopy_coverage"] = union_area / (image_area + _EPS)
        values["plant.mean_region_area"] = total_area / (len(regions) + _EPS)

        # Instance-only traits: defined only when regions are per-leaf.
        if regions.is_per_leaf:
            values["plant.leaf_count"] = float(len(regions))
            values["plant.wilted_leaf_ratio"] = self._wilted_ratio(features)
        else:
            values["plant.leaf_count"] = None
            values["plant.wilted_leaf_ratio"] = None

        clean = self._coerce_finite(values)
        return PlantFeatures(values=clean, region_count=len(regions), per_region=tuple(features))

    @staticmethod
    def _coerce_finite(values: dict[str, float | None]) -> dict[str, float | None]:
        """Coerce non-finite plant-level values to 0.0 so nothing degenerate reaches a model.

        Per-region features are already coerced in ``FeatureExtractor.extract``; the derived keys
        (canopy coverage, mean region area) bypass that, so they are coerced too.
        """
        clean: dict[str, float | None] = {}
        coerced: list[str] = []
        for key, value in values.items():
            if value is None:
                clean[key] = None
            elif math.isfinite(value):
                clean[key] = float(value)
            else:
                clean[key] = 0.0
                coerced.append(key)
        if coerced:
            logger.warning("aggregation coerced non-finite feature(s) to 0.0: %s", coerced)
        return clean

    def _reduce(
        self, key: str, vals: list[float], weights: list[float], policy: Mapping[str, str]
    ) -> float:
        if policy.get(key) == "sum":
            return float(sum(vals))
        weight_sum = sum(weights)
        if weight_sum <= 0:  # pragma: no cover: regions always have area > 0; defensive only
            logger.warning("region areas sum to 0 for %r; falling back to unweighted mean", key)
            return float(np.mean(vals)) if vals else 0.0
        return float(np.average(vals, weights=weights))

    def _wilted_ratio(self, features: Sequence[FeatureVector]) -> float | None:
        senescence: list[float] = []
        for fv in features:
            parts: list[float] = []
            for key in self.senescence_keys:
                value = fv.values.get(key)
                if value is None:
                    return None  # required trait not available; leave undefined
                parts.append(value)
            senescence.append(sum(parts))
        if not senescence:
            return None
        wilted = sum(1 for s in senescence if s >= self.wilt_senescence_threshold)
        return wilted / len(senescence)
