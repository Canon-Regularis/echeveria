"""The feature-extraction contract and a composite that merges several extractors.

Design notes:
- ``FeatureExtractor`` uses a *template method*: subclasses implement ``_compute``, and the
  base ``extract`` enforces the invariant for every subtype — keys are namespaced and every value
  is a finite float. A subtype cannot violate the contract, which is what LSP requires.
- Extractors declare which of their traits are *extensive* (summed across regions) vs intensive
  (area-weighted mean) via ``reduction_policy``, so the aggregator never has to hardcode key names.
- Both ``FeatureExtractor`` and ``CompositeFeatureExtractor`` satisfy ``FeatureExtraction``,
  so the pipeline can depend on the abstraction (interface segregation / dependency inversion).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from skimage.measure import regionprops

from phytovision.types import FeatureVector, Image, Region

logger = logging.getLogger(__name__)


@runtime_checkable
class FeatureExtraction(Protocol):
    """The minimal surface the pipeline needs from anything that produces features."""

    def extract(self, image: Image, region: Region) -> FeatureVector: ...

    def reduction_policy(self) -> dict[str, str]:
        """Map produced feature keys to ``"sum"`` (extensive) or ``"mean"`` (intensive)."""
        ...


def finite(value: object) -> float:
    """Coerce to a finite float; NaN/inf (from degenerate regions) become 0.0."""
    f = float(value)  # type: ignore[arg-type]
    return f if math.isfinite(f) else 0.0


@lru_cache(maxsize=4)
def _region_props_cached(mask_bytes: bytes, shape: tuple[int, ...]) -> Any:
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
    return regionprops(mask.astype(np.int32))[0]


def single_region_props(region: Region) -> Any:
    """regionprops for the whole region as ONE labelled object (even if disconnected).

    Cached by mask content so geometry and morphology extractors share one computation per region.
    """
    return _region_props_cached(region.mask.tobytes(), region.mask.shape)


def prop(props: object, *names: str) -> float:
    """Read a regionprops attribute, tolerating scikit-image's renamed properties."""
    for name in names:
        if hasattr(props, name):
            return float(getattr(props, name))
    raise AttributeError(f"regionprops has none of {names}")


class FeatureExtractor(ABC):
    """Base class for one family of per-region features (geometry, colour, texture, morphology)."""

    namespace: ClassVar[str] = ""
    # Local (un-namespaced) trait names that are extensive and should be summed across regions.
    extensive: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def _compute(self, image: Image, region: Region) -> dict[str, float]: ...

    def extract(self, image: Image, region: Region) -> FeatureVector:
        if not self.namespace:
            raise TypeError(f"{type(self).__name__} must set a class-level `namespace`")
        raw = self._compute(image, region)
        values: dict[str, float] = {}
        coerced = 0
        for key, value in raw.items():
            fval = float(value)
            if not math.isfinite(fval):
                fval = 0.0
                coerced += 1
            values[f"{self.namespace}.{key}"] = fval
        if coerced:
            logger.warning(
                "%s coerced %d non-finite feature value(s) to 0.0 for region %s (%s)",
                type(self).__name__,
                coerced,
                region.id,
                region.label,
            )
        return FeatureVector(region_id=region.id, values=values)

    def reduction_policy(self) -> dict[str, str]:
        return {f"{self.namespace}.{name}": "sum" for name in self.extensive}


class CompositeFeatureExtractor:
    """Runs a sequence of extractors and merges their (namespaced) outputs for a region."""

    def __init__(self, extractors: Sequence[FeatureExtraction]) -> None:
        if not extractors:
            raise ValueError("CompositeFeatureExtractor needs at least one extractor")
        self.extractors: tuple[FeatureExtraction, ...] = tuple(extractors)

    def extract(self, image: Image, region: Region) -> FeatureVector:
        merged = FeatureVector(region_id=region.id, values={})
        for extractor in self.extractors:
            merged = merged.merged_with(extractor.extract(image, region))
        return merged

    def reduction_policy(self) -> dict[str, str]:
        policy: dict[str, str] = {}
        for extractor in self.extractors:
            policy.update(extractor.reduction_policy())
        return policy
