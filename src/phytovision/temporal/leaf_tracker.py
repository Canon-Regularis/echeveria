"""Track leaf identity across a plant's frames, so per-leaf trajectories exist.

``Region.id`` is a per-image index, so the same physical leaf gets a different id per frame. This
assigns stable identities by matching leaves between consecutive frames: a Hungarian assignment on
normalized centroid distance and region area, minting a new identity for a leaf that appears and
ending one that disappears. It turns a sequence of per-image ``RegionSet`` objects into per-leaf
tracks, so the forecasters and the survival model can run on one leaf over time, not only the whole
plant. It matches on geometry alone, so it never needs the masks to carry features.

Per-leaf tracking is only meaningful under a leaf-instance region provider (the watershed
segmenter), where a frame holds several leaf regions; under the whole-plant provider every frame is
one region, so every frame trivially tracks as the same single identity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phytovision.models.base import StressModel
from phytovision.temporal.history import Observation
from phytovision.types import AnalysisReport, PlantFeatures, Region, RegionSet

# Reject a match whose combined normalized cost exceeds this: a leaf that moved a little matches, a
# different leaf across the frame does not. Centroids are normalized by image size, so it is scaled.
_DEFAULT_MAX_COST = 0.2
# How much a difference in area fraction adds to the centroid distance in the match cost.
_DEFAULT_AREA_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class _Descriptor:
    """A leaf's position and size for matching, normalized to the image so it is scale-invariant."""

    centroid_row: float
    centroid_col: float
    area_fraction: float


@dataclass(frozen=True, slots=True)
class LeafTracks:
    """Stable leaf identities over a sequence of frames.

    ``assignments[f]`` gives the global leaf id of each region in frame ``f``, in region order, to
    align with the report's per-region features. ``n_leaves`` is the count of distinct identities.
    """

    assignments: tuple[tuple[int, ...], ...]
    n_leaves: int

    def frames_for(self, leaf_id: int) -> list[tuple[int, int]]:
        """The ``(frame_index, region_index)`` positions where this leaf identity appears."""
        return [
            (frame, region)
            for frame, ids in enumerate(self.assignments)
            for region, assigned in enumerate(ids)
            if assigned == leaf_id
        ]

    @property
    def leaf_ids(self) -> list[int]:
        return sorted({leaf_id for ids in self.assignments for leaf_id in ids})


class LeafTracker:
    """Assign stable leaf identities across frames by matching centroids and areas."""

    def __init__(
        self, max_cost: float = _DEFAULT_MAX_COST, area_weight: float = _DEFAULT_AREA_WEIGHT
    ) -> None:
        self.max_cost = max_cost
        self.area_weight = area_weight

    def track(self, region_sets: Sequence[RegionSet]) -> LeafTracks:
        """Link the regions of each frame to a global leaf identity."""
        assignments: list[tuple[int, ...]] = []
        previous: list[tuple[int, _Descriptor]] = []
        next_id = 0
        for region_set in region_sets:
            current = [_describe(region, region_set.image_shape) for region in region_set.regions]
            if not previous:
                ids = list(range(next_id, next_id + len(current)))
                next_id += len(current)
            else:
                ids, next_id = self._match(previous, current, next_id)
            assignments.append(tuple(ids))
            previous = list(zip(ids, current, strict=True))
        return LeafTracks(tuple(assignments), next_id)

    def _match(
        self, previous: list[tuple[int, _Descriptor]], current: list[_Descriptor], next_id: int
    ) -> tuple[list[int], int]:
        from scipy.optimize import linear_sum_assignment

        ids: list[int | None] = [None] * len(current)
        if previous and current:
            cost = np.array(
                [[self._cost(prev, now) for now in current] for _, prev in previous], dtype=float
            )
            # Gate over-threshold edges out of the optimization with a large finite sentinel (scipy
            # rejects inf and nan), so a disappearing leaf cannot steal a valid match by lowering
            # the global sum. Assigned edges are still rejected on their original cost below.
            gated = np.where(cost <= self.max_cost, cost, self.max_cost * 1e6)
            rows, cols = linear_sum_assignment(gated)
            for row, col in zip(rows, cols, strict=True):
                if cost[row, col] <= self.max_cost:
                    ids[col] = previous[row][0]  # inherit the previous frame's identity
        for index, leaf_id in enumerate(ids):
            if leaf_id is None:  # an unmatched region is a newly appeared leaf
                ids[index] = next_id
                next_id += 1
        return [leaf_id for leaf_id in ids if leaf_id is not None], next_id

    def _cost(self, a: _Descriptor, b: _Descriptor) -> float:
        centroid = math.hypot(a.centroid_row - b.centroid_row, a.centroid_col - b.centroid_col)
        return centroid + self.area_weight * abs(a.area_fraction - b.area_fraction)


def _describe(region: Region, image_shape: tuple[int, int]) -> _Descriptor:
    rows, cols = np.nonzero(region.mask)
    height, width = image_shape
    return _Descriptor(
        centroid_row=float(rows.mean()) / height,
        centroid_col=float(cols.mean()) / width,
        area_fraction=float(region.mask.sum()) / float(height * width),
    )


def build_leaf_histories(
    reports: Sequence[AnalysisReport],
    timestamps: Sequence[str],
    model: StressModel,
    tracker: LeafTracker | None = None,
) -> dict[int, list[Observation]]:
    """Turn a plant's per-frame reports into one observation sequence per tracked leaf.

    Each leaf's score is the stress model read on that leaf's own features, so the histories feed a
    forecaster and the survival model directly, one leaf at a time. The reports must come from a
    leaf-instance pipeline, so ``plant_features.per_region`` holds one vector per leaf.
    """
    if len(reports) != len(timestamps):
        raise ValueError("reports and timestamps must be the same length")
    matcher = tracker or LeafTracker()
    tracks = matcher.track([report.regions for report in reports])

    histories: dict[int, list[Observation]] = {}
    for frame, (report, timestamp) in enumerate(zip(reports, timestamps, strict=True)):
        per_region = report.plant_features.per_region
        for region_index, leaf_id in enumerate(tracks.assignments[frame]):
            features = dict(per_region[region_index].values)
            score = model.predict(PlantFeatures.from_values(features)).score
            observation = Observation(f"leaf_{leaf_id}", timestamp, score, features)
            histories.setdefault(leaf_id, []).append(observation)
    return histories
