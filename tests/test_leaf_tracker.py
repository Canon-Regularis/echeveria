"""Cross-frame leaf tracking and per-leaf trajectory building."""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.pipeline import Pipeline
from phytovision.regions.base import region_from_mask
from phytovision.temporal import (
    FeatureHistory,
    LeafTracker,
    build_leaf_histories,
    stress_forecast,
)
from phytovision.types import RegionSet

_SIZE = 64


def _disk(center: tuple[int, int], radius: int = 6) -> np.ndarray:
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def _frame(centers: list[tuple[int, int]]) -> RegionSet:
    regions = tuple(region_from_mask(i, "leaf", _disk(center)) for i, center in enumerate(centers))
    return RegionSet(regions, "leaf", (_SIZE, _SIZE))


def test_identity_is_preserved_under_slight_movement() -> None:
    # Two leaves that drift a little keep their identities across frames.
    frames = [
        _frame([(30, 18), (30, 46)]),
        _frame([(31, 20), (29, 45)]),
        _frame([(30, 17), (31, 47)]),
    ]
    tracks = LeafTracker().track(frames)
    assert tracks.n_leaves == 2
    assert all(assignment == (0, 1) for assignment in tracks.assignments)
    assert tracks.frames_for(0) == [(0, 0), (1, 0), (2, 0)]
    assert tracks.leaf_ids == [0, 1]


def test_a_new_leaf_gets_a_fresh_identity() -> None:
    tracks = LeafTracker().track([_frame([(30, 18)]), _frame([(30, 18), (30, 46)])])
    assert tracks.assignments == ((0,), (0, 1))  # the second leaf is a new identity
    assert tracks.n_leaves == 2


def test_a_disappearing_leaf_does_not_crash_or_reassign() -> None:
    tracks = LeafTracker().track([_frame([(30, 18), (30, 46)]), _frame([(30, 19)])])
    assert tracks.assignments == ((0, 1), (0,))  # the surviving leaf keeps id 0
    assert tracks.n_leaves == 2


def test_a_disappearing_leaf_does_not_steal_a_valid_match() -> None:
    # A leaf about to vanish must not steal the global Hungarian match of a leaf that moved only a
    # little, which would re-mint the live leaf as a new identity. Over-threshold edges are gated
    # out of the optimization, so the cheap move is chosen and only the vanished leaf is dropped.
    frames = [_frame([(7, 7), (32, 33)]), _frame([(32, 36), (57, 57)])]
    tracks = LeafTracker(max_cost=0.2).track(frames)
    # The moved leaf keeps id 1; only the genuinely new leaf mints a fresh id.
    assert tracks.assignments == ((0, 1), (1, 2))
    assert tracks.n_leaves == 3


def test_a_leaf_that_jumps_across_the_frame_is_a_new_identity() -> None:
    # A single leaf that teleports beyond the match threshold is treated as a new leaf.
    tracks = LeafTracker(max_cost=0.2).track([_frame([(10, 10)]), _frame([(55, 55)])])
    assert tracks.assignments == ((0,), (1,))
    assert tracks.n_leaves == 2


def test_whole_plant_frames_track_as_one_identity() -> None:
    # One region per frame (the whole-plant case) trivially tracks as the same single leaf.
    plant = [_frame([(32, 32)]) for _ in range(4)]
    tracks = LeafTracker().track(plant)
    assert tracks.n_leaves == 1
    assert all(assignment == (0,) for assignment in tracks.assignments)


# --- per-leaf trajectories through the real pipeline ---


def _two_leaf_image(shift: int) -> np.ndarray:
    size = 128
    image = np.ones((size, size, 3), np.float32) * 0.1
    yy, xx = np.mgrid[0:size, 0:size]
    for col in (40 + shift, 88 + shift):
        image[(yy - 64) ** 2 + (xx - col) ** 2 <= 16**2] = np.array([0.15, 0.6, 0.15], np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _one_leaf_image() -> np.ndarray:
    size = 128
    image = np.ones((size, size, 3), np.float32) * 0.1
    yy, xx = np.mgrid[0:size, 0:size]
    image[(yy - 64) ** 2 + (xx - 40) ** 2 <= 16**2] = np.array([0.15, 0.6, 0.15], np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def test_build_leaf_histories_yields_one_sequence_per_leaf() -> None:
    pipeline = Pipeline.from_config({"region_provider": "leaf-instance"})
    reports = [pipeline.analyze(_two_leaf_image(shift)) for shift in (0, 3, -2)]
    timestamps = ["2026-03-01", "2026-03-02", "2026-03-03"]
    histories = build_leaf_histories(reports, timestamps, HeuristicStressModel())
    assert len(histories) == 2
    for observations in histories.values():
        assert len(observations) == 3  # each leaf seen in every frame
        assert all(obs.plant_id.startswith("leaf_") for obs in observations)
        assert all(0.0 <= obs.stress_score <= 1.0 for obs in observations)


def test_build_leaf_histories_truncates_a_vanished_leaf() -> None:
    # When a leaf vanishes, its trajectory ends; the surviving leaf keeps every frame. This covers
    # the shorter-sequence path and confirms per-leaf histories stay in ascending timestamp order.
    pipeline = Pipeline.from_config({"region_provider": "leaf-instance"})
    reports = [
        pipeline.analyze(_two_leaf_image(0)),
        pipeline.analyze(_two_leaf_image(2)),
        pipeline.analyze(_one_leaf_image()),
    ]
    timestamps = ["2026-03-01", "2026-03-02", "2026-03-03"]
    histories = build_leaf_histories(reports, timestamps, HeuristicStressModel())
    assert sorted(len(obs) for obs in histories.values()) == [2, 3]
    for observations in histories.values():
        stamps = [obs.timestamp for obs in observations]
        assert stamps == sorted(stamps)  # ascending timestamp order
        assert stamps == timestamps[: len(stamps)]  # exactly the frames it appeared in


def test_a_per_leaf_history_feeds_a_forecaster() -> None:
    # The point of per-leaf tracking: a forecaster runs on one leaf, not only the whole plant.
    pipeline = Pipeline.from_config({"region_provider": "leaf-instance"})
    reports = [pipeline.analyze(_two_leaf_image(shift)) for shift in (0, 2, -1, 1)]
    histories = build_leaf_histories(
        reports, ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"], HeuristicStressModel()
    )
    leaf_id, observations = next(iter(sorted(histories.items())))
    forecast = stress_forecast(f"leaf_{leaf_id}", observations, horizons=(1, 3))
    assert set(forecast.projected_scores) == {1, 3}
    assert 0.0 <= forecast.confidence <= 1.0


def test_build_leaf_histories_rejects_mismatched_lengths() -> None:
    pipeline = Pipeline.from_config({"region_provider": "leaf-instance"})
    reports = [pipeline.analyze(_two_leaf_image(0))]
    with pytest.raises(ValueError, match="same length"):
        build_leaf_histories(reports, ["2026-03-01", "2026-03-02"], HeuristicStressModel())


# --- the Observation.per_region extension ---


def test_observation_per_region_defaults_empty_and_record_populates_it(healthy_image) -> None:
    from phytovision.temporal.history import Observation

    assert Observation("p", "t", 0.5).per_region == ()  # backward-compatible default

    history = FeatureHistory()
    report = Pipeline.from_config({"region_provider": "leaf-instance"}).analyze(_two_leaf_image(0))
    observation = history.record("plant", "2026-03-01", report)
    assert len(observation.per_region) == len(report.regions)  # one vector per leaf region
    assert all("colour.gcc_mean" in vector for vector in observation.per_region)
