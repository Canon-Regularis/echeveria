"""Regression tests for a batch of edge-case robustness fixes.

The preprocessor tolerates a float image with a stray over-one pixel, mask cleanup keeps a diagonal
structure, the config schema rejects an unknown component key, the conformal quantile survives float
rounding at an exact rank, step-zero watering is applied, and a prevalent survival case is excluded.
"""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.exceptions import ConfigError


def test_preprocessor_keeps_a_float_image_with_a_stray_over_one_pixel() -> None:
    from phytovision.preprocessing.basic import ResizeNormalizePreprocessor

    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    image[0, 0, 0] = 1.0001  # a stray pixel just over 1.0 (float rounding), not an 8-bit image
    out = ResizeNormalizePreprocessor().process(image)
    # The old max > 1.0 test would divide the whole frame by 255 and darken it ~255x; it must not.
    assert 0.4 < float(out.mean()) < 0.6


def test_clean_mask_keeps_a_diagonal_line() -> None:
    from phytovision.segmentation.cleanup import clean_mask

    shape = (100, 100)  # min_object_fraction 0.002 -> min_size 20
    mask = np.zeros(shape, dtype=bool)
    for i in range(30):
        mask[10 + i, 10 + i] = True  # a 30-pixel diagonal line: one 8-connected object
    out = clean_mask(mask, shape, closing_radius=0)
    # With 4-connectivity each diagonal pixel is its own size-1 object and the line vanishes; the
    # 8-connected cleanup keeps it as one object above the threshold.
    assert int(out.sum()) == 30


def test_config_schema_rejects_unknown_component_keys() -> None:
    from phytovision.config_schema import PipelineConfig

    with pytest.raises(ConfigError, match="unknown key"):
        PipelineConfig.from_mapping({"model": {"name": "heuristic", "bias": 0.5}})  # forgot params
    with pytest.raises(ConfigError, match="unknown key"):
        PipelineConfig.from_mapping({"model": {"name": "heuristic", "parms": {"x": 1}}})  # typo


def test_conformal_quantile_survives_float_rounding_at_an_exact_rank() -> None:
    from phytovision.models.conformal import conformal_quantile

    scores = list(range(1, 1000))  # n = 999, so the k-th smallest score equals k
    # (n + 1)(1 - alpha) = 1000 * 0.941 = 941 exactly; float error must not push k to 942.
    assert conformal_quantile(scores, alpha=0.059) == 941


def test_simulate_applies_watering_at_step_zero() -> None:
    from phytovision.simulation import DryDownParams, simulate_plant

    params = DryDownParams(
        n_steps=3,
        process_noise=0.0,
        observation_noise=0.0,
        feature_noise=0.0,
        initial_stress=0.8,
        watering_steps=(0,),
        watering_amount=0.5,
    )
    plant = simulate_plant("a", params, np.random.default_rng(0), decline_rate=0.1)
    # Watering at step 0 reduces the initial state (0.8 - 0.5 = 0.3) rather than being ignored.
    assert plant.latent[0] == pytest.approx(0.3)


def test_derive_records_excludes_a_frame_zero_crosser() -> None:
    from phytovision.models.survival.cohort import derive_records
    from phytovision.temporal import FeatureHistory
    from phytovision.temporal.history import Observation

    history = FeatureHistory()
    for step, score in enumerate([0.90, 0.90]):  # already over the cut at the first frame
        history.add(
            Observation(
                plant_id="prevalent", timestamp=f"2024-01-{step + 1:02d}", stress_score=score
            )
        )
    for step, score in enumerate(
        [0.20, 0.30, 0.70, 0.90]
    ):  # crosses later, keeps the cohort non-empty
        history.add(
            Observation(
                plant_id="incident", timestamp=f"2024-02-{step + 1:02d}", stress_score=score
            )
        )

    ids = {record.plant_id for record in derive_records(history).records}
    assert "prevalent" not in ids  # a prevalent case has no pre-event window, so it is excluded
    assert "incident" in ids
