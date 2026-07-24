"""Unit tests for the small shared helpers used across the package.

Covers the numeric helpers, the least-squares line fit, the dataset-loader base, the evaluation
label and prediction helpers, the segmentation threshold helper, and the ``Reason`` marker.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phytovision._num import EPS, as_float, clip01, normalize01
from phytovision.datasets.base import (
    InMemoryDataset,
    Sample,
    require_directory,
    resolve_root,
)
from phytovision.evaluation._common import binary_labels, predict_labels, trainable_model_names
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.segmentation.plant._threshold_base import threshold_field
from phytovision.temporal._fit import fit_line, slope
from phytovision.types import Reason


def test_clip01_bounds() -> None:
    assert clip01(-0.5) == 0.0
    assert clip01(1.5) == 1.0
    assert clip01(0.3) == 0.3


def test_normalize01_scales_and_clamps() -> None:
    assert normalize01(0.35, 0.2, 0.5) == pytest.approx(0.5)
    assert normalize01(0.1, 0.2, 0.5) == 0.0  # below the range clamps to 0
    assert normalize01(0.9, 0.2, 0.5) == 1.0  # above the range clamps to 1


def test_as_float_uses_default_only_for_none() -> None:
    assert as_float(None, 3.0) == 3.0
    assert as_float(2, 0.0) == pytest.approx(2.0)
    assert isinstance(as_float(2, 0.0), float)
    assert EPS > 0.0


def test_fit_line_recovers_a_known_line() -> None:
    m, b, r2 = fit_line([0.0, 2.0, 4.0, 6.0])
    assert m == pytest.approx(2.0)
    assert b == pytest.approx(0.0)
    assert r2 == pytest.approx(1.0)


def test_fit_line_flat_series_has_zero_slope_and_full_r2() -> None:
    m, b, r2 = fit_line([1.0, 1.0, 1.0])
    assert m == pytest.approx(0.0)
    assert b == pytest.approx(1.0)
    assert r2 == 1.0  # a flat series has no variance to explain


def test_slope_sign_follows_direction() -> None:
    assert slope([0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert slope([3.0, 2.0, 1.0, 0.0]) == pytest.approx(-1.0)


def test_require_directory_returns_path_or_raises(tmp_path: Path) -> None:
    assert require_directory(tmp_path, "root") == tmp_path
    with pytest.raises(NotADirectoryError):
        require_directory(tmp_path / "missing", "root")


def test_resolve_root_prefers_explicit_over_default(tmp_path: Path) -> None:
    default = tmp_path / "manifest_dir"
    assert resolve_root(None, default) == default
    assert resolve_root(tmp_path / "images", default) == tmp_path / "images"


def test_in_memory_dataset_iterates_and_lists_labels() -> None:
    class _Mem(InMemoryDataset):
        def __init__(self, samples: list[Sample]) -> None:
            self._samples = samples

    loader = _Mem(
        [
            Sample(image_path="a", label="wilted"),
            Sample(image_path="b", label=None),
            Sample(image_path="c", label="healthy"),
            Sample(image_path="d", label="wilted"),
        ]
    )
    assert len(loader) == 4
    assert [s.image_path for s in loader] == ["a", "b", "c", "d"]
    assert loader.labels == ["healthy", "wilted"]  # sorted, deduplicated, None excluded


def test_binary_labels_maps_healthy_to_zero() -> None:
    rows = [
        SimpleNamespace(label="healthy"),
        SimpleNamespace(label="wilted"),
        SimpleNamespace(label=None),
    ]
    assert binary_labels(rows, "healthy") == [0, 1, 1]  # type: ignore[arg-type]


def test_predict_labels_thresholds_scores() -> None:
    model = HeuristicStressModel()
    # A strongly green plant reads healthy (0); an empty feature vector reads not-healthy (1).
    labels = predict_labels(model, [{"colour.gcc_mean": 0.42}, {}])
    assert labels == [0, 1]


def test_predict_label_uses_the_shared_healthy_rule() -> None:
    # The fitted-model evaluate modes (--cv, --transfer, --importance) must classify a score exactly
    # as the single-pass mode does. A private 0.5 cut measured a decision rule the tool never ships
    # and flipped every row scoring in [0.33, 0.5) between the two modes on identical data.
    from phytovision.evaluation._common import predict_label
    from phytovision.models.base import bucket_label
    from phytovision.types import StressAssessment

    class _Stub:
        def __init__(self, score: float) -> None:
            self.score = score

        def predict(self, features: object) -> StressAssessment:
            return StressAssessment(self.score, 0.5, bucket_label(self.score), "stub")

    for score in (0.0, 0.2, 0.32, 0.33, 0.4, 0.49, 0.5, 0.66, 1.0):
        single_pass = 0 if bucket_label(score) == "healthy" else 1
        assert predict_label(_Stub(score), {}) == single_pass


def test_trainable_model_names_excludes_the_heuristic() -> None:
    names = trainable_model_names()
    assert set(names) == {"gradient-boosted", "ensemble"}
    assert "heuristic" not in names


def test_threshold_field_splits_and_guards_degenerate_input() -> None:
    assert not threshold_field(np.zeros((4, 4))).any()  # uniform field has no foreground
    assert not threshold_field(np.full((4, 4), np.nan)).any()  # a non-finite field is empty
    split = threshold_field(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert split.sum() == 2  # the high half is foreground


def test_reason_marker_reflects_direction() -> None:
    up = Reason(feature="f", direction="increases", contribution=0.5, value=0.3, description="d")
    down = Reason(feature="f", direction="decreases", contribution=-0.5, value=0.3, description="d")
    assert up.marker == "+"
    assert down.marker == "-"


def test_to_unit_rgb_is_the_one_range_rule() -> None:
    # Every converter in the package (preprocessor, quality, renderers, saliency, occlusion) must
    # answer "is this 8-bit or already normalized" identically, or an image the pipeline scores
    # correctly gets rendered or explained as a different image.
    from phytovision._num import to_unit_rgb

    # A float frame with a stray pixel just over 1.0 is still a [0, 1] image, not an 8-bit one.
    stray = np.full((4, 4, 3), 0.4, np.float32)
    stray[0, 0] = 1.2
    assert to_unit_rgb(stray).mean() == pytest.approx(0.4, abs=0.05)

    # An integer frame is scaled by its own dtype range, so 16-bit is not read as 8-bit.
    u16 = np.full((4, 4, 3), int(0.4 * 65535), np.uint16)
    assert to_unit_rgb(u16).mean() == pytest.approx(0.4, abs=0.01)
    u8 = np.full((4, 4, 3), 102, np.uint8)
    assert to_unit_rgb(u8).mean() == pytest.approx(0.4, abs=0.01)

    # A clearly 8-bit float frame is still divided by 255.
    eight_bit = np.full((4, 4, 3), 102.0, np.float64)
    assert to_unit_rgb(eight_bit).mean() == pytest.approx(0.4, abs=0.01)
    assert to_unit_rgb(stray).shape == (4, 4, 3)  # always H x W x 3 in [0, 1]
    assert to_unit_rgb(stray).min() >= 0.0
    assert to_unit_rgb(stray).max() <= 1.0
