"""End-to-end pipeline behaviour."""

from __future__ import annotations

import numpy as np

from phytovision.pipeline import Pipeline
from phytovision.types import AnalysisReport


def test_analyze_returns_valid_report(healthy_image) -> None:
    report = Pipeline.default().analyze(healthy_image)

    assert isinstance(report, AnalysisReport)
    assert report.regions.kind == "plant"
    assert 0.0 <= report.stress.score <= 1.0
    assert 0.0 <= report.stress.confidence <= 1.0
    assert report.stress.label in {"healthy", "mild", "stressed"}


def test_summary_is_json_shaped(healthy_image) -> None:
    summary = Pipeline.default().analyze(healthy_image).summary()
    assert set(summary) >= {"region_kind", "region_count", "stress", "top_reasons"}
    assert set(summary["stress"]) == {"score", "confidence", "label", "model"}


def test_healthy_scores_lower_than_stressed(healthy_image, stressed_image) -> None:
    pipeline = Pipeline.default()
    healthy = pipeline.analyze(healthy_image).stress.score
    stressed = pipeline.analyze(stressed_image).stress.score
    assert healthy < stressed


def test_analyze_records_per_stage_timing(healthy_image) -> None:
    report = Pipeline.default().analyze(healthy_image)
    for stage in ("preprocess", "segment", "regions", "extract", "model", "explain", "total"):
        assert stage in report.timing_ms
        assert report.timing_ms[stage] >= 0.0
    assert "timing_ms" in report.summary()


def test_degenerate_thin_images_do_not_crash(healthy_image) -> None:
    # A 1-pixel-thin mask (a 1xN image, or a wide photo the preprocessor shrinks to 1px on a side)
    # must not raise a raw scikit-image ValueError; it should still give a finite, valid report.
    pipeline = Pipeline.default()
    rng = np.random.default_rng(0)
    for shape in ((1, 1, 3), (1, 200, 3), (200, 1, 3), (3, 5000, 3)):
        report = pipeline.analyze(rng.integers(0, 255, shape, dtype=np.uint8))
        assert 0.0 <= report.stress.score <= 1.0 and 0.0 <= report.stress.confidence <= 1.0
        assert all(np.isfinite(v) for v in report.plant_features.defined().values())


def test_accepts_ndarray_and_path(tmp_path, healthy_image) -> None:
    from PIL import Image as PILImage

    # ndarray path
    Pipeline.default().analyze(healthy_image)

    # file path
    out = tmp_path / "plant.png"
    PILImage.fromarray((healthy_image * 255).astype(np.uint8)).save(out)
    report = Pipeline.default().analyze(out)
    assert report.image_path == str(out)
