"""Temporal ingest: build a FeatureHistory (and per-plant trends) from a tagged image manifest."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from phytovision.datasets.manifest import CsvManifestLoader
from phytovision.pipeline import Pipeline
from phytovision.temporal import build_history, plant_trends


def _save(path, image) -> None:
    PILImage.fromarray((image * 255).astype(np.uint8)).save(path)


def test_build_history_and_trends_from_a_manifest(tmp_path, healthy_image, stressed_image) -> None:
    _save(tmp_path / "a.png", healthy_image)  # the earlier, healthier plant
    _save(tmp_path / "b.png", stressed_image)  # the later, stressed plant
    manifest = tmp_path / "m.csv"
    # Rows are out of chronological order so the assertions exercise the timestamp sort, not the
    # manifest's row order: the later stressed image is listed first.
    manifest.write_text("image_path,plant_id,timestamp\nb.png,p1,2026-03-02\na.png,p1,2026-03-01\n")

    history = build_history(Pipeline.default(), CsvManifestLoader(manifest))
    assert history.plant_ids == ["p1"]
    series = history.series_for("p1")
    assert [obs.timestamp for obs in series] == ["2026-03-01", "2026-03-02"]  # chronological
    assert (
        series[0].stress_score < series[1].stress_score
    )  # healthy (earlier) below stressed (later)

    trend = plant_trends(history)["p1"]
    assert trend.n == 2
    # A healthy plant followed by a stressed one reads as rising stress.
    assert trend.end_score > trend.start_score
    assert trend.direction == "rising"


def test_build_history_skips_untagged_samples(tmp_path, healthy_image) -> None:
    _save(tmp_path / "a.png", healthy_image)
    _save(tmp_path / "b.png", healthy_image)
    manifest = tmp_path / "m.csv"
    # The second row has no plant_id or timestamp, so it cannot join a time series and is skipped.
    manifest.write_text("image_path,plant_id,timestamp\na.png,p1,2026-03-01\nb.png,,\n")

    history = build_history(Pipeline.default(), CsvManifestLoader(manifest))
    assert len(history) == 1
    assert history.plant_ids == ["p1"]


def test_build_history_skips_a_tagged_sample_whose_image_is_missing(
    tmp_path, healthy_image
) -> None:
    _save(tmp_path / "a.png", healthy_image)
    manifest = tmp_path / "m.csv"
    # The second row is fully tagged but its image is missing, so analysis fails and it is skipped.
    manifest.write_text(
        "image_path,plant_id,timestamp\na.png,p1,2026-03-01\ngone.png,p1,2026-03-02\n"
    )

    history = build_history(Pipeline.default(), CsvManifestLoader(manifest))
    assert [obs.timestamp for obs in history.series_for("p1")] == ["2026-03-01"]


def test_build_history_skips_a_decompression_bomb(tmp_path, healthy_image, monkeypatch) -> None:
    import PIL.Image

    _save(tmp_path / "a.png", healthy_image)  # 128x128, analyzes cleanly
    _save(tmp_path / "big.png", np.zeros((210, 210, 3), np.float32))  # trips PIL's bomb guard below
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "image_path,plant_id,timestamp\na.png,p1,2026-03-01\nbig.png,p1,2026-03-02\n"
    )
    # PIL raises DecompressionBombError above 2x this. a.png stays under it; big.png trips it.
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 20000)

    history = build_history(Pipeline.default(), CsvManifestLoader(manifest))
    # The bomb image is skipped rather than crashing the batch, and the good sample survives.
    assert [obs.timestamp for obs in history.series_for("p1")] == ["2026-03-01"]
