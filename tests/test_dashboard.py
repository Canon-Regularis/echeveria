"""Dashboard helpers. The Streamlit render() needs a running server, so we test the pure logic it
delegates to: decoding, reason rows, contribution series, disease, timing, and observations."""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image as PILImage

from phytovision.dashboard import (
    contribution_series,
    decode_image,
    disease_series,
    forecast_band,
    forecast_points,
    observation_table,
    quality_banner,
    reason_rows,
    timing_rows,
)
from phytovision.exceptions import InvalidImageError
from phytovision.pipeline import Pipeline
from phytovision.quality import QualityAssessment
from phytovision.serving import attach_heads
from phytovision.temporal import Forecast, Observation


@pytest.fixture
def report(stressed_image):
    return Pipeline.default().analyze(stressed_image)


def test_quality_banner_is_none_when_usable(report) -> None:
    assert quality_banner(report) is None


def test_quality_banner_summarises_warnings(report) -> None:
    warning = "image detail is very low, so the score may be unreliable"
    unusable = replace(
        report,
        quality=QualityAssessment(False, ("blurry",), (warning,), 0.0, 0.3, 0.2),
    )
    banner = quality_banner(unusable)
    assert banner is not None
    assert "Low input quality" in banner
    assert warning in banner


def _png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray((image * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_roundtrips_a_valid_png(healthy_image) -> None:
    decoded = decode_image(_png_bytes(healthy_image))
    assert decoded.shape == healthy_image.shape
    assert decoded.dtype == np.uint8


def test_decode_image_rejects_junk_bytes() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")


def test_decode_image_rejects_a_decompression_bomb(healthy_image, monkeypatch) -> None:
    import PIL.Image

    # Force PIL to treat a normal image as a bomb; it must become a clean domain error.
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 4)
    with pytest.raises(InvalidImageError):
        decode_image(_png_bytes(healthy_image))


def test_reason_rows_expose_the_expected_columns(report) -> None:
    assert report.explanation.reasons  # a stressed plant must produce drivers to explain
    rows = reason_rows(report)
    assert len(rows) == len(report.explanation.reasons)
    assert set(rows[0]) == {"feature", "value", "effect on stress", "contribution", "why"}


def test_contribution_series_is_aligned_and_sorted_by_magnitude(report) -> None:
    assert len(report.explanation.reasons) >= 2  # need >=2 drivers for ordering to be meaningful
    features, contributions = contribution_series(report)
    assert len(features) == len(contributions) == len(report.explanation.reasons)
    magnitudes = [abs(value) for value in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_contribution_series_reorders_scrambled_reasons(report) -> None:
    # The explainer already sorts reasons strongest-first, so feed them in the wrong (ascending)
    # order to prove the magnitude sort genuinely lives in contribution_series, not just upstream.
    ascending = tuple(sorted(report.explanation.reasons, key=lambda r: abs(r.contribution)))
    scrambled = replace(report, explanation=replace(report.explanation, reasons=ascending))
    features, contributions = contribution_series(scrambled)

    magnitudes = [abs(value) for value in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # The strongest driver leads, and its feature stays paired with its own contribution.
    strongest = max(ascending, key=lambda r: abs(r.contribution))
    assert features[0] == strongest.feature
    assert contributions[0] == strongest.contribution


def test_disease_series_reads_the_head_output(stressed_image) -> None:
    report = attach_heads(Pipeline.default(), disease=True).analyze(stressed_image)
    labels, probabilities = disease_series(report)
    assert set(labels) == {"healthy", "lesion-like"}
    assert len(probabilities) == len(labels)
    assert sum(probabilities) == pytest.approx(1.0)


def test_disease_series_is_empty_without_the_head(report) -> None:
    # The default pipeline attaches no head, so there is nothing to plot.
    assert disease_series(report) == ([], [])


def test_observation_table_rows(report) -> None:
    observations = [
        Observation("p1", "2026-03-01", 0.123456),
        Observation("p1", "2026-03-02", 0.7),
    ]
    rows = observation_table(observations)
    assert rows == [
        {"timestamp": "2026-03-01", "stress_score": 0.1235},
        {"timestamp": "2026-03-02", "stress_score": 0.7},
    ]


def test_forecast_points_are_ascending_by_horizon() -> None:
    forecast = Forecast("p", 0.1, 0.5, {3: 0.8, 1: 0.6, 7: 1.0}, 2, 0.5, "note")
    steps, scores = forecast_points(forecast)
    assert steps == [1, 3, 7]
    assert scores == [0.6, 0.8, 1.0]


def test_forecast_band_returns_bounds_for_horizons_with_intervals() -> None:
    forecast = Forecast(
        "p",
        0.1,
        0.5,
        {1: 0.6, 3: 0.8},
        2,
        0.5,
        "note",
        lower={1: 0.55, 3: 0.70},
        upper={1: 0.65, 3: 0.90},
    )
    horizons, lower, upper = forecast_band(forecast)
    assert horizons == [1, 3]
    assert lower == [0.55, 0.70]
    assert upper == [0.65, 0.90]


def test_forecast_band_is_empty_without_intervals() -> None:
    forecast = Forecast("p", 0.1, 0.5, {1: 0.6}, None, 0.1, "note")
    assert forecast_band(forecast) == ([], [], [])


def test_timing_rows_from_a_timed_report(report) -> None:
    # Pipeline.analyze records per-stage timing, so a real report yields stage/ms rows.
    assert report.timing_ms
    rows = timing_rows(report)
    assert {row["stage"] for row in rows} == set(report.timing_ms)
    # Pin each ms value back to the report, so a wrong stage->ms mapping cannot pass.
    for row in rows:
        assert row["ms"] == round(report.timing_ms[row["stage"]], 1)
        assert row["ms"] >= 0.0
