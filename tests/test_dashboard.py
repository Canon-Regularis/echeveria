"""Dashboard helpers (F26). The Streamlit render() needs a running server, so we test the pure
logic it delegates to: image decoding, reason rows, and the bar-chart contribution series."""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image as PILImage

from phytovision.dashboard import contribution_series, decode_image, reason_rows
from phytovision.exceptions import InvalidImageError
from phytovision.pipeline import Pipeline


@pytest.fixture
def report(stressed_image):
    return Pipeline.default().analyze(stressed_image)


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
