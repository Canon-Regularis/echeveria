"""Regression tests for a batch of IO, dtype, and numeric edge-case fixes.

A non-finite image is rejected, colour math survives uint8, interior edge density excludes a curved
boundary, the dataset loaders fail cleanly on non-UTF-8 input (or tolerate it), an empty images-root
falls back to the default, the linear forecast caps its time-to-stressed and keeps a non-zero band
at the ceiling, and a corrupt model file is wrapped in a clean error.
"""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.exceptions import ConfigError, InvalidImageError
from phytovision.regions.base import region_from_mask


def test_analyze_rejects_a_non_finite_image() -> None:
    from phytovision.pipeline import Pipeline

    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    image[0, 0, 0] = np.nan
    with pytest.raises(
        InvalidImageError
    ):  # a NaN pixel is rejected loudly, not silently corrupting
        Pipeline.default().analyze(image)


def test_colour_features_do_not_overflow_on_uint8() -> None:
    from phytovision.phenotyping.colour import ColourFeatures

    image = np.full((16, 16, 3), 200, dtype=np.uint8)  # r + g + b = 600 wraps in uint8
    values = ColourFeatures().extract(image, region_from_mask(0, "plant", np.ones((16, 16), bool)))
    # gcc = g / (r + g + b) is 1/3 for a grey pixel; the uint8 wrap-around gave ~2.27.
    assert values.values["colour.gcc_mean"] == pytest.approx(1.0 / 3.0, abs=0.01)


def test_interior_edge_density_zero_for_a_solid_disk() -> None:
    from phytovision.phenotyping.texture import TextureFeatures

    image = np.full((40, 40, 3), 0.1, dtype=np.float32)
    yy, xx = np.mgrid[0:40, 0:40]
    disk = (yy - 20) ** 2 + (xx - 20) ** 2 <= 12**2
    image[disk] = 0.6  # a uniform-colour disk: zero internal texture, only a curved outline
    values = TextureFeatures().extract(image, region_from_mask(0, "plant", disk)).values
    # The 3x3 erosion excludes the diagonal boundary the cross element used to leak into the value.
    assert values["texture.edge_density"] < 1e-6


def test_coco_loader_rejects_non_utf8(tmp_path) -> None:
    from phytovision.datasets.coco import CocoDetectionLoader

    path = tmp_path / "a.json"
    path.write_bytes(
        '{"images":[],"categories":[{"id":1,"name":"Aloe café"}],"annotations":[]}'.encode("cp1252")
    )
    with pytest.raises(ConfigError):
        CocoDetectionLoader(path)


def test_manifest_loader_rejects_non_utf8(tmp_path) -> None:
    from phytovision.datasets.manifest import CsvManifestLoader

    path = tmp_path / "m.csv"
    path.write_bytes("image_path,source\na.png,Aloe café farm\n".encode("cp1252"))
    with pytest.raises(ConfigError):
        list(CsvManifestLoader(path))


def test_yolo_loader_tolerates_a_non_utf8_label(tmp_path) -> None:
    from phytovision.datasets.yolo import YoloDetectionLoader

    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"image")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_bytes(
        b"0 0.5 0.5 0.2 0.3\n\x80 corrupt\n"
    )  # valid line + a stray byte

    samples = list(YoloDetectionLoader(str(images)))  # must not raise
    assert len(samples) == 1
    assert samples[0].extra["boxes"] == [{"category": "0", "bbox": [0.5, 0.5, 0.2, 0.3]}]


def test_resolve_root_treats_empty_string_as_unset(tmp_path) -> None:
    from pathlib import Path

    from phytovision.datasets.base import resolve_root

    default = tmp_path / "default"
    assert resolve_root("", default) == default  # empty string falls back to the default folder
    assert resolve_root(None, default) == default
    assert resolve_root("explicit", default) == Path("explicit")


def test_linear_forecast_caps_time_to_stressed() -> None:
    from phytovision.temporal.forecast import forecast_scores

    forecast = forecast_scores("p", [0.5, 0.5, 0.5, 0.5 + 1e-9], (1, 3, 7))
    # A near-flat slope used to project a crossing hundreds of millions of steps out; it is now
    # None, matching the richer forecasters' capped search.
    assert forecast.steps_to_stressed is None


def test_linear_forecast_interval_does_not_collapse_at_the_ceiling() -> None:
    from phytovision.temporal.forecast import forecast_scores

    forecast = forecast_scores("p", [0.80, 0.84, 0.88, 0.92, 0.96], (1, 3, 7))
    for horizon in (1, 3, 7):
        # Centring the band on the clipped projection keeps a non-zero width even at the ceiling.
        assert forecast.upper[horizon] - forecast.lower[horizon] > 0.0


def test_read_envelope_wraps_a_corrupt_file(tmp_path) -> None:
    pytest.importorskip("joblib")
    from phytovision.models.persistence import read_envelope

    path = tmp_path / "bad.joblib"
    path.write_bytes(
        b"\x1f\x8b\x08\x00" + b"garbage"
    )  # a truncated gzip magic, not a real envelope
    with pytest.raises(ConfigError):  # a decompressor OSError becomes a clean ConfigError
        read_envelope(path)
