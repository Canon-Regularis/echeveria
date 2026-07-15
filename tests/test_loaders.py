"""CSV-manifest and YOLO dataset loaders, plus the DATASET_LOADERS registry."""

from __future__ import annotations

import pytest

from phytovision.datasets.manifest import CsvManifestLoader
from phytovision.datasets.yolo import YoloDetectionLoader
from phytovision.exceptions import ConfigError
from phytovision.registries import DATASET_LOADERS


def test_csv_manifest_loads_samples(tmp_path) -> None:
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,label,source\na.png,healthy,ds1\nb.png,wilted,ds1\n")
    samples = list(CsvManifestLoader(manifest))

    assert len(samples) == 2
    assert samples[0].label == "healthy"
    assert samples[0].source == "ds1"
    assert samples[0].image_path == str(tmp_path / "a.png")
    assert samples[0].split is None  # absent column stays None
    # Absent optional columns must yield None (row.get, not row[...]), not raise KeyError.
    assert samples[0].plant_id is None
    assert samples[0].timestamp is None


def test_csv_tsv_delimiter_and_blank_rows(tmp_path) -> None:
    manifest = tmp_path / "m.tsv"
    manifest.write_text("image_path\tlabel\na.png\thealthy\n\t\n")
    samples = list(CsvManifestLoader(manifest))
    assert len(samples) == 1  # the blank row is skipped
    assert samples[0].label == "healthy"


def test_csv_manifest_carries_temporal_metadata(tmp_path) -> None:
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "image_path,plant_id,timestamp\na.png,plant-1,2026-03-01\nb.png,plant-1,2026-03-02\n"
    )
    samples = list(CsvManifestLoader(manifest))
    assert [s.plant_id for s in samples] == ["plant-1", "plant-1"]
    assert [s.timestamp for s in samples] == ["2026-03-01", "2026-03-02"]


def test_csv_missing_image_column_errors(tmp_path) -> None:
    manifest = tmp_path / "m.csv"
    manifest.write_text("path,label\na.png,healthy\n")
    with pytest.raises(ConfigError, match="image_path"):
        CsvManifestLoader(manifest)


def _yolo_dataset(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "img1.png").touch()
    (labels / "img1.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.05 0.05\n")
    return images, labels


def test_yolo_loads_boxes_with_class_names(tmp_path) -> None:
    images, labels = _yolo_dataset(tmp_path)
    loader = YoloDetectionLoader(images, labels, class_names=["leaf", "flower"])
    sample = list(loader)[0]
    boxes = sample.extra["boxes"]

    assert sample.label is None  # detection data has no image-level label
    assert len(boxes) == 2
    assert boxes[0]["category"] == "leaf"
    assert boxes[0]["bbox"] == [0.5, 0.5, 0.2, 0.3]
    assert boxes[1]["category"] == "flower"
    assert loader.categories == ["leaf", "flower"]


def test_yolo_missing_labels_yield_empty_boxes(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "img1.png").touch()
    loader = YoloDetectionLoader(images)  # default labels dir does not exist
    assert list(loader)[0].extra["boxes"] == []


def test_yolo_rejects_a_non_directory(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        YoloDetectionLoader(tmp_path / "nope")


def test_dataset_loaders_registry(tmp_path) -> None:
    assert {"folder", "directory", "coco", "csv", "yolo"} <= set(DATASET_LOADERS.names())
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,label\na.png,healthy\n")
    loader = DATASET_LOADERS.create("csv", manifest_path=str(manifest))
    assert len(loader) == 1
