"""Dataset loaders."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image as PILImage

from phytovision.datasets.coco import CocoDetectionLoader
from phytovision.datasets.folder import FolderClassificationLoader
from phytovision.exceptions import ConfigError


def test_folder_loader_reads_classes_and_metadata(tmp_path) -> None:
    for label in ("healthy", "wilted"):
        class_dir = tmp_path / label
        class_dir.mkdir()
        for i in range(2):
            PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(class_dir / f"{i}.png")

    loader = FolderClassificationLoader(tmp_path, source="unit-test", license="CC-BY-4.0")

    assert len(loader) == 4
    assert loader.labels == ["healthy", "wilted"]
    sample = next(iter(loader))
    assert sample.source == "unit-test"
    assert sample.license == "CC-BY-4.0"
    assert sample.label in {"healthy", "wilted"}


def test_coco_loader_reads_boxes_and_categories(tmp_path) -> None:
    PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "Healthy"}, {"id": 2, "name": "Unhealthy"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"image_id": 1, "category_id": 2, "bbox": [5, 5, 3, 3]},
        ],
    }
    annotations = tmp_path / "_annotations.coco.json"
    annotations.write_text(json.dumps(coco))

    loader = CocoDetectionLoader(annotations, source="roboflow")

    assert len(loader) == 1
    assert loader.categories == ["Healthy", "Unhealthy"]
    sample = next(iter(loader))
    assert sample.label is None
    assert sample.source == "roboflow"
    assert sample.image_path.endswith("a.png")
    boxes = sample.extra["boxes"]
    assert isinstance(boxes, list)
    assert {box["category"] for box in boxes} == {"Healthy", "Unhealthy"}


def test_coco_invalid_json_is_a_clean_config_error(tmp_path) -> None:
    # A truncated or corrupt download must name the file, not raise a bare JSONDecodeError.
    bad = tmp_path / "bad.coco.json"
    bad.write_text("{not valid json")
    with pytest.raises(ConfigError, match="could not parse"):
        CocoDetectionLoader(bad)


def test_coco_non_object_top_level_is_a_clean_config_error(tmp_path) -> None:
    listy = tmp_path / "list.coco.json"
    listy.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ConfigError, match="JSON object"):
        CocoDetectionLoader(listy)


def test_coco_missing_required_key_is_a_clean_config_error(tmp_path) -> None:
    # An image entry without file_name must be a clean, file-named error, not a bare KeyError.
    malformed = tmp_path / "m.coco.json"
    malformed.write_text(json.dumps({"images": [{"id": 1}], "annotations": [], "categories": []}))
    with pytest.raises(ConfigError, match="malformed"):
        CocoDetectionLoader(malformed)


def test_coco_empty_object_yields_no_samples(tmp_path) -> None:
    empty = tmp_path / "e.coco.json"
    empty.write_text(json.dumps({}))
    loader = CocoDetectionLoader(empty)
    assert len(loader) == 0 and loader.categories == []
