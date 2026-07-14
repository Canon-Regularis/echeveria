"""Dataset loaders."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image as PILImage

from phytovision.datasets.coco import CocoDetectionLoader
from phytovision.datasets.folder import FolderClassificationLoader


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
