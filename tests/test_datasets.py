"""Folder-per-class dataset loader."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

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
