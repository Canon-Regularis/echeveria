"""Loader for COCO-format object-detection exports, such as those from Roboflow.

Reads a COCO annotations JSON, one Sample per image, with each object's box and category in
``Sample.extra["boxes"]``. Detection data has no single image-level label, so ``Sample.label`` stays
None. This is the ingestion path for the aloe stress detection sets and for a future leaf module.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from phytovision.datasets.base import DatasetLoader, Sample
from phytovision.exceptions import ConfigError


class CocoDetectionLoader(DatasetLoader):
    def __init__(
        self,
        annotations_path: str | Path,
        images_root: str | Path | None = None,
        source: str | None = None,
        license: str | None = None,
        split: str | None = None,
    ) -> None:
        annotations = Path(annotations_path)
        try:
            data = json.loads(annotations.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"could not parse COCO file {annotations}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"COCO file {annotations} must be a JSON object at the top level")
        root = Path(images_root) if images_root is not None else annotations.parent

        # A malformed export (missing a required id/bbox/file_name, or a non-object entry) becomes a
        # clean ConfigError naming the file, rather than a bare KeyError or TypeError.
        try:
            self._samples, self._categories = _parse(data, root, split, source, license)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ConfigError(f"malformed COCO file {annotations}: {exc}") from exc

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def categories(self) -> list[str]:
        """All object categories declared in the annotations file."""
        return list(self._categories)


def _parse(
    data: dict[str, Any],
    root: Path,
    split: str | None,
    source: str | None,
    license: str | None,
) -> tuple[list[Sample], list[str]]:
    """Build samples and the category list from a parsed COCO object; raises on a missing key."""
    category_names = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    categories = sorted(str(name) for name in category_names.values())

    boxes_by_image: dict[int, list[dict[str, object]]] = {}
    for ann in data.get("annotations", []):
        boxes_by_image.setdefault(ann["image_id"], []).append(
            {
                "bbox": ann["bbox"],
                "category": category_names.get(ann["category_id"], str(ann["category_id"])),
            }
        )

    samples = [
        Sample(
            image_path=str(root / img["file_name"]),
            split=split,
            source=source,
            license=license,
            extra={
                "boxes": boxes_by_image.get(img["id"], []),
                "width": img.get("width"),
                "height": img.get("height"),
            },
        )
        for img in data.get("images", [])
    ]
    return samples, categories
