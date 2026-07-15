"""Loader for a YOLO-format detection export (images plus one ``.txt`` label file per image).

Each label file has one row per object: ``class cx cy w h`` with the box normalized to [0, 1] and
centred. Class names are passed in (a YOLO ``data.yaml`` lists them), so this needs no YAML parser;
without names the numeric class id is used. Boxes land in ``Sample.extra["boxes"]`` like the COCO
loader, and ``Sample.label`` stays None because detection data has no single image-level label.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from phytovision.datasets.base import IMAGE_SUFFIXES, DatasetLoader, Sample


class YoloDetectionLoader(DatasetLoader):
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path | None = None,
        class_names: Sequence[str] | None = None,
        source: str | None = None,
        license: str | None = None,
        split: str | None = None,
    ) -> None:
        images = Path(images_dir)
        if not images.is_dir():
            raise NotADirectoryError(f"images directory is not a directory: {images}")
        labels = Path(labels_dir) if labels_dir is not None else images.parent / "labels"
        names = list(class_names) if class_names is not None else None
        self._categories = names

        self._samples: list[Sample] = []
        for image_path in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label_file = labels / f"{image_path.stem}.txt"
            boxes = _parse_labels(label_file, names) if label_file.exists() else []
            self._samples.append(
                Sample(
                    image_path=str(image_path),
                    split=split,
                    source=source,
                    license=license,
                    extra={"boxes": boxes},
                )
            )

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def categories(self) -> list[str] | None:
        """The class names, if they were provided."""
        return list(self._categories) if self._categories is not None else None


def _parse_labels(path: Path, names: list[str] | None) -> list[dict[str, object]]:
    boxes: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        bbox = [float(value) for value in parts[1:5]]  # normalized centre x, y, width, height
        known = names is not None and 0 <= class_id < len(names)
        category = names[class_id] if known else str(class_id)  # type: ignore[index]
        boxes.append({"category": category, "bbox": bbox})
    return boxes
