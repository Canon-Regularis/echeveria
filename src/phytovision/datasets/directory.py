"""Loader for a plain directory of images (no class-folder layout required).

Use this for a folder of plant photos to batch-analyze. If an image sits in a subdirectory, that
subdirectory's name becomes its label, so a ``root/<label>/<image>`` layout also works; images
directly under the root have no label.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from phytovision.datasets.base import IMAGE_SUFFIXES, DatasetLoader, Sample


class ImageDirectoryLoader(DatasetLoader):
    def __init__(
        self,
        root: str | Path,
        recursive: bool = True,
        source: str | None = None,
        license: str | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"not a directory: {self.root}")
        paths = self.root.rglob("*") if recursive else self.root.iterdir()
        self._samples = [
            Sample(
                image_path=str(path),
                label=(path.parent.name if path.parent != self.root else None),
                source=source,
                license=license,
            )
            for path in sorted(paths)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
