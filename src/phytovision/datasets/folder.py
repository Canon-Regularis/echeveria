"""Folder-per-class classification loader.

Matches the layout of several v1 datasets (Kaggle healthy/wilted, Mendeley aloe classes):
``root/<label>/<image>``. Each subdirectory name is the class label.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from phytovision.datasets.base import IMAGE_SUFFIXES, DatasetLoader, Sample


class FolderClassificationLoader(DatasetLoader):
    def __init__(
        self,
        root: str | Path,
        source: str | None = None,
        license: str | None = None,
        split: str | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"dataset root is not a directory: {self.root}")
        self.source = source
        self.license = license
        self.split = split
        self._samples = self._scan()

    def _scan(self) -> list[Sample]:
        samples: list[Sample] = []
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in IMAGE_SUFFIXES:
                    samples.append(
                        Sample(
                            image_path=str(img),
                            label=class_dir.name,
                            split=self.split,
                            source=self.source,
                            license=self.license,
                        )
                    )
        return samples

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def labels(self) -> list[str]:
        return sorted({s.label for s in self._samples if s.label is not None})
