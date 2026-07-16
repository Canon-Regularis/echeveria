"""Folder-per-class classification loader.

Matches the layout of several v1 datasets (Kaggle healthy/wilted, Mendeley aloe classes):
``root/<label>/<image>``. Each subdirectory name is the class label.
"""

from __future__ import annotations

from pathlib import Path

from phytovision.datasets.base import (
    IMAGE_SUFFIXES,
    InMemoryDataset,
    Sample,
    require_directory,
)


class FolderClassificationLoader(InMemoryDataset):
    def __init__(
        self,
        root: str | Path,
        source: str | None = None,
        license: str | None = None,
        split: str | None = None,
    ) -> None:
        self.root = require_directory(root, "dataset root")
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
