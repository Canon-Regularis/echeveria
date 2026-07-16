"""Unified dataset access.

Objective 0: give every dataset one loading interface so downstream code depends on ``Sample``
objects, not on Kaggle/Roboflow/Mendeley layout quirks. License/domain metadata travels with each
sample (it must not be lost between download and training).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Image file extensions the folder loaders recognize.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def require_directory(path: str | Path, description: str) -> Path:
    """Return ``path`` as a :class:`Path`; raise ``NotADirectoryError`` if it is not a directory."""
    resolved = Path(path)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {resolved}")
    return resolved


def resolve_root(images_root: str | Path | None, default: Path) -> Path:
    """Return ``images_root`` as a path, or ``default`` when no explicit root is given."""
    return Path(images_root) if images_root is not None else default


@dataclass(frozen=True, slots=True)
class Sample:
    image_path: str
    label: str | None = None  # e.g. "healthy" / "wilted"
    split: str | None = None  # "train" / "val" / "test"
    source: str | None = None  # dataset name / URL
    license: str | None = None
    # Temporal-tracking metadata: which plant this is, and when the image was taken. A sortable
    # timestamp (ISO-8601 works) lets the feature-history store order a plant's observations.
    plant_id: str | None = None
    timestamp: str | None = None
    # A measured water-status value (soil moisture, leaf water content, or similar), when a dataset
    # ships one. Units are dataset specific; validate scores the stress score against it.
    target: float | None = None
    extra: dict[str, object] = field(default_factory=dict)


class DatasetLoader(ABC):
    """Iterates a dataset as ``Sample`` objects with attached provenance."""

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class InMemoryDataset(DatasetLoader):
    """A loader backed by an eagerly built list of samples.

    Subclasses build ``self._samples`` in their constructor; iteration, length, and the label set
    live here, so each loader only has to parse its own on-disk format.
    """

    _samples: list[Sample]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def labels(self) -> list[str]:
        """The sorted set of non-empty labels across the samples."""
        return sorted({sample.label for sample in self._samples if sample.label is not None})
