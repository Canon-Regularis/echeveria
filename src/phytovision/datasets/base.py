"""Unified dataset access.

Objective 0: give every dataset one loading interface so downstream code depends on ``Sample``
objects, not on Kaggle/Roboflow/Mendeley layout quirks. License/domain metadata travels with each
sample (it must not be lost between download and training).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

# Image file extensions the folder loaders recognize.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


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
    extra: dict[str, object] = field(default_factory=dict)


class DatasetLoader(ABC):
    """Iterates a dataset as ``Sample`` objects with attached provenance."""

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]: ...

    @abstractmethod
    def __len__(self) -> int: ...
