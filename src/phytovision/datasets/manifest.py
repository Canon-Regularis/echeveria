"""Loader for a CSV or TSV manifest that maps columns to ``Sample`` fields.

A manifest is the most portable dataset format: one row per image, with columns for the path and any
provenance. Column names are configurable so it reads exports from other tools without renaming.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from phytovision.datasets.base import DatasetLoader, Sample
from phytovision.exceptions import ConfigError


class CsvManifestLoader(DatasetLoader):
    def __init__(
        self,
        manifest_path: str | Path,
        images_root: str | Path | None = None,
        image_column: str = "image_path",
        label_column: str = "label",
        split_column: str = "split",
        source_column: str = "source",
        license_column: str = "license",
        plant_id_column: str = "plant_id",
        timestamp_column: str = "timestamp",
    ) -> None:
        manifest = Path(manifest_path)
        root = Path(images_root) if images_root is not None else manifest.parent
        delimiter = "\t" if manifest.suffix.lower() in {".tsv", ".tab"} else ","
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None or image_column not in reader.fieldnames:
                raise ConfigError(f"manifest {manifest} has no {image_column!r} column")
            self._samples = [
                Sample(
                    image_path=str(root / image_value),
                    label=_clean(row.get(label_column)),
                    split=_clean(row.get(split_column)),
                    source=_clean(row.get(source_column)),
                    license=_clean(row.get(license_column)),
                    plant_id=_clean(row.get(plant_id_column)),
                    timestamp=_clean(row.get(timestamp_column)),
                )
                for row in reader
                if (image_value := _clean(row.get(image_column)))
            ]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


def _clean(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None
