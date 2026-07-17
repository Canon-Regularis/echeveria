"""Loader for a CSV or TSV manifest that maps columns to ``Sample`` fields.

A manifest is the most portable dataset format: one row per image, with columns for the path and any
provenance. Column names are configurable so it reads exports from other tools without renaming.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from phytovision.datasets.base import InMemoryDataset, Sample, resolve_root
from phytovision.exceptions import ConfigError


class CsvManifestLoader(InMemoryDataset):
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
        target_column: str = "target",
    ) -> None:
        manifest = Path(manifest_path)
        root = resolve_root(images_root, manifest.parent)
        delimiter = "\t" if manifest.suffix.lower() in {".tsv", ".tab"} else ","
        # utf-8-sig drops the BOM that Excel and pandas prepend, so the first column name matches.
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            header = reader.fieldnames
            if header is not None:
                # Strip header whitespace so "image_path, plant_id" (spaces after commas) maps.
                reader.fieldnames = [name.strip() for name in header]
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
                    target=_clean_float(row.get(target_column), manifest, target_column),
                )
                for row in reader
                if (image_value := _clean(row.get(image_column)))
            ]


def _clean(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _clean_float(value: str | None, manifest: Path, column: str) -> float | None:
    """Parse a numeric column, tolerating a blank cell; a non-numeric or non-finite value is a clean
    error. ``float`` accepts ``nan``/``inf``, which would silently poison a whole regression report,
    so those are rejected here too."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        raise ConfigError(
            f"manifest {manifest} has a non-numeric {column!r} value: {text!r}"
        ) from None
    if not math.isfinite(parsed):
        raise ConfigError(f"manifest {manifest} has a non-finite {column!r} value: {text!r}")
    return parsed
