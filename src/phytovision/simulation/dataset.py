"""Assemble synthetic plants into a cohort, and move it to and from disk.

A cohort is many :class:`SyntheticSeries` drawn from one seed, so a run reproduces byte for byte.
The cohort serializes two ways. The manifest is one row per observation: it carries the columns
``CsvManifestLoader`` already reads (image path, label, plant id, timestamp, source, target), plus
the observed ``stress_score`` and the synthetic feature columns. The events table is one row per
plant with its duration and a censoring flag, which is what the survival model consumes. Reading the
manifest back rebuilds a ``FeatureHistory`` directly from the score and feature columns, so the
forecasters and the benchmark run on synthetic data with no image files on disk.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phytovision.exceptions import ConfigError
from phytovision.models.base import bucket_label
from phytovision.simulation.drydown import (
    DryDownParams,
    SyntheticSeries,
    feature_keys,
    simulate_plant,
)
from phytovision.temporal.history import FeatureHistory, Observation

# The provenance tag written into every synthetic row, so a synthetic cohort is never mistaken for
# measured data downstream.
SYNTHETIC_SOURCE = "synthetic-drydown"

# The manifest starts with the columns CsvManifestLoader reads, then the observed score, then the
# feature columns. The events table is one row per plant for the survival model.
_MANIFEST_META = (
    "image_path",
    "label",
    "plant_id",
    "timestamp",
    "source",
    "target",
    "stress_score",
)
_EVENT_FIELDS = ("plant_id", "decline_rate", "duration", "event_time", "event_observed", "censored")


@dataclass(frozen=True, slots=True)
class SyntheticCohort:
    """A batch of synthetic plants drawn from one seed under one set of parameters."""

    series: tuple[SyntheticSeries, ...]
    params: DryDownParams

    def __len__(self) -> int:
        return len(self.series)


def simulate_cohort(
    n_plants: int, params: DryDownParams | None = None, seed: int = 0
) -> SyntheticCohort:
    """Draw ``n_plants`` independent dry-downs. The same seed reproduces the cohort exactly."""
    if n_plants < 1:
        raise ConfigError("a cohort needs at least one plant")
    settings = params or DryDownParams()
    # Spawn one child seed per plant so plant i draws the same stream regardless of cohort size.
    children = np.random.SeedSequence(seed).spawn(n_plants)
    series = tuple(
        simulate_plant(f"plant_{i:03d}", settings, np.random.default_rng(child))
        for i, child in enumerate(children)
    )
    return SyntheticCohort(series, settings)


def cohort_history(cohort: SyntheticCohort) -> FeatureHistory:
    """Collect every plant's observations into a ``FeatureHistory`` for temporal analysis."""
    history = FeatureHistory()
    for plant in cohort.series:
        for observation in plant.observations:
            history.add(observation)
    return history


def manifest_rows(cohort: SyntheticCohort) -> Iterator[dict[str, object]]:
    """One row per observation, with the loader columns plus the score and feature columns."""
    for plant in cohort.series:
        for step, observation in enumerate(plant.observations):
            # Bucket the rounded score that the row stores, not the full-precision one, so the label
            # and the stress_score column never straddle a cut (e.g. 0.6599996 stored as 0.66).
            score = round(observation.stress_score, 6)
            row: dict[str, object] = {
                "image_path": f"synthetic/{plant.plant_id}/{step:03d}.png",
                "label": bucket_label(score),
                "plant_id": plant.plant_id,
                "timestamp": observation.timestamp,
                "source": SYNTHETIC_SOURCE,
                "target": round(plant.latent[step], 6),
                "stress_score": score,
            }
            row.update({key: round(value, 6) for key, value in observation.features.items()})
            yield row


def event_rows(cohort: SyntheticCohort) -> Iterator[dict[str, object]]:
    """One row per plant with its duration, event time, and censoring flag for survival analysis."""
    for plant in cohort.series:
        yield {
            "plant_id": plant.plant_id,
            "decline_rate": round(plant.decline_rate, 6),
            # +1 to match the survival contract: duration is a 1-based observation count (>= 1), the
            # same convention derive_records uses, so a crossing at the first step is 1, never 0.
            "duration": plant.duration + 1,
            "event_time": plant.event_time,
            "event_observed": int(not plant.censored),
            "censored": int(plant.censored),
        }


def write_manifest(cohort: SyntheticCohort, path: str | Path) -> Path:
    """Write the per-observation manifest. Returns the path it wrote."""
    fieldnames = [*_MANIFEST_META, *feature_keys()]
    return _write_csv(path, fieldnames, manifest_rows(cohort))


def write_events(cohort: SyntheticCohort, path: str | Path) -> Path:
    """Write the per-plant events table. Returns the path it wrote."""
    return _write_csv(path, list(_EVENT_FIELDS), event_rows(cohort))


def load_history(manifest_path: str | Path) -> FeatureHistory:
    """Rebuild a ``FeatureHistory`` from a synthetic manifest, reading scores and features directly.

    This is the no-image path: it needs the ``plant_id``, ``timestamp``, and ``stress_score``
    columns, and treats every namespaced column (one containing a dot) as a feature.
    """
    manifest = Path(manifest_path)
    history = FeatureHistory()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = {"plant_id", "timestamp", "stress_score"} - set(fields)
        if missing:
            raise ConfigError(f"manifest {manifest} is missing column(s): {sorted(missing)}")
        feature_columns = [name for name in fields if "." in name]
        for row in reader:
            features = {
                name: _numeric(manifest, name, row[name])
                for name in feature_columns
                if (row.get(name) or "").strip()
            }
            history.add(
                Observation(
                    plant_id=row["plant_id"],
                    timestamp=row["timestamp"],
                    stress_score=_numeric(manifest, "stress_score", row["stress_score"]),
                    features=features,
                )
            )
    return history


def _numeric(manifest: Path, column: str, value: str) -> float:
    """Parse a manifest cell to a finite float, or raise a clean ConfigError naming the column."""
    try:
        parsed = float(value)
    except ValueError:
        raise ConfigError(
            f"manifest {manifest} has a non-numeric {column!r} value: {value!r}"
        ) from None
    if not math.isfinite(parsed):
        raise ConfigError(f"manifest {manifest} has a non-finite {column!r} value: {value!r}")
    return parsed


def _write_csv(path: str | Path, fieldnames: list[str], rows: Iterator[dict[str, object]]) -> Path:
    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return out
