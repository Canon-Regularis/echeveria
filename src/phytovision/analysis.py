"""Run the pipeline over a dataset and collect one result row per image.

This is the shared engine behind the ``batch`` export, training, and evaluation: it iterates any
``DatasetLoader``, analyzes each image, and yields a flat row with the sample's metadata, the stress
result, and the full feature vector. Images that fail to analyze are logged and skipped rather than
aborting the whole run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from phytovision.datasets.base import DatasetLoader
from phytovision.exceptions import PhytoVisionError
from phytovision.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    """One image's result: provenance, the stress prediction, and its feature vector."""

    image_path: str
    label: str | None
    split: str | None
    source: str | None
    score: float
    confidence: float
    stress_label: str
    model: str
    features: dict[str, float]

    def as_record(self) -> dict[str, object]:
        """Flatten to a single dict (features merged in) for CSV/JSON rows."""
        record: dict[str, object] = {
            "image_path": self.image_path,
            "label": self.label,
            "split": self.split,
            "source": self.source,
            "score": self.score,
            "confidence": self.confidence,
            "stress_label": self.stress_label,
            "model": self.model,
        }
        record.update(self.features)
        return record


_BASE_COLUMNS = [
    "image_path",
    "label",
    "split",
    "source",
    "score",
    "confidence",
    "stress_label",
    "model",
]


def feature_table(rows: Iterable[AnalysisRow]) -> tuple[list[str], list[dict[str, object]]]:
    """Materialize rows into ``(fieldnames, records)``.

    ``fieldnames`` is the metadata columns followed by the sorted union of feature keys, so a CSV
    has a stable, complete header even if some rows differ.
    """
    records = [row.as_record() for row in rows]
    feature_keys = sorted({key for record in records for key in record} - set(_BASE_COLUMNS))
    return _BASE_COLUMNS + feature_keys, records


def analyze_dataset(pipeline: Pipeline, loader: DatasetLoader) -> Iterator[AnalysisRow]:
    """Analyze every sample in ``loader`` with ``pipeline``, yielding one ``AnalysisRow`` each.

    Samples whose image cannot be read or analyzed are logged at WARNING and skipped.
    """
    for sample in loader:
        try:
            report = pipeline.analyze(sample.image_path)
        except (FileNotFoundError, PhytoVisionError) as exc:
            logger.warning("skipping %s: %s", sample.image_path, exc)
            continue
        yield AnalysisRow(
            image_path=sample.image_path,
            label=sample.label,
            split=sample.split,
            source=sample.source,
            score=report.stress.score,
            confidence=report.stress.confidence,
            stress_label=report.stress.label,
            model=report.stress.model_name,
            features=report.plant_features.defined(),
        )
