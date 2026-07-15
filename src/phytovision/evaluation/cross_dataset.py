"""Leave-one-dataset-out transfer evaluation (F14).

Train on every dataset but one, test on the held-out one, and repeat for each dataset. This is the
honest test of whether the model learned water stress or one dataset's quirks: a model that only
memorized artifacts scores well within a dataset but poorly on a new one. Datasets are identified by
``Sample.source``, which every row already carries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from phytovision.analysis import AnalysisRow
from phytovision.evaluation._common import feature_keys_of, fit_predict_labels, model_factory
from phytovision.evaluation.metrics import BinaryMetrics, binary_metrics
from phytovision.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TransferMatrix:
    """Held-out accuracy per dataset when trained on all the others."""

    entries: tuple[tuple[str, BinaryMetrics], ...]

    @property
    def datasets(self) -> list[str]:
        return [name for name, _ in self.entries]

    @property
    def mean_accuracy(self) -> float:
        """Mean of the held-out accuracies: how well the model transfers on average."""
        if not self.entries:
            return 0.0
        return float(np.mean([metrics.accuracy for _, metrics in self.entries]))

    def metrics_for(self, dataset: str) -> BinaryMetrics:
        return dict(self.entries)[dataset]


def leave_one_dataset_out(
    rows: Iterable[AnalysisRow],
    *,
    healthy_label: str = "healthy",
    model: str = "gradient-boosted",
) -> TransferMatrix:
    """Hold out each dataset in turn, training on the rest, and score the held-out dataset.

    Folds whose training data would carry only one class are skipped with a warning.

    :param model: which trainable model to fit (``gradient-boosted`` or ``ensemble``).
    :raises ConfigError: if fewer than two datasets are present, or the model cannot train.
    :raises ImportError: if the ``ml`` extra (scikit-learn) is not installed.
    """
    factory = model_factory(model)  # resolve early so an untrainable model fails before any work
    rows = [row for row in rows if row.source is not None]
    sources = sorted({row.source for row in rows if row.source is not None})
    if len(sources) < 2:
        raise ConfigError("leave-one-dataset-out needs at least two datasets (sources)")

    entries: list[tuple[str, BinaryMetrics]] = []
    for held in sources:
        train_rows = [row for row in rows if row.source != held]
        test_rows = [row for row in rows if row.source == held]
        train_labels = [_label(row, healthy_label) for row in train_rows]
        if len(set(train_labels)) < 2:
            logger.warning("skipping held-out dataset %s: training data has a single class", held)
            continue
        keys = feature_keys_of([row.features for row in train_rows])
        predictions = fit_predict_labels(
            [row.features for row in train_rows],
            train_labels,
            [row.features for row in test_rows],
            keys,
            factory,
        )
        test_labels = [_label(row, healthy_label) for row in test_rows]
        entries.append((held, binary_metrics(test_labels, predictions)))

    if not entries:
        raise ConfigError("no dataset could be held out with two training classes")
    return TransferMatrix(tuple(entries))


def _label(row: AnalysisRow, healthy_label: str) -> int:
    return 0 if row.label == healthy_label else 1
