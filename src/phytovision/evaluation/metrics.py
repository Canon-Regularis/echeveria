"""Binary classification metrics for the ``evaluate`` command.

Dependency-light on purpose: evaluating the heuristic model should not require the ``ml`` extra.
Class 1 is "stressed" (the positive class); class 0 is "healthy".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Confusion-matrix counts and the scores derived from them."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> BinaryMetrics:
    """Confusion-matrix metrics for two equal-length 0/1 sequences (1 = stressed = positive)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = fn = tn = 0
    for true, pred in zip(y_true, y_pred, strict=True):
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
        else:
            tn += 1
    return BinaryMetrics(tp=tp, fp=fp, fn=fn, tn=tn)
