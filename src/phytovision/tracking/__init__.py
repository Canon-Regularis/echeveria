"""Experiment tracking: log a benchmark run to MLflow when the ``tracking`` extra is installed.

Tracking is opt-in and lazily imported, so the benchmark runs and prints its table with no tracking
dependency. When MLflow is present and requested, one run records the benchmark parameters and every
forecaster's per-horizon metrics, so comparisons are reproducible and diffable across changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phytovision.evaluation.benchmark import BenchmarkResult

_EXTRA_HINT = 'experiment tracking needs the tracking extra: pip install -e ".[tracking]"'


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the optional tracking extra
        raise ImportError(_EXTRA_HINT) from exc
    return mlflow


def log_benchmark(
    result: BenchmarkResult,
    params: dict[str, object],
    experiment: str = "forecaster-benchmark",
) -> None:
    """Log the benchmark parameters and per-forecaster metrics as one MLflow run."""
    mlflow = _require_mlflow()
    mlflow.set_experiment(experiment)  # pragma: no cover - needs the tracking extra installed
    with mlflow.start_run():  # pragma: no cover - needs the tracking extra installed
        mlflow.log_params(params)
        mlflow.log_param("interval_level", result.interval_level)
        if result.skipped:
            mlflow.log_param("skipped", ",".join(result.skipped))
        for score in result.scores:
            prefix = f"{score.name}.h{score.horizon}"
            mlflow.log_metric(f"{prefix}.crps", score.crps)
            mlflow.log_metric(f"{prefix}.pinball", score.pinball)
            mlflow.log_metric(f"{prefix}.coverage", score.coverage)
            mlflow.log_metric(f"{prefix}.mean_width", score.mean_width)
