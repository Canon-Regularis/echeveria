"""Micro-benchmark for the classical no-ML pipeline.

The core selling point is a pipeline that runs with no ML stack, so speed matters. Run the real
benchmark with ``pytest tests/benchmarks --benchmark-enable --no-cov``. Under the default run
benchmarking is disabled, and the test just executes once as a correctness check.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from phytovision.pipeline import Pipeline  # noqa: E402


def _blob() -> np.ndarray:
    size, radius = 128, 42
    image = np.ones((size, size, 3), np.float32) * np.array((0.1, 0.1, 0.1), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius**2
    image[disk] = np.array((0.15, 0.6, 0.15), np.float32)
    return image


def test_pipeline_analyze_benchmark(benchmark) -> None:
    image = _blob()
    pipeline = Pipeline.default()
    report = benchmark(pipeline.analyze, image)
    assert report.stress.label in {"healthy", "mild", "stressed"}
