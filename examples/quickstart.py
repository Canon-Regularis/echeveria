"""Quickstart: analyze one image and print the explained water-stress verdict.

Run with a real photo:
    python examples/quickstart.py path/to/plant.jpg

With no argument it uses a synthetic green plant, so the example runs with only the base install.
"""

from __future__ import annotations

import sys

import numpy as np

from phytovision import Pipeline


def _synthetic_plant() -> np.ndarray:
    size, radius = 128, 42
    image = np.ones((size, size, 3), np.float32) * np.array((0.1, 0.1, 0.1), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius**2
    image[disk] = np.array((0.15, 0.6, 0.15), np.float32)
    return image


def main() -> None:
    source: str | np.ndarray = sys.argv[1] if len(sys.argv) > 1 else _synthetic_plant()
    report = Pipeline.default().analyze(source)

    stress = report.stress
    print(
        f"Water-stress: {stress.label.upper()}  "
        f"score={stress.score:.2f}  confidence={stress.confidence:.2f}"
    )
    for reason in report.explanation.reasons[:3]:
        print(f"  {reason.direction}: {reason.description} ({reason.feature}={reason.value:.3f})")


if __name__ == "__main__":
    main()
