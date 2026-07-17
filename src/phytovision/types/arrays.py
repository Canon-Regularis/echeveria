"""Array type aliases for the pipeline contract; the runtime is a plain ``numpy.ndarray``."""

from __future__ import annotations

import numpy as np

Image = np.ndarray  # H x W x 3, RGB. uint8 for raw input, float32 in [0, 1] once preprocessed.
Mask = np.ndarray  # H x W, boolean. True = foreground / inside region.
