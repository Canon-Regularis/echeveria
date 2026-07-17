"""Time-series cross-validation splits that never leak the future into the past.

The classification cross-validation in ``crossval.py`` shuffles rows, which is wrong for a sequence:
a fold must train only on the past and test on the future. An expanding window does that. The train
set is always a prefix ending at the origin; the test set is the next ``horizon`` steps. So the
largest training index is always below the smallest test index, by construction.
"""

from __future__ import annotations

from phytovision.exceptions import ConfigError

Split = tuple[list[int], list[int]]


def expanding_window_splits(
    n_samples: int, min_train: int = 4, horizon: int = 1, step: int = 1
) -> list[Split]:
    """Expanding-window splits over ``n_samples`` points.

    Each split trains on indices ``0 .. origin-1`` and tests on ``origin .. origin+horizon-1``,
    stepping the origin forward by ``step``. A trailing window shorter than ``horizon`` is kept, so
    every reachable future point is scored once.
    """
    if min_train < 2:
        raise ConfigError("expanding-window splits need at least two training points")
    if horizon < 1 or step < 1:
        raise ConfigError("horizon and step must be at least one")

    splits: list[Split] = []
    origin = min_train
    while origin < n_samples:
        test = list(range(origin, min(origin + horizon, n_samples)))
        if not test:
            break
        splits.append((list(range(origin)), test))
        origin += step
    return splits
