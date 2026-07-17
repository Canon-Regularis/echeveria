"""One-call global seeding, so a whole run is reproducible from a single seed.

The stochastic stages that matter each already take an explicit seed: the simulator spawns per-plant
generators, the cross-validation splits and the gradient-boosted model take a ``random_state``, and
the forecasters that draw randomness pass a fixed seed. This utility covers the rest: any stage that
falls back to Python's ``random`` module or numpy's legacy global generator. Call it once at the
start of a run to make those paths deterministic too. It does not reseed a
``numpy.random.Generator`` a stage already constructed with its own seed; those stay independent.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` module and numpy's legacy global generator from one seed.

    Python's ``random`` accepts any integer, but numpy's legacy generator requires a value in
    ``[0, 2**32)``, so the seed is reduced into that range for numpy. A negative or very large
    ``--seed`` is therefore reproducible rather than a crash.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
