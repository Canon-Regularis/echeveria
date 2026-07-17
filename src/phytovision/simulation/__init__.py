"""Synthetic dry-down simulation: labelled sequences to train and benchmark against.

The public surface is the parameters, the per-plant and cohort builders, the manifest and events IO,
and the no-image manifest reader. Everything here is labelled synthetic; see ``SYNTHETIC_SOURCE``.
"""

from __future__ import annotations

from phytovision.simulation.dataset import (
    SYNTHETIC_SOURCE,
    SyntheticCohort,
    cohort_history,
    event_rows,
    load_history,
    manifest_rows,
    simulate_cohort,
    write_events,
    write_manifest,
)
from phytovision.simulation.drydown import (
    DryDownParams,
    SyntheticSeries,
    feature_keys,
    simulate_plant,
)

__all__ = [
    "SYNTHETIC_SOURCE",
    "DryDownParams",
    "SyntheticCohort",
    "SyntheticSeries",
    "cohort_history",
    "event_rows",
    "feature_keys",
    "load_history",
    "manifest_rows",
    "simulate_cohort",
    "simulate_plant",
    "write_events",
    "write_manifest",
]
