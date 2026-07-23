"""The ``simulate`` command: write a synthetic dry-down cohort to a manifest and an events table.

The output is labelled synthetic in every ``source`` cell and in this command's notice. It exists to
unblock the forecasters, the survival model, and the benchmark, none of which have real succulent
time series to fit; it is not a substitute for measured data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phytovision.cli._shared import fail
from phytovision.exceptions import PhytoVisionError
from phytovision.simulation import (
    DryDownParams,
    simulate_cohort,
    write_events,
    write_manifest,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "simulate", help="write a synthetic dry-down cohort (manifest plus events table)"
    )
    parser.add_argument("--out", required=True, metavar="FILE", help="manifest output path (.csv)")
    parser.add_argument(
        "--events",
        metavar="FILE",
        help="events-table output path (defaults to the manifest name with an _events suffix)",
    )
    parser.add_argument("--plants", type=int, default=30, help="number of synthetic plants")
    parser.add_argument("--steps", type=int, default=20, help="observations per plant")
    parser.add_argument(
        "--decline-rate", type=float, default=0.12, help="base latent decline rate per step"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (same seed, same cohort)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    # A non-positive count would otherwise write one observation per plant (the initial state) and a
    # report claiming a step count that never happened, so reject it before doing any work.
    if args.steps < 1:
        return fail("--steps must be at least 1")
    if args.plants < 1:
        return fail("--plants must be at least 1")
    params = DryDownParams(n_steps=args.steps, base_decline_rate=args.decline_rate)
    try:
        cohort = simulate_cohort(args.plants, params, seed=args.seed)
        manifest = write_manifest(cohort, args.out)
        events = write_events(cohort, args.events or _default_events_path(args.out))
    except (OSError, PhytoVisionError) as exc:
        return fail(str(exc))

    wilted = sum(1 for plant in cohort.series if not plant.censored)
    censored = len(cohort) - wilted
    print("Synthetic dry-down cohort; every row is labelled synthetic, not measured data.")
    print(f"wrote {len(cohort)} plants x {args.steps} steps to {manifest}")
    print(f"wrote {len(cohort)} event rows to {events} ({wilted} wilted, {censored} censored)")
    return 0


def _default_events_path(manifest_out: str) -> Path:
    """Place the events table beside the manifest, tagged with an ``_events`` suffix."""
    out = Path(manifest_out)
    return out.with_name(f"{out.stem}_events{out.suffix or '.csv'}")
