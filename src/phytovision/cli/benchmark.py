"""The ``benchmark`` command: rank the forecasters over a synthetic cohort with time-series CV.

It reads a cohort manifest written by ``simulate`` (one that carries a stress_score column), runs
every forecaster over expanding-window origins, and prints a table ranked by CRPS within each
horizon. The scores are synthetic-trained: they compare forecasters against each other on generated
data, not against a validated succulent prognosis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phytovision.cli._shared import fail, parse_horizons, write_table
from phytovision.evaluation.benchmark import benchmark_forecasters
from phytovision.exceptions import PhytoVisionError
from phytovision.registries import FORECASTERS
from phytovision.simulation import load_history
from phytovision.temporal.forecast import valid_interval_level

_TABLE_FIELDS = [
    "horizon",
    "forecaster",
    "crps",
    "crps_lo",
    "crps_hi",
    "pinball",
    "coverage",
    "mean_width",
    "n",
]


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "benchmark", help="rank forecasters over a synthetic cohort with time-series CV"
    )
    parser.add_argument("manifest", help="a `simulate` cohort manifest with a stress_score column")
    parser.add_argument("--horizons", default="1,3,7", help="comma-separated horizons in steps")
    parser.add_argument(
        "--forecasters", help="comma-separated forecaster names (default: every registered one)"
    )
    parser.add_argument(
        "--min-train", type=int, default=4, help="training points before the first origin"
    )
    parser.add_argument(
        "--interval-level", type=float, default=0.9, help="nominal prediction-interval coverage"
    )
    parser.add_argument("--out", metavar="FILE", help="write the ranked table to .csv or .json")
    parser.add_argument(
        "--mlflow", action="store_true", help="log the run to MLflow (needs the tracking extra)"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if not valid_interval_level(args.interval_level):
        return fail("--interval-level must be in (0, 1)")
    if args.min_train < 2:
        return fail("--min-train must be at least 2")
    horizons = parse_horizons(args.horizons)
    names = _selected_names(args.forecasters)
    if names is not None and (unknown := [name for name in names if name not in FORECASTERS]):
        return fail(f"unknown forecaster(s) {unknown}; available: {FORECASTERS.names()}")

    try:
        history = load_history(args.manifest)
    except (OSError, PhytoVisionError) as exc:
        return fail(str(exc))
    if not history.plant_ids:
        return fail(f"no plant series in {args.manifest}")

    result = benchmark_forecasters(history, names, horizons, args.min_train, args.interval_level)
    rows = result.table()
    if not rows:
        return fail("no forecasts were scored; check the horizons and the min-train setting")

    print("Synthetic-trained forecaster comparison; scores rank models, not a validated prognosis.")
    _print_table(rows)
    if result.skipped:
        print(f"skipped (missing extra): {', '.join(result.skipped)}")

    if args.out:
        try:
            write_table(Path(args.out), _TABLE_FIELDS, rows)
        except OSError as exc:
            return fail(str(exc))
        print(f"wrote the ranked table to {args.out}")

    if args.mlflow:
        try:
            from phytovision.tracking import log_benchmark

            log_benchmark(result, {"plants": len(history.plant_ids), "min_train": args.min_train})
        except ImportError as exc:
            return fail(str(exc))
        print("logged the benchmark to MLflow")
    return 0


def _selected_names(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


def _print_table(rows: list[dict[str, object]]) -> None:
    head = ("horizon", "forecaster", "crps", "pinball", "cover", "width")
    print(f"\n{head[0]:>7}  {head[1]:<16}  {head[2]:>8}  {head[3]:>8}  {head[4]:>6}  {head[5]:>6}")
    for row in rows:
        print(
            f"{row['horizon']:>7}  {str(row['forecaster']):<16}  {row['crps']:>8}  "
            f"{row['pinball']:>8}  {row['coverage']:>6}  {row['mean_width']:>6}"
        )
