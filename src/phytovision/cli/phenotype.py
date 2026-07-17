"""The ``phenotype`` command: high-throughput trajectory phenotyping over a timestamped manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from phytovision.cli._shared import (
    add_pipeline_args,
    build_pipeline,
    fail,
    parse_horizons,
    write_table,
)
from phytovision.datasets.manifest import CsvManifestLoader
from phytovision.exceptions import PhytoVisionError
from phytovision.registries import DROUGHT_STAGE_MODELS, FORECASTERS
from phytovision.temporal import (
    build_history,
    plant_early_warnings,
    plant_forecasts,
    plant_trends,
)
from phytovision.types import PlantFeatures


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "phenotype", help="high-throughput trajectory phenotyping over a timestamped manifest"
    )
    parser.add_argument("manifest", help="CSV/TSV manifest with plant_id and timestamp columns")
    parser.add_argument(
        "--images-root", metavar="DIR", help="image folder (defaults to the manifest folder)"
    )
    parser.add_argument("--out", required=True, metavar="FILE", help="output path, .csv or .json")
    parser.add_argument(
        "--horizons", default="1,3,7", help="comma-separated forecast horizons in observation steps"
    )
    parser.add_argument(
        "--forecaster",
        default="linear-trend",
        choices=FORECASTERS.names(),
        help="trajectory forecaster (richer models report a prediction interval per horizon)",
    )
    add_pipeline_args(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    horizons = parse_horizons(args.horizons)
    try:
        pipeline = build_pipeline(args)
        loader = CsvManifestLoader(args.manifest, args.images_root or None)
        forecaster = FORECASTERS.create(args.forecaster)
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    history = build_history(pipeline, loader)
    if not history.plant_ids:
        return fail(f"no plant-tagged, timestamped samples in {args.manifest}")

    trends = plant_trends(history)
    warnings = plant_early_warnings(history)
    forecasts = plant_forecasts(history, horizons, forecaster)
    stage_model = DROUGHT_STAGE_MODELS.create("rule-based")

    forecast_columns = [f"forecast_h{h}" for h in horizons]
    interval_columns = [f"forecast_h{h}_{bound}" for h in horizons for bound in ("lo", "hi")]
    fieldnames = [
        "plant_id",
        "n",
        "first_timestamp",
        "last_timestamp",
        "latest_score",
        "latest_stage",
        "trend_direction",
        "trend_slope",
        "early_warning_flagged",
        "steps_to_stressed",
        *forecast_columns,
        *interval_columns,
        "forecast_method",
        "forecast_confidence",
    ]
    records: list[dict[str, object]] = []
    for plant_id in history.plant_ids:
        series = history.series_for(plant_id)
        latest = series[-1]
        forecast = forecasts[plant_id]
        latest_features = PlantFeatures.from_values(latest.features)
        row: dict[str, object] = {
            "plant_id": plant_id,
            "n": len(series),
            "first_timestamp": series[0].timestamp,
            "last_timestamp": latest.timestamp,
            "latest_score": round(latest.stress_score, 4),
            "latest_stage": stage_model.stage(latest_features)["stage"],
            "trend_direction": trends[plant_id].direction,
            "trend_slope": round(trends[plant_id].slope, 6),
            "early_warning_flagged": warnings[plant_id].flagged,
            "steps_to_stressed": forecast.steps_to_stressed,
            "forecast_method": forecast.method,
            "forecast_confidence": round(forecast.confidence, 4),
        }
        for h in horizons:
            row[f"forecast_h{h}"] = round(forecast.projected_scores.get(h, 0.0), 4)
            if h in forecast.lower:
                row[f"forecast_h{h}_lo"] = round(forecast.lower[h], 4)
                row[f"forecast_h{h}_hi"] = round(forecast.upper[h], 4)
        records.append(row)

    out = Path(args.out)
    try:
        write_table(out, fieldnames, records)
    except OSError as exc:
        return fail(str(exc))
    print(f"wrote {len(records)} plant trajectory row(s) to {out}")
    return 0
