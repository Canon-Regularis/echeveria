"""The ``phenotype`` command: high-throughput trajectory phenotyping over a timestamped manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from phytovision.cli._shared import (
    add_pipeline_args,
    build_pipeline,
    fail,
    parse_horizons,
    write_table,
)
from phytovision.datasets.manifest import CsvManifestLoader
from phytovision.exceptions import InsufficientDataError, PhytoVisionError
from phytovision.models.survival import fit_cohort_survival
from phytovision.registries import DROUGHT_STAGE_MODELS, FORECASTERS, SURVIVAL_MODELS
from phytovision.temporal import (
    build_history,
    plant_early_warnings,
    plant_forecasts,
    plant_trends,
)
from phytovision.types import PlantFeatures

if TYPE_CHECKING:
    from phytovision.models.survival import SurvivalFit

_SURVIVAL_FIELDS = ["median_time_to_wilt", "time_to_wilt_lo", "time_to_wilt_hi", "survival_basis"]


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
    parser.add_argument(
        "--survival-model",
        default="weibull-aft",
        choices=SURVIVAL_MODELS.names(),
        help="survival model for a per-plant median time-to-wilt (needs the stats extra)",
    )
    parser.add_argument(
        "--survival-window", type=int, default=3, help="early observations used as covariates"
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
    survival_fit = _survival_or_notice(history, args.survival_model, args.survival_window)

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
        *_SURVIVAL_FIELDS,
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
            **survival_row(survival_fit, plant_id),
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


def _survival_or_notice(history: object, model: str, window: int) -> SurvivalFit | None:
    """Fit the cohort survival, or print a notice and skip it when the stats extra is absent."""
    try:
        return fit_cohort_survival(history, model, window)  # type: ignore[arg-type]
    except ImportError:
        print('survival omitted: install the stats extra (pip install -e ".[stats]")')
        return None
    except InsufficientDataError:
        print("survival omitted: no plant has two or more observations")
        return None


def survival_row(fit: SurvivalFit | None, plant_id: str) -> dict[str, object]:
    """The four survival columns for one plant: blanks and a basis when survival is unavailable.

    The two unavailable cases are named honestly: ``unavailable-stats-extra`` when the fit could not
    run at all, ``insufficient-observations`` when the fit ran but this plant was dropped for having
    fewer than two observations, so the missing extra is never falsely implicated.
    """
    if fit is None:
        return _blank_survival("unavailable-stats-extra")
    if plant_id not in fit.per_plant:
        return _blank_survival("insufficient-observations")
    plant = fit.per_plant[plant_id]
    return {
        "median_time_to_wilt": "" if plant.median is None else round(plant.median, 3),
        "time_to_wilt_lo": "" if plant.lower is None else round(plant.lower, 3),
        "time_to_wilt_hi": "" if plant.upper is None else round(plant.upper, 3),
        "survival_basis": plant.basis,
    }


def _blank_survival(basis: str) -> dict[str, object]:
    blanks: dict[str, object] = dict.fromkeys(_SURVIVAL_FIELDS[:3], "")
    blanks["survival_basis"] = basis
    return blanks
