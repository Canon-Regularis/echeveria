"""Helpers shared by the command modules: argument wiring, pipeline building, and output.

Keeping these in one place lets each command module stay focused on its own parser and handler.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from phytovision.config import read_config
from phytovision.models.base import StressModel
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.persistence import load_saved
from phytovision.pipeline import Pipeline
from phytovision.registries import EXPLAINERS, SEGMENTERS, STRESS_MODELS
from phytovision.temporal import DEFAULT_HORIZONS


def fail(message: str) -> int:
    """Print an error to stderr and return the process exit code for a handled failure."""
    print(f"error: {message}", file=sys.stderr)
    return 2


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        # Scope debug to our own logger so -v doesn't surface third-party (PIL, ...) noise.
        logging.getLogger("phytovision").setLevel(logging.DEBUG)


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default="heuristic", choices=STRESS_MODELS.names(), help="stress model"
    )
    parser.add_argument(
        "--segmenter", default="exg-otsu", choices=SEGMENTERS.names(), help="plant segmenter"
    )
    parser.add_argument(
        "--explainer",
        default="feature-contribution",
        choices=EXPLAINERS.names(),
        help="explanation method",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="pipeline config (.toml/.json); overrides --model/--segmenter",
    )
    parser.add_argument(
        "--model-path", metavar="FILE", help="load a trained .joblib model (overrides --model)"
    )
    parser.add_argument(
        "--strict-schema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fail (or, with --no-strict-schema, do not fail) if a loaded model's feature schema "
        "differs from the live extractor output; unset keeps the loaded model's own policy",
    )


def build_pipeline(args: argparse.Namespace, model: StressModel | None = None) -> Pipeline:
    if args.config:
        pipeline = Pipeline.from_config(read_config(args.config))
    else:
        pipeline = Pipeline.from_names(
            model=args.model, segmenter=args.segmenter, explainer=args.explainer
        )
    chosen = model
    if chosen is None and args.model_path:
        chosen = stress_model_from_path(args.model_path)
    if chosen is not None:
        # Only override the model's own drift policy when the flag was given; unset (None) keeps the
        # strict_schema the model was saved with, rather than silently resetting it to lenient.
        strict = getattr(args, "strict_schema", None)
        if strict is not None:
            apply_strict_schema(chosen, strict)
        pipeline = pipeline.with_model(chosen)
    return pipeline


def apply_strict_schema(model: StressModel, strict: bool) -> None:
    """Set the schema-drift mode on a loaded model and, for an ensemble, on its members."""
    if hasattr(model, "strict_schema"):
        model.strict_schema = strict
    for member in getattr(model, "members", ()):
        apply_strict_schema(member, strict)


def stress_model_from_path(path: str) -> StressModel:
    """The stress model to run in the pipeline; a conformal file contributes its wrapped model."""
    loaded = load_saved(path)
    return loaded.model if isinstance(loaded, SplitConformalClassifier) else loaded


def write_table(out: Path, fieldnames: list[str], records: list[dict[str, object]]) -> None:
    """Write records to the output path as JSON (.json) or CSV (any other suffix).

    Both formats project each record onto ``fieldnames`` so every row carries the identical column
    set: the CSV writer already fills a missing key with a blank, and the JSON branch mirrors that
    with ``null`` rather than emitting objects whose keys vary by row (e.g. a plant with a
    degenerate forecast that has no interval columns).
    """
    if out.suffix.lower() == ".json":
        rows = [{key: record.get(key) for key in fieldnames} for record in records]
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(records)


def parse_horizons(text: str) -> tuple[int, ...]:
    """Parse a comma-separated horizon list, falling back to the defaults on empty or bad input.

    Positive horizons only, deduplicated and sorted, so a stray "1,1,3" cannot create duplicate
    forecast columns.
    """
    try:
        horizons = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        return DEFAULT_HORIZONS
    return tuple(sorted({h for h in horizons if h > 0})) or DEFAULT_HORIZONS
