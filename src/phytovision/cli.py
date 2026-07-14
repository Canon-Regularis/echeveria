"""Command-line surface: ``phytovision [-v] {analyze,batch} ...``."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from phytovision.analysis import analyze_dataset, feature_table
from phytovision.datasets.directory import ImageDirectoryLoader
from phytovision.exceptions import ConfigError, PhytoVisionError
from phytovision.io import load_image
from phytovision.pipeline import Pipeline
from phytovision.registries import SEGMENTERS, STRESS_MODELS
from phytovision.visualize import render_overlay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phytovision",
        description="Explainable plant water-stress analysis.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging on stderr"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyze one image for water stress")
    analyze.add_argument("image", help="path to an RGB image")
    _add_pipeline_args(analyze)
    analyze.add_argument("--json", action="store_true", help="emit the JSON summary only")
    analyze.add_argument("--features", action="store_true", help="include the full feature vector")
    analyze.add_argument("--save-overlay", metavar="PNG", help="write an annotated overlay image")

    batch = sub.add_parser("batch", help="analyze every image in a folder and export a table")
    batch.add_argument(
        "directory", help="folder of images (recurses; subfolder names become labels)"
    )
    batch.add_argument("--out", required=True, metavar="FILE", help="output path, .csv or .json")
    _add_pipeline_args(batch)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "batch":
        return _batch(args)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover - argparse guards this
    return 2


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default="heuristic", choices=STRESS_MODELS.names(), help="stress model"
    )
    parser.add_argument(
        "--segmenter", default="exg-otsu", choices=SEGMENTERS.names(), help="plant segmenter"
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="pipeline config (.toml/.json); overrides --model/--segmenter",
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        # Scope debug to our own logger so -v doesn't surface third-party (PIL, ...) noise.
        logging.getLogger("phytovision").setLevel(logging.DEBUG)


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    if args.config:
        return Pipeline.from_config(_load_config(args.config))
    return Pipeline.from_names(model=args.model, segmenter=args.segmenter)


def _load_config(path: str) -> Mapping[str, object]:
    file = Path(path)
    text = file.read_text(encoding="utf-8")  # raises FileNotFoundError if missing
    suffix = file.suffix.lower()
    if suffix not in {".toml", ".json"}:
        raise ConfigError(f"config must be .toml or .json: {file}")
    try:
        data = tomllib.loads(text) if suffix == ".toml" else json.loads(text)
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not parse config {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a table/object at the top level: {file}")
    return data


def _analyze(args: argparse.Namespace) -> int:
    try:
        pipeline = _build_pipeline(args)
        report = pipeline.analyze(args.image)
        if args.save_overlay:
            render_overlay(load_image(args.image), report).save(args.save_overlay)
    except (OSError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = report.summary()
        if args.features:
            payload["features"] = report.plant_features.defined()
        print(json.dumps(payload, indent=2))
        return 0

    stress = report.stress
    print(
        f"Water-stress: {stress.label.upper()}  "
        f"score={stress.score:.2f}  confidence={stress.confidence:.2f}  "
        f"(model: {stress.model_name})"
    )
    print(f"Regions analysed: {len(report.regions)} ({report.regions.kind})")
    if report.explanation.reasons:
        print("Top reasons:")
        for reason in report.explanation.reasons:
            marker = "+" if reason.direction == "increases" else "-"
            print(f"  [{marker}] {reason.feature}={reason.value:.3f} - {reason.description}")
    if args.features:
        print("Features:")
        for key, value in sorted(report.plant_features.defined().items()):
            print(f"  {key} = {value:.4f}")
    if args.save_overlay:
        print(f"Overlay written to {args.save_overlay}")
    return 0


def _batch(args: argparse.Namespace) -> int:
    try:
        pipeline = _build_pipeline(args)
        loader = ImageDirectoryLoader(args.directory)
    except (OSError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    fieldnames, records = feature_table(analyze_dataset(pipeline, loader))
    if not records:
        print(f"error: no images found in {args.directory}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    else:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(records)
    print(f"wrote {len(records)} row(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
