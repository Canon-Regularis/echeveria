"""Command-line surface: ``phytovision [-v] {analyze,batch,train,evaluate} ...``."""

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
from phytovision.datasets.folder import FolderClassificationLoader
from phytovision.evaluation.metrics import binary_metrics
from phytovision.exceptions import ConfigError, PhytoVisionError
from phytovision.io import load_image
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
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

    train = sub.add_parser("train", help="train a model on a labelled folder and save it")
    train.add_argument("directory", help="labelled folder: root/<label>/<image>")
    train.add_argument("--out", required=True, metavar="FILE", help="output model path (.joblib)")
    train.add_argument(
        "--segmenter", default="exg-otsu", choices=SEGMENTERS.names(), help="plant segmenter"
    )
    train.add_argument(
        "--healthy-label", default="healthy", help="label treated as the healthy class"
    )

    evaluate = sub.add_parser("evaluate", help="score a model on a labelled folder")
    evaluate.add_argument("directory", help="labelled folder: root/<label>/<image>")
    _add_pipeline_args(evaluate)
    evaluate.add_argument(
        "--healthy-label", default="healthy", help="label treated as the healthy class"
    )

    serve = sub.add_parser("serve", help="run the HTTP API (needs the 'api' extra)")
    serve.add_argument("--host", default="127.0.0.1", help="bind host")
    serve.add_argument("--port", type=int, default=8000, help="bind port")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    handlers = {
        "analyze": _analyze,
        "batch": _batch,
        "train": _train,
        "evaluate": _evaluate,
        "serve": _serve,
    }
    return handlers[args.command](args)


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
    parser.add_argument(
        "--model-path", metavar="FILE", help="load a trained .joblib model (overrides --model)"
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        # Scope debug to our own logger so -v doesn't surface third-party (PIL, ...) noise.
        logging.getLogger("phytovision").setLevel(logging.DEBUG)


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    if args.config:
        pipeline = Pipeline.from_config(_load_config(args.config))
    else:
        pipeline = Pipeline.from_names(model=args.model, segmenter=args.segmenter)
    if args.model_path:
        pipeline = pipeline.with_model(GradientBoostedStressModel.load(args.model_path))
    return pipeline


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
    except (OSError, ImportError, PhytoVisionError) as exc:
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
    except (OSError, ImportError, PhytoVisionError) as exc:
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


def _train(args: argparse.Namespace) -> int:
    try:
        loader = FolderClassificationLoader(args.directory)
        pipeline = Pipeline.from_names(segmenter=args.segmenter)  # model is unused for extraction
    except (OSError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = list(analyze_dataset(pipeline, loader))
    if not rows:
        print(f"error: no images found in {args.directory}", file=sys.stderr)
        return 2

    labels = [0 if row.label == args.healthy_label else 1 for row in rows]
    if len(set(labels)) < 2:
        found = sorted({str(row.label) for row in rows})
        print(f"error: need at least two classes; found labels {found}", file=sys.stderr)
        return 2

    feature_dicts = [row.features for row in rows]
    feature_keys = sorted({key for record in feature_dicts for key in record})
    try:
        model = GradientBoostedStressModel(feature_keys=feature_keys).fit(feature_dicts, labels)
        model.save(args.out)
    except (OSError, ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"trained on {len(rows)} image(s); saved model to {args.out}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    try:
        pipeline = _build_pipeline(args)
        loader = FolderClassificationLoader(args.directory)
    except (OSError, ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = list(analyze_dataset(pipeline, loader))
    if not rows:
        print(f"error: no images found in {args.directory}", file=sys.stderr)
        return 2

    y_true = [0 if row.label == args.healthy_label else 1 for row in rows]
    y_pred = [
        0 if row.stress_label == "healthy" else 1 for row in rows
    ]  # mild/stressed -> positive
    metrics = binary_metrics(y_true, y_pred)
    print(
        f"images: {metrics.n}   accuracy: {metrics.accuracy:.3f}   "
        f"precision: {metrics.precision:.3f}   recall: {metrics.recall:.3f}   f1: {metrics.f1:.3f}"
    )
    print("confusion (rows = true, cols = predicted):")
    print("                 healthy   stressed")
    print(f"  true healthy   {metrics.tn:>7}   {metrics.fp:>8}")
    print(f"  true stressed  {metrics.fn:>7}   {metrics.tp:>8}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print('error: serving needs the "api" extra: pip install -e ".[api]"', file=sys.stderr)
        return 2
    uvicorn.run("phytovision.api:app", host=args.host, port=args.port)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
