"""Command-line surface: ``phytovision [-v] {analyze,batch,train,evaluate,serve} ...``."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from phytovision.analysis import AnalysisRow, analyze_dataset, feature_table
from phytovision.datasets.directory import ImageDirectoryLoader
from phytovision.datasets.folder import FolderClassificationLoader
from phytovision.evaluation._common import to_plant_features
from phytovision.evaluation.cross_dataset import leave_one_dataset_out
from phytovision.evaluation.crossval import grouped_stratified_cv
from phytovision.evaluation.importance import permutation_importance
from phytovision.evaluation.metrics import binary_metrics
from phytovision.exceptions import ConfigError, PhytoVisionError
from phytovision.explainability.counterfactual import counterfactuals
from phytovision.io import load_image
from phytovision.models.base import StressModel
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.persistence import build_manifest, load_saved, save_model
from phytovision.models.stress.ensemble import EnsembleStressModel
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.pipeline import Pipeline
from phytovision.registries import EXPLAINERS, SEGMENTERS, STRESS_MODELS
from phytovision.visualize import render_overlay

_TRAINABLE_MODELS = ("gradient-boosted", "ensemble")


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
    analyze.add_argument("--timing", action="store_true", help="show per-stage wall-clock timing")
    analyze.add_argument(
        "--counterfactual",
        action="store_true",
        help="report the smallest feature changes that would flip the verdict",
    )
    analyze.add_argument(
        "--conformal",
        action="store_true",
        help="print the conformal label set (needs a model saved with train --calibrate)",
    )

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
        "--model",
        default="gradient-boosted",
        choices=_TRAINABLE_MODELS,
        help="model to train",
    )
    train.add_argument(
        "--segmenter", default="exg-otsu", choices=SEGMENTERS.names(), help="plant segmenter"
    )
    train.add_argument(
        "--healthy-label", default="healthy", help="label treated as the healthy class"
    )
    train.add_argument(
        "--calibrate",
        type=float,
        metavar="FRAC",
        help="hold out FRAC of the data to calibrate a conformal wrapper and save it",
    )
    train.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="conformal miscoverage rate, so coverage is 1 - alpha (with --calibrate)",
    )

    evaluate = sub.add_parser("evaluate", help="score a model on labelled folders")
    evaluate.add_argument(
        "directory",
        nargs="+",
        help="labelled folder(s): root/<label>/<image>; several enable --transfer or grouped --cv",
    )
    _add_pipeline_args(evaluate)
    evaluate.add_argument(
        "--healthy-label", default="healthy", help="label treated as the healthy class"
    )
    mode = evaluate.add_mutually_exclusive_group()
    mode.add_argument(
        "--cv",
        type=int,
        metavar="N",
        help="run N-fold grouped stratified cross-validation instead of a single pass",
    )
    mode.add_argument(
        "--transfer",
        action="store_true",
        help="leave-one-dataset-out across the folders (each folder is one dataset)",
    )
    mode.add_argument(
        "--importance",
        action="store_true",
        help="report global permutation feature importance for a trained model",
    )

    serve = sub.add_parser("serve", help="run the HTTP API (needs the 'api' extra)")
    serve.add_argument("--host", default="127.0.0.1", help="bind host")
    serve.add_argument("--port", type=int, default=8000, help="bind port")
    serve.add_argument(
        "--config", metavar="FILE", help="pipeline config (.toml/.json) for the served app"
    )
    serve.add_argument(
        "--model-path", metavar="FILE", help="trained or calibrated .joblib model to serve"
    )

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
        action="store_true",
        help="fail if a loaded model's feature schema differs from the live extractor output",
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        # Scope debug to our own logger so -v doesn't surface third-party (PIL, ...) noise.
        logging.getLogger("phytovision").setLevel(logging.DEBUG)


def _build_pipeline(args: argparse.Namespace, model: StressModel | None = None) -> Pipeline:
    if args.config:
        pipeline = Pipeline.from_config(_load_config(args.config))
    else:
        pipeline = Pipeline.from_names(
            model=args.model, segmenter=args.segmenter, explainer=args.explainer
        )
    chosen = model
    if chosen is None and args.model_path:
        chosen = _stress_model_from_path(args.model_path)
    if chosen is not None:
        _apply_strict_schema(chosen, getattr(args, "strict_schema", False))
        pipeline = pipeline.with_model(chosen)
    return pipeline


def _apply_strict_schema(model: StressModel, strict: bool) -> None:
    """Set the schema-drift mode on a loaded model and, for an ensemble, on its members."""
    if hasattr(model, "strict_schema"):
        model.strict_schema = strict
    for member in getattr(model, "members", ()):
        _apply_strict_schema(member, strict)


def _stress_model_from_path(path: str) -> StressModel:
    """The stress model to run in the pipeline; a conformal file contributes its wrapped model."""
    loaded = load_saved(path)
    return loaded.model if isinstance(loaded, SplitConformalClassifier) else loaded


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
    conformal: SplitConformalClassifier | None = None
    try:
        override: StressModel | None = None
        if args.model_path:
            loaded = load_saved(args.model_path)
            if isinstance(loaded, SplitConformalClassifier):
                conformal, override = loaded, loaded.model
            else:
                override = loaded
        if args.conformal and conformal is None:
            print("error: --conformal needs a model saved with train --calibrate", file=sys.stderr)
            return 2
        pipeline = _build_pipeline(args, model=override)
        report = pipeline.analyze(args.image)
        changes = (
            counterfactuals(pipeline.model, report.plant_features) if args.counterfactual else []
        )
        if args.save_overlay:
            render_overlay(load_image(args.image), report).save(args.save_overlay)
    except (OSError, ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    conformal_set = conformal.predict_set(report.plant_features) if conformal else None

    if args.json:
        payload = report.summary()
        if args.features:
            payload["features"] = report.plant_features.defined()
        if conformal_set is not None:
            payload["conformal"] = {
                "labels": list(conformal_set.labels),
                "alpha": conformal_set.alpha,
            }
        if args.counterfactual:
            payload["counterfactuals"] = [
                {
                    "feature": cf.feature,
                    "from": round(cf.current_value, 4),
                    "to": round(cf.target_value, 4),
                    "target_label": cf.target_label,
                }
                for cf in changes
            ]
        print(json.dumps(payload, indent=2))
        return 0

    stress = report.stress
    print(
        f"Water-stress: {stress.label.upper()}  "
        f"score={stress.score:.2f}  confidence={stress.confidence:.2f}  "
        f"(model: {stress.model_name})"
    )
    if conformal_set is not None:
        coverage = round((1.0 - conformal_set.alpha) * 100)
        members = " or ".join(conformal_set.labels) if conformal_set.labels else "(empty)"
        print(f"Conformal set ({coverage}% coverage): {{{members}}}")
    print(f"Regions analysed: {len(report.regions)} ({report.regions.kind})")
    if args.timing and report.timing_ms:
        print("Timing (ms):")
        for stage, elapsed in report.timing_ms.items():
            print(f"  {stage:<12} {elapsed:.1f}")
    if report.explanation.reasons:
        print("Top reasons:")
        for reason in report.explanation.reasons:
            marker = "+" if reason.direction == "increases" else "-"
            print(f"  [{marker}] {reason.feature}={reason.value:.3f} - {reason.description}")
    if args.counterfactual:
        if changes:
            print("To change the verdict:")
            for cf in changes:
                verb = "raise" if cf.target_value > cf.current_value else "lower"
                print(
                    f"  {verb} {cf.feature} from {cf.current_value:.3f} to "
                    f"{cf.target_value:.3f} -> {cf.target_label.upper()}"
                )
        else:
            print("No single change to an interpretable bounded feature flips the verdict.")
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
    if args.calibrate is not None and not 0.0 < args.calibrate < 1.0:
        print("error: --calibrate must be a fraction in (0, 1)", file=sys.stderr)
        return 2
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
    train_idx, calib_idx = _train_calibration_split(labels, args.calibrate)
    if len({labels[i] for i in train_idx}) < 2:
        print(
            "error: not enough data to keep both classes after the calibration split",
            file=sys.stderr,
        )
        return 2

    manifest = build_manifest(
        feature_keys=feature_keys,
        sources=[row.source for row in rows],
        extra={"model": args.model, "trained_on": len(train_idx)},
    )
    try:
        model = _fit_model(
            args.model,
            feature_keys,
            [feature_dicts[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )
        if calib_idx:
            wrapper = SplitConformalClassifier(model, alpha=args.alpha).calibrate(
                [to_plant_features(feature_dicts[i]) for i in calib_idx],
                [labels[i] for i in calib_idx],
            )
            wrapper.save(args.out, manifest=manifest)
        else:
            save_model(model, args.out, manifest=manifest)
    except (OSError, ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if calib_idx:
        print(
            f"trained {args.model} on {len(train_idx)} image(s), calibrated on {len(calib_idx)}; "
            f"saved to {args.out}"
        )
    else:
        print(f"trained {args.model} on {len(train_idx)} image(s); saved model to {args.out}")
    return 0


def _fit_model(
    name: str,
    feature_keys: Sequence[str],
    feature_dicts: Sequence[dict[str, float]],
    labels: Sequence[int],
) -> StressModel:
    """Fit the requested trainable model. An ensemble pairs the heuristic with a trained member."""
    trained = GradientBoostedStressModel(feature_keys=feature_keys).fit(
        list(feature_dicts), list(labels)
    )
    if name == "ensemble":
        return EnsembleStressModel([HeuristicStressModel(), trained])
    return trained


def _train_calibration_split(
    labels: Sequence[int], fraction: float | None
) -> tuple[list[int], list[int]]:
    """Split row indices into (train, calibration): a deterministic per-class holdout."""
    if fraction is None:
        return list(range(len(labels))), []
    calib: set[int] = set()
    for cls in sorted(set(labels)):
        members = [i for i, label in enumerate(labels) if label == cls]
        take = max(1, math.ceil(len(members) * fraction))
        calib.update(members[:take])
    train = [i for i in range(len(labels)) if i not in calib]
    return train, sorted(calib)


def _evaluate(args: argparse.Namespace) -> int:
    if args.transfer and len(args.directory) < 2:
        print("error: --transfer needs at least two folders", file=sys.stderr)
        return 2
    try:
        pipeline = _extraction_pipeline(args)
        rows = _load_labelled_rows(pipeline, args.directory)
    except (OSError, ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print(f"error: no images found in {', '.join(args.directory)}", file=sys.stderr)
        return 2

    if args.transfer:
        return _evaluate_transfer(rows, args)
    if args.cv is not None:
        return _evaluate_cv(rows, args)
    if args.importance:
        return _evaluate_importance(rows, args)
    return _evaluate_single(rows, args)


def _extraction_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the pipeline that extracts features for ``evaluate``.

    Single-pass evaluation compares the pipeline model's own prediction, so it honors ``--model``
    and ``--model-path``. The retraining modes (cross-validation, transfer, importance) fit a model
    themselves, so the extraction model is irrelevant; force a buildable default there.
    """
    if args.cv is None and not args.transfer and not args.importance:
        return _build_pipeline(args)
    if args.config:
        config = dict(_load_config(args.config))
        config.pop("model", None)  # the evaluated model comes from --model, not this pipeline
        return Pipeline.from_config(config)
    return Pipeline.from_names(segmenter=args.segmenter)


def _load_labelled_rows(pipeline: Pipeline, directories: list[str]) -> list[AnalysisRow]:
    """Analyze every folder; each row's ``source`` is set so datasets can be told apart."""
    sources = [Path(directory).name for directory in directories]
    if len(set(sources)) != len(sources):
        raise ConfigError("dataset folders must have distinct names to identify each dataset")
    rows: list[AnalysisRow] = []
    for directory, source in zip(directories, sources, strict=True):
        loader = FolderClassificationLoader(directory, source=source)
        rows.extend(analyze_dataset(pipeline, loader))
    return rows


def _evaluate_single(rows: list[AnalysisRow], args: argparse.Namespace) -> int:
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


def _evaluation_model(args: argparse.Namespace) -> str:
    """The trainable model for --cv/--transfer. The heuristic default maps to gradient-boosted."""
    return "gradient-boosted" if args.model == "heuristic" else args.model


def _evaluate_transfer(rows: list[AnalysisRow], args: argparse.Namespace) -> int:
    model = _evaluation_model(args)
    try:
        matrix = leave_one_dataset_out(rows, healthy_label=args.healthy_label, model=model)
    except (ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"leave-one-dataset-out ({model}; train on the rest, test on each held-out dataset):")
    for name in matrix.datasets:
        metrics = matrix.metrics_for(name)
        print(f"  {name:<24} n={metrics.n:<4} accuracy={metrics.accuracy:.3f}  f1={metrics.f1:.3f}")
    print(f"  mean held-out accuracy: {matrix.mean_accuracy:.3f}")
    return 0


def _evaluate_cv(rows: list[AnalysisRow], args: argparse.Namespace) -> int:
    model = _evaluation_model(args)
    try:
        result = grouped_stratified_cv(
            rows, healthy_label=args.healthy_label, n_splits=args.cv, model=model
        )
    except (ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    lo, hi = result.accuracy_ci95
    print(f"{result.n_splits}-fold {result.strategy} cross-validation ({model})")
    print(
        f"  accuracy: {result.mean_accuracy:.3f} +/- {result.std_accuracy:.3f}  "
        f"(95% CI {lo:.3f}..{hi:.3f})"
    )
    print(f"  f1:       {result.mean_f1:.3f}")
    return 0


def _evaluate_importance(rows: list[AnalysisRow], args: argparse.Namespace) -> int:
    model = _evaluation_model(args)
    try:
        ranked = permutation_importance(rows, healthy_label=args.healthy_label, model=model)
    except (ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"permutation feature importance ({model}), top features:")
    for key, importance in ranked[:10]:
        print(f"  {key:<28} {importance:+.4f}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print('error: serving needs the "api" extra: pip install -e ".[api]"', file=sys.stderr)
        return 2

    # Validate the chosen pipeline here so a bad path is a clean error, not a traceback when the
    # app is imported by string below (create_app runs at import and reads these files).
    try:
        if args.config:
            _load_config(args.config)
        if args.model_path:
            load_saved(args.model_path)
    except (OSError, ImportError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import os

    if args.config:
        os.environ["PHYTOVISION_CONFIG"] = str(Path(args.config))
    if args.model_path:
        os.environ["PHYTOVISION_MODEL_PATH"] = str(Path(args.model_path))
    uvicorn.run("phytovision.api:app", host=args.host, port=args.port)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
