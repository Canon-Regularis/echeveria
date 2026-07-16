"""The ``evaluate`` command: score a model on labelled folders.

Supports a single pass, grouped cross-validation, leave-one-dataset-out transfer, and permutation
feature importance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phytovision.analysis import AnalysisRow, analyze_dataset
from phytovision.cli._shared import add_pipeline_args, build_pipeline, fail
from phytovision.config import read_config
from phytovision.datasets.folder import FolderClassificationLoader
from phytovision.evaluation._common import binary_labels
from phytovision.evaluation.cross_dataset import leave_one_dataset_out
from phytovision.evaluation.crossval import grouped_stratified_cv
from phytovision.evaluation.importance import permutation_importance
from phytovision.evaluation.metrics import binary_metrics
from phytovision.exceptions import ConfigError, PhytoVisionError
from phytovision.pipeline import Pipeline


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("evaluate", help="score a model on labelled folders")
    parser.add_argument(
        "directory",
        nargs="+",
        help="labelled folder(s): root/<label>/<image>; several enable --transfer or grouped --cv",
    )
    add_pipeline_args(parser)
    parser.add_argument(
        "--healthy-label", default="healthy", help="label treated as the healthy class"
    )
    mode = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        "--seed", type=int, metavar="N", help="seed the model and folds so the run is reproducible"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.transfer and len(args.directory) < 2:
        return fail("--transfer needs at least two folders")
    try:
        pipeline = _extraction_pipeline(args)
        rows = _load_labelled_rows(pipeline, args.directory)
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    if not rows:
        return fail(f"no images found in {', '.join(args.directory)}")

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
        return build_pipeline(args)
    if args.config:
        config = dict(read_config(args.config))
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
    y_true = binary_labels(rows, args.healthy_label)
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
        matrix = leave_one_dataset_out(
            rows, healthy_label=args.healthy_label, model=model, seed=args.seed
        )
    except (ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

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
            rows, healthy_label=args.healthy_label, n_splits=args.cv, model=model, seed=args.seed
        )
    except (ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

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
    extra = {"seed": args.seed} if args.seed is not None else {}  # keep the default seed otherwise
    try:
        ranked = permutation_importance(
            rows, healthy_label=args.healthy_label, model=model, **extra
        )
    except (ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    print(f"permutation feature importance ({model}), top features:")
    for key, importance in ranked[:10]:
        print(f"  {key:<28} {importance:+.4f}")
    return 0
