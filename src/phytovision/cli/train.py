"""The ``train`` command: fit a trainable model on a labelled folder and save it."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from phytovision.analysis import analyze_dataset
from phytovision.cli._shared import fail
from phytovision.datasets.folder import FolderClassificationLoader
from phytovision.evaluation._common import (
    binary_labels,
    feature_keys_of,
    model_factory,
    trainable_model_names,
)
from phytovision.exceptions import PhytoVisionError
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.persistence import build_manifest, save_model
from phytovision.pipeline import Pipeline
from phytovision.registries import SEGMENTERS
from phytovision.types import PlantFeatures


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="train a model on a labelled folder and save it")
    parser.add_argument("directory", help="labelled folder: root/<label>/<image>")
    parser.add_argument("--out", required=True, metavar="FILE", help="output model path (.joblib)")
    parser.add_argument(
        "--model",
        default="gradient-boosted",
        choices=trainable_model_names(),
        help="model to train",
    )
    parser.add_argument(
        "--segmenter", default="exg-otsu", choices=SEGMENTERS.names(), help="plant segmenter"
    )
    parser.add_argument(
        "--healthy-label", default="healthy", help="label treated as the healthy class"
    )
    parser.add_argument(
        "--calibrate",
        type=float,
        metavar="FRAC",
        help="hold out FRAC of the data to calibrate a conformal wrapper and save it",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="conformal miscoverage rate, so coverage is 1 - alpha (with --calibrate)",
    )
    parser.add_argument(
        "--seed", type=int, metavar="N", help="seed the model so the fit is reproducible"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.calibrate is not None and not 0.0 < args.calibrate < 1.0:
        return fail("--calibrate must be a fraction in (0, 1)")
    try:
        loader = FolderClassificationLoader(args.directory)
        pipeline = Pipeline.from_names(segmenter=args.segmenter)  # model is unused for extraction
    except (OSError, PhytoVisionError) as exc:
        return fail(str(exc))

    rows = list(analyze_dataset(pipeline, loader))
    if not rows:
        return fail(f"no images found in {args.directory}")

    labels = binary_labels(rows, args.healthy_label)
    if len(set(labels)) < 2:
        found = sorted({str(row.label) for row in rows})
        return fail(f"need at least two classes; found labels {found}")

    feature_dicts = [row.features for row in rows]
    feature_keys = feature_keys_of(feature_dicts)
    train_idx, calib_idx = _train_calibration_split(labels, args.calibrate)
    if len({labels[i] for i in train_idx}) < 2:
        return fail("not enough data to keep both classes after the calibration split")

    manifest = build_manifest(
        feature_keys=feature_keys,
        sources=[row.source for row in rows],
        seed=args.seed,
        extra={"model": args.model, "trained_on": len(train_idx)},
    )
    try:
        model = model_factory(args.model, seed=args.seed)(
            feature_keys,
            [feature_dicts[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )
        if calib_idx:
            wrapper = SplitConformalClassifier(model, alpha=args.alpha).calibrate(
                [PlantFeatures.from_values(feature_dicts[i]) for i in calib_idx],
                [labels[i] for i in calib_idx],
            )
            wrapper.save(args.out, manifest=manifest)
        else:
            save_model(model, args.out, manifest=manifest)
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    if calib_idx:
        print(
            f"trained {args.model} on {len(train_idx)} image(s), calibrated on {len(calib_idx)}; "
            f"saved to {args.out}"
        )
    else:
        print(f"trained {args.model} on {len(train_idx)} image(s); saved model to {args.out}")
    return 0


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
