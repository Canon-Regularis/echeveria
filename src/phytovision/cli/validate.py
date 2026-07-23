"""The ``validate`` command: score the stress model against a manifest of measured water status.

The stress score is an RGB proxy, so this reports how well it tracks measured values, not that it
equals them. The public measured-water-status datasets are non-succulent, so a result over them is
cross-species transfer and indicative, not a validated succulent diagnosis.
"""

from __future__ import annotations

import argparse

from phytovision.analysis import analyze_dataset
from phytovision.cli._shared import add_pipeline_args, build_pipeline, fail
from phytovision.datasets.manifest import CsvManifestLoader
from phytovision.evaluation._common import binary_labels
from phytovision.evaluation.calibration import brier_score, reliability_curve
from phytovision.evaluation.regression import regression_metrics
from phytovision.exceptions import PhytoVisionError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "validate", help="score the model against a manifest of measured water status"
    )
    parser.add_argument("manifest", help="CSV/TSV manifest with an image_path and a target column")
    parser.add_argument(
        "--images-root", metavar="DIR", help="image folder (defaults to the manifest folder)"
    )
    parser.add_argument(
        "--target-column", default="target", help="manifest column holding the measured value"
    )
    parser.add_argument(
        "--healthy-label", default="healthy", help="label treated as the healthy class"
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="number of reliability-curve bins over [0, 1]"
    )
    add_pipeline_args(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.bins < 1:
        return fail("--bins must be at least 1")
    try:
        pipeline = build_pipeline(args)
        loader = CsvManifestLoader(
            args.manifest, args.images_root or None, target_column=args.target_column
        )
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    rows = list(analyze_dataset(pipeline, loader))
    if not rows:
        return fail(f"no analysable images in {args.manifest}")

    print("The stress score is an RGB proxy; treat these numbers as indicative, not validated.")

    # The reliability curve and Brier score need the healthy/stressed label, which is optional in a
    # manifest. Build them only from labelled rows, mirroring the regression block below: otherwise
    # an unlabelled row reads as a true stressed event (None != healthy) and fabricates a curve that
    # asserts every image was stressed.
    labelled = [row for row in rows if row.label is not None]
    if labelled:
        scores = [row.score for row in labelled]
        events = binary_labels(labelled, args.healthy_label)
        curve = reliability_curve(scores, events, n_bins=args.bins)
        print(f"\nreliability of the score as P(stressed) over {len(labelled)} labelled image(s):")
        print("  bin  mean_score  observed  count")
        for i, (mean, observed, count) in enumerate(
            zip(curve.mean_score, curve.observed_rate, curve.counts, strict=True)
        ):
            if count:
                print(f"  {i:>3}  {mean:>10.3f}  {observed:>8.3f}  {count:>5}")
        print(f"  Brier score: {brier_score(scores, events):.4f}")
    else:
        print("\nno 'label' values found, so the reliability curve is skipped")

    targeted_scores = [row.score for row in rows if row.target is not None]
    targeted_values = [row.target for row in rows if row.target is not None]
    if targeted_values:
        metrics = regression_metrics(targeted_scores, targeted_values)
        count = len(targeted_values)
        print(f"\nscore vs {args.target_column} over {count} image(s) with a measured value:")
        print(f"  RMSE: {metrics.rmse:.4f}   MAE: {metrics.mae:.4f}   R2: {metrics.r2:.4f}")
    else:
        print(f"\nno {args.target_column!r} values found, so the regression is skipped")
    return 0
