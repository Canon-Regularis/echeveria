"""The ``analyze`` command: run the pipeline on one image and print the verdict and drivers."""

from __future__ import annotations

import argparse
import json

from phytovision.cli._shared import add_pipeline_args, build_pipeline, fail
from phytovision.exceptions import PhytoVisionError
from phytovision.explainability.counterfactual import counterfactuals
from phytovision.io import load_image
from phytovision.models.base import StressModel
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.persistence import load_saved
from phytovision.serving import attach_heads
from phytovision.visualize import render_overlay


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("analyze", help="analyze one image for water stress")
    parser.add_argument("image", help="path to an RGB image")
    add_pipeline_args(parser)
    parser.add_argument("--json", action="store_true", help="emit the JSON summary only")
    parser.add_argument("--features", action="store_true", help="include the full feature vector")
    parser.add_argument("--save-overlay", metavar="PNG", help="write an annotated overlay image")
    parser.add_argument("--timing", action="store_true", help="show per-stage wall-clock timing")
    parser.add_argument(
        "--disease",
        action="store_true",
        help="attach the placeholder disease-appearance head (not a validated diagnostic)",
    )
    parser.add_argument(
        "--drought-stage",
        action="store_true",
        help="attach the drought-stage head (a literature-motivated rule set, not a diagnosis)",
    )
    parser.add_argument(
        "--counterfactual",
        action="store_true",
        help="report the smallest feature changes that would flip the verdict",
    )
    parser.add_argument(
        "--conformal",
        action="store_true",
        help="print the conformal label set (needs a model saved with train --calibrate)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
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
            return fail("--conformal needs a model saved with train --calibrate")
        pipeline = build_pipeline(args, model=override)
        pipeline = attach_heads(pipeline, disease=args.disease, drought_stage=args.drought_stage)
        report = pipeline.analyze(args.image)
        changes = (
            counterfactuals(pipeline.model, report.plant_features) if args.counterfactual else []
        )
        if args.save_overlay:
            render_overlay(load_image(args.image), report).save(args.save_overlay)
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

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
        if report.head_outputs:
            payload["head_outputs"] = report.head_outputs
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
    for head_name, output in report.head_outputs.items():
        print(f"Head '{head_name}': {output}")
    if args.timing and report.timing_ms:
        print("Timing (ms):")
        for stage, elapsed in report.timing_ms.items():
            print(f"  {stage:<12} {elapsed:.1f}")
    if report.explanation.reasons:
        print("Top reasons:")
        for reason in report.explanation.reasons:
            print(f"  [{reason.marker}] {reason.feature}={reason.value:.3f}: {reason.description}")
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
