"""Command-line surface: ``phytovision [-v] analyze <image> [--model ...] [--json]``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from phytovision.exceptions import PhytoVisionError
from phytovision.pipeline import Pipeline
from phytovision.registries import SEGMENTERS, STRESS_MODELS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phytovision",
        description="Explainable plant water-stress analysis.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging on stderr"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyze an image for water stress")
    analyze.add_argument("image", help="path to an RGB image")
    analyze.add_argument(
        "--model", default="heuristic", choices=STRESS_MODELS.names(), help="stress model"
    )
    analyze.add_argument(
        "--segmenter", default="exg-otsu", choices=SEGMENTERS.names(), help="plant segmenter"
    )
    analyze.add_argument("--json", action="store_true", help="emit the JSON summary only")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return _analyze(args.image, model=args.model, segmenter=args.segmenter, as_json=args.json)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        # Scope debug to our own logger so -v doesn't surface third-party (PIL, ...) noise.
        logging.getLogger("phytovision").setLevel(logging.DEBUG)


def _analyze(path: str, *, model: str, segmenter: str, as_json: bool) -> int:
    try:
        pipeline = Pipeline.from_names(model=model, segmenter=segmenter)
        report = pipeline.analyze(path)
    except (FileNotFoundError, PhytoVisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(report.summary(), indent=2))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
