"""The ``batch`` command: analyze every image in a folder and export a feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from phytovision.analysis import analyze_dataset, feature_table
from phytovision.cli._shared import add_pipeline_args, build_pipeline, fail, write_table
from phytovision.datasets.directory import ImageDirectoryLoader
from phytovision.exceptions import PhytoVisionError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "batch", help="analyze every image in a folder and export a table"
    )
    parser.add_argument(
        "directory", help="folder of images (recurses; subfolder names become labels)"
    )
    parser.add_argument("--out", required=True, metavar="FILE", help="output path, .csv or .json")
    add_pipeline_args(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        pipeline = build_pipeline(args)
        loader = ImageDirectoryLoader(args.directory)
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    fieldnames, records = feature_table(analyze_dataset(pipeline, loader))
    if not records:
        return fail(f"no images found in {args.directory}")

    out = Path(args.out)
    write_table(out, fieldnames, records)
    print(f"wrote {len(records)} row(s) to {out}")
    return 0
