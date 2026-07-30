#!/usr/bin/env python
"""Reduce a YOLO or COCO detection export to a per-image classification manifest.

The in-domain succulent datasets in docs/DATASETS.md (the aloe stress sets) ship as detection
exports (bounding boxes), but the stress model trains on one label per image. This assigns each
image a single label from its boxes (the majority box category, ties broken alphabetically so the
result is deterministic) and writes the ``image_path,label,source`` manifest that CsvManifestLoader
reads. An image with no boxes takes ``--no-box-label`` (default ``healthy``), or is dropped with
``--drop-empty``. It reuses the package's own detection loaders, so the parsing matches what the
pipeline already understands.

Examples:
  python scripts/detection_to_manifest.py --format yolo --images data/train/images \\
      --data-yaml data/data.yaml --source aloe-wwmar --out data/train.csv
  python scripts/detection_to_manifest.py --format coco \\
      --annotations data/_annotations.coco.json --source aloe-y-v11 --out data/train.csv

Then, e.g.:  phytovision evaluate <folder>  is for folder-per-class data; for a manifest use it as a
cohort for `benchmark`/`phenotype`, or train from a labelled folder. See docs/DATA_COLLECTION.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from phytovision.datasets.coco import CocoDetectionLoader
from phytovision.datasets.yolo import YoloDetectionLoader


def image_label(boxes: list[dict[str, object]], no_box_label: str) -> str:
    """The single image-level label: the most common box category, or ``no_box_label`` if empty."""
    if not boxes:
        return no_box_label
    counts = Counter(str(box["category"]) for box in boxes)
    # Sort by descending count then category name, so ties resolve the same way on every run.
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def yolo_class_names(data_yaml: str) -> list[str] | None:
    """Best-effort read of the ``names:`` field from a YOLO ``data.yaml`` (no pyyaml dependency).

    Handles the inline list ``names: [a, b]``, the block list (``- a`` per line), and the index map
    (``0: a`` per line). Returns None if it cannot parse, so the loader falls back to numeric ids.
    """
    lines = Path(data_yaml).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("names:"):
            continue
        after = line.split("names:", 1)[1].strip()
        if after.startswith("["):  # inline list: names: [a, b]
            inner = after.strip("[]")
            return [name.strip().strip("'\"") for name in inner.split(",") if name.strip()]
        names: list[str] = []
        for follow in lines[index + 1 :]:
            if follow.strip() and not follow[:1].isspace():
                break  # a dedented line is the next top-level key, so the names block is over
            stripped = follow.strip()
            if not stripped:
                continue
            value = stripped.split(":", 1)[1] if ":" in stripped else stripped.lstrip("- ")
            names.append(value.strip().strip("'\""))
        return names or None
    return None


def build_loader(args: argparse.Namespace) -> CocoDetectionLoader | YoloDetectionLoader:
    if args.format == "yolo":
        names: list[str] | None = None
        if args.names:
            names = [name.strip() for name in args.names.split(",") if name.strip()]
        elif args.data_yaml:
            names = yolo_class_names(args.data_yaml)
        return YoloDetectionLoader(args.images, args.labels, names, source=args.source)
    return CocoDetectionLoader(args.annotations, args.images, source=args.source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detection export -> classification manifest.")
    parser.add_argument("--format", choices=["yolo", "coco"], required=True)
    parser.add_argument("--images", help="images directory (yolo) or images root (coco)")
    parser.add_argument("--labels", help="yolo labels directory (default: <images>/../labels)")
    parser.add_argument("--annotations", help="coco annotations json")
    parser.add_argument("--data-yaml", help="yolo data.yaml, to read the class names")
    parser.add_argument("--names", help="comma-separated class names (overrides --data-yaml)")
    parser.add_argument("--source", help="provenance tag written to every row")
    parser.add_argument(
        "--no-box-label", default="healthy", help="label for an image that has no boxes"
    )
    parser.add_argument("--drop-empty", action="store_true", help="skip images that have no boxes")
    parser.add_argument("--out", required=True, help="manifest CSV to write")
    args = parser.parse_args(argv)

    if args.format == "yolo" and not args.images:
        parser.error("--format yolo needs --images")
    if args.format == "coco" and not args.annotations:
        parser.error("--format coco needs --annotations")

    loader = build_loader(args)
    rows: list[dict[str, str]] = []
    empty = 0
    for sample in loader:
        boxes = (sample.extra or {}).get("boxes", [])
        if not boxes and args.drop_empty:
            empty += 1
            continue
        rows.append(
            {
                "image_path": sample.image_path,
                "label": image_label(boxes, args.no_box_label),
                "source": args.source or "",
            }
        )

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")
    print("label counts:", dict(sorted(Counter(row["label"] for row in rows).items())))
    if args.drop_empty and empty:
        print(f"dropped {empty} image(s) with no boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
