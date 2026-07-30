#!/usr/bin/env python
"""Join an images folder and a per-image measurements CSV into a phytovision manifest.

For a dataset that pairs images with a measured water-status value (soil moisture, water potential,
relative water content), like the lettuce set in docs/DATASETS.md, this remaps a metadata CSV's
columns into the manifest schema ``image_path,label,plant_id,timestamp,source,target`` and resolves
each image path against ``--images``, so ``phytovision validate`` can score the RGB stress proxy
against the measured target. One row of the metadata CSV must correspond to one image.

If your measured values live in a separate sensor log (one reading per timestamp, not per image),
first join them to your image list by nearest timestamp into a per-image CSV, then run this. That
join is dataset-specific and intentionally out of scope here.

Example:
  python scripts/measurements_to_manifest.py --images data/rgb --measurements data/readings.csv \\
      --image-col file --target-col soil_moisture --plant-col plant --timestamp-col time \\
      --source lettuce-294zk6k5wf --out data/lettuce.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_FIELDNAMES = ["image_path", "label", "plant_id", "timestamp", "source", "target"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Images + measurements CSV -> manifest.")
    parser.add_argument("--images", required=True, help="directory holding the images")
    parser.add_argument("--measurements", required=True, help="CSV with one row per image")
    parser.add_argument("--image-col", default="image", help="column holding the image filename")
    parser.add_argument("--target-col", help="column holding the measured value (-> target)")
    parser.add_argument("--label-col", help="column holding a class label (-> label)")
    parser.add_argument("--plant-col", help="column holding the plant id (-> plant_id)")
    parser.add_argument("--timestamp-col", help="column holding the timestamp (-> timestamp)")
    parser.add_argument("--source", help="provenance tag written to every row")
    parser.add_argument("--out", required=True, help="manifest CSV to write")
    parser.add_argument(
        "--skip-missing", action="store_true", help="skip a row whose image is not on disk"
    )
    args = parser.parse_args(argv)

    images = Path(args.images)
    source_csv = Path(args.measurements)
    rows: list[dict[str, str]] = []
    missing = 0
    with source_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = {name.strip() for name in (reader.fieldnames or [])}
        for flag, column in (
            ("--image-col", args.image_col),
            ("--target-col", args.target_col),
            ("--label-col", args.label_col),
            ("--plant-col", args.plant_col),
            ("--timestamp-col", args.timestamp_col),
        ):
            if column and column not in columns:
                parser.error(
                    f"{flag} {column!r} is not a column in {source_csv}; has {sorted(columns)}"
                )
        for raw in reader:
            record = {name.strip(): (value or "").strip() for name, value in raw.items()}
            filename = record.get(args.image_col, "")
            if not filename:
                continue
            image_path = images / filename
            if not image_path.exists():
                missing += 1
                if args.skip_missing:
                    continue
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": record.get(args.label_col, "") if args.label_col else "",
                    "plant_id": record.get(args.plant_col, "") if args.plant_col else "",
                    "timestamp": record.get(args.timestamp_col, "") if args.timestamp_col else "",
                    "source": args.source or "",
                    "target": record.get(args.target_col, "") if args.target_col else "",
                }
            )

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")
    if missing:
        note = "skipped" if args.skip_missing else "kept (image not found on disk)"
        print(f"{missing} row(s) had a missing image: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
