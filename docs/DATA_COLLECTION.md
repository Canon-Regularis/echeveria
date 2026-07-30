# Data collection and manifest protocol

echeveria ships no succulent data (see [DATASETS.md](DATASETS.md)), so every model is trained and
benchmarked on the synthetic dry-down simulator and labelled as such. This document is how you bring
real data in: what to collect, the manifest schema every command reads, how to convert the public
datasets into it, and the commands to run once you have it. The ingestion is already built, so once
your data matches the schema below it drops straight in.

## What to collect, and what each tier unlocks

There are three tiers of label, in increasing value. You do not need all three; pick the one that
matches your goal.

1. **Class labels (`healthy` / `wilted`)**, one per image. Trains and evaluates the stress model
   (`train`, `evaluate`). Easiest to produce: sort images into two folders.
2. **Timestamped series per plant.** Unlocks the forecasters and the survival model on real
   trajectories (`phenotype`). You do not hand-label events: the crossing to the stressed cut and the
   censoring are derived from the score itself. You only need a `plant_id` and a `timestamp` per image.
3. **A measured continuous target** (soil moisture, water potential, relative water content) paired
   with each image. This is what `validate` scores the RGB proxy against, and the only tier that turns
   the proxy from plausible into validated. Hardest to produce: it needs an instrument, not the eye.

Honesty holds throughout: an RGB score is a proxy read from pixels, not a measurement. Tier 3 is what
lets you check it against ground truth instead of trusting it.

## The manifest schema

Most commands read a CSV (or TSV) manifest through `CsvManifestLoader`. One row per image. Column
names are configurable, but the defaults are:

| Column | Meaning | Required by |
| --- | --- | --- |
| `image_path` | path to the image, absolute or relative to the manifest (or to `--images-root`) | every manifest command |
| `label` | class label; `healthy` is the healthy class (rename with `--healthy-label`) | `validate` reliability curve |
| `plant_id` | stable id of the plant across its images | `phenotype` |
| `timestamp` | capture time; any sortable format (`2026-03-01`, ISO 8601) | `phenotype` |
| `target` | the measured value to score the score against | `validate` regression |
| `source` | provenance tag, so datasets can be told apart | optional, recommended |

A folder-per-class dataset (`root/<label>/<image>`) needs no manifest at all: `train` and `evaluate`
read it directly. See [`examples/manifest_sample.csv`](../examples/manifest_sample.csv) for a filled-in
manifest.

## Capturing your own dry-down (tiers 1 and 2)

A controlled dry-down is the cheapest way to get real in-domain data:

- **One plant per pot, one id per plant.** Put the id in the filename (`p03_2026-03-01.jpg`) or in a
  `plant_id` column. Keep it stable across the whole series.
- **Shoot on a schedule** (for example daily) from a fixed distance and angle, in even, consistent
  light, against a plain background. The segmenter finds the plant by colour and shape, so a busy or
  green background makes its job harder.
- **Record the capture time** per image (`timestamp`). Order, not spacing, is what the trend uses, but
  real timestamps keep the series honest.
- **Water and withhold on a known schedule**, and note it. If you can, log a measurement at each shot
  (a cheap soil-moisture probe is enough) to earn tier 3.
- **Label loosely for tier 1** (`healthy` vs `wilted`); the model buckets the continuous score itself.

Two plants over a week already give you a real trajectory to forecast. A cohort of a dozen over a
dry-down gives you a real benchmark.

## Plugging in an existing dataset

The public sets in [DATASETS.md](DATASETS.md) come in three shapes; here is how each becomes usable.

**Folder-per-class** (Kaggle healthy/wilted #1, Mendeley aloe classes #19/#20): already the shape
`train`/`evaluate` want. Point the command at the root.

```bash
phytovision evaluate path/to/dataset      # root/<label>/<image>
phytovision train    path/to/dataset --out model.joblib
```

**Detection export** (the in-domain aloe stress sets #3/#4, Roboflow YOLO or COCO): these have boxes,
not one label per image, so reduce them with the converter. It assigns each image the majority box
category and writes a `image_path,label,source` manifest.

```bash
# YOLO export (images/, labels/*.txt, data.yaml)
python scripts/detection_to_manifest.py --format yolo \
    --images data/train/images --data-yaml data/data.yaml \
    --source aloe-wwmar --out data/train.csv

# COCO export (Roboflow _annotations.coco.json)
python scripts/detection_to_manifest.py --format coco \
    --annotations data/_annotations.coco.json --source aloe-y-v11 --out data/train.csv

# then, since the aloe class is "Healthy" (capitalised):
phytovision validate data/train.csv --healthy-label Healthy
```

**Images plus a measurements CSV** (the lettuce thermal + soil-moisture set #15, the tier-3 case):
join them into a manifest with a `target` column, then validate against it.

```bash
python scripts/measurements_to_manifest.py \
    --images data/rgb --measurements data/readings.csv \
    --image-col file --target-col soil_moisture \
    --plant-col plant --timestamp-col time \
    --source lettuce-294zk6k5wf --out data/lettuce.csv

phytovision validate  data/lettuce.csv    # reliability curve + RMSE/MAE/R2 vs the measured target
phytovision phenotype data/lettuce.csv --out traj.csv   # per-plant trajectory forecast
```

If your measured values are a separate sensor log (one reading per timestamp, not per image), join
them to your image list by nearest timestamp into a per-image CSV first; that join is dataset-specific.

## Commands, by what you have

| You have | Command | What you get |
| --- | --- | --- |
| a labelled folder | `train` | a fitted gradient-boosted model |
| a labelled folder | `evaluate` | accuracy, precision/recall/F1, confusion (add `--cv`, `--transfer`, `--importance`) |
| a manifest with `label` | `validate` | a reliability curve and Brier score of the RGB score |
| a manifest with `target` | `validate` | the above plus RMSE, MAE, and R2 against the measured value |
| a manifest with `plant_id` + `timestamp` | `phenotype` | per-plant trend, early warning, forecast, and time-to-wilt |
| a synthetic cohort manifest | `benchmark` | the forecasters ranked by CRPS, pinball, and coverage |

A quick end-to-end sanity check on the shipped synthetic simulator, no download needed:

```bash
phytovision simulate  --out cohort.csv --plants 12 --seed 0
phytovision benchmark cohort.csv
```

## Licensing

Check each dataset's license in [DATASETS.md](DATASETS.md#licensing) before you train on or
redistribute anything. Several of the recommended sets are CC BY 4.0; a few are unclear or restricted.
Keep the restricted ones out of any model you plan to ship.
