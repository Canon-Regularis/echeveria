# echeveria

echeveria detects water stress (wilting) in succulents from ordinary RGB photos. It finds the plant in
the image, measures a set of traits like colour, shape and texture, and turns those into a stress score
with the reasons attached.

The Python package is named `phytovision`. The commands and imports below use that name.

## What works today

| Step | Status | Notes |
| --- | --- | --- |
| Preprocessing | done | resize and normalize |
| Plant segmentation | done | Excess-Green plus Otsu threshold, no training needed |
| Feature extraction | done | geometry, colour, texture, morphology |
| Aggregation | done | one feature vector per plant |
| Water-stress model | done | an interpretable rule-based scorer by default; a trainable gradient-boosted model is also available |
| Explanations | done | ranked list of features that pushed the score up or down |
| Per-leaf segmentation | planned | needs a hand-labelled succulent dataset that does not exist yet |
| Disease, temporal, dashboard, API | planned | interfaces are in place, no implementation yet |

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Run it

From the command line:

```bash
phytovision analyze path/to/plant.jpg
```

Add `-v` for debug logging and `--json` for machine-readable output.

From Python:

```python
from phytovision import Pipeline

report = Pipeline.default().analyze("plant.jpg")
print(report.stress.score, report.stress.label)

for reason in report.explanation.reasons:
    print(reason.feature, reason.direction, reason.description)
```

## Changing how it runs

Each step is a separate, replaceable part: preprocessing, plant segmentation, feature extraction,
aggregation, the stress model, and the explainer. Pick parts by name:

```python
Pipeline.from_names(model="gradient-boosted", segmenter="exg-otsu")
```

Or build the whole thing from a config dictionary (or a file you load into one) with
`Pipeline.from_config({...})`. The CLI exposes `--model` and `--segmenter`. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the parts fit together and how to add your own.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the code is laid out and the two ideas that make it
  extensible.
- [docs/OBJECTIVES.md](docs/OBJECTIVES.md): what the project is trying to achieve, the plan, and the
  open risks.
- [docs/DATASETS.md](docs/DATASETS.md): the datasets we checked, what each one is good for, and their
  licenses.

## Datasets

The full, vetted list is in [docs/DATASETS.md](docs/DATASETS.md). The recommended set to build the first
version on, chosen for being succulent, permissively licensed, and downloadable today:

1. [Aloe Vera Health Detection (wwmar v22)](https://universe.roboflow.com/aloe-vera-health-detection/aloe-vera-health-detection-wwmar/dataset/22): aloe stress and health, CC BY 4.0.
2. [Aloevera Health Detection (Y-V11 v8)](https://universe.roboflow.com/aloe-vera-health-detection/aloevera-health-detection-y-v11/dataset/8): the largest aloe stress set, CC BY 4.0.
3. [Healthy and Wilted Houseplant Images](https://www.kaggle.com/datasets/russellchan/healthy-and-wilted-houseplant-images): a broad healthy/wilted baseline (license unclear).
4. [Aloe Vera Diseases (background-removed)](https://data.mendeley.com/datasets/cksmdjw8gy/2): useful for separating plant from background, plus disease, CC BY 4.0.
5. [Lettuce thermal and soil moisture](https://data.mendeley.com/datasets/294zk6k5wf/2): the one set that pairs images with measured water status, CC BY 4.0.

## License

The code is Apache-2.0 (see [LICENSE](LICENSE)). The datasets have their own licenses, some of them
restrictive. Check the licensing table in [docs/DATASETS.md](docs/DATASETS.md#licensing) before you
train on or share anything.
