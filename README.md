# echeveria

[![CI](https://github.com/Matthew-Miezaniec/echeveria/actions/workflows/ci.yml/badge.svg)](https://github.com/Matthew-Miezaniec/echeveria/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/phytovision.svg)](https://pypi.org/project/phytovision/)

echeveria detects water stress (wilting) in succulents from ordinary RGB photos. It finds the plant in
the image and measures traits like colour, shape, and texture. It scores the stress, then lists the
features that drove that score. The scores are proxies read from pixels, so treat them as
indicative signals and confirm anything important against the plant.

The Python package is named `phytovision`. The commands and imports below use that name.

## Feature set

### Core single-image analysis

The pipeline runs the same seven stages every time: preprocess, plant segmentation, region selection,
feature extraction, aggregation, the stress model, and the explainer. The classical path needs no
machine-learning stack; it runs on the base dependencies alone.

- Preprocessing: resize so the longest side is at most 1024 pixels, then normalize to float in [0, 1].
- Plant segmentation: Excess-Green plus Otsu by default (`exg-otsu`), with a colour-agnostic CIELab
  chroma segmenter for red, purple, or blue plants (`lab-chroma`).
- Region selection: a whole-plant region by default (`whole-plant`), or per-leaf regions from a
  classical watershed splitter (`leaf-instance`).
- Feature extraction: namespaced, combinable families: geometry, colour, texture, and morphology by
  default, plus an opt-in `skeleton` family (silhouette medial-axis descriptors) that is registered but
  kept out of the default stack, so it never changes the shipped feature schema.
- Aggregation: one plant-level feature vector per image (`plant-level`), so the model never depends on
  the region count.
- Stress model: an interpretable heuristic scorer by default (`heuristic`, no training); a trainable
  gradient-boosted model (`gradient-boosted`); and a soft-voting `ensemble` of the two.
- Explanations: a ranked list of the features that pushed the score up or down (`feature-contribution`),
  or exact SHAP attributions with a completeness check (`shap`).

### Explanations and uncertainty

- Feature-contribution reasons: each driver gives the feature, its direction, and its signed
  contribution, plus a plain-language description. Some descriptions add a physiological note.
- SHAP explanations: TreeSHAP attributions for the gradient-boosted model, plus an additivity error so
  you can see how close the attribution is to completeness.
- Counterfactuals: the smallest change to one interpretable feature that would flip the verdict.
- Conformal prediction: split conformal calibration produces a label set with a coverage guarantee, so
  a borderline case can return both labels.

### Physiology proxies

The drought-stage head also reports three physiology proxies derived from the pigment, turgor, and
necrosis markers: a water-potential (deficit) index, a stomatal-conductance index, and a transpiration
index. They are crude RGB proxies, not measurements (the water-potential proxy is ordinal, not MPa;
the other two are relative indices, not fluxes), and they add interpretive grounding, not independent
signal, so they are never fed to the stress model or the forecaster.

### Secondary heads

These run after the stress model over the same plant features. Each one is a literature-motivated
prior, and its output carries a disclaimer that says so.

- Disease appearance: a placeholder baseline that estimates how lesion-like the plant looks from
  browning and surface irregularity. It has no disease training data behind it, so it demonstrates the
  head seam.
- Drought stage: a literature-motivated rule set that maps pigment change, turgor loss, and necrosis to
  a stage (well-watered, early-stress, moderate, or severe). It names the markers that drove the stage.

### Temporal analysis and forecasting

Given a timestamped series of images of one plant, echeveria reports how the stress moves over time.

- Stress trend: a direction and a slope fitted to the stress score over observation order.
- Pigment early warning: flags a plant whose RGB pigment-stress proxy is rising while its stress score
  is still below the stressed cut. Deterioration shows up before the verdict flips.
- Probabilistic forecast: pluggable forecasters that each project the stress score forward with a
  prediction interval per horizon. Pick one by name (`--forecaster`): a linear-trend baseline, a
  state-space local linear trend and an ARIMA model (the `stats` extra), and a Gaussian process and a
  Bayesian ridge (the `ml` extra). Each also estimates the steps to the stressed cut and a confidence
  that falls as the horizon grows.
- Survival analysis: a median time-to-wilt per plant with a band, handling plants that never wilt in
  the observed window (right censoring). A Kaplan-Meier cohort baseline, a Weibull accelerated-failure
  model (default), and a Cox model register under `SURVIVAL_MODELS`; the event is derived from the
  observed stress crossing, and the estimate surfaces on `phenotype`, `/trend`, and the dashboard.
- Per-leaf tracking: a `LeafTracker` gives each leaf a stable identity across a plant's frames
  (Hungarian matching on centroid and area), so `build_leaf_histories` produces one trajectory per
  leaf and the forecasters and the survival model can run on a single leaf, not only the whole plant.
- High-throughput phenotyping: the `phenotype` command reads a manifest of many plants over time and
  writes one trajectory row per plant, with the per-horizon interval columns and the time-to-wilt band.
- Synthetic data and benchmarking: no labelled succulent time series exists, so `simulate` generates a
  seeded dry-down cohort (labelled synthetic), `benchmark` ranks every forecaster over it with
  time-series cross-validation and proper scoring rules (CRPS, pinball loss, interval coverage), and a
  survival leaderboard ranks the survival models by held-out concordance.

### Training, evaluation, and persistence

- Training: fit a gradient-boosted or ensemble model from a labelled folder, optionally holding out a
  fraction to calibrate a conformal wrapper.
- Cross-validation: grouped, stratified k-fold, grouped by dataset source so one dataset never lands in
  both train and test.
- Transfer evaluation: leave-one-dataset-out across several folders. It measures how well the model
  carries over to a dataset it never trained on.
- Feature importance: global permutation importance for a trained model.
- Metrics: accuracy, precision, recall, F1, and a confusion matrix.
- Persistence: a type-tagged joblib envelope with a provenance manifest, so the heuristic, the
  gradient-boosted model, ensembles, and the conformal wrapper all round-trip the same way.

### Dataset loaders

One loading interface returns `Sample` objects with their provenance attached, so downstream code does
not depend on any single dataset layout. Loaders ship for a folder-per-class layout (`folder`), a plain
directory of images (`directory`), a CSV or TSV manifest (`csv`), COCO detection exports (`coco`), and
YOLO detection exports (`yolo`). A malformed file becomes a clean error naming the file.

### Serving

- HTTP API (needs the `api` extra): a FastAPI app with `GET /health`, `POST /analyze` (with optional
  disease and drought-stage heads and a conformal set), `POST /overlay` (an annotated image), and
  `POST /trend` (per-plant trends, early warnings, and forecasts).
- Dashboard (needs the `dashboard` extra): a Streamlit terminal with an ANALYZE tab (verdict, overlay,
  drivers, the disease and drought-stage panels, timing, and features) and a TEMPORAL tab (stress over
  time, the early warning, and the projected forecast).

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

The core single-image path installs with the base dependencies alone. Heavier features live behind
extras: `ml` (the gradient-boosted model, SHAP, persistence, and two forecasters), `stats` (the
state-space and ARIMA forecasters), `tracking` (MLflow benchmark logging), `api`, `dashboard`, and
`docs`. Install `.[all]` for every runtime feature, or `.[dev]` for the test, lint, and type-check
tooling.

## Command line

| Command | What it does |
| --- | --- |
| `analyze` | analyze one image for water stress |
| `batch` | analyze every image in a folder and export a table |
| `train` | train a model on a labelled folder and save it |
| `evaluate` | score a model on labelled folders (single pass, `--cv`, `--transfer`, or `--importance`) |
| `serve` | run the HTTP API (needs the `api` extra) |
| `dashboard` | run the Streamlit dashboard (needs the `dashboard` extra) |
| `phenotype` | high-throughput trajectory phenotyping over a timestamped manifest |
| `simulate` | write a synthetic dry-down cohort (a manifest plus an events table) |
| `benchmark` | rank the forecasters over a synthetic cohort with time-series cross-validation |

```bash
phytovision analyze path/to/plant.jpg
phytovision analyze path/to/plant.jpg --json --features --disease --drought-stage --physiology
phytovision analyze path/to/plant.jpg --counterfactual --conformal --save-overlay overlay.png
phytovision analyze path/to/plant.jpg --save-saliency pigment.png --save-occlusion occlusion.png
```

Add `-v` for debug logging and `--json` for machine-readable output.

## From Python

```python
from phytovision import Pipeline

report = Pipeline.default().analyze("plant.jpg")
print(report.stress.score, report.stress.label)

for reason in report.explanation.reasons:
    print(reason.marker, reason.feature, reason.description)
```

## Changing how it runs

Every stage is a separate, replaceable part registered by name: preprocessor, plant segmenter, region
provider, feature extractors, aggregator, stress model, and explainer, plus the optional disease and
drought-stage heads. Pick parts by name:

```python
Pipeline.from_names(model="gradient-boosted", segmenter="exg-otsu", explainer="shap")
```

Or build the whole thing from a config dictionary, or from a config file you load into one, with
`Pipeline.from_config({...})`. The config is validated against a typed schema, so a mistyped slot name
is rejected rather than silently ignored, and the resolved config is diffable for reproducibility.
Adding a new implementation means registering it; nothing in the orchestrator changes. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the parts fit together and how to add your own.

For reproducible runs, the commands that draw randomness take `--seed`, which threads into the
per-stage generators and seeds the global RNGs from one place; see the reproducibility notes in
[CONTRIBUTING.md](CONTRIBUTING.md), which also cover optional MLflow tracking and DVC data versioning.

## Honesty and limits

- echeveria works from RGB photos only. The multispectral and thermal fusion that some water-status
  research uses is out of scope.
- The stress score, the disease panel, the drought stage, and the forecast are proxies read from
  ordinary pixels; treat them as indicative signals, and confirm anything important against the plant.
- A trained per-leaf model is deferred until a hand-labelled succulent dataset exists. The classical
  watershed splitter covers per-leaf regions in the meantime.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the code is laid out and the two ideas that make it
  extensible.
- [docs/OBJECTIVES.md](docs/OBJECTIVES.md): what the project is trying to achieve, the plan, and the
  open risks.
- [docs/DATASETS.md](docs/DATASETS.md): the datasets we checked, what each one is good for, and their
  licenses.
- [MODEL_CARD.md](MODEL_CARD.md): the intended use, the proxies, and the caveats behind each output.

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
