# Model card: echeveria water-stress model

This card describes the water-stress models shipped in `phytovision` (the package behind echeveria).
It follows the intent of a standard model card: what the model is for, how it works, and where it
should not be trusted.

## Overview

Given one RGB photo of a succulent, the pipeline segments the plant, measures about 45 phenotypic
features (colour, geometry, texture, morphology), and produces a water-stress score in [0, 1], a bucket
label (`healthy`, `mild`, `stressed`), a confidence, and an explanation. Three stress models are
available:

- `heuristic` (default, v1): a transparent, monotonic weighted model over the features. It needs no
  training and runs with the base dependencies only. Its thresholds are documented priors.
- `gradient-boosted`: a scikit-learn `HistGradientBoostingClassifier` trained on labelled images. Needs
  the `ml` extra and your own labelled data.
- `ensemble`: soft-voting over the heuristic and a trained model.

## Intended use

- Research and education in plant phenotyping.
- Screening a folder of succulent photos for likely water stress, as one input among many.
- A reproducible, explainable baseline that others can extend (new segmenters, features, or models).

## Out of scope

- It is not a diagnostic tool and must not be the sole basis for an irrigation or agronomic decision.
- The priors and any trained model are tuned for echeveria-like succulents. Other species, growth
  stages, or imaging setups need re-validation or retraining.
- Water stress is only one cause of visible change. Disease, nutrient deficiency, light stress, and
  natural senescence can look similar and are not disambiguated here.

## Data

- The heuristic is not trained. Its weights encode agronomic priors (greenness and turgor lower stress;
  yellowing, browning, textural irregularity, and outline concavity raise it).
- The gradient-boosted and ensemble models train on user-provided labelled folders
  (`root/<label>/<image>`). Candidate datasets and their licenses are listed in
  [docs/DATASETS.md](docs/DATASETS.md); provenance and license travel with every sample.

## Evaluation

- `phytovision evaluate <folder>` reports accuracy, precision, recall, F1, and a confusion matrix.
- `--cv N` runs grouped stratified cross-validation (grouped by dataset source), reported as a mean
  with a confidence interval.
- `--transfer` runs leave-one-dataset-out, the honest test of whether a model learned water stress or
  dataset artifacts.
- `--importance` reports global permutation feature importance.

## Uncertainty and explainability

- Confidence from the models is a heuristic, not a calibrated probability.
- `train --calibrate` and `analyze --conformal` provide split conformal prediction: a label set with a
  distribution-free coverage guarantee (the true label falls in the set at least `1 - alpha` of the
  time over fresh data).
- Explanations: per-feature contributions (default), SHAP (`--explainer shap`, needs the `ml` extra),
  counterfactuals (`analyze --counterfactual`), and global importance.

## Limitations and risks

- Segmentation quality caps every downstream feature. The default Excess-Green segmenter is weak on
  red, purple, or blue succulents; use `--segmenter lab-chroma` for coloured plants.
- Inputs are RGB, so only RGB-derived vegetation indices are available (no NDVI or multispectral).
- A trained model depends on the exact extractor stack it was trained with. If the stack changes, the
  live features drift from the trained schema. Use `--strict-schema` to fail loudly instead of
  predicting on a partly-missing feature vector.
- The heuristic thresholds are priors, not calibrated on your plants. Calibrate or train before relying
  on the exact bucket boundaries.
- Trusting the output blindly risks over- or under-watering. Treat it as a screening signal, confirm
  with direct inspection, and validate on your own plants before operational use.

## How to reproduce

```bash
pip install -e ".[ml]"
phytovision train data/labelled --model gradient-boosted --calibrate 0.2 --out model.joblib
phytovision evaluate data/labelled --model gradient-boosted --cv 5
phytovision analyze plant.jpg --model-path model.joblib --conformal --explainer shap
```
