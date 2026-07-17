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
  yellowing, browning, reddening, textural irregularity, and outline concavity raise it).
- The gradient-boosted and ensemble models train on user-provided labelled folders
  (`root/<label>/<image>`). Candidate datasets and their licenses are listed in
  [docs/DATASETS.md](docs/DATASETS.md); provenance and license travel with every sample.

## Physiological basis

The visible markers this project measures are grounded in succulent drought physiology, notably the
study "Responses of Succulents to Drought: Comparative Analysis of Four Sedum Species" (Scientia
Horticulturae, 2019). It documents a progressive drought response: pigment change (chlorophyll
degradation and anthocyanin accumulation) and oxidative stress appear first, then tissue-water loss,
turgor and anatomical change, and eventual collapse, with pigment changes preceding complete collapse
and tolerant species carrying different signatures.

The RGB features stand in for those mechanisms as proxies:

| Drought mechanism | RGB proxy features |
| --- | --- |
| Chlorophyll degradation (greenness loss) | `colour.gcc_mean`, `colour.exg_mean`, `colour.greenness_ratio` |
| Yellowing / browning (senescence, necrosis) | `colour.yellow_fraction`, `colour.brown_fraction` |
| Anthocyanin accumulation (reddening) | `colour.red_fraction` |
| Pigment saturation loss | `colour.saturation_mean`, `colour.value_mean` |
| Turgor loss / leaf deformation | `geometry.solidity`, `morphology.concavity`, `morphology.radial_variation` |
| Surface texture change | the `texture.*` family |

Two optional heads build on this progression: `analyze --drought-stage` names an ordinal stage
(`well-watered`, `early-stress`, `moderate`, `severe`) from the pattern of markers, and the temporal
early warning flags a plant whose pigment stress is rising while its overall score is still below the
stressed cut-off (the pigment-before-collapse signal). Explanations also cite the mechanism behind
each driver (for example "yellowing raises the estimate (chlorophyll degradation)").

Honesty caveats:

- These are RGB proxies, not measurements. The paper measures chlorophyll content, oxidative markers,
  and relative water content with destructive laboratory assays; none are observable from a photo. A
  high `red_fraction` is reddening in the image, not a measured anthocyanin concentration.
- The drought-stage rules and thresholds are literature-motivated priors, not fitted to labelled staged
  data. No Sedum drought dataset is catalogued in [docs/DATASETS.md](docs/DATASETS.md), so the stage is
  indicative, not validated.
- Reddening is confounded: many succulents redden under light stress or are naturally red or purple, so
  `red_fraction` carries a deliberately modest weight.
- Species differ. Tolerant and sensitive species show different signatures, which this single model
  does not yet account for (see the deferred species objective in [docs/OBJECTIVES.md](docs/OBJECTIVES.md)).

## High-throughput phenotyping and forecasting

Following the multi-sensor, multi-temporal high-throughput phenotyping literature (for example
"Multi-Sensor and Multi-temporal High-Throughput Phenotyping for Water Stress", arXiv 2402.18751,
2024), water stress is treated as a trajectory, not a single snapshot. echeveria's pipeline is:

    RGB image -> segmentation -> shape + colour + texture -> water-stress score -> forecast

Over a `plant_id`/`timestamp` manifest, `phytovision phenotype` builds a per-plant trajectory and the
API `/trend` and dashboard TEMPORAL tab surface it: the trend, the pigment early warning, and a
forecast (a projected stress score per horizon, an estimated steps-to-stressed, and a confidence).

The forecast is pluggable. Every forecaster reports a mean and a prediction interval per horizon, and
all register under `FORECASTERS`, so a caller selects one by name (`phenotype --forecaster`,
`/trend?forecaster=`, the dashboard picker):

- `linear-trend` (default): the least-squares extrapolation, with an ordinary-least-squares interval.
- `state-space`: a local linear trend model (statsmodels, the `stats` extra).
- `arima`: an ARIMA model with native prediction intervals (the `stats` extra).
- `gaussian-process` and `bayesian-ridge`: scikit-learn models over the trend (the `ml` extra).

### Synthetic data and the benchmark

No labelled succulent time series exists, so the advanced forecasters have nothing real to fit or
score against. `phytovision simulate` fills that gap with a seeded dry-down simulator: a latent stress
state rises under a decline forcing, the observed score is a noisy readout, a wilt event fires when the
latent state crosses the stressed cut, and each step carries a real-namespace feature vector. Every
row is labelled synthetic in its `source`. `phytovision benchmark` then runs every forecaster over a
synthetic cohort with time-series cross-validation and ranks them by CRPS, pinball loss, and
prediction-interval coverage. On the simulator the state-space and ARIMA models beat the naive linear
baseline, whose interval undercovers at longer horizons.

Honesty caveats:

- The forecast is an extrapolation of the recent stress trend, not a validated prognostic. A prediction
  interval is an uncertainty estimate, not a measurement. Confidence decays the further ahead it
  projects and grows with trajectory length.
- The simulator is a small generative tool for benchmark data, not a plant physiology model. Any
  forecaster fitted or benchmarked on it is validated against synthetic data, not real succulents.
- A distribution-free interval is available separately (split-conformal regression), which reaches
  close to its nominal coverage on the simulator where a parametric interval may not.
- Each observation is treated as one time step, so horizons are days only under daily sampling.
- Inputs are RGB. The cited work fuses RGB with multispectral sensors; multispectral fusion is out of
  scope here (no hardware or data). A `Sample.extra["modality"]` tag is reserved for that future work.

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
- Forecasts carry a prediction interval per horizon, and split-conformal regression turns any point
  forecaster's residuals into a distribution-free interval. The `benchmark` command scores forecasters
  with proper scoring rules (CRPS, pinball loss) and reports interval coverage against width.
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
