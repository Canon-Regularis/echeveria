# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dry-down simulator (`phytovision simulate`): a seeded generative model of a succulent dry-down that
  emits synthetic labelled sequences (a per-observation manifest plus a per-plant events table with
  durations and censoring). It exists because no labelled succulent time series is available to fit or
  benchmark the temporal models; every row is labelled synthetic in its `source`.
- Probabilistic trajectory forecasting: a `TrajectoryForecaster` contract and a `FORECASTERS` registry
  with five pluggable models (a linear-trend baseline, a local-linear-trend state-space model, an ARIMA
  model, a Gaussian process, and a Bayesian ridge). Each reports a prediction interval per horizon.
  `Forecast` now carries per-horizon lower/upper bounds and the method name; the intervals thread into
  the `phenotype` columns, the `/trend` forecast block (`/trend?forecaster=`), and the dashboard
  projection as a shaded band.
- Probabilistic evaluation: CRPS, pinball loss, prediction-interval coverage and width, and PIT
  (`evaluation/probabilistic.py`); split-conformal regression intervals extending the conformal
  primitive from label space to residual space (`evaluation/conformal_regression.py`); and an
  expanding-window time-series cross-validation splitter (`evaluation/timeseries.py`).
- Forecaster benchmark (`phytovision benchmark`): runs every registered forecaster over a synthetic
  cohort with time-series cross-validation and ranks them by CRPS, pinball, and coverage, with optional
  MLflow logging behind the new `tracking` extra.
- Survival analysis of time-to-wilt: a `SurvivalModel` contract and a `SURVIVAL_MODELS` registry with a
  Kaplan-Meier cohort baseline, a Weibull accelerated-failure-time model (the default), and a Cox
  proportional-hazards model, all over events derived from the observed stress crossing (never the
  latent) and two observable covariates. It surfaces as a `survival` block on `/trend`
  (`/trend?survival_model=`), four columns on the `phenotype` table (`--survival-model`), and a survival
  curve on the dashboard, plus a held-out concordance leaderboard (`benchmark_survival_models`). Needs
  the `stats` extra (lifelines); estimates are synthetic-trained RGB proxies, not validated prognoses.
- Per-leaf tracking across frames: a `LeafTracker` (Hungarian assignment on centroid and area) that
  gives each leaf a stable identity over a plant's sequence, `build_leaf_histories` that turns a
  leaf-instance pipeline's reports into one observation sequence per leaf (so the forecasters and the
  survival model run per leaf, not only per plant), and an optional `per_region` field on `Observation`.
- Physiology proxies: water-potential, stomatal-conductance, and transpiration RGB-proxy indices
  derived from the existing pigment, turgor, and necrosis markers, surfaced in the `drought_stage` head
  output (a `physiology` block plus a `physiology_basis` caveat) and grounded with `_PHYSIOLOGY` notes.
  They are crude proxies, not measurements (the water-potential proxy is ordinal, not MPa; conductance
  and transpiration are relative indices, not fluxes); they add interpretive grounding, not independent
  signal, and are never fed to the stress model or the forecaster.
- Skeleton morphology features: a `SkeletonFeatures` extractor (`skeleton` namespace) over the region
  silhouette (skimage `medial_axis`), reporting skeleton length, branch and endpoint counts, medial
  thickness, and tortuosity. It is registered but kept out of the default stack, so the shipped feature
  schema does not drift; it describes the silhouette, not the vein network.
- New optional extras: `stats` (statsmodels and lifelines, for the statistical forecasters and the
  survival models) and `tracking` (mlflow, for benchmark logging), both added to the `all`
  self-reference and the CI install lists.

## [0.2.0] (2026-07-16)

### Added
- Registry-driven pipeline construction: `Pipeline.from_config()` / `Pipeline.from_names()` and CLI
  `--model` / `--segmenter` / `--explainer` selection.
- Dataset tool: `batch`, `train`, and `evaluate` CLI commands; folder, image-directory, and COCO
  loaders; a shared `analyze_dataset` engine.
- Models: gradient-boosted (`ml` extra) and soft-voting ensemble, plus a type-tagged persistence
  envelope (`save_model`/`load_model`) with a provenance manifest.
- Honest uncertainty and evaluation: split conformal prediction (`train --calibrate`,
  `analyze --conformal`), grouped stratified cross-validation (`evaluate --cv`), and
  leave-one-dataset-out transfer (`evaluate --transfer`).
- Explainability: SHAP explainer (`--explainer shap`), completeness (additivity) check, counterfactuals
  (`analyze --counterfactual`), and global permutation importance (`evaluate --importance`).
- Correctness safeguards: a model schema-drift guard (`--strict-schema`, `ModelSchemaError`) and a
  runtime feature contract (`PlantFeatures` finiteness, declared ranges).
- Reach: a Lab-chroma segmenter for coloured succulents, a FastAPI service (`api` extra, `serve`),
  and annotated overlays.
- Leaf-instance segmentation: a classical watershed leaf segmenter (the `leaf-instance` region
  provider) that unlocks per-leaf traits with no training.
- More dataset loaders: CSV/TSV manifest and YOLO detection loaders, all selectable through a
  `DATASET_LOADERS` registry.
- Optional disease-appearance head, a documented placeholder that is not a validated diagnostic,
  reachable from the CLI (`analyze --disease`), the API (`/analyze?disease=true`), and the dashboard.
- Temporal tracking: a per-plant `FeatureHistory` store and a stress-trend fit, with `Sample` carrying
  optional `plant_id`/`timestamp`, a `build_history` ingest from any tagged loader, a stateless
  `/trend` batch endpoint, and a dashboard time-series view.
- Interactive Streamlit dashboard (`dashboard` extra, `phytovision dashboard`): a neutral-dark tabbed
  terminal with an ANALYZE tab (verdict, overlay, drivers, disease, timing, features) and a TEMPORAL
  tab (stress-over-time trend from a CSV manifest).
- Drought-physiology grounding (Sedum 2019 study): a `colour.red_fraction` anthocyanin/reddening
  feature, physiology-grounded explanations (drivers cite their mechanism), a rule-based drought-stage
  head (`analyze --drought-stage`, `/analyze?drought_stage=true`, and the dashboard) naming an ordinal
  stage from the marker pattern, and a pigment-before-collapse temporal early warning (in `/trend` and
  the dashboard TEMPORAL tab). All are RGB proxies and literature-motivated priors; see MODEL_CARD.md.
- High-throughput phenotyping and forecasting (2024 HTP study, arXiv 2402.18751): a trend-extrapolation
  forecaster implementing the reserved `LeafDeathPredictor` (projected stress per horizon,
  steps-to-stressed, confidence), a `phytovision phenotype` command exporting a per-plant trajectory
  table over a manifest, and a `forecast` field on `/trend` plus a projection in the dashboard TEMPORAL
  tab. The forecast is a labelled trend extrapolation, not a validated prognostic; RGB-only (the study's
  multispectral fusion is deferred).
- Observability: per-stage timing on `AnalysisReport` (`analyze --timing`), a pytest-benchmark harness,
  and a `MODEL_CARD.md`.
- Verification: a registry-driven substitutability contract suite, Hypothesis property tests,
  metamorphic invariants, and mutation-testing / dependency-audit / release CI workflows.
- Exception hierarchy rooted at `PhytoVisionError` (`InvalidImageError`, `ContractViolationError`,
  `SegmentationError`, `ModelNotFittedError`, `ModelSchemaError`, `ConfigError`).
- Structured `logging` at pipeline stage boundaries and at every silent fallback.
- Optional post-model **head** seam (`Pipeline.add_head`, `AnalysisReport.head_outputs`).
- Extractor-declared reduction policy (`sum` vs `mean`) so the aggregator no longer hardcodes keys.
- Shared `validate_rgb_image` input validation at the pipeline entry point.
- PEP 561 `py.typed` marker; full public API re-exported from `phytovision`.
- CI (GitHub Actions), pre-commit config, coverage gate, and Dependabot.
- Input and segmentation quality gate: a per-analysis `quality` assessment (blur, uniformity, and
  segmentation coverage) on `AnalysisReport` and its `summary()`, surfaced in the CLI, the API
  disclaimer, and the dashboard, so an unanalysable image is flagged rather than scored silently.
- Reproducibility: `train --seed` and `evaluate --seed`, threaded into the gradient-boosted model and
  the cross-validation folds and recorded in the persistence manifest.
- Spatial pigment saliency: a per-pixel map of the colour drivers behind the score
  (`analyze --save-saliency`, the `/saliency` API route, and the dashboard), labelled an RGB proxy.
- Scientific validation: a `validate` command reporting a reliability curve, Brier score, and
  RMSE/MAE/R2 of the score against a measured water-status target, with a numeric `target` column on
  the CSV manifest and on `Sample`. The datasets remain non-succulent, so results are indicative.
- Presentation: README status badges, a docs build-and-deploy CI workflow (GitHub Pages), Docker
  images and a compose file for the API and dashboard, and an example notebook.

### Changed
- `Explainer.explain` now types the model as `StressModel` and logs when explanations are unavailable.
- `GradientBoostedStressModel` raises `ModelNotFittedError` instead of using a bare `assert`.
- Packaging modernized to PEP 639 SPDX license metadata; `all` extra is self-referential.
- Version is single-sourced from installed distribution metadata.

## [0.1.0] (2026-07-12)

### Added
- Initial explainable water-stress pipeline: preprocessing, Excess-Green segmentation, whole-plant
  region provider, geometry/colour/texture/morphology feature extractors, plant-level aggregation,
  interpretable heuristic stress model, and feature-contribution explainer.
- Reserved interfaces for the future leaf-instance, disease, and temporal modules.
