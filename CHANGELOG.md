# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Reproducibility: a typed, validated config schema (`config_schema.PipelineConfig`) that
  `Pipeline.from_config` routes through, so an unknown top-level config key is rejected (a mistyped
  slot name fails loudly instead of silently) and the resolved config is diffable via `as_dict()`; a
  one-call `seeding.set_global_seed`, applied in the CLI entry point for any `--seed` command so the
  global RNGs are seeded alongside the per-stage seeds; and a CONTRIBUTING section documenting the
  seeding, the validated config, MLflow tracking, and optional DVC data versioning with a sample config.
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
- Model-agnostic occlusion saliency: `occlusion.occlusion_saliency` reruns the pipeline over occluded
  patches of the plant and paints each patch by how far hiding it moves the stress score, so it
  localizes any driver (shape and texture, not only colour) and works for any model, unlike the
  colour-only pigment saliency. It reruns the pipeline once per patch, so it lives behind a flag:
  `analyze --save-occlusion` and `visualize.render_occlusion_overlay`. Every value is an RGB proxy of
  the score's source, not a measurement.
- Standalone physiology head: `PhysiologyHead` (`analyze --physiology`, `/analyze?physiology=true`, and
  a dashboard panel) reports the water-potential, stomatal-conductance, and transpiration proxies on
  their own, without running the drought-stage rule set. The proxies derive from the same drought
  markers, so the standalone head and the `drought_stage` block always agree; every value stays a crude
  RGB proxy, not a measurement.
- New optional extras: `stats` (statsmodels and lifelines, for the statistical forecasters and the
  survival models) and `tracking` (mlflow, for benchmark logging), both added to the `all`
  self-reference and the CI install lists.

### Fixed
- A project-wide bug sweep closed a set of latent defects: the config schema now rejects a malformed
  (non-mapping) `params` instead of silently dropping it; `read_config` validates the extension before
  reading and wraps a non-UTF-8 file as a clean `ConfigError`; manifest and cohort readers reject a
  non-finite (`nan`/`inf`) or non-numeric cell rather than poisoning a regression report or crashing;
  `validate --bins` and `benchmark --interval-level` reject out-of-range values cleanly instead of
  dumping a traceback; the forecasters' time-to-stressed search no longer reports a too-late crossing
  when a horizon above its 30-step window is requested, and the linear forecaster and `forecast_scores`
  now reject an out-of-range interval level like the others do; the excess-green segmenter no longer
  integer-overflows on a uint8 image; the covariate survival model now actually drops a constant column
  (the standard-deviation floor had made the drop unreachable); the gradient-boosted verdict uses the
  shared bucket cuts so it agrees with the other models at a score; the synthetic events table emits a
  1-based duration matching the survival contract (never a 0 that a parametric fit rejects); a survival
  cohort with no repeated observations degrades to a clear note on `/trend` and the dashboard instead of
  raising; the folder loader skips a directory whose name ends in an image suffix; and the saliency and
  occlusion overlays tint a mid-strength pixel with its true colour at a strength-scaled opacity, rather
  than applying the magnitude twice and pulling it toward grey.
- A second sweep closed the deferred findings: the GLCM and edge-density texture descriptors now measure
  interior surface texture only (foreground-to-foreground co-occurrences, and an eroded interior for the
  edge density), so a concave silhouette no longer leaks into them; the quality and preprocessing scalers
  are dtype-aware, so a uint8 near-black frame and a higher-bit-depth image are normalized by their true
  range rather than a value guess or a hard-coded 255; the model reader reports an unreadable or
  incompatible file as a clean error instead of mislabelling it a missing extra; `analyze` emits the
  conformal set only when `--conformal` is passed; the survival leaderboard names a model it could not
  score instead of dropping it; and `normalize01` collapses a degenerate range to its clamp rather than
  dividing by zero.
- An intensive multi-round hunt closed a further set: the GLCM descriptor averages only scan directions
  that carry foreground pairs, so a thin (single-row or single-column) region is no longer deflated; a
  degenerate single-observation forecast now carries an empty interval, matching its contract, and
  `phenotype` omits survival with a notice instead of crashing when no plant has two observations; the
  forecaster benchmark builds each model at the requested coverage level rather than the default; the
  YOLO loader skips a directory whose name ends in an image suffix; the mask cleanup keeps a component
  exactly at the size threshold instead of deleting it; `hue_mean` is a circular mean, so a red hue
  split across the wraparound no longer averages to its opposite; and `benchmark --min-train` is
  validated.
- A third hunt closed nine more: the watershed leaves reassign a dropped sub-threshold basin to the
  nearest kept leaf, so per-leaf regions still tile the plant and the aggregate area and canopy
  coverage no longer undercount; the aggregator averages hue on the circle across leaves (a new
  `circular` reduction policy), completing the `hue_mean` fix for the multi-leaf case; the
  gradient-boosted SHAP output is oriented to match its values and base, so completeness holds when
  the positive label is class 0; a model forecaster reports its current level from the last
  observation rather than an unrelated linear fit, keeping it consistent with its own projection;
  grouped cross-validation derives each fold's feature schema from the training rows only, so a
  dataset-specific feature no longer leaks into the model or breaks the fit; the YOLO loader skips a
  non-finite class id instead of crashing with an `OverflowError`; `read_config` accepts a UTF-8 byte
  order mark; the simulator buckets the label from the rounded score it stores, so the two never
  straddle a cut; and the dashboard survival band steps like the Kaplan-Meier curve rather than
  sloping between event times.
- Forecasting and survival correctness: the Gaussian-process forecaster folds the detrending line's
  extrapolation uncertainty into its band, so a far horizon is no longer overconfident; the
  state-space reader coerces a degenerate (two-point) forecast to the expected shape rather than
  silently falling back to a mislabelled linear interval; the survival covariate window is capped at
  the crossing, so a plant that wilts within the warmup window no longer leaks its outcome into the
  "early" covariates and the held-out concordance; the concordance sentinel for a plant with no
  in-window median now sits above every finite predicted median, so it ranks as the longest-surviving
  instead of below an over-extrapolated one; and `load_history` reports a clean error on a truncated
  manifest row rather than crashing with a `TypeError`.
- Edge-case robustness across the codebase: the preprocessor no longer misreads a normalized float
  image with a stray over-one pixel as 8-bit and darkens the whole frame; mask cleanup uses
  8-connectivity, so a thin diagonal structure is kept as one object instead of vanishing; the config
  schema rejects an unknown key inside a component spec (a forgotten `params` wrapper or a typo) rather
  than silently dropping the override; the split-conformal quantile subtracts a rounding epsilon so an
  exact rank does not widen every set by one; the gradient-boosted model gives no attribution to a
  schema-drifted (absent) feature and orients its SHAP output for a class-0 positive label; the
  simulator applies a step-zero watering event instead of ignoring it; and the survival cohort excludes
  a prevalent plant (already over the cut at its first frame) that has no pre-event window.
- IO, dtype, and numeric hardening: a non-finite (NaN/inf) image is rejected loudly at validation
  rather than silently corrupting the analysis; the colour features normalize their pixels so a uint8
  image no longer overflows the green-chromatic denominator; the interior edge density erodes with a
  3x3 element matching the Sobel support, so a curved silhouette no longer leaks into it; the COCO and
  CSV manifest loaders raise a clean error on a non-UTF-8 file and the YOLO loader tolerates a stray
  byte per line; an empty `images_root` falls back to the default folder instead of the working
  directory; the default linear forecast caps its time-to-stressed like the richer forecasters and
  centres its interval on the clipped projection so the band never collapses to zero width at the
  ceiling; the `/trend` unknown-forecaster error drops its stray quotes; `phenotype` fails cleanly when
  a selected forecaster's extra is absent; and a corrupt or truncated model file is wrapped in a clean
  error instead of leaking a decompressor `OSError`.
- Interval, aggregation, and contract hardening: the Gaussian-process, Bayesian-ridge, and state-space
  forecasters centre their prediction band on the clipped mean, so a projection past the [0, 1] ceiling
  no longer collapses the band to zero width and hands the probabilistic scorer a near-certain sigma,
  the same fix the linear baseline already carries; plant-level orientation reduces as an axial angle (a
  new `axial` reduction policy), so leaves straddling the plus-or-minus 90-degree seam no longer average
  to the wrong axis; the SHAP explainer skips a schema-drifted feature instead of citing it with a NaN
  value that would reach the JSON digest; image validation rejects a non-numeric dtype as a clean
  `InvalidImageError` rather than a raw `TypeError`; `add_head` rejects a duplicate head name and the
  config schema rejects duplicate feature extractors, so a name collision fails loudly at build time
  instead of silently dropping a head output or crashing every `analyze`; grouped cross-validation keeps
  rows with no source together in one synthetic group rather than crashing on a mix of `None` and string
  sources; image loading applies the EXIF orientation tag, so a portrait photo is analysed upright and
  its orientation feature matches the photo as viewed; and a loadable-but-malformed model envelope (a
  non-mapping state or manifest) is wrapped in a clean `ConfigError` rather than leaking a `TypeError`.
- Numeric edges at the seams and the extremes: a circular mean whose vector lands a hair below the
  wraparound no longer rounds up to exactly 1.0, which is outside the documented `[0, 1)` hue range and
  sits at the opposite end of the linear feature range from the value it should report; this is fixed
  in both the per-region `circular_hue_mean` and the plant-level circular reduction, so two reds
  straddling the seam stay red at every level. The default linear forecast reports no time-to-stressed
  on a degenerate (non-finite) fit instead of crashing with a `ValueError` from `math.ceil(NaN)`,
  matching what the pluggable forecasters already did. The heuristic model's logistic saturates to 0.0
  or 1.0 at an extreme configured bias rather than raising `OverflowError` from `math.exp`. And the
  COCO loader stringifies a per-box category the same way it stringifies the declared category list, so
  a numeric category name still matches the vocabulary it belongs to.
- CLI, API, survival, and forecasting robustness across the serving surfaces: `validate` builds the
  reliability curve and Brier score only from labelled rows, so an unlabelled manifest no longer reads
  every image as a true stressed event and fabricates a catastrophic-miscalibration report; `analyze`
  rejects a `--save-*` path with no image extension up front instead of crashing with a raw Pillow
  `ValueError` after the work is done; `simulate` rejects a non-positive `--steps` rather than writing
  one observation per plant under a step count that never happened; `attach_heads` is idempotent, so a
  served pipeline that already carries a head plus the matching request flag returns a clean response
  rather than a 500 from the duplicate-name guard; the API refuses to build with an uncalibrated
  conformal wrapper instead of raising on every request; the Cox model predicts a single-record cohort
  (a leave-one-out survival fold) rather than raising `AttributeError` on the scalar lifelines squeezes
  a one-row frame to; a loadable model envelope whose state is incomplete or wrong-typed is wrapped in a
  clean `ConfigError` rather than a raw `KeyError`; an interval level a hair below 1.0 is rejected at
  construction rather than crashing inside `NormalDist`; and the state-space forecaster floors its band
  width like the other forecasters, so a boundary-solution fit on a smooth series no longer reports a
  near-zero-width interval.
- Honest reporting and state hygiene across survival, forecasting, and the CLI: a prevalent plant
  (already over the stressed cut at its first frame) and an all-prevalent cohort are named distinctly
  from a genuinely too-short series, in the `phenotype` `survival_basis` column and the
  `InsufficientDataError` message, so a ten-observation plant is never mislabelled "insufficient
  observations"; the survival leaderboard distinguishes "no fold had enough events" from "no fold had a
  comparable pair" (all durations tied); a covariate survival model that fails at predict time degrades
  only that one call to a baseline fitted on its training cohort and keeps the trained fitter, rather
  than refitting on the predict-time cohort and nulling the fitter so every later prediction returned
  that cohort's median; a statistical forecaster that could not fit and fell back to the linear
  interval is flagged (`Forecast.degraded`) and surfaced in the benchmark's `fallbacks`, so a row whose
  numbers are partly the fallback is visible rather than read as pure model output; a non-finite score
  in a forecast series is rejected with a `ContractViolationError` instead of silently projecting a
  confident 0.0; the `phenotype` JSON output carries the same column set on every object (`null` where
  a value is absent) as the CSV does, rather than objects whose keys vary by row; the `--strict-schema`
  flag is tri-state, so an unset flag keeps a loaded model's own drift policy instead of silently
  resetting it to lenient; and `benchmark --mlflow` reports a tracking-store failure (a read-only
  directory or an unreachable server) as a clean error rather than a traceback that discards the ranked
  table.
- The Bayesian-ridge forecaster floors its predictive spread at the same residual-std minimum the
  linear, Gaussian-process, and state-space forecasters use, so a smooth series of a few noisy readings
  no longer reports an interval tens of times too narrow that claims near-certainty and tanks its
  benchmark coverage. It was the last forecaster missing the floor.

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
