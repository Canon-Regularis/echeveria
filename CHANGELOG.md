# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Registry-driven pipeline construction: `Pipeline.from_config()` / `Pipeline.from_names()` and CLI
  `--model` / `--segmenter` selection.
- Exception hierarchy rooted at `PhytoVisionError` (`InvalidImageError`, `ContractViolationError`,
  `SegmentationError`, `ModelNotFittedError`, `ConfigError`).
- Structured `logging` at pipeline stage boundaries and at every silent fallback.
- Optional post-model **head** seam (`Pipeline.add_head`, `AnalysisReport.head_outputs`).
- Extractor-declared reduction policy (`sum` vs `mean`) so the aggregator no longer hardcodes keys.
- Shared `validate_rgb_image` input validation at the pipeline entry point.
- PEP 561 `py.typed` marker; full public API re-exported from `phytovision`.
- CI (GitHub Actions), pre-commit config, coverage gate, and Dependabot.

### Changed
- `Explainer.explain` now types the model as `StressModel` and logs when explanations are unavailable.
- `GradientBoostedStressModel` raises `ModelNotFittedError` instead of using a bare `assert`.
- Packaging modernized to PEP 639 SPDX license metadata; `all` extra is self-referential.
- Version is single-sourced from installed distribution metadata.

## [0.1.0] - 2026-07-12

### Added
- Initial explainable water-stress pipeline: preprocessing, Excess-Green segmentation, whole-plant
  region provider, geometry/colour/texture/morphology feature extractors, plant-level aggregation,
  interpretable heuristic stress model, and feature-contribution explainer.
- Reserved interfaces for the future leaf-instance, disease, and temporal modules.
