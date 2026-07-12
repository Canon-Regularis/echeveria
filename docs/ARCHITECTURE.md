# Architecture

The code runs an image through a fixed sequence of steps. Each step is a small, replaceable part with a
clear job. This page explains the steps, the two design choices that make the project easy to extend,
and what a new part has to do to fit in.

## The steps, in order

1. **Preprocess.** Resize the image and scale its values to a standard range.
2. **Segment the plant.** Produce a true/false mask marking which pixels are plant.
3. **Choose what to measure.** Turn the plant mask into one or more regions to measure.
4. **Measure each region.** Compute geometry, colour, texture and morphology numbers per region.
5. **Combine.** Reduce the per-region numbers to a single set of numbers for the whole plant.
6. **Score.** Feed those numbers to a water-stress model, which returns a score and a confidence.
7. **Explain.** List the features that pushed the score up or down.

The result of a run is an `AnalysisReport`: the mask, the regions, the plant numbers, the score, and the
explanation.

## The two ideas that make it extensible

**What gets measured is decided in one place: the region provider (step 3).** Today the provider hands
the rest of the pipeline a single region, the whole plant. A future leaf module will hand it several
regions, one per leaf. Every step after that works on "a region" and does not care how many there are.
So per-leaf measurements can be added later by writing one new part, without touching the code that
measures, combines, scores, or explains. Counts that only make sense per leaf (leaf count, wilted-leaf
ratio) are left empty for now and fill in on their own once a leaf provider is in use.

**The model on top reads the combined plant numbers, nothing else.** The water-stress model takes that
one set of numbers and returns a score. A disease or growth model later would read the same numbers. So
you can add or swap the model without changing anything that produced the numbers.

## Building and choosing a pipeline

Three ways to build one:

- `Pipeline.default()` gives the standard setup.
- `Pipeline.from_names(model="gradient-boosted", segmenter="exg-otsu")` swaps parts by their registered
  name.
- `Pipeline.from_config({...})` takes a full config (names and their parameters), which you can load
  from a TOML or JSON file.

Registered names live in `registries.py`. To add a part, write it, register it there under a name, and
it becomes selectable from all three builders and from the CLI. The `with_*` methods (for example
`with_model`) return a copy of a pipeline with one part swapped, and `add_head` attaches an optional
extra analysis that runs after the model.

## Where things live

| Area | What it does | Main types |
| --- | --- | --- |
| `types.py` | The data passed between steps | `Image`, `Mask`, `Region`, `RegionSet`, `FeatureVector`, `PlantFeatures`, `StressAssessment`, `Explanation`, `AnalysisReport` |
| `registry.py`, `registries.py` | Look parts up by name | `Registry`, `SEGMENTERS`, `STRESS_MODELS`, ... |
| `preprocessing/` | Resize and normalize | `Preprocessor`, `ResizeNormalizePreprocessor` |
| `segmentation/plant/` | Find the plant (foreground) | `PlantSegmenter`, `ExGThresholdSegmenter` |
| `segmentation/leaves/` | Split the plant into leaves (planned) | `LeafInstanceSegmenter` |
| `regions/` | Decide what gets measured | `RegionProvider`, `WholePlantRegionProvider`, `LeafInstanceRegionProvider` |
| `phenotyping/` | Measure traits per region | `FeatureExtractor`, `GeometryFeatures`, `ColourFeatures`, `TextureFeatures`, `MorphologyFeatures` |
| `phenotyping/aggregation/` | Combine region traits into one plant vector | `FeatureAggregator`, `PlantLevelAggregator` |
| `models/stress/` | Score water stress | `StressModel`, `HeuristicStressModel`, `GradientBoostedStressModel` |
| `models/disease/`, `models/temporal/` | Planned model types | interfaces only |
| `explainability/` | Explain a score in plain terms | `Explainer`, `FeatureContributionExplainer` |
| `datasets/` | Load datasets through one interface | `DatasetLoader`, `FolderClassificationLoader` |
| `pipeline.py` | Run the steps in order and wire them | `Pipeline` |
| `exceptions.py` | The error types the code raises | `PhytoVisionError` and its subclasses |
| `validation.py` | Check the input image once | `validate_rgb_image` |
| `cli.py` | The command-line tool | `main` |

## What a new part has to do

- **Plant segmenter:** return a true/false mask the same height and width as the image, where true means
  plant.
- **Region provider:** return at least one region. Each region has a mask the size of the image with at
  least one pixel set, and a label saying what it is (`plant` or `leaf`).
- **Feature extractor:** take an image and a region and return a flat set of numbers. Do not change the
  inputs. Prefix each key with the extractor's own name (for example `geometry.area`) so different
  extractors never clash. It also declares which of its numbers should be summed across regions rather
  than averaged.
- **Aggregator:** do not assume how many regions there are. Leave per-leaf-only fields empty when the
  regions are not leaves.
- **Stress model:** return a score between 0 and 1 and a confidence between 0 and 1.

## Errors and validation

Everything the library raises is a `PhytoVisionError`, so you can catch its failures without also
catching unrelated bugs. The common ones are `InvalidImageError` (bad, empty, or missing image),
`ModelNotFittedError` (a trainable model was used before it was trained), and `ConfigError` (a config
asked for a part that is not registered). Each also inherits the matching built-in, so
`InvalidImageError` is a `ValueError` and `ModelNotFittedError` is a `RuntimeError`, and older `except`
blocks keep working.

The input image is checked once, at the pipeline entry, by `validate_rgb_image`. The steps rely on that
instead of each re-checking the shape.

## Logging

Modules log through the standard `logging` module and never print. The pipeline logs each step at debug
level. It logs a warning whenever it falls back, for example when the green threshold finds no plant and
it drops back to a saturation threshold, or when a bad feature value is replaced with zero. The
application decides where the logs go. The CLI's `-v` flag turns on debug output for echeveria's own
logs and leaves other libraries quiet.
