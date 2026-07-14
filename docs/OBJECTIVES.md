# Objectives and roadmap

echeveria detects water stress in succulents from RGB photos. The first version works at the level of
the whole plant: find the plant, measure it, and score how stressed it looks, with reasons. The steps
that do the finding, measuring, and explaining are reusable, so two later goals can be added on top
without redoing them: detecting disease, and predicting when a leaf will die.

The long-term goal, end to end: separate the plant from the background, separate individual leaves,
measure useful traits, estimate health, detect water stress, and later detect disease and predict leaf
death over time. Every prediction should come with a plain reason.

## Objective labels used in the dataset list

The dataset list in [DATASETS.md](DATASETS.md) tags each dataset with what it can help with:

- `seg-plant`: separating the plant from the background.
- `seg-leaf`: separating each leaf from the others.
- `pheno`: measured physiological values (like water content or leaf area) to check computed features
  against.
- `water-stress`: a label or signal for whether the plant is water-stressed.
- `disease`: a label for plant disease.
- `temporal`: repeated photos of the same plant over time.

## The objectives

Each objective says what "done" looks like and which datasets feed it.

### Objective 0: Load datasets through one interface

Read the different dataset layouts (Kaggle folders, Roboflow exports, Mendeley CSV plus images, Zenodo
masks) into one shared format, keeping each sample's source and license attached.

Done when any dataset in the list loads through the same interface, every sample carries its source and
license, and there is a held-out succulent test split.

### Objective 1: Separate the plant from the background

Produce a foreground mask. The default uses Excess-Green plus an Otsu threshold, which needs no
training. A learned model (U-Net, DeepLabV3+, Mask R-CNN, YOLO segmentation) can replace it later.

Done when the mask overlaps hand-labelled succulent images well (roughly 0.9 IoU or better).

Data: the background-removed aloe pairs (#19) and PlantVillage's segmented variant (#18) give foreground
examples. The cactus and houseplant box sets (#5, #2) help a plant locator. Boxes are not masks, so a
few real succulent masks are still needed to check the result.

### Objective 2: Separate each leaf (planned)

Deferred from the first version. There is no hand-labelled succulent leaf dataset, and every public
leaf-segmentation set is flat-leaved (Arabidopsis, tobacco, cauliflower), which is a poor match for a
tight succulent rosette. When resumed, it plugs in as a new region provider and the rest of the pipeline
does not change.

Prerequisite: a small hand-labelled Echeveria leaf set, about 100 to 300 images.

Data for method work: KOMATSUNA (#7), CVPPP (#6), MSU-PID (#8), GrowliFlower (#9), ALED (#13), and the
synthetic UPGen (#10).

### Objective 3: Measure each region

Compute per region: geometry (area, perimeter, solidity, aspect ratio, circularity, elongation), colour
(RGB, HSV and LAB stats, greenness indices, yellow and brown pixel share, saturation), texture (local
binary patterns, GLCM contrast, homogeneity, energy and entropy, edge density), and morphology
(roughness, concavity, curvature).

Done when the feature code is deterministic, unit-tested, and runs on any set of regions. Keeping it
independent of how many regions there are is what lets the leaf module drop in later for free.

Data: lettuce thermal and soil moisture (#15), wheat fluorescence (#14), and avocado/olive/grape (#16)
to calibrate against measured values. None are succulents, so treat them as guidance, not as ground
truth for Echeveria.

### Objective 4: Combine into one set of plant numbers

Reduce the per-region numbers to one vector for the plant: total leaf area, average greenness and colour
stats, canopy coverage, texture and morphology summaries. Leaf count and wilted-leaf ratio need per-leaf
regions, so they stay empty until the leaf module exists.

Done when one plant feature vector is produced, with the per-leaf-only fields explicitly marked empty in
the first version.

### Objective 5: Score water stress, with reasons

Run two models: an interpretable one (the default) and a trainable gradient-boosted one for comparison.
The output is a score, a confidence, and the features that drove it.

Done when both beat a trivial baseline on a held-out succulent split, the interpretable model gives
feature-level reasons, and water stress and disease stay separate tasks so their labels do not leak into
each other.

Data: the aloe health sets (#3, #4) and wilted-houseplant sets (#1, #2), with the physiological sets
(#14, #15, #16) to validate against.

### Objective 6: Explain the score

For the interpretable model, list the features that pushed the score up or down. For a vision model
later, use saliency or Grad-CAM.

Done when every prediction comes with a reason a person can read.

### Objective 7: Dashboard and API

A view of the plant, its numbers, the stress score, and highlighted problem areas, plus a programmatic
API over the same output.

## The main decision: whole plant now, leaves later

The first version measures the whole plant as one unit and does not separate leaves. This was a
deliberate call (2026-07-12): there is no time to hand-label a succulent leaf dataset, and none exists to
reuse. The public leaf datasets are all flat-leaved plants, which do not transfer to a thick,
self-occluding succulent rosette.

The important part is that this does not close the door. The measuring, combining, scoring, and
explaining code all work on regions, not on "the plant" specifically. When a leaf module is added, it
becomes the part that hands over regions (one per leaf), and the per-leaf numbers (leaf count,
wilted-leaf ratio) start filling in on their own. To resume: hand-label 100 to 300 Echeveria images,
then pretrain on the public leaf sets listed above.

## Future work

Designed so the current pipeline does not need to change:

- **Track the same leaf across days**, using the same-plant sequences in KOMATSUNA, MSU-PID, ALED, and
  GrowliFlower.
- **Growth metrics**: growth rate, leaves appearing and disappearing, a rough biomass proxy.
- **Predict leaf death**: a model over each leaf's history that estimates the chance it dies within 7,
  14, or 30 days. This needs repeated photos of the same succulents over time, which do not exist yet.
  Setting up a fixed camera for a controlled dry-down is the way to get that data. UNL cotton (#17) and
  ALED (#13) are the closest existing matches.
- **Disease detection**: swap the model on top and reuse everything before it. Data: the aloe disease
  sets (#19, #20, #21), with PlantVillage (#18) as extra.

## Development order

1. Ingestion: the shared loader, the internal format, and the held-out succulent test split.
2. Plant segmentation: the foreground mask, which becomes the single region.
3. Region-agnostic feature extraction: the geometry, colour, texture, and morphology functions, wired
   through the whole-plant region provider.
4. Aggregation: the plant feature vector, with per-leaf-only fields left empty.
5. Water-stress models: the interpretable one and the trainable one.
6. Explanations.
7. Dashboard and API.

Later, without a rewrite: the leaf module (a new region provider), temporal tracking and leaf-death
prediction (once there is longitudinal succulent data), and disease detection (a new model on top).

## Recommended first datasets

The smallest set to build an end-to-end first version, chosen for being succulent, permissively
licensed, and downloadable today:

1. Aloe Vera Health Detection wwmar v22 (#3): aloe stress and health, CC BY 4.0.
2. Aloevera Health Detection Y-V11 v8 (#4): the largest aloe stress set, CC BY 4.0.
3. Healthy and Wilted Houseplant Images (#1): a broad healthy/wilted baseline, license unclear.
4. Aloe Vera Diseases, background-removed (#19): foreground supervision plus disease, CC BY 4.0.
5. Lettuce thermal and soil moisture (#15): the one set that pairs images with measured water status,
   CC BY 4.0.

The leaf datasets (KOMATSUNA, CVPPP, MSU-PID, ALED, UPGen) come in only when the leaf module is resumed.

## Risks and open decisions

Listed by how much they threaten the main goal.

**Resolved: leaves are out of the first version (decided 2026-07-12).** There is no succulent leaf-mask
data, public sets are all flat-leaved, and the one succulent instance-segmentation set (#12) is about 41
images and probably marks whole plants, not leaves. The first version measures the whole plant. Leaves
plug in later as a new region provider. See the decision section above.

**No succulent physiological ground truth.** All the measured-value datasets are lettuce, wheat,
avocado, or cotton, which are ordinary C3 plants. Succulents store water internally and behave
differently, so those datasets are guidance, not ground truth for Echeveria. Mitigation: a small
in-house loop, weighing pots for water content, measuring leaf area, and using a cheap thermal camera,
to anchor the features on real succulents.

**Water-stress labels are coarse and noisy.** "Wilt" mixes drought with over-watering, root rot, and
disease. There is no severity scale and no link to measured water status. The aloe detection sets also
put stress and disease in the same boxes, so their labels can leak into each other. Mitigation: define
one clear water-stress labelling scheme up front and keep it separate from disease.

**Some "plant segmentation" data is the wrong kind.** Several sets tagged for plant segmentation are
bounding boxes or background-removed images, not real masks. Do not train segmentation on them. Check
the plant mask against real masks.

**The original README dataset list was too narrow.** The six datasets it started with only covered water
stress and disease. Leaf segmentation, physiological data, and time-series data were missing, and are
now in the dataset list. One of the six, the generic "Dataset Aloevera" Roboflow set, serves no goal and
was dropped.

**There is no in-domain data for the time-series work.** Same-plant sequences exist for other species,
which is fine for prototyping, but nothing succulent and nothing with graded leaf-death labels. This is
fine for a later goal, but the camera rig for a controlled dry-down should be set up early so data
starts accumulating while the first version is built.
