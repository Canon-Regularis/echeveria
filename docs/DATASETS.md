# Datasets

This is the list of datasets we checked for echeveria: what each one contains, what it is good for, and
its license. Everything was checked against its live page on 2026-07-12 (that it exists, is accessible,
its type, and its license). Where a fact could not be read from the page, it says so. Counts, class
lists, and licenses change over time, so re-check before you train on anything.

## Legend

**Type:** `classification` (one folder per class), `detection` (bounding boxes), `semantic-seg` (a
foreground mask), `instance-seg` (a mask per object), `time-series`, `tabular` (measurements), or
`mixed`.

**Good for:** which goals it can help with (`seg-plant`, `seg-leaf`, `pheno`, `water-stress`, `disease`,
`temporal`). These labels are defined in [OBJECTIVES.md](OBJECTIVES.md#objective-labels-used-in-the-dataset-list).

**Domain:** how close it is to the target. `succulent` is in-domain. `other` is a different potted or
rosette plant. `far` is a field crop or a lab leaf, which is a big jump from a succulent.

## The list at a glance

| # | Dataset | Type | Good for | License | Domain |
| --- | --- | --- | --- | --- | --- |
| **core water stress and in-domain succulent** | | | | | |
| 1 | [Healthy and Wilted Houseplant Images](https://www.kaggle.com/datasets/russellchan/healthy-and-wilted-houseplant-images) | classification | water-stress | unverified | other |
| 2 | [Houseplant Dataset (wilted/unhealthy)](https://universe.roboflow.com/seojin-jang/houseplant-dataset) | detection | water-stress | CC BY 4.0 | other |
| 3 | [Aloe Vera Health Detection wwmar v22](https://universe.roboflow.com/aloe-vera-health-detection/aloe-vera-health-detection-wwmar/dataset/22) | detection | water-stress, disease | CC BY 4.0 | succulent |
| 4 | [Aloevera Health Detection Y-V11 v8](https://universe.roboflow.com/aloe-vera-health-detection/aloevera-health-detection-y-v11/dataset/8) | detection | water-stress, disease | CC BY 4.0 | succulent |
| 5 | [Cactus species (incl. Echeveria)](https://universe.roboflow.com/chayada-im3ic/cactus-species) | detection | seg-plant\* | CC BY 4.0 | succulent |
| **plant and leaf segmentation** | | | | | |
| 6 | [CVPPP LSC/LCC (A1-A5)](https://www.plant-phenotyping.org/CVPPP2017-challenge) | instance-seg | seg-leaf, seg-plant, pheno | academic only, gated | far |
| 7 | [KOMATSUNA](https://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) | instance-seg, RGB-D | seg-leaf, temporal | unstated (cite paper) | far |
| 8 | [MSU-PID](http://cvlab.cse.msu.edu/project-pid-database.html) | instance-seg, multi-modal | seg-leaf, temporal, pheno | none stated (cite paper) | far |
| 9 | [GrowliFlower](http://rs.ipb.uni-bonn.de/data/growliflower/) | instance-seg | seg-leaf, seg-plant, pheno, temporal | public (cite paper) | far |
| 10 | [UPGen (synthetic)](https://github.com/csiro-robotics/UPGen) | instance-seg | seg-leaf | CSIRO non-commercial | synthetic |
| 11 | [Roboflow leaf-seg sets (e.g. giovi)](https://universe.roboflow.com/giovi/leaf-segmentation-uxlob) | instance-seg | seg-leaf | CC BY 4.0 (varies) | other |
| 12 | [Cactus type (instance-seg)](https://universe.roboflow.com/student-1-tgqaq/cactus-type/dataset/2) | instance-seg | seg-plant, seg-leaf? | unverified | succulent (tiny) |
| 13 | [Aberystwyth Leaf Evaluation (ALED)](https://zenodo.org/records/168158) | instance-seg, timelapse | seg-leaf, seg-plant, pheno, temporal | CC BY 4.0 | far |
| **physiological ground truth** | | | | | |
| 14 | [Wheat chlorophyll-fluorescence water stress](https://data.mendeley.com/datasets/2mpd7d3vry/2) | classification | water-stress, pheno | CC BY 4.0 | far |
| 15 | [Lettuce thermal, RGB, soil moisture](https://data.mendeley.com/datasets/294zk6k5wf/2) | mixed (thermal, CSV) | pheno, water-stress, temporal, seg-plant | CC BY 4.0 | other |
| 16 | [Avocado/olive/grape hyperspectral dehydration](https://doi.org/10.6084/m9.figshare.26950660) | mixed (hyperspectral, tabular) | pheno, water-stress | CC BY 4.0 | far |
| **future time-series and leaf-death prediction** | | | | | |
| 17 | [UNL Cotton multimodal drought time-series](https://pmc.ncbi.nlm.nih.gov/articles/PMC9947149/) | time-series | temporal, water-stress, pheno | on request only | far |
| | *KOMATSUNA (7), MSU-PID (8), GrowliFlower (9), ALED (13), and Lettuce (15) are also same-plant time series.* | | | | |
| **disease detection** | | | | | |
| 18 | [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | classification | disease, seg-plant\* | ambiguous (CC BY 3.0 / CC0 claimed) | far |
| 19 | [Aloe Vera Diseases (bg-removed)](https://data.mendeley.com/datasets/cksmdjw8gy/2) | classification | disease | CC BY 4.0 | succulent |
| 20 | [Aloe Vera Leaf Disease Detection](https://data.mendeley.com/datasets/7w6t4zx33n/1) | classification | disease, water-stress\* | CC BY 4.0 | succulent |
| 21 | [Plant Diagnosis AI 6 (Aloe Vera)](https://universe.roboflow.com/dynamite-duelers-2/plant-diagnosis-ai-6--aloe-vera) | detection | disease | CC BY 4.0 | succulent |
| **Dropped** | | | | | |
| 22 | [Dataset Aloevera](https://universe.roboflow.com/dataset-aloe-vera/dataset-aloevera) | detection | none | CC BY 4.0 | succulent |

\* A weak or derived fit. See the notes below.

## core water stress and in-domain succulent

### 1. Healthy and Wilted Houseplant Images (Kaggle, russellchan)

About 904 images, split into healthy and wilted folders. No boxes and no masks. Mixed indoor species
scraped from web searches, so backgrounds vary and a classifier should generalize. Caveats: the license
could not be read (the page needs JavaScript), and the images were scraped, so redistribution rights are
unclear. The wilted label is a single yes/no with no severity. Fine as a first transfer-learning
baseline, not for anything you plan to redistribute.

### 2. Houseplant Dataset, wilted and unhealthy (Roboflow, seojin-jang)

About 849 images, detection, roughly 3 classes centred on healthy versus unhealthy or wilted leaves.
CC BY 4.0. The boxes locate the plant, which helps a plant-vs-background detector. Community set of
unknown label quality. Boxes only, so it is not segmentation data.

### 3. Aloe Vera Health Detection wwmar v22 (Roboflow), succulent

450 images, detection, CC BY 4.0. Classes: Healthy, Unhealthy, Environmental-stress, Disease-infected,
and leaf-demage (a misspelling of leaf-damage). The class list is auto-generated and messy, with a stray
"SOWj" token. Aloe is a succulent, so this is real in-domain signal for the main task. The
Environmental-stress class is general abiotic stress, not specifically water. Small.

### 4. Aloevera Health Detection Y-V11 v8 (Roboflow), succulent

About 3,169 images including augmentation (about 1,864 source), detection, CC BY 4.0. Four classes:
Healthy, Disease Infected, Environmental stress, Leaf Damage, with a train/valid/test split. The largest
in-domain aloe set. Reported detection quality is low (about 36% mAP@50), which points to noisy or thin
annotations. Stress and disease share the same label space, so watch for the two tasks bleeding
together.

### 5. Cactus species, including Echeveria (Roboflow, chayada), succulent

600 images, detection, CC BY 4.0. Ten succulent and cactus species, including Echeveria runyonii, plus
Haworthia, Gymnocalycium, Mammillaria, and others. This is the only dataset that actually contains the
project's namesake genus. Every plant is healthy (it labels species, not stress), so use it to pretrain
a succulent plant locator or foreground model, not for stress labels. Boxes only, so it supports plant
segmentation only loosely.

## plant and leaf segmentation

Leaf segmentation is out of the first version (see the decision in
[OBJECTIVES.md](OBJECTIVES.md#risks-and-open-decisions)). These sets are for the leaf module later. The
first version's need to separate plant from background is met instead by the background-removed pairs
(#19) and the box sets (#2, #5).

When you resume leaf segmentation, note that every research-grade leaf set here is flat-leaved
(Arabidopsis, tobacco, cauliflower, komatsuna). A succulent is a tight rosette of thick, curved,
overlapping leaves, which is a large jump. Use these to build and pretrain the method, but do not expect
them to produce leaf masks that work on succulents without a small in-domain set to fine-tune on.

### 6. CVPPP Leaf Segmentation Challenge, A1-A5 (plant-phenotyping.org)

The standard benchmark for per-leaf masks. PNG masks with one integer per leaf, plus leaf-count ground
truth. Arabidopsis and tobacco, top-down. Blocker: the data is for academic and challenge use only,
needs registration, and forbids commercial use. Good for method development, not usable as-is in a
product. Per-subset counts (about 810 train and 275 test) come from papers, not the live page.

### 7. KOMATSUNA (Kyushu University LIMU)

1,560 images (1,080 multi-view and 480 RGB-D) with 6,184 per-leaf instances. Real per-leaf masks, depth,
and sequential growth stages, so it also helps tracking. Instant download, no registration; the license
is not stated, so cite the 2017 ICCV workshop paper. The best practical starting point for a first
leaf-segmentation baseline: real masks, depth to help separate overlapping leaves, and free.

### 8. MSU-PID (Michigan State CVLab)

Arabidopsis (16 plants imaged hourly for about 9 days) and bean, in four modalities (fluorescence,
infrared, RGB, depth). Per-leaf masks plus leaf-tip points and cross-time tracking labels. About 318 MB,
no gating; the page has an SSL certificate quirk that may warn a browser, but the download itself is
fine. No formal license, citation requested. One of the few sets with both masks and leaf tracking, so
it also serves the time-series work.

### 9. GrowliFlower (University of Bonn, IPB)

Cauliflower filmed from a drone over two seasons. Both per-leaf and whole-plant masks, plus stem
annotations, with measured traits for 740 plants, in RGB and multispectral. Public download, cite the
paper. It covers leaf and plant segmentation, physiological traits, and time series, but field-scale
drone imagery looks very different from a close-up potted plant.

### 10. UPGen, synthetic (CSIRO Robotics)

Domain-randomized synthetic leaf masks: unlimited and perfectly labelled, and shown to transfer (about
88% SBD on CVPPP A1-A4). License: CSIRO non-commercial. The synthetic dataset itself is listed as
"available soon"; a pretrained Mask R-CNN and the generator code are available now. Use it to pretrain
or augment leaf segmentation when real succulent masks are scarce. Not shippable in a commercial product
under this license.

### 11. Roboflow leaf-segmentation sets (for example giovi/leaf-segmentation-uxlob)

Instant COCO or YOLOv8-seg polygon exports. The giovi set is about 345 images, 2 classes, CC BY 4.0.
Others (about 322 and about 2,390) were not individually re-checked. The fastest way to a runnable
leaf-segmentation baseline, but crowd-sourced, inconsistent, and rarely succulent. Check each one in the
browser first.

### 12. Cactus type, instance segmentation (Roboflow, student-1), succulent

The only succulent instance-segmentation set found, but only about 41 images, low confidence, and the
masks probably mark whole cactus bodies rather than individual leaves. Proof-of-concept or fine-tuning
only.

### 13. Aberystwyth Leaf Evaluation Dataset, ALED (Zenodo)

Arabidopsis rosettes filmed every 15 minutes from about day 21 to day 55. 56 annotated ground-truth
images covering 916 hand-marked plants, with per-leaf regions and destructive-harvest measurements.
CC BY 4.0, about 63.7 GB. The rosette shape is structurally closer to a succulent than a flat crop, and
the timelapse plus senescence make it useful for the time-series work too. Large download; the exact
mask format is only confirmed after downloading.

## physiological ground truth, to calibrate and check features

These pair images with measured water status, so computed features (greenness, area, thermal, texture)
can be checked against real values instead of trusted blindly. None is a succulent. Succulents store
water internally and behave differently in thermal and spectral terms, so use these as guidance, not as
ground truth for Echeveria. Anchor the main task with your own succulent measurements (see the risks in
[OBJECTIVES.md](OBJECTIVES.md#risks-and-open-decisions)).

### 14. Wheat chlorophyll-fluorescence water stress (Mendeley, 2mpd7d3vry)

2,880 chlorophyll-fluorescence images (24 per day over a 60-day dry-down), split into Control and
Drought, CC BY 4.0. Fluorescence is a direct signal of photosynthetic stress. Specialized capture, not
ordinary RGB.

### 15. Lettuce thermal, RGB, and soil moisture (Mendeley, 294zk6k5wf)

Two potted lettuces (one watered, one stressed), with thermal-infrared and RGB images plus hourly
soil-moisture readings in CSV, twice a day for 6 days, CC BY 4.0. It genuinely pairs images with a
measured water-status signal and is a time series. Only 2 plants, so little variety, but it is the
closest thing to an image-and-measurement pairing in a pot setting.

### 16. Avocado, olive, and grape hyperspectral leaf dehydration (figshare, 26950660)

Cut leaves imaged across 5 drying stages, hyperspectral 350 to 2500 nm plus multispectral, paired with
per-leaf weight, chlorophyll, and nitrogen tables. The license is CC BY 4.0 (the paper says BY-NC-ND,
but the figshare item v2 is BY 4.0). About 1.18 GB in a single RAR. Good for comparing water-sensitive
spectral features against plain RGB features.

## future time-series and leaf-death prediction

The time-series work needs sequences of the same plant. The list already has several: KOMATSUNA (growth
stages with tracking labels), MSU-PID (hourly, with tracking labels), ALED (15-minute timelapse with
senescence and harvest data), GrowliFlower (seasonal), and Lettuce (a 6-day dry-down). What is missing is
any succulent sequence and any graded leaf-death labels in-domain.

### 17. UNL Cotton multimodal drought-stress time-series (paper PMC9947149)

Daily sequences of the same cotton plants going from healthy to drought (control versus stressed), in
visible, infrared, and hyperspectral, with trait time-series, pixel-level stress labels, and
drought-onset timing. The best match for the stress-progression and leaf-death intent. Not a public
download; the raw data is available from the authors on request, with no license, so treat it as a
request-based option with low confidence. Leaves are not tracked individually, so it does not help leaf
segmentation.

## disease detection

Disease is well covered and stays a secondary goal.

### 18. PlantVillage (GitHub, spMohanty)

About 54,306 images, 38 crop-disease classes across 14 species, one folder per class, with a
background-removed variant. The license is unclear (no LICENSE file; third parties cite CC BY 3.0 or
CC0). Known to have a lab-background bias that hurts real-world transfer. Use for extra pretraining only.

### 19. Aloe Vera Diseases, classification (Mendeley, cksmdjw8gy), succulent

2,307 originals, plus 2,450 background-removed PNGs and 9,000 augmented, in classes Fresh, Rot, and Rust,
CC BY 4.0. The background-removed and original pairs give near-free foreground supervision (not true
masks). Rot looks like degradation or senescence.

### 20. Aloe Vera Leaf Disease Detection (Mendeley, 7w6t4zx33n), succulent

2,500 images, 5 classes: Aloe Rust, Anthracnose, Leaf Spot, SunBurn, and Healthy, CC BY 4.0. SunBurn is
a heat and light stress, useful for telling stress apart from disease, but it is not a controlled
water-deficit label.

### 21. Plant Diagnosis AI 6, Aloe Vera (Roboflow), succulent

About 139 images, one class (browning), detection, CC BY 4.0. Small, and browning mixes disease,
sunburn, and over-watering. A minor extra signal.

## Dropped

### 22. Dataset Aloevera (Roboflow, dataset-aloe-vera)

264 images, one generic Plants bounding-box class, with no health, stress, or disease labels. It serves
none of the goals, so it was removed from the README list.

## Licensing

| Status | Datasets |
| --- | --- |
| Permissive (CC BY 4.0) | 2, 3, 4, 5, 11 (varies), 13, 14, 15, 16, 19, 20, 21, 22 |
| Unclear, check first | 1 (unreadable), 18 (no LICENSE file) |
| Cite only, no formal license | 7, 8, 9 |
| Restricted (academic, non-commercial, or gated) | 6 (academic only and gated), 10 (CSIRO non-commercial), 17 (on request) |

If echeveria ever ships as a product, keep the restricted datasets out of the training set for any
shipped model, or get written permission.

## Physiology references

Not datasets, but the literature the feature and staging design rests on:

- "Responses of Succulents to Drought: Comparative Analysis of Four Sedum Species" (Scientia
  Horticulturae, 2019). Documents the progressive drought response (pigment change and oxidative stress
  first, then tissue-water loss and turgor change, then collapse; pigment changes precede collapse;
  tolerant species differ). It grounds the colour, texture, and deformation features, the rule-based
  drought-stage head, and the pigment early warning. See MODEL_CARD.md, Physiological basis.
- "Multi-Sensor and Multi-temporal High-Throughput Phenotyping for Water Stress" (arXiv 2402.18751,
  2024). A crop study whose RGB -> multispectral -> time series -> ML -> stress-prediction methodology
  transfers to succulents. It motivates the trajectory framing, the `phenotype` command, and the
  forecast. echeveria uses the RGB and time-series parts; the multispectral fusion is out of scope.

No in-domain succulent time-series dataset is catalogued above, so the drought-stage rules and the
forecast are literature-motivated, not fitted to labelled staged data. A controlled dry-down capture
(see the time-series risk in OBJECTIVES.md) would be the way to calibrate and validate them.
