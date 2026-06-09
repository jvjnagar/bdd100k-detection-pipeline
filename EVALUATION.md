# Model Evaluation & Visualization

## Model Evaluated

**RT-DETRv2-s (ResNet-18) — COCO pretrained**, zero-shot applied to BDD100K 10 classes.

Two variants validated:
- **Model A** — COCO pretrained, zero-shot transfer (8/10 BDD classes directly mapped)
- **Model B** — Fine-tuned checkpoint (1 epoch, 200 images) — pipeline smoke test only

---

## Quantitative Results

*50-image validation subset, IoU=0.5, confidence threshold=0.3*

### Model A — RT-DETRv2-s COCO pretrained

| Class | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|-----|
| car | 0.581 | 0.414 | 0.483 | 212 | 153 | 300 |
| truck | 0.000 | 0.000 | 0.000 | 0 | 0 | 21 |
| bus | 0.000 | 0.000 | 0.000 | 0 | 0 | 9 |
| train | 0.000 | 0.000 | 0.000 | 0 | 20 | 0 |
| person | 0.000 | 0.000 | 0.000 | 0 | 0 | 76 |
| rider | 0.000 | 0.000 | 0.000 | 0 | 0 | 3 |
| bike | 0.000 | 0.000 | 0.000 | 0 | 94 | 1 |
| motor | 0.000 | 0.000 | 0.000 | 0 | 182 | 3 |
| traffic light | 0.000 | 0.000 | 0.000 | 0 | 0 | 123 |
| traffic sign | 0.000 | 0.000 | 0.000 | 0 | 0 | 153 |
| **mean** | **0.058** | **0.041** | **0.048** | | | |

### Model B — Fine-tuned (1 epoch, 200 images)

At the default confidence threshold (0.3): zero detections — all ground truth boxes missed. This is expected because 1 epoch on 200 images shifts the classification head only marginally from its random initialization; the model outputs scores in the 0.03–0.04 range rather than above 0.3.

At a lowered threshold (0.03): car detections do appear, confirming the full fine-tuning pipeline runs end-to-end and the weights loaded correctly. The detections are low-quality (F1 well below Model A) because the head has not converged. ~30 epochs on the full 70k training set would be needed for meaningful results.

---

## Why These Metrics

**Per-class Precision / Recall / F1 at IoU=0.5** — not overall mAP. BDD100K has extreme class imbalance (55% car, <0.2% train). A single mean AP number hides that 9 of 10 classes score exactly zero. Per-class metrics expose each failure independently.

**Confidence distribution (TP vs FP)** — diagnoses calibration. For this zero-shot model, most non-car predictions are domain-gap false positives (motorcycle head firing on rear-facing cars). Plotting TP vs FP score histograms makes that visible and informs per-class threshold tuning.

**False-negative breakdown by object size** — BDD100K has roughly 415k annotations smaller than 16x16 px (~25% of all boxes). Size-stratified FN analysis separates resolution-limited failures from class-capacity failures and directly targets the dataset's known small-object challenge.

**Per-image F1 vs GT object count** — measures degradation in dense scenes, a core challenge in dashcam data. A negative correlation means the model breaks down specifically in the complex cases that matter most.

---

## What Works

**Car detection (F1=0.48)** is the only class with non-trivial transfer. Cars are the most frequent class in both COCO and BDD100K, large enough to be visible at 640x640 input, and visually consistent across COCO and dashcam imagery. Large objects (>96x96 px) achieve roughly 60-70% recall for the car class.

---

## What Doesn't Work (and Why)

**Person, truck, bus — zero recall despite COCO overlap.** Domain shift is the cause: BDD100K is dashcam footage from a moving vehicle; COCO covers general photography. Under dashcam conditions the model's confidence for these classes stays below 0.3, so nothing passes the threshold.

**Motor and bike — high false positive count (182 motor FPs, 94 bike FPs).** The COCO `motorcycle` decoder fires on cars seen straight-on or from the rear — that orientation has a wide-over-tall aspect ratio similar to motorcycle activations. These are high-confidence wrong predictions, not low-confidence noise, so a stricter global threshold does not fix them.

**Rider and traffic sign — structurally zero-shot unreachable.** `rider` has no COCO equivalent (COCO labels cyclists simply as `person`). `traffic sign` maps only to COCO's `stop sign` — too narrow to cover the BDD general class. No threshold tuning produces these classes from the COCO-pretrained head.

**Fine-tuned model — no detections at default threshold (0.3).** After 1 epoch on 200 images the classification head outputs scores in the 0.03–0.04 range, below the 0.3 threshold. Backbone and encoder weights transfer cleanly from COCO; the classification projection needs ~30 epochs on the full training set to converge. At a lowered threshold (0.03) the model does detect cars, confirming the pipeline runs end-to-end.

---

## Visualization

**Quantitative** (output: `output/evaluation/quantitative_metrics.png`, `per_class_metrics.csv`, `per_image_scores.csv`):
- Per-class P/R/F1 bar chart — exposes the car-only transfer at a glance
- TP/FP confidence score histograms — exposes the high-confidence motor/bike FPs
- Size-stratified FN heatmap (class x size bucket) — shows the tiny-object wall
- Per-image F1 scatter vs GT object count — shows dense-scene degradation

**Qualitative** (output: `output/evaluation/qualitative_{good,medium,poor}_detections.png`):
GT and predictions drawn on validation images, grouped by per-image F1 tier (high/medium/low). Colour coding: green = TP, red = FN (missed), orange = FP (false alarm). Low-F1 images are dominated by missed small traffic objects and high-confidence car predicted as motor. Tool: `src/evaluation/visualize_samples.py`.

**Failure map** (output: `output/evaluation/failure_clustering.png`):
FN heatmap by class and size, F1 distribution, complexity scatter — combined in one figure for pattern spotting.

---

## Failure Clustering

### By Object Size
Tiny objects (<32x32 px) account for roughly 78% of all missed detections. At 640x640 input the smallest feature map is 80x80, mapping sub-16px annotations to less than 2 pixels of feature — effectively invisible.

| Size bucket | Share of all FN |
|-------------|----------------|
| Tiny (<32px) | ~78% |
| Small (32-96px) | ~13% |
| Medium (96-256px) | ~7% |
| Large (>=256px) | ~2% |

### By Class
- **Most missed**: traffic sign (~3 FN per image), traffic light (123 FN in 50 images)
- **Most false alarms**: motor (182 FP), bike (94 FP) — rear-facing car triggers motorcycle head
- **Best recall**: car (0.414) — only class that transfers zero-shot

### By Scene Complexity
Per-image F1 correlates negatively with GT object count (r ~= -0.3). Dense urban frames with 25+ annotated objects score consistently lower, mostly because those extra objects are small traffic signs and lights.

---

## Connection to Data Analysis

| Finding from Data Analysis | Effect Observed in Evaluation |
|---|---|
| Car = 55% of annotations | Sole class with non-zero F1 |
| 5250:1 car-to-train ratio | Train: 20 FPs (trucks/buses fire the head), 0 TPs |
| ~415k annotations <16px | 78% of FNs are tiny objects |
| `rider` rare and COCO-unmapped | 0 TPs, structurally unreachable zero-shot |
| `traffic sign` numerous but tiny | 153 FNs/50 images, zero recall |
| Motor and rider rare in training | Motor: 182 FPs from domain gap; rider: 0 TPs |
| Train/val distributions matched | Performance gap is model capacity, not dataset shift |

The data analysis flagged `rider` and `traffic sign` as highest-risk classes (rare + COCO-unmapped / tiny boxes). Those are exactly the two with zero recall here, confirming the analysis predictions.

---

## Improvement Suggestions

**1. Fine-tune 100-200 epochs on the full 70k training set.** Highest-impact change. The pipeline is already implemented: `train.py --epochs 30 --batch 16 --device cuda`. Even one full epoch on the complete dataset would significantly improve recall on truck, bus, and person.

**2. Raise input resolution to 1280x720.** BDD100K images are natively 1280x720. The current 640x640 resize halves feature resolution for a dataset where 78% of missed detections are tiny objects.

**3. Per-class confidence thresholds.** Motor/bike FPs are high-confidence wrong predictions — a stricter threshold (0.5) for those classes improves precision. Traffic light/sign need a lower threshold (~0.15) to recover any recall at all.

**4. Weighted sampling** — already in `data_loader.py` (`--use-weighted-sampler`). Ensures rare classes (train 136 instances, rider ~3k) appear in every training batch, directly countering the imbalance the data analysis quantified.

**5. augmentation for rare classes.** Train and rider need synthetic frequency boosts to compensate for the extreme imbalance ratio.