# BDD100K Object Detection — Data Analysis, Model & Evaluation

## Overview

End-to-end object detection pipeline on the [BDD100K dataset](https://www.bdd100k.com/), covering the **10 detection classes** with bounding boxes:

| # | Class | # | Class |
|---|-------|---|-------|
| 1 | car | 6 | rider |
| 2 | truck | 7 | bike |
| 3 | bus | 8 | motor |
| 4 | train | 9 | traffic light |
| 5 | person | 10 | traffic sign |


## Quick Start

### Prerequisites

- **Docker**
- **Optional GPU:** NVIDIA GPU + [NVIDIA Container Toolkit]
- **BDD100K dataset:** [Download here](https://www.bdd100k.com/) — 100K Images (~5.3 GB) + Labels (~107 MB)

### 1. Fetch RT-DETRv2 source and pretrained weights

`third_party/` and `weights/` are **not committed to Git**. The `detect`, `train`, and `evaluate` stages **fetch them automatically on first run** if they are missing — no manual step required.

> **Note:** mount `third_party` and `weights` as writable volumes (no `:ro`) so the container can populate them on first run. Pre-create the directories on the host to avoid Docker creating them as root:
> ```bash
> mkdir -p third_party weights
> ```

To pre-fetch before running (e.g. for offline deployment), or to choose a different model variant:

```bash
bash scripts/setup_rtdetrv2.sh              # default: rtdetrv2-s (fastest)
bash scripts/setup_rtdetrv2.sh rtdetrv2-l   # larger model, higher accuracy
```

The script sparse-clones [`lyuwenyu/RT-DETR`](https://github.com/lyuwenyu/RT-DETR) into `third_party/` and downloads the pretrained checkpoint (~75 MB) into `weights/`.

**Persistence:** mount both as host volumes (shown in the `docker run` commands below). The container writes the files to your host on first run — subsequent runs skip the download entirely.

> Available variants: `rtdetrv2-s` · `rtdetrv2-m-r34` · `rtdetrv2-m` · `rtdetrv2-l` · `rtdetrv2-x`

### 2. Data layout

```
data/
└── bdd100k/
    ├── images/100k/{train,val}/      # .jpg images
    └── annotations/
        ├── bdd100k_train_coco.json
        └── bdd100k_val_coco.json
```

### 3. Build the Docker image

```bash
docker build -t bdd-pipeline .
```

Builds with no `apt` packages (`opencv-python-headless` is self-contained), so it works behind restrictive corporate proxies.

---

## Running Pipeline Stages

A single image runs every stage. The first argument selects the stage:

| Arg | Task | What runs |
|-----|------|-----------|
| `analyze` *(default)* | 1 | Full data analysis pipeline |
| `dashboard` | 1 | Interactive Dash app (port 8050) |
| `visualize` | 1 | Annotated interesting-sample grids |
| `dataloader` | 2 | PyTorch DataLoader demo + class weights |
| `detect` | 2 | RT-DETRv2 single-image inference |
| `train` | 2 | Fine-tune RT-DETRv2 on BDD100K |
| `evaluate` | 3 | Metrics + qualitative failure analysis |
| `compare` | 3 | Multi-model comparison (RT-DETRv2 + YOLOv8) |
| `bash` | — | Interactive shell |

---

## Task 1 — Data Analysis

### Full analysis pipeline

```bash
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  bdd-pipeline analyze
```

Parses train (70k) and val (10k) annotations, computes class distributions, split comparisons, bounding-box statistics, and anomaly detection. All outputs saved to `output/`.

### Interactive dashboard

```bash
docker run --rm -p 8050:8050 \
  -v /path-to-bdd-data/data:/app/data:ro \
  bdd-pipeline dashboard
```

Open **http://localhost:8050** in your browser. The dashboard covers:
1. **Class Distribution** — instances per class (train & val)
2. **Train / Val Split Analysis** — distribution shift per class
3. **Anomalies & Patterns** — imbalance ratios, tiny-box breakdown, per-class deep-dive

### Interesting-sample visualization

```bash
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  bdd-pipeline visualize
```

---

## Task 2 — Model & Training

**Chosen model: RT-DETRv2** (official [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), run in pure PyTorch with no Ultralytics dependency). See [MODEL_SELECTION.md](MODEL_SELECTION.md) for the full reasoning.

### COCO → BDD class mapping (zero-shot transfer)

The pretrained weights are COCO-trained; **8 of BDD's 10 classes map directly**:

| BDD | ← COCO | BDD | ← COCO |
|-----|--------|-----|--------|
| car | car | person | person |
| truck | truck | bike | bicycle |
| bus | bus | motor | motorcycle |
| train | train | traffic light | traffic light |

`rider` and `traffic sign` have no COCO equivalent — they require fine-tuning.

### Single-image inference

```bash
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/output:/app/output \
  bdd-pipeline detect --image /app/data/bdd100k/images/100k/val/<id>.jpg --bdd-only
```

### Fine-tune on BDD100K

```bash
# CPU smoke test (1 epoch, 200 images):
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/output:/app/output \
  bdd-pipeline train --device cpu --epochs 1 --batch 2 --train-images 200

# GPU full run:
docker run --rm --gpus all \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/output:/app/output \
  bdd-pipeline train --device cuda --epochs 30 --batch 16
```

Checkpoints are saved to `output/rtdetrv2_bdd100k/`.

### PyTorch DataLoader demo

```bash
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  bdd-pipeline dataloader --split val --max-images 200 --weighted-sampler
```

`src/model/data_loader.py` provides a `BDD100KDetectionDataset` with class-weighted sampling to counter the extreme imbalance found in the analysis (≈5250:1 car-to-train ratio).

---

## Task 3 — Evaluation & Visualization

See [EVALUATION.md](EVALUATION.md) for the full analysis and improvement suggestions.

### Baseline evaluation (pretrained RT-DETRv2)

```bash
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/output:/app/output \
  bdd-pipeline evaluate
```

### Compare with a fine-tuned checkpoint

```bash
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/output:/app/output \
  bdd-pipeline evaluate \
    --compare-with rtdetrv2-bdd:output/rtdetrv2_bdd100k/best.pth
```

Repeat `--compare-with` to compare multiple models in a single run. A `model_comparison.png` side-by-side chart is generated automatically whenever any comparison model is present.

**Other flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--score-thresh` | `0.3` | Confidence threshold for detections |
| `--images` | `500` | Max validation images to evaluate |
| `--compare-with` | — | Backend:weights spec (repeatable) |

Outputs written to `output/evaluation/`:

| File | Description |
|------|-------------|
| `quantitative_metrics.png` | Precision / recall / F1 per class |
| `per_class_metrics.csv` | Raw per-class numbers |
| `per_image_scores.csv` | Per-image TP / FP / FN |
| `qualitative_*.png` | GT vs predictions on good / medium / poor images |
| `failure_clustering.png` | Missed objects by class × size |
| `evaluation_report.json` | Full structured report |
| `model_comparison.png` | Side-by-side F1/P/R chart (multi-model, if `--compare-with` used) |
| `model_comparison.csv` | Tabular comparison across all evaluated models |

### Multi-model comparison (`compare`)

Dedicated command that runs **RT-DETRv2 COCO pretrained**, **RT-DETRv2 fine-tuned**, and **YOLOv8s** side-by-side, producing a qualitative grid (same images, all predictions overlaid) and grouped quantitative bars.  
YOLOv8s weights are auto-downloaded by `ultralytics` on first run (~22 MB, stored in `/tmp/Ultralytics`).

```bash
docker run --rm \
  -v /path-to-bdd-data/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/output:/app/output \
  bdd-pipeline compare \
    --finetuned-weights output_/rtdetrv2_bdd100k/best.pth \
    --yolo-weights yolov8s.pt \
    --images 200
```

Pass `--yolo-weights none` to skip YOLO, or `--finetuned-weights none` to skip the fine-tuned model.  
Use a local BDD100K-trained YOLO checkpoint (e.g. from `ultralytics train data=yolo_dataset/bdd100k.yaml model=yolov8s.pt`) for a fair domain-matched comparison.

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | `200` | Validation images per model |
| `--score-thresh` | `0.3` | Confidence threshold |
| `--finetuned-weights` | `output_/rtdetrv2_bdd100k/best.pth` | RT-DETRv2 BDD checkpoint |
| `--yolo-weights` | `yolov8s.pt` | YOLO checkpoint (auto-downloaded) |
| `--qual-images` | `6` | Images in the qualitative grid |

Outputs written to `output/comparison/`:

| File | Description |
|------|-------------|
| `quantitative_comparison.png` | Grouped F1 / P / R bars for all models per class |
| `qualitative_comparison.png` | Side-by-side grid: GT + each model's predictions on the same images |
| `summary.csv` | Mean F1 / P / R and per-class F1 for every model |
| `comparison_report.json` | Full per-class metrics for all models |
