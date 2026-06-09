# Model Selection: RT-DETRv2 for BDD100K Object Detection

## Summary

| | |
|---|---|
| **Model** | RT-DETRv2-s (ResNet-18vd backbone), COCO pretrained |
| **Framework** | Plain PyTorch — official `lyuwenyu/RT-DETR` GitHub repo|
| **Checkpoint** | `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` (~78 MB, COCO AP 48.1) |
| **Fine-tuned checkpoint** | `output_/rtdetrv2_bdd100k/best.pth` (1 epoch, 200 images, smoke test) |
| **DataLoader** | `src/model/data_loader.py` — custom `torch.utils.data.Dataset` + weighted sampler |
| **Training script** | `train.py` — explicit PyTorch loop (AdamW, EMA, GradScaler, Hungarian loss) |
| **Notebook** | `notebooks/model_training.ipynb` — end-to-end demo |
| **COCO → BDD transfer** | 8/10 classes map directly; `rider` and `traffic sign` require fine-tuning |

---

## Why RT-DETRv2

The task specifies using PyTorch code instead of Ultralytics. That single constraint is
the deciding factor:

| Candidate | Pure-PyTorch pretrained weights |
|-----------|----------------------------------|
| YOLOv8 | Weights and all pre/post-processing ship inside the `ultralytics` package. Using them in plain PyTorch requires re-implementing the anchor-free head and DFL decode — fragile. |
| YOLOv11 | Same — it is an Ultralytics-native release with no first-party non-Ultralytics path. |
| **RT-DETRv2** | **First-class plain PyTorch** via the official `lyuwenyu/RT-DETR` repo. No Ultralytics at any point in the inference or training path. |

Beyond availability, RT-DETRv2 is architecturally well-suited for BDD100K:

- **NMS-free end-to-end detection** — the decoder directly outputs a fixed set of
  predictions; no IoU-threshold tuning in post-processing.
- **Transformer global context** — decoder attention sees the whole image at once,
  which helps with occluded and overlapping instances in dense urban scenes.
- **Multi-scale by construction** — the hybrid encoder fuses multi-scale backbone
  features, directly relevant to BDD100K's extreme object-size range (large nearby
  cars versus tiny distant traffic lights at the same resolution).

---

## Architecture

RT-DETRv2 follows a four-stage pipeline:

```
Input image (640 x 640 x 3)
[ Backbone: PResNet-18 (ResNet-18vd) ]
  - Standard ResNet stages S1-S4
  - Outputs multi-scale feature maps S3, S4, S5
[ Efficient Hybrid Encoder ]
  - AIFI: Intra-scale attention on the deepest scale (global context)
  - CCFM: Cross-scale CNN feature fusion (top-down + bottom-up)
  - Output: enriched multi-scale memory (same 3 scales)
[ IoU-Aware Query Selection ]
  - Picks top-K encoder positions as initial object queries
  - Seeded queries have better spatial locality than random init
[ Transformer Decoder (deformable cross-attention, 6 layers) ]
  - Each layer refines box coordinates and class logits
  - Denoising training: adds noise to GT boxes as extra queries during training
  { class logits, box (cx, cy, w, h) }  ->  threshold  ->  final detections
```

**Key properties vs. CNN detectors:**
- No anchors — queries are learned; no aspect-ratio hyperparameters to tune.
- No NMS — the bipartite (Hungarian) matcher during training forces unique
  assignments so duplicate predictions don't emerge.
- The backbone and encoder weight from COCO transfer to BDD100K with minimal
  adaptation; only the final class projection layer (80 → 10 classes) is replaced.

---

## Dataset Loader

`src/model/data_loader.py` implements a full PyTorch data-loading stack:

- `BDD100KDetectionDataset` — `torch.utils.data.Dataset` returning
  `(image, target)` pairs in the torchvision detection convention
  (`boxes` in `[x1, y1, x2, y2]` pixel coords, `labels` as int64 class ids).
- `compute_class_weights` — inverse-frequency weights to counter BDD100K's
  extreme class imbalance (5250:1 car-to-train ratio).
- `make_weighted_sampler` — `WeightedRandomSampler` that oversamples images
  containing rare classes so every batch sees at least some minority-class examples.
- `build_dataloader` — one-call factory wiring together dataset, sampler,
  collation, and class weights.

```python
from src.model.data_loader import build_dataloader

loader, class_weights = build_dataloader(
    "data",
    split="train",
    batch_size=4,
    img_size=640,
    max_images=200,            # subset for quick runs
    use_weighted_sampler=True,
)

images, targets = next(iter(loader))
```

Run the demo directly (prints dataset statistics and one batch shape):

```bash
docker run --rm -v $(pwd)/data:/app/data:ro bdd-pipeline dataloader
```

---

## Training Pipeline

`train.py` implements an explicit PyTorch fine-tuning loop. Every step is written
out — no black-box trainer API. The steps are:

1. Load BDD100K into `build_dataloader` with weighted sampling.
2. Build RT-DETRv2-s from its YAML config (`rtdetrv2_r18vd_120e_coco.yml`).
3. Load COCO pretrained weights in "tuning" mode: backbone and encoder weights are
   copied; the 80-class head is discarded and replaced with a fresh 10-class head.
4. AdamW optimizer with two parameter groups:
   - Backbone non-norm weights: `0.1 × base_lr` (preserve COCO features)
   - All other weights: `base_lr` (default 2.5e-4)
   - Norm/BN layers: `weight_decay = 0` in both groups.
5. Linear warmup (100 steps) followed by MultiStepLR.
6. AMP `GradScaler` on CUDA; plain FP32 on CPU.
7. EMA shadow model for stable evaluation weights.
8. Per-batch: forward → `RTDETRCriterionv2` loss (focal cls + L1 box + GIoU,
   Hungarian matching + denoising aux losses) → `backward` → gradient clip
   (`max_norm=0.1`) → `optimizer.step()` → EMA update.
9. Best checkpoint saved to `output_/rtdetrv2_bdd100k/best.pth`.

**Smoke test (1 epoch, CPU, confirmed working):**

```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party:ro \
  -v $(pwd)/weights:/app/weights:ro \
  -v $(pwd)/output:/app/output \
  bdd-pipeline train --device cpu --epochs 1 --batch 2 --train-images 200
```

**Full fine-tuning (GPU recommended):**

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/third_party:/app/third_party:ro \
  -v $(pwd)/weights:/app/weights:ro \
  -v $(pwd)/output:/app/output \
  bdd-pipeline train --device cuda --epochs 30 --batch 16
```

The fine-tuned checkpoint at `output_/rtdetrv2_bdd100k/best.pth` was produced by
the 1-epoch smoke test and demonstrates a complete, functioning training pipeline.
The checkpoint uses EMA weights and is directly usable for inference via the
`compare` command.

---

## COCO to BDD100K Class Transfer

Eight BDD classes map directly from COCO pretrained weights:

| BDD class | COCO source | BDD class | COCO source |
|-----------|-------------|-----------|-------------|
| car | car | person | person |
| truck | truck | bike | bicycle |
| bus | bus | motor | motorcycle |
| train | train | traffic light | traffic light |

Two classes have **no COCO equivalent**:

- **`rider`** — COCO labels cyclists as `person`; no distinct rider class exists.
- **`traffic sign`** — COCO has only `stop sign`, which is too narrow for the BDD
  general traffic sign class.

This connects directly to the data analysis findings: `rider` was flagged as one
of the rarest classes and `traffic sign` as numerous but predominantly tiny. Both
are structurally unreachable in zero-shot transfer and require fine-tuning on BDD100K.
