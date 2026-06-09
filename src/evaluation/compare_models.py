from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.parser import load_split
from src.model.class_map import ID_TO_CLASS
from src.model.detector import (
    Detection,
    build_detector,
    correct_domain_gap_labels,
    filter_to_bdd,
)

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "bdd100k" / "images" / "100k" / "val"
OUTPUT_DIR = Path("output") / "comparison"

BDD_CLASSES = list(ID_TO_CLASS.values())

# Per-class colours used consistently across all figures
_COLORS: Dict[str, Tuple[float, float, float]] = {
    "car": (0.0, 0.447, 0.741),
    "truck": (0.851, 0.325, 0.098),
    "bus": (0.929, 0.694, 0.125),
    "train": (0.494, 0.184, 0.557),
    "person": (0.467, 0.675, 0.188),
    "rider": (0.302, 0.745, 0.933),
    "bike": (0.635, 0.078, 0.184),
    "motor": (1.0, 0.6, 0.0),
    "traffic light": (0.298, 0.741, 0.929),
    "traffic sign": (1.0, 0.388, 0.278),
}

def _iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _match(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_confs, iou_thr=0.5):
    matched = set()
    tp, fp, fn = [], [], []
    order = np.argsort(-np.array(pred_confs)) if pred_confs else []
    for i in order:
        pb, pc, cf = pred_boxes[i], pred_classes[i], pred_confs[i]
        best_iou, best_gi = 0.0, -1
        for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if gi in matched or gc != pc:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thr and best_gi >= 0:
            tp.append({"class": pc, "conf": cf})
            matched.add(best_gi)
        else:
            fp.append({"class": pc, "conf": cf})
    for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
        if gi not in matched:
            fn.append({"class": gc})
    return tp, fp, fn


def _per_class_metrics(all_tp, all_fp, all_fn) -> Dict[str, Dict]:
    metrics = {}
    for cls in BDD_CLASSES:
        tp = sum(1 for x in all_tp if x["class"] == cls)
        fp = sum(1 for x in all_fp if x["class"] == cls)
        fn = sum(1 for x in all_fn if x["class"] == cls)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        metrics[cls] = {"precision": p, "recall": r, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}
    return metrics

def _infer(
    detector,
    val_images: List[Path],
    apply_domain_correction: bool,
    batch_size: int,
) -> Tuple[Dict, Dict[str, Dict]]:
    """Run batch inference and return (dets_map, per_class_metrics)."""
    raw_map = detector.batch_predict_paths(
        [str(p) for p in val_images],
        batch_size=batch_size,
        device="auto",
    )
    ann_dict = {p.stem: _ANN_DICT[p.stem] for p in val_images if p.stem in _ANN_DICT}

    all_tp, all_fp, all_fn = [], [], []
    for img_path in val_images:
        ann = ann_dict.get(img_path.stem)
        if ann is None:
            continue
        raw = raw_map.get(str(img_path), [])
        if apply_domain_correction:
            raw = correct_domain_gap_labels(raw)
        dets = filter_to_bdd(raw)
        raw_map[str(img_path)] = dets  # replace with BDD-filtered list in-place

        tp, fp, fn = _match(
            [[b.x1, b.y1, b.x2, b.y2] for b in ann.bboxes],
            [b.category for b in ann.bboxes],
            [[d.x1, d.y1, d.x2, d.y2] for d in dets],
            [d.bdd_label for d in dets],
            [d.score for d in dets],
        )
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)

    return raw_map, _per_class_metrics(all_tp, all_fp, all_fn)


# Module-level annotation dict set by main() so helpers can access it.
_ANN_DICT: Dict = {}

def _plot_quantitative(model_metrics: Dict[str, Dict], out_dir: Path) -> None:
    n = len(model_metrics)
    x = np.arange(len(BDD_CLASSES))
    w = 0.75 / n
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "Multi-Model Comparison on BDD100K Validation (IoU=0.5, conf≥0.3)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    metric_labels = [("f1", "F1 Score"), ("precision", "Precision"), ("recall", "Recall")]
    for ax, (key, ylabel) in zip(axes, metric_labels):
        for i, (lbl, metrics) in enumerate(model_metrics.items()):
            vals = [metrics[c][key] for c in BDD_CLASSES]
            offset = (i - (n - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, label=lbl,
                          color=cmap(i), alpha=0.85, edgecolor="white", linewidth=0.5)
            # Annotate non-zero bars with the value
            for bar, v in zip(bars, vals):
                if v > 0.02:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6,
                    )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(BDD_CLASSES, rotation=40, ha="right", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "quantitative_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")

def _draw_boxes(
    ax,
    img: np.ndarray,
    dets: List[Detection],
    gt_boxes: Optional[List] = None,
    gt_classes: Optional[List] = None,
    title: str = "",
    show_gt: bool = False,
) -> None:
    """Render an image with bounding boxes onto a matplotlib axis."""
    ax.imshow(img)
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")

    if show_gt and gt_boxes:
        for box, cls in zip(gt_boxes, gt_classes):
            x1, y1, x2, y2 = (int(v) for v in box)
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)

    for det in dets:
        color = _COLORS.get(det.bdd_label, (0.5, 0.5, 0.5))
        rect = plt.Rectangle(
            (det.x1, det.y1), det.x2 - det.x1, det.y2 - det.y1,
            linewidth=1.8, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            det.x1, max(det.y1 - 3, 0),
            f"{det.bdd_label} {det.score:.2f}",
            fontsize=5.5, color="white",
            bbox=dict(facecolor=color, alpha=0.75, pad=1, linewidth=0),
        )


def _pick_representative_images(
    val_images: List[Path], ann_dict: Dict, n: int = 6
) -> List[Path]:
    """Select images with varied class coverage for a more informative grid."""
    scored = []
    for p in val_images:
        ann = ann_dict.get(p.stem)
        if ann is None:
            continue
        classes = {b.category for b in ann.bboxes}
        n_boxes = len(ann.bboxes)
        # Score: prefer images with multiple classes and a reasonable box count
        score = len(classes) * 3 + min(n_boxes, 20)
        scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:n]]


def _plot_qualitative(
    model_dets: Dict[str, Dict],
    sample_images: List[Path],
    ann_dict: Dict,
    out_dir: Path,
) -> None:
    """One figure: rows = images, cols = [GT only] + [each model]."""
    model_labels = list(model_dets.keys())
    n_cols = 1 + len(model_labels)   # GT column + one per model
    n_rows = len(sample_images)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.0 * n_rows),
    )
    # Ensure axes is always 2-D even for 1 row
    if n_rows == 1:
        axes = [axes]

    col_headers = ["Ground Truth"] + model_labels
    for j, header in enumerate(col_headers):
        axes[0][j].set_title(header, fontsize=9, fontweight="bold", pad=4)

    for row, img_path in enumerate(sample_images):
        stem = img_path.stem
        ann = ann_dict.get(stem)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            for col in range(n_cols):
                axes[row][col].axis("off")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_boxes = [[b.x1, b.y1, b.x2, b.y2] for b in ann.bboxes] if ann else []
        gt_classes = [b.category for b in ann.bboxes] if ann else []
        n_gt = len(gt_boxes)

        # GT column (row label = filename)
        _draw_boxes(
            axes[row][0], img_rgb, [],
            gt_boxes=gt_boxes, gt_classes=gt_classes,
            title=f"{stem[:30]} (GT={n_gt})",
            show_gt=True,
        )

        # One column per model
        for col, lbl in enumerate(model_labels, start=1):
            dets = model_dets[lbl].get(str(img_path), [])
            n_pred = len(dets)
            _draw_boxes(
                axes[row][col], img_rgb, dets,
                title=f"preds={n_pred}",
            )

    # Legend: GT style + class colours
    legend_handles = [
        mpatches.Patch(edgecolor="lime", facecolor="none",
                       linestyle="--", linewidth=1.5, label="Ground Truth")
    ]
    legend_handles += [
        mpatches.Patch(facecolor=c, label=cls)
        for cls, c in _COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=6,
        fontsize=8, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = out_dir / "qualitative_comparison.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare RT-DETRv2 pretrained, RT-DETRv2 fine-tuned, and YOLOv8 "
            "side-by-side on the BDD100K validation set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--images", type=int, default=200,
        help="Number of validation images to evaluate each model on.",
    )
    p.add_argument(
        "--score-thresh", type=float, default=0.3,
        help="Minimum confidence threshold for all models.",
    )
    p.add_argument(
        "--finetuned-weights", default="output_/rtdetrv2_bdd100k/best.pth",
        help="Path to BDD100K fine-tuned RT-DETRv2 checkpoint (from train.py). "
             "Pass 'none' to skip.",
    )
    p.add_argument(
        "--yolo-weights", default="yolov8s.pt",
        help=(
            "YOLO checkpoint to use.  Defaults to 'yolov8s.pt', which "
            "ultralytics downloads automatically (COCO pretrained, 80 classes). "
            "Pass a local .pt file trained on BDD100K for a fair comparison. "
            "Pass 'none' to skip YOLO."
        ),
    )
    p.add_argument(
        "--finetuned-score-thresh", type=float, default=0.03,
        help=(
            "Confidence threshold applied specifically to the fine-tuned "
            "RT-DETRv2 model.  After only 1-2 epochs the decoder scores are "
            "low-calibrated (typically 0.02-0.05), so a lower threshold than "
            "the default 0.3 is needed to see any detections."
        ),
    )
    p.add_argument(
        "--qual-images", type=int, default=6,
        help="Number of images to include in the qualitative grid.",
    )
    return p.parse_args()

def main() -> None:
    global _ANN_DICT

    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Model Comparison: RT-DETRv2 vs YOLOv8")
    print("=" * 60)

    # Load annotations
    print("\nLoading validation annotations ...")
    val_anns = load_split(str(DATA_DIR), "val")
    _ANN_DICT = {ann.filename: ann for ann in val_anns}

    val_images = [
        p for p in sorted(IMAGE_DIR.glob("*.jpg"))[: args.images]
        if p.stem in _ANN_DICT
    ]
    print(f"Evaluating on {len(val_images)} images.")

    try:
        import torch as _t
        _gpu = _t.cuda.is_available()
    except Exception:
        _gpu = False
    batch = 8 if _gpu else 4

    model_metrics: Dict[str, Dict] = {}
    model_dets: Dict[str, Dict] = {}

    # ── Model 1: RT-DETRv2 COCO pretrained ──────────────────────────────────
    print("\n[1] RT-DETRv2-s  (COCO pretrained, zero-shot) ...")
    rtdetr = build_detector(backend="rtdetrv2-gh", score_thresh=args.score_thresh)
    dets_map, metrics = _infer(rtdetr, val_images, apply_domain_correction=True, batch_size=batch)
    label_a = "RT-DETRv2 COCO"
    model_metrics[label_a] = metrics
    model_dets[label_a] = dets_map
    mean_f1 = np.mean([v["f1"] for v in metrics.values()])
    print(f"   mean F1={mean_f1:.4f}  "
          f"car F1={metrics['car']['f1']:.3f}  "
          f"motor FP={metrics['motor']['fp']}")

    # ── Model 2: RT-DETRv2 fine-tuned ───────────────────────────────────────
    if args.finetuned_weights and args.finetuned_weights.lower() != "none":
        ft_path = Path(args.finetuned_weights)
        if ft_path.exists():
            print(f"\n[2] RT-DETRv2-s  (fine-tuned: {ft_path.name}) ...")
            ft_det = build_detector(
                backend="rtdetrv2-bdd",
                score_thresh=args.finetuned_score_thresh,
                finetuned_weights=str(ft_path),
            )
            dets_map_ft, metrics_ft = _infer(
                ft_det, val_images, apply_domain_correction=False, batch_size=batch
            )
            label_b = f"RT-DETRv2 fine-tuned ({ft_path.stem})"
            model_metrics[label_b] = metrics_ft
            model_dets[label_b] = dets_map_ft
            total_tp = sum(v["tp"] for v in metrics_ft.values())
            print(f"   total TP={total_tp}  (expected ~0 for 1-epoch smoke test)")
        else:
            print(f"\n[2] Skipping fine-tuned model — weights not found: {ft_path}")
    else:
        print("\n[2] Fine-tuned model skipped (--finetuned-weights none).")

    # ── Model 3: YOLOv8 ─────────────────────────────────────────────────────
    if args.yolo_weights and args.yolo_weights.lower() != "none":
        print(f"\n[3] YOLOv8  (weights: {args.yolo_weights}) ...")
        print("    (ultralytics will auto-download 'yolov8s.pt' if not cached)")
        try:
            from src.model.yolo_detector import YOLODetector

            yolo_det = YOLODetector(
                weights=args.yolo_weights,
                score_thresh=args.score_thresh,
            )
            dets_map_y, metrics_y = _infer(
                yolo_det, val_images,
                apply_domain_correction=not yolo_det._is_bdd_model,
                batch_size=batch,
            )
            label_c = f"YOLOv8 ({Path(args.yolo_weights).stem})"
            model_metrics[label_c] = metrics_y
            model_dets[label_c] = dets_map_y
            mean_f1_y = np.mean([v["f1"] for v in metrics_y.values()])
            print(f"   mean F1={mean_f1_y:.4f}  "
                  f"car F1={metrics_y['car']['f1']:.3f}")
        except ImportError as exc:
            print(f"   Skipping YOLOv8 — {exc}")
        except Exception as exc:
            print(f"   YOLOv8 inference failed: {exc}")
    else:
        print("\n[3] YOLOv8 skipped (--yolo-weights none).")

    if not model_metrics:
        print("\nNo models were evaluated. Check arguments and try again.")
        return

    print("\n[4] Generating quantitative comparison chart ...")
    _plot_quantitative(model_metrics, OUTPUT_DIR)

    print("[5] Generating qualitative comparison grid ...")
    sample_imgs = _pick_representative_images(val_images, _ANN_DICT, args.qual_images)
    _plot_qualitative(model_dets, sample_imgs, _ANN_DICT, OUTPUT_DIR)

    rows = []
    for lbl, metrics in model_metrics.items():
        row = {
            "model": lbl,
            "mean_f1": round(float(np.mean([v["f1"] for v in metrics.values()])), 4),
            "mean_precision": round(float(np.mean([v["precision"] for v in metrics.values()])), 4),
            "mean_recall": round(float(np.mean([v["recall"] for v in metrics.values()])), 4),
        }
        for cls in BDD_CLASSES:
            row[f"f1_{cls}"] = round(metrics[cls]["f1"], 4)
            row[f"tp_{cls}"] = metrics[cls]["tp"]
            row[f"fp_{cls}"] = metrics[cls]["fp"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("  Saved: summary.csv")

    # Detailed JSON
    report = {
        "images_evaluated": len(val_images),
        "score_threshold": args.score_thresh,
        "models": {
            lbl: {cls: {k: round(float(v), 4) for k, v in m.items()}
                  for cls, m in metrics.items()}
            for lbl, metrics in model_metrics.items()
        },
    }
    with open(OUTPUT_DIR / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  Saved: comparison_report.json")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'mean F1':>8} {'mean P':>8} {'mean R':>8}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['model']:<45} {row['mean_f1']:>8.4f} "
              f"{row['mean_precision']:>8.4f} {row['mean_recall']:>8.4f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
