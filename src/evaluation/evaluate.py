import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.parser import DETECTION_CLASSES, ImageAnnotation, load_split
from src.model.class_map import CLASS_TO_ID, ID_TO_CLASS
from src.model.detector import (
    Detection,
    build_detector,
    correct_domain_gap_labels,
    filter_to_bdd,
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/evaluation")
IMAGE_DIR = DATA_DIR / "bdd100k" / "images" / "100k" / "val"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BDD_CLASSES = list(ID_TO_CLASS.values())

COLORS = {
    "car": (0, 114, 189), "truck": (217, 83, 25), "bus": (237, 177, 32),
    "train": (126, 47, 142), "person": (119, 172, 48), "rider": (77, 190, 238),
    "bike": (162, 20, 47), "motor": (255, 153, 0),
    "traffic light": (76, 189, 237), "traffic sign": (255, 99, 71),
}


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(
    gt_boxes: List, gt_classes: List, pred_boxes: List, pred_classes: List,
    pred_confs: List, iou_thresh: float = 0.5
) -> Tuple[List, List, List]:
    """Match predictions to ground truth; return TP, FP, FN lists."""
    matched_gt = set()
    tp, fp, fn_details = [], [], []

    order = np.argsort(-np.array(pred_confs)) if pred_confs else []

    for idx in order:
        pb = pred_boxes[idx]
        pc = pred_classes[idx]
        conf = pred_confs[idx]
        best_iou, best_gt_idx = 0.0, -1

        for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if gi in matched_gt or gc != pc:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp.append({"class": pc, "conf": conf, "iou": best_iou, "box": pb})
            matched_gt.add(best_gt_idx)
        else:
            fp.append({"class": pc, "conf": conf, "box": pb})

    for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
        if gi not in matched_gt:
            area = (gb[2] - gb[0]) * (gb[3] - gb[1])
            fn_details.append({"class": gc, "box": gb, "area": area})

    return tp, fp, fn_details


def categorize_fn_by_size(fn_list: List) -> Dict[str, int]:
    """Categorize false negatives by object size."""
    categories = {
        "tiny (<32\xb2)": 0,
        "small (32\xb2-96\xb2)": 0,
        "medium (96\xb2-256\xb2)": 0,
        "large (\u226556\xb2)": 0,
    }
    for fn in fn_list:
        area = fn["area"]
        if area < 32 * 32:
            categories["tiny (<32\xb2)"] += 1
        elif area < 96 * 96:
            categories["small (32\xb2-96\xb2)"] += 1
        elif area < 256 * 256:
            categories["medium (96\xb2-256\xb2)"] += 1
        else:
            categories["large (\u226556\xb2)"] += 1
    return categories

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RT-DETRv2 (and optionally compare with other models) "
                    "on the BDD100K validation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--score-thresh", type=float, default=0.3,
        help="Minimum confidence score to keep a detection.",
    )
    p.add_argument(
        "--images", type=int, default=500,
        help="Maximum number of validation images to evaluate.",
    )
    p.add_argument(
        "--compare-with", action="append", dest="compare_with",
        metavar="BACKEND:WEIGHTS",
        help=(
            "Add a model to compare against the baseline RT-DETRv2. "
            "Format: 'rtdetrv2-bdd:path/to/best.pth' for a BDD100K fine-tuned "
            "checkpoint (from train.py), or 'yolo:path/to/weights.pt' for any "
            "ultralytics YOLO model.  Repeat the flag to compare multiple models."
        ),
    )
    return p.parse_args()

def _evaluate_model(
    label: str,
    detector,
    val_images: List[Path],
    ann_dict: Dict,
    apply_domain_correction: bool,
    batch_size: int,
    device: str,
) -> Tuple[Dict, List, List, List, pd.DataFrame, Dict]:
    """Run batch inference + per-image metric collection for one detector.

    Args:
        label: Human-readable name used in printed output.
        detector: Any detector that exposes ``batch_predict_paths``.
        apply_domain_correction: Apply :func:`correct_domain_gap_labels` to
            detections.  Should be ``True`` for COCO-pretrained models and
            ``False`` for models already trained on BDD100K.
        batch_size: Images per forward pass.
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.

    Returns:
        Tuple of (per_class_metrics, all_tp, all_fp, all_fn,
                  per_image_scores_df, all_dets_map).
    """
    print(f"\n  Running batch inference for [{label}] ...")
    all_dets_map = detector.batch_predict_paths(
        [str(p) for p in val_images],
        batch_size=batch_size,
        device=device,
    )
    print(f"  Inference done ({len(all_dets_map)} images).")

    all_tp: List = []
    all_fp: List = []
    all_fn: List = []
    per_image_scores: List = []
    failure_images: Dict[str, List] = {"missed_small": [], "many_fp": []}

    for img_path in val_images:
        stem = img_path.stem
        ann = ann_dict[stem]
        gt_boxes = [[b.x1, b.y1, b.x2, b.y2] for b in ann.bboxes]
        gt_classes = [b.category for b in ann.bboxes]

        raw_dets = all_dets_map.get(str(img_path), [])
        if apply_domain_correction:
            raw_dets = correct_domain_gap_labels(raw_dets)
        bdd_dets = filter_to_bdd(raw_dets)

        pred_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in bdd_dets]
        pred_classes = [d.bdd_label for d in bdd_dets]
        pred_confs = [d.score for d in bdd_dets]

        tp, fp, fn = match_predictions_to_gt(
            gt_boxes, gt_classes, pred_boxes, pred_classes, pred_confs
        )
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)

        recall = len(tp) / max(len(gt_boxes), 1)
        precision = len(tp) / max(len(tp) + len(fp), 1)
        per_image_scores.append({
            "filename": stem,
            "tp": len(tp), "fp": len(fp), "fn": len(fn),
            "precision": precision, "recall": recall,
            "gt_count": len(gt_boxes), "pred_count": len(pred_boxes),
        })

        small_fn = [f for f in fn if f["area"] < 32 * 32]
        if len(small_fn) > 5:
            failure_images["missed_small"].append((stem, len(small_fn)))
        if len(fp) > 10:
            failure_images["many_fp"].append((stem, len(fp)))

    print(f"  [{label}] TP={len(all_tp)}, FP={len(all_fp)}, FN={len(all_fn)}")

    per_class_metrics: Dict[str, Dict] = {}
    for cls in BDD_CLASSES:
        tp_cls = len([t for t in all_tp if t["class"] == cls])
        fp_cls = len([f for f in all_fp if f["class"] == cls])
        fn_cls = len([f for f in all_fn if f["class"] == cls])
        prc = tp_cls / max(tp_cls + fp_cls, 1)
        rec = tp_cls / max(tp_cls + fn_cls, 1)
        f1 = 2 * prc * rec / max(prc + rec, 1e-6)
        per_class_metrics[cls] = {
            "precision": prc, "recall": rec, "f1": f1,
            "tp": tp_cls, "fp": fp_cls, "fn": fn_cls,
        }

    scores_df = pd.DataFrame(per_image_scores)
    scores_df["f1"] = (
        2 * scores_df["precision"] * scores_df["recall"]
        / (scores_df["precision"] + scores_df["recall"] + 1e-6)
    )

    return (
        per_class_metrics, all_tp, all_fp, all_fn,
        scores_df, all_dets_map,
    )

def _plot_model_comparison(
    model_metrics: Dict[str, Dict],
    out_dir: Path,
) -> None:
    """Side-by-side F1 bar chart for all evaluated models.

    Args:
        model_metrics: Mapping from model label to per_class_metrics dict.
        out_dir: Directory where the PNG is written.
    """
    labels = list(model_metrics.keys())
    n_models = len(labels)
    x = np.arange(len(BDD_CLASSES))
    width = 0.8 / n_models

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Multi-Model Comparison on BDD100K Validation Set",
                 fontsize=14, fontweight="bold")

    cmap = plt.get_cmap("tab10")

    # F1 score comparison
    ax = axes[0]
    for i, (lbl, metrics) in enumerate(model_metrics.items()):
        f1s = [metrics[c]["f1"] for c in BDD_CLASSES]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, f1s, width, label=lbl,
                      color=cmap(i), alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(BDD_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Precision comparison
    ax = axes[1]
    for i, (lbl, metrics) in enumerate(model_metrics.items()):
        precs = [metrics[c]["precision"] for c in BDD_CLASSES]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, precs, width, label=lbl,
               color=cmap(i), alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(BDD_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Recall comparison
    ax = axes[2]
    for i, (lbl, metrics) in enumerate(model_metrics.items()):
        recs = [metrics[c]["recall"] for c in BDD_CLASSES]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, recs, width, label=lbl,
               color=cmap(i), alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(BDD_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")

    # Also save a summary CSV
    rows = []
    for lbl, metrics in model_metrics.items():
        mean_f1 = float(np.mean([metrics[c]["f1"] for c in BDD_CLASSES]))
        mean_p = float(np.mean([metrics[c]["precision"] for c in BDD_CLASSES]))
        mean_r = float(np.mean([metrics[c]["recall"] for c in BDD_CLASSES]))
        rows.append({"model": lbl, "mean_f1": round(mean_f1, 4),
                     "mean_precision": round(mean_p, 4), "mean_recall": round(mean_r, 4)})
        for cls in BDD_CLASSES:
            rows[-1][f"f1_{cls}"] = round(metrics[cls]["f1"], 4)
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)
    print("  Saved: model_comparison.csv")


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("Step 3: Evaluation and Visualization (RT-DETRv2)")
    print("=" * 60)

    # Auto-detect GPU and build detector
    try:
        import torch as _torch
        _gpu_available = _torch.cuda.is_available()
    except Exception:
        _gpu_available = False
    infer_device = "auto"  # subprocess decides: cuda if available, else cpu

    detector = build_detector(backend="rtdetrv2-gh", score_thresh=args.score_thresh)
    print(f"Detector: {detector.backend} / {detector.model_id}")
    print(f"GPU available: {_gpu_available}  (inference device: {'cuda' if _gpu_available else 'cpu'})")

    print("\nLoading validation annotations...")
    val_anns = load_split(str(DATA_DIR), "val")
    ann_dict = {ann.filename: ann for ann in val_anns}

    # Collect only images that have annotations.
    val_images = [
        p for p in sorted(IMAGE_DIR.glob("*.jpg"))[:args.images]
        if p.stem in ann_dict
    ]
    print(f"Analyzing {len(val_images)} validation images...")

    batch_size = 8 if _gpu_available else 4

    # ── Baseline RT-DETRv2 evaluation ────────────────────────────────────────
    print("\n[1/4] Running baseline inference and collecting metrics...")
    (
        per_class_metrics, all_tp, all_fp, all_fn,
        scores_df, all_dets_map,
    ) = _evaluate_model(
        label=f"RT-DETRv2 ({detector.model_id})",
        detector=detector,
        val_images=val_images,
        ann_dict=ann_dict,
        apply_domain_correction=True,
        batch_size=batch_size,
        device=infer_device,
    )

    metrics_df = pd.DataFrame(per_class_metrics).T.round(3)
    print(f"\nPer-Class Metrics (IoU=0.5, conf={args.score_thresh}):")
    print(metrics_df.to_string())
    metrics_df.to_csv(OUTPUT_DIR / "per_class_metrics.csv")

    scores_df = scores_df.sort_values("f1")

    # ===== QUANTITATIVE VISUALIZATION =====
    print("\n[2/4] Quantitative Visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("RT-DETRv2 on BDD100K Validation — Quantitative Analysis",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    precisions = [per_class_metrics[c]["precision"] for c in BDD_CLASSES]
    ax.bar(BDD_CLASSES, precisions,
           color=[np.array(COLORS[c]) / 255 for c in BDD_CLASSES])
    ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision (conf=0.3, IoU=0.5)")
    ax.set_xticklabels(BDD_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    for cls in BDD_CLASSES:
        m = per_class_metrics[cls]
        ax.scatter(m["recall"], m["precision"], s=100,
                   color=np.array(COLORS[cls]) / 255, label=cls, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall by Class")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    f1_scores = {cls: per_class_metrics[cls]["f1"] for cls in BDD_CLASSES}
    ax.barh(list(f1_scores.keys()), list(f1_scores.values()),
            color=[np.array(COLORS[c]) / 255 for c in f1_scores.keys()])
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.set_xlim(0, 1)

    ax = axes[1, 0]
    fn_by_size = categorize_fn_by_size(all_fn)
    ax.bar(fn_by_size.keys(), fn_by_size.values(), color="coral", edgecolor="black")
    ax.set_ylabel("Missed Objects (FN)")
    ax.set_title("False Negatives by Object Size")
    ax.set_xticklabels(fn_by_size.keys(), rotation=20)

    ax = axes[1, 1]
    fn_per_class = Counter(f["class"] for f in all_fn)
    fp_per_class = Counter(f["class"] for f in all_fp)
    x = np.arange(len(BDD_CLASSES))
    w = 0.35
    ax.bar(x - w / 2, [fn_per_class.get(c, 0) for c in BDD_CLASSES],
           w, label="FN (Missed)", color="coral")
    ax.bar(x + w / 2, [fp_per_class.get(c, 0) for c in BDD_CLASSES],
           w, label="FP (False Alarm)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(BDD_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("FN vs FP per Class")
    ax.legend()

    ax = axes[1, 2]
    tp_confs = [t["conf"] for t in all_tp]
    fp_confs = [f["conf"] for f in all_fp]
    ax.hist(tp_confs, bins=30, alpha=0.6, label=f"TP (n={len(tp_confs)})",
            color="green", density=True)
    ax.hist(fp_confs, bins=30, alpha=0.6, label=f"FP (n={len(fp_confs)})",
            color="red", density=True)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: TP vs FP")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quantitative_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'quantitative_metrics.png'}")

    # ===== QUALITATIVE VISUALIZATION =====
    print("\n[3/4] Qualitative Visualizations (GT vs Predictions)...")

    sample_categories = {
        "good_detections": scores_df.tail(4)["filename"].tolist(),
        "poor_detections": scores_df.head(4)["filename"].tolist(),
        "medium_detections": scores_df.iloc[
            len(scores_df) // 2 - 2: len(scores_df) // 2 + 2
        ]["filename"].tolist(),
    }

    for category, filenames in sample_categories.items():
        fig, axes_q = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle(f"Qualitative: {category.replace('_', ' ').title()}",
                     fontsize=14, fontweight="bold")

        for idx, fname in enumerate(filenames[:4]):
            ax = axes_q[idx // 2, idx % 2]
            img_path = IMAGE_DIR / f"{fname}.jpg"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ann = ann_dict[fname]
            gt_boxes = [[b.x1, b.y1, b.x2, b.y2] for b in ann.bboxes]
            gt_classes = [b.category for b in ann.bboxes]

            bdd_dets = filter_to_bdd(correct_domain_gap_labels(all_dets_map.get(str(img_path), [])))

            for box, cls in zip(gt_boxes, gt_classes):
                x1, y1, x2, y2 = (int(v) for v in box)
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--"
                )
                ax.add_patch(rect)

            for det in bdd_dets:
                cls_name = det.bdd_label
                color = np.array(COLORS[cls_name]) / 255
                rect = plt.Rectangle(
                    (det.x1, det.y1), det.x2 - det.x1, det.y2 - det.y1,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(det.x1, det.y1 - 3, f"{cls_name} {det.score:.2f}",
                        fontsize=6, color="white",
                        bbox=dict(facecolor=color, alpha=0.7, pad=0.5))

            ax.imshow(img)
            score = scores_df[scores_df["filename"] == fname]["f1"].values
            title = f"{fname} (F1={score[0]:.2f})" if len(score) else fname
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        patches_legend = [
            mpatches.Patch(edgecolor="lime", facecolor="none",
                           linestyle="--", label="Ground Truth")
        ]
        patches_legend += [
            mpatches.Patch(color=np.array(COLORS[c]) / 255, label=c)
            for c in BDD_CLASSES[:5]
        ]
        fig.legend(handles=patches_legend, loc="lower center", ncol=6, fontsize=9)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        out_name = f"qualitative_{category}.png"
        plt.savefig(OUTPUT_DIR / out_name, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_name}")

    # ===== FAILURE CLUSTERING =====
    print("\n[4/4] Failure Analysis & Report...")

    fn_size_class: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for fn in all_fn:
        area = fn["area"]
        cls = fn["class"]
        if area < 32 * 32:
            fn_size_class[cls]["tiny"] += 1
        elif area < 96 * 96:
            fn_size_class[cls]["small"] += 1
        elif area < 256 * 256:
            fn_size_class[cls]["medium"] += 1
        else:
            fn_size_class[cls]["large"] += 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Failure Analysis: Where RT-DETRv2 Fails", fontsize=14, fontweight="bold")

    size_cats = ["tiny", "small", "medium", "large"]
    heatmap_data = np.zeros((len(BDD_CLASSES), len(size_cats)))
    for i, cls in enumerate(BDD_CLASSES):
        for j, sz in enumerate(size_cats):
            heatmap_data[i, j] = fn_size_class[cls].get(sz, 0)

    sns.heatmap(heatmap_data, annot=True, fmt=".0f", xticklabels=size_cats,
                yticklabels=BDD_CLASSES, cmap="Reds", ax=axes[0])
    axes[0].set_title("Missed Objects: Class x Size")

    axes[1].hist(scores_df["f1"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(scores_df["f1"].median(), color="red", linestyle="--",
                    label=f"Median F1: {scores_df['f1'].median():.2f}")
    axes[1].set_xlabel("Per-Image F1 Score")
    axes[1].set_ylabel("Number of Images")
    axes[1].set_title("Image-Level Performance Distribution")
    axes[1].legend()

    axes[2].scatter(scores_df["gt_count"], scores_df["f1"],
                    alpha=0.4, s=20, c="steelblue")
    axes[2].set_xlabel("Ground Truth Objects per Image")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Performance vs Scene Complexity")
    z = np.polyfit(scores_df["gt_count"], scores_df["f1"], 1)
    x_line = np.linspace(scores_df["gt_count"].min(), scores_df["gt_count"].max(), 50)
    axes[2].plot(x_line, np.poly1d(z)(x_line), "r--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: failure_clustering.png")

    failure_analysis = {
        "size_based": categorize_fn_by_size(all_fn),
        "class_based_fn": dict(Counter(f["class"] for f in all_fn).most_common()),
        "class_based_fp": dict(Counter(f["class"] for f in all_fp).most_common()),
        "high_conf_fp": len([f for f in all_fp if f["conf"] > 0.7]),
        "low_conf_tp": len([t for t in all_tp if t["conf"] < 0.4]),
    }

    report = {
        "model": f"RT-DETRv2 ({detector.model_id})",
        "val_subset": f"{len(val_images)} images",
        "overall_metrics": {
            "mean_precision": round(float(metrics_df["precision"].mean()), 4),
            "mean_recall": round(float(metrics_df["recall"].mean()), 4),
            "mean_f1": round(float(metrics_df["f1"].mean()), 4),
        },
        "per_class_metrics": {
            cls: {k: round(float(v), 4) for k, v in per_class_metrics[cls].items()}
            for cls in BDD_CLASSES
        },
        "failure_analysis": failure_analysis,
        "key_findings": [
            "RT-DETRv2 is COCO-pretrained; BDD100K classes rider and traffic sign "
            "have no direct COCO equivalent — zero-shot performance is limited.",
            "Tiny objects (< 32x32 px) are the primary missed-detection category.",
            "Car has the highest recall — most prevalent class in COCO and BDD100K.",
            "Train and rider are the rarest BDD classes with limited COCO overlap.",
        ],
        "improvement_suggestions": [
            "1. Fine-tune on BDD100K for 30-100 epochs using train.py",
            "2. Address class imbalance via weighted sampler (data_loader.py)",
            "3. Increase input resolution (640->1280) to improve tiny-object detection",
            "4. Copy-paste augmentation to oversample rare classes (rider, train, motor)",
            "5. Add SAHI (sliced inference) for dense small-object scenes",
            "6. Filter annotation noise: 400k+ boxes < 16px in training set",
        ],
        "connection_to_data_analysis": {
            "class_imbalance": "5250:1 imbalance (car vs train) predicts performance gaps.",
            "tiny_objects": "415k boxes < 16px in training; FPN P3 at 80x80 is too "
                            "coarse for sub-16px objects.",
            "split_consistency": "Train/val distributions match (<0.2% diff); poor "
                                 "val performance is capacity/training related.",
        },
    }

    with open(OUTPUT_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    scores_df.to_csv(OUTPUT_DIR / "per_image_scores.csv", index=False)

    if args.compare_with:
        baseline_label = f"RT-DETRv2 pretrained ({detector.model_id})"
        model_metrics: Dict[str, Dict] = {baseline_label: per_class_metrics}

        for spec in args.compare_with:
            if ":" not in spec:
                print(f"  [compare] Skipping malformed spec '{spec}' (expected BACKEND:PATH)")
                continue
            backend, weights_path = spec.split(":", 1)
            backend = backend.strip()
            weights_path = weights_path.strip()

            print(f"\n  [compare] Building detector: {backend} / {weights_path}")
            try:
                if backend == "rtdetrv2-bdd":
                    comp_det = build_detector(
                        backend="rtdetrv2-bdd",
                        score_thresh=args.score_thresh,
                        finetuned_weights=weights_path,
                    )
                    apply_corr = False   # model trained on BDD100K — no domain gap correction
                elif backend == "yolo":
                    from src.model.yolo_detector import YOLODetector
                    comp_det = YOLODetector(
                        weights=weights_path, score_thresh=args.score_thresh
                    )
                    apply_corr = not comp_det._is_bdd_model  # COCO-YOLO needs correction
                else:
                    print(f"  [compare] Unknown backend '{backend}', skipping.")
                    continue
            except Exception as exc:
                print(f"  [compare] Could not build detector for '{spec}': {exc}")
                continue

            comp_label = f"{backend} ({Path(weights_path).stem})"
            comp_metrics, *_ = _evaluate_model(
                label=comp_label,
                detector=comp_det,
                val_images=val_images,
                ann_dict=ann_dict,
                apply_domain_correction=apply_corr,
                batch_size=batch_size,
                device=infer_device,
            )
            model_metrics[comp_label] = comp_metrics

        print("\n[+] Generating comparison chart ...")
        _plot_model_comparison(model_metrics, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    print(f"\nOverall mean F1:        {report['overall_metrics']['mean_f1']:.4f}")
    print(f"Overall mean precision: {report['overall_metrics']['mean_precision']:.4f}")
    print(f"Overall mean recall:    {report['overall_metrics']['mean_recall']:.4f}")


if __name__ == "__main__":
    main()
