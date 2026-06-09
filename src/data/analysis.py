from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .parser import DETECTION_CLASSES, BoundingBox, ImageAnnotation


def class_distribution(annotations: List[ImageAnnotation]) -> pd.DataFrame:
    """Compute the number of bounding boxes per detection class.

    Args:
        annotations: List of image annotations.

    Returns:
        DataFrame with columns ['class', 'count', 'percentage'].
    """
    counter = Counter()
    for ann in annotations:
        for bbox in ann.bboxes:
            counter[bbox.category] += 1

    total = sum(counter.values())
    rows = []
    for cls in DETECTION_CLASSES:
        count = counter.get(cls, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        rows.append({"class": cls, "count": count, "percentage": round(pct, 2)})

    df = pd.DataFrame(rows)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    return df


def split_comparison(
    train_anns: List[ImageAnnotation], val_anns: List[ImageAnnotation]
) -> pd.DataFrame:
    """Compare class distributions between train and val splits.

    Args:
        train_anns: Training set annotations.
        val_anns: Validation set annotations.

    Returns:
        DataFrame with per-class counts and percentages for both splits.
    """
    train_dist = class_distribution(train_anns).rename(
        columns={"count": "train_count", "percentage": "train_pct"}
    )
    val_dist = class_distribution(val_anns).rename(
        columns={"count": "val_count", "percentage": "val_pct"}
    )

    merged = train_dist.merge(val_dist, on="class", how="outer").fillna(0)
    merged["ratio_diff"] = abs(merged["train_pct"] - merged["val_pct"])
    return merged


def bbox_size_stats(annotations: List[ImageAnnotation]) -> pd.DataFrame:
    """Compute bounding box size statistics per class.

    Categorizes boxes into: tiny (<32x32), small (<96x96),
    medium (<256x256), and large (>=256x256).

    Args:
        annotations: List of image annotations.

    Returns:
        DataFrame with size category counts per class.
    """
    records = []
    for ann in annotations:
        for bbox in ann.bboxes:
            area = bbox.area
            if area < 32 * 32:
                size_cat = "tiny"
            elif area < 96 * 96:
                size_cat = "small"
            elif area < 256 * 256:
                size_cat = "medium"
            else:
                size_cat = "large"

            records.append(
                {
                    "class": bbox.category,
                    "width": bbox.width,
                    "height": bbox.height,
                    "area": area,
                    "aspect_ratio": bbox.aspect_ratio,
                    "size_category": size_cat,
                }
            )

    return pd.DataFrame(records)


def co_occurrence_matrix(
    annotations: List[ImageAnnotation],
) -> pd.DataFrame:
    """Compute class co-occurrence matrix across images.

    Entry (i, j) = number of images containing both class i and class j.

    Args:
        annotations: List of image annotations.

    Returns:
        Square DataFrame with co-occurrence counts.
    """
    matrix = np.zeros((len(DETECTION_CLASSES), len(DETECTION_CLASSES)), dtype=int)
    cls_to_idx = {cls: i for i, cls in enumerate(DETECTION_CLASSES)}

    for ann in annotations:
        present = set(bbox.category for bbox in ann.bboxes)
        present_list = [cls for cls in present if cls in cls_to_idx]
        for i, cls_a in enumerate(present_list):
            for cls_b in present_list[i:]:
                idx_a = cls_to_idx[cls_a]
                idx_b = cls_to_idx[cls_b]
                matrix[idx_a][idx_b] += 1
                if idx_a != idx_b:
                    matrix[idx_b][idx_a] += 1

    return pd.DataFrame(matrix, index=DETECTION_CLASSES, columns=DETECTION_CLASSES)


def images_per_class(
    annotations: List[ImageAnnotation],
) -> Dict[str, int]:
    """Count number of images containing at least one instance of each class.

    Args:
        annotations: List of image annotations.

    Returns:
        Dictionary mapping class name to image count.
    """
    counts = defaultdict(int)
    for ann in annotations:
        classes_in_image = set(bbox.category for bbox in ann.bboxes)
        for cls in classes_in_image:
            counts[cls] += 1
    return dict(counts)


def annotations_per_image_stats(
    annotations: List[ImageAnnotation],
) -> Dict[str, float]:
    """Compute statistics on the number of annotations per image.

    Args:
        annotations: List of image annotations.

    Returns:
        Dictionary with min, max, mean, median, std of bbox count per image.
    """
    counts = [len(ann.bboxes) for ann in annotations]
    if not counts:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}

    return {
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "mean": round(float(np.mean(counts)), 2),
        "median": round(float(np.median(counts)), 2),
        "std": round(float(np.std(counts)), 2),
        "total_images": len(counts),
        "images_with_annotations": sum(1 for c in counts if c > 0),
    }
