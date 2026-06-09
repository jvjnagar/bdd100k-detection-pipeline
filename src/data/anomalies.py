from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .parser import DETECTION_CLASSES, ImageAnnotation


def find_outlier_images(
    annotations: List[ImageAnnotation], threshold: float = 3.0
) -> Dict[str, List[ImageAnnotation]]:
    """Find images with anomalous annotation counts using z-score.

    Args:
        annotations: List of image annotations.
        threshold: Z-score threshold for outlier detection.

    Returns:
        Dictionary with 'high' and 'low' outlier image lists.
    """
    counts = np.array([len(ann.bboxes) for ann in annotations])
    if len(counts) == 0 or counts.std() == 0:
        return {"high": [], "low": []}

    z_scores = (counts - counts.mean()) / counts.std()

    high_outliers = [
        ann for ann, z in zip(annotations, z_scores) if z > threshold
    ]
    low_outliers = [
        ann
        for ann, z, c in zip(annotations, z_scores, counts)
        if z < -threshold and c > 0
    ]

    return {"high": high_outliers, "low": low_outliers}


def detect_tiny_boxes(
    annotations: List[ImageAnnotation], min_pixels: int = 16
) -> List[Tuple[str, str, float]]:
    """Detect suspiciously small bounding boxes that may be annotation errors.

    Args:
        annotations: List of image annotations.
        min_pixels: Minimum width or height in pixels.

    Returns:
        List of (filename, category, area) tuples for tiny boxes.
    """
    tiny = []
    for ann in annotations:
        for bbox in ann.bboxes:
            if bbox.width < min_pixels or bbox.height < min_pixels:
                tiny.append((ann.filename, bbox.category, bbox.area))
    return tiny


def detect_class_imbalance(
    annotations: List[ImageAnnotation],
) -> pd.DataFrame:
    """Analyze class imbalance and compute imbalance ratio.

    The imbalance ratio is max_count / class_count for each class.

    Args:
        annotations: List of image annotations.

    Returns:
        DataFrame with class, count, and imbalance_ratio columns.
    """
    from collections import Counter

    counter = Counter()
    for ann in annotations:
        for bbox in ann.bboxes:
            counter[bbox.category] += 1

    if not counter:
        return pd.DataFrame(columns=["class", "count", "imbalance_ratio"])

    max_count = max(counter.values())
    rows = []
    for cls in DETECTION_CLASSES:
        count = counter.get(cls, 0)
        ratio = max_count / count if count > 0 else float("inf")
        rows.append({"class": cls, "count": count, "imbalance_ratio": round(ratio, 2)})

    return pd.DataFrame(rows).sort_values("imbalance_ratio", ascending=False)


def find_interesting_samples(
    annotations: List[ImageAnnotation], top_n: int = 10
) -> Dict[str, List[str]]:
    """Identify interesting/unique samples per class.

    Finds images where a particular class dominates, or images with
    rare classes, or images with the most diverse set of classes.

    Args:
        annotations: List of image annotations.
        top_n: Number of samples to return per category.

    Returns:
        Dictionary with categories of interesting samples and filenames.
    """
    results = {}

    # Images with most annotations overall
    sorted_by_count = sorted(annotations, key=lambda a: len(a.bboxes), reverse=True)
    results["most_annotations"] = [a.filename for a in sorted_by_count[:top_n]]

    # Images with most diverse classes
    sorted_by_diversity = sorted(
        annotations,
        key=lambda a: len(set(b.category for b in a.bboxes)),
        reverse=True,
    )
    results["most_diverse"] = [a.filename for a in sorted_by_diversity[:top_n]]

    # Images containing rare classes (train, rider, motor)
    rare_classes = ["train", "rider", "motor"]
    for rare_cls in rare_classes:
        imgs_with_rare = [
            ann.filename
            for ann in annotations
            if any(b.category == rare_cls for b in ann.bboxes)
        ]
        results[f"contains_{rare_cls}"] = imgs_with_rare[:top_n]

    # Images where a single class dominates (>80% of boxes)
    dominated = []
    for ann in annotations:
        if len(ann.bboxes) < 5:
            continue
        from collections import Counter

        class_counts = Counter(b.category for b in ann.bboxes)
        most_common_cls, most_common_count = class_counts.most_common(1)[0]
        if most_common_count / len(ann.bboxes) > 0.8:
            dominated.append((ann.filename, most_common_cls, len(ann.bboxes)))

    dominated.sort(key=lambda x: x[2], reverse=True)
    results["single_class_dominated"] = [
        f"{fn} ({cls}, {cnt} boxes)" for fn, cls, cnt in dominated[:top_n]
    ]

    return results
