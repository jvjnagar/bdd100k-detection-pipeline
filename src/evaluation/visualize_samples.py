import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.data.parser import DETECTION_CLASSES, ImageAnnotation, load_split

# Color palette for the 10 detection classes (BGR for cv2, RGB for matplotlib)
CLASS_COLORS = {
    "car": (0, 114, 189),
    "truck": (217, 83, 25),
    "bus": (237, 177, 32),
    "train": (126, 47, 142),
    "person": (119, 172, 48),
    "rider": (77, 190, 238),
    "bike": (162, 20, 47),
    "motor": (255, 153, 0),
    "traffic light": (76, 189, 237),
    "traffic sign": (255, 99, 71),
}

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "bdd100k" / "images" / "100k" / "train"
OUTPUT_DIR = Path("output") / "interesting_samples"


def draw_bboxes_on_image(
    image: np.ndarray, annotation: ImageAnnotation
) -> np.ndarray:
    """Draw bounding boxes with class labels on an image."""
    img = image.copy()
    for bbox in annotation.bboxes:
        color = CLASS_COLORS.get(bbox.category, (200, 200, 200))
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = bbox.category
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def load_image(filename: str) -> np.ndarray:
    """Load an image by its annotation filename."""
    img_path = IMAGE_DIR / f"{filename}.jpg"
    if not img_path.exists():
        # Return blank placeholder
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    img = cv2.imread(str(img_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_grid_visualization(
    annotations_dict: Dict[str, ImageAnnotation],
    filenames: List[str],
    title: str,
    output_path: Path,
    max_images: int = 6,
) -> None:
    """Create a grid of annotated images and save to file."""
    filenames = [f for f in filenames if f in annotations_dict][:max_images]
    if not filenames:
        print(f"  Skipping '{title}' - no matching images found")
        return

    n = len(filenames)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, fname in enumerate(filenames):
        r, c = divmod(idx, cols)
        ann = annotations_dict[fname]
        img = load_image(fname)
        img_annotated = draw_bboxes_on_image(img, ann)

        axes[r, c].imshow(img_annotated)
        class_counts = Counter(b.category for b in ann.bboxes)
        subtitle = ", ".join(f"{cls}:{cnt}" for cls, cnt in class_counts.most_common(4))
        axes[r, c].set_title(f"{fname}\n[{subtitle}]", fontsize=8)
        axes[r, c].axis("off")

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    # Legend
    patches = [mpatches.Patch(color=np.array(c) / 255.0, label=cls)
               for cls, c in CLASS_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def find_per_class_unique_samples(
    annotations: List[ImageAnnotation], top_n: int = 6
) -> Dict[str, List[str]]:
    """Find images with the highest instance count for each class."""
    per_class: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for ann in annotations:
        class_counts = Counter(b.category for b in ann.bboxes)
        for cls, count in class_counts.items():
            per_class[cls].append((count, ann.filename))

    results = {}
    for cls in DETECTION_CLASSES:
        sorted_imgs = sorted(per_class.get(cls, []), reverse=True)
        results[cls] = [fname for _, fname in sorted_imgs[:top_n]]
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training annotations...")
    train_anns = load_split(str(DATA_DIR), "train")
    if train_anns is None:
        print("ERROR: Could not load training data")
        sys.exit(1)

    # Build lookup dict
    ann_dict = {ann.filename: ann for ann in train_anns}
    print(f"Loaded {len(train_anns)} images\n")

    # === 1. Most annotated images ===
    print("[1/6] Most annotated images (highest bbox count)...")
    sorted_by_count = sorted(train_anns, key=lambda a: len(a.bboxes), reverse=True)
    create_grid_visualization(
        ann_dict,
        [a.filename for a in sorted_by_count[:6]],
        "Most Annotated Images (Highest Bounding Box Count)",
        OUTPUT_DIR / "most_annotated.png",
    )

    # === 2. Most diverse images ===
    print("[2/6] Most diverse images (most class variety)...")
    sorted_by_diversity = sorted(
        train_anns,
        key=lambda a: len(set(b.category for b in a.bboxes)),
        reverse=True,
    )
    create_grid_visualization(
        ann_dict,
        [a.filename for a in sorted_by_diversity[:6]],
        "Most Diverse Images (Most Different Classes Present)",
        OUTPUT_DIR / "most_diverse.png",
    )

    # === 3. Rare class samples ===
    print("[3/6] Rare class samples (train, rider, motor)...")
    for rare_cls in ["train", "rider", "motor"]:
        imgs_with_rare = [
            ann for ann in train_anns
            if any(b.category == rare_cls for b in ann.bboxes)
        ]
        # Sort by count of the rare class in descending order
        imgs_with_rare.sort(
            key=lambda a: sum(1 for b in a.bboxes if b.category == rare_cls),
            reverse=True,
        )
        create_grid_visualization(
            ann_dict,
            [a.filename for a in imgs_with_rare[:6]],
            f"Rare Class: '{rare_cls}' — Images with Most Instances",
            OUTPUT_DIR / f"rare_class_{rare_cls}.png",
        )

    # === 4. Single-class dominated images ===
    print("[4/6] Single-class dominated images (>80% one class)...")
    dominated = []
    for ann in train_anns:
        if len(ann.bboxes) >= 10:
            class_counts = Counter(b.category for b in ann.bboxes)
            top_cls, top_cnt = class_counts.most_common(1)[0]
            ratio = top_cnt / len(ann.bboxes)
            if ratio > 0.85:
                dominated.append((ann.filename, top_cls, len(ann.bboxes), ratio))
    dominated.sort(key=lambda x: x[2], reverse=True)
    create_grid_visualization(
        ann_dict,
        [fn for fn, _, _, _ in dominated[:6]],
        "Single-Class Dominated Images (>85% one class, ≥10 boxes)",
        OUTPUT_DIR / "single_class_dominated.png",
    )

    # === 5. Per-class top samples ===
    print("[5/6] Per-class unique samples (highest instance count)...")
    per_class = find_per_class_unique_samples(train_anns, top_n=6)
    for cls, filenames in per_class.items():
        safe_name = cls.replace(" ", "_")
        create_grid_visualization(
            ann_dict,
            filenames,
            f"Class '{cls}' — Images with Most Instances",
            OUTPUT_DIR / f"top_samples_{safe_name}.png",
        )

    # === 6. Summary statistics figure ===
    print("[6/6] Creating summary statistics figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Interesting Sample Statistics", fontsize=14, fontweight="bold")

    # Per-class: how many images contain each class
    class_image_counts = defaultdict(int)
    for ann in train_anns:
        for cls in set(b.category for b in ann.bboxes):
            class_image_counts[cls] += 1
    classes = DETECTION_CLASSES
    img_counts = [class_image_counts.get(c, 0) for c in classes]
    axes[0, 0].barh(classes, img_counts, color=[np.array(CLASS_COLORS[c]) / 255 for c in classes])
    axes[0, 0].set_xlabel("Number of Images")
    axes[0, 0].set_title("Images Containing Each Class")

    # Distribution of class diversity per image
    diversities = [len(set(b.category for b in ann.bboxes)) for ann in train_anns]
    axes[0, 1].hist(diversities, bins=range(1, 12), edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Number of Distinct Classes")
    axes[0, 1].set_ylabel("Number of Images")
    axes[0, 1].set_title("Class Diversity per Image")

    # Max instances of each class in a single image
    max_per_class = {}
    for ann in train_anns:
        cc = Counter(b.category for b in ann.bboxes)
        for cls, cnt in cc.items():
            max_per_class[cls] = max(max_per_class.get(cls, 0), cnt)
    max_counts = [max_per_class.get(c, 0) for c in classes]
    axes[1, 0].barh(classes, max_counts, color=[np.array(CLASS_COLORS[c]) / 255 for c in classes])
    axes[1, 0].set_xlabel("Max Instances in Single Image")
    axes[1, 0].set_title("Maximum Per-Image Instance Count")

    # Dominated images breakdown
    dom_by_class = defaultdict(int)
    for fn, cls, cnt, ratio in dominated:
        dom_by_class[cls] += 1
    dom_classes = list(dom_by_class.keys())
    dom_counts = [dom_by_class[c] for c in dom_classes]
    axes[1, 1].bar(dom_classes, dom_counts, color=[np.array(CLASS_COLORS.get(c, (150, 150, 150))) / 255 for c in dom_classes])
    axes[1, 1].set_ylabel("Number of Images")
    axes[1, 1].set_title("Single-Class Dominated Images by Class")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'summary_statistics.png'}")

    print(f"\nDone! All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
